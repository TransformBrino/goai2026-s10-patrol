/**
 * @file ros_command_interface.hpp
 * @brief 通过 ROS 话题下发速度指令与状态切换，替代 KeyboardInterface。
 *
 * 加这个接口有三个原因：
 *   1. WSL / 无头机器上没有可用的 tty 或 /dev/input，键盘那条路走不通；
 *   2. 自动化评测需要脚本化地发指令，人按键盘没法复现；
 *   3. 这正是感知/导航组要接进来的位置——他们发 /cmd_vel，运控这边只认这个话题。
 *
 * 话题：
 *   /cmd_vel     geometry_msgs/Twist   linear.x / linear.y / angular.z
 *   /robot_mode  std_msgs/UInt8        目标状态，沿用 RobotMotionState 编码
 *                                      (1=StandingUp, 2=JointDamping, 4=LieDown, 6=RLControl)
 *
 * 默认行为与 KeyboardInterface 逐位对齐（上限 1.0/0.6/1.0、不限斜率），
 * 这样测出来的基线才是"原厂"基线。所有增强都通过环境变量显式打开：
 *
 *   S10_CMD_MAX_VX      前进速度上限   默认 1.0（键盘里写死的值）
 *   S10_CMD_MAX_VY      侧移速度上限   默认 0.6
 *   S10_CMD_MAX_WZ      偏航角速度上限 默认 1.0
 *   S10_CMD_SLEW_V      线速度斜率限制 m/s^2，<=0 表示不限（默认，等同键盘的阶跃）
 *   S10_CMD_SLEW_W      角速度斜率限制 rad/s^2，同上
 *   S10_CMD_TIMEOUT_MS  看门狗超时，超时后指令归零。默认 500ms，<=0 关闭
 *
 * @author  运控组
 * @date    2026-08-02
 */
#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include "basic_function.hpp"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

using namespace interface;
using namespace types;

class RosCommandInterface : public UserCommandInterface {
private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr mode_sub_;
    /// /climb_mode：1 = 切到专家策略翻越高坎，0 = 用通用策略。
    /// 导航层知道机器人位置和航点，由它在接近那道 37.7cm 坎时置位。
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr climb_sub_;
    /// /height_scan：187 维高程图，策略观测的第 58~244 维。
    /// 由感知侧（订阅雷达点云 + 位姿、跑累积局部地图）算好后发过来。
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr hscan_sub_;
    rclcpp::executors::SingleThreadedExecutor exec_;

    std::thread spin_thread_, shape_thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex m_;
    float in_vx_ = 0.f, in_vy_ = 0.f, in_wz_ = 0.f;  ///< 最近一次收到的原始指令
    double last_cmd_ms_ = -1.0;                      ///< 最近一次收到 /cmd_vel 的时刻

    // 高程图缓冲。和 /cmd_vel 分开加锁 —— 187 个 float 的拷贝比指令长得多，
    // 共用一把锁会让 ShapeLoop（200Hz 整形线程）跟着一起等。
    static constexpr int kHeightScanDim = 187;
    mutable std::mutex hscan_m_;
    std::array<float, kHeightScanDim> hscan_{};
    double hscan_ms_ = -1.0;                         ///< 最近一帧的到达时刻
    long   hscan_n_ = 0;
    /// 超过这个年龄就当没有。感知 50Hz（20ms 一帧），给 5 倍余量。
    /// 宁可回落到"按平地走"，也不能拿几百毫秒前的地形做决策 ——
    /// 1 m/s 时 200ms 就是 20cm，整整两格。
    double hscan_max_age_ms_ = 100.0;

    float out_vx_ = 0.f, out_vy_ = 0.f, out_wz_ = 0.f;  ///< 限幅+限斜率后真正下发的
    std::atomic<float> climb_flag_{0.f};                ///< 经 reserved_scale 传给状态机
    /// 相位时钟：进入专家区的时刻（ms）。<0 表示未激活。
    /// 翻越是多秒时序动作，而盲策略每帧看到的状态相似却要输出序列的不同段。
    /// 载体借用 side_vel_scale（观测第 7 维）—— 它只进观测、不驱动控制，
    /// 本项目全程置零，是个免费通道，不必改网络结构或观测维度。
    /// 由 S10_CLIMB_CLOCK=1 开启；专家策略必须是带 --phase-clock 训出来的，
    /// 否则这一维会给它一个训练时没见过的信号。
    std::atomic<double> climb_t0_{-1.0};
    bool climb_clock_ = false;

    // 可调参数
    float max_vx_, max_vy_, max_wz_;
    float slew_v_, slew_w_;
    double timeout_ms_;

    static float EnvF(const char *key, float dft) {
        const char *v = std::getenv(key);
        if (!v || *v == '\0') return dft;
        return static_cast<float>(std::atof(v));
    }

    static double NowMs() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /// 朝 target 逼近，每步最多走 max_delta。max_delta<=0 表示直接跳变。
    static float Approach(float cur, float target, float max_delta) {
        if (max_delta <= 0.f) return target;
        float d = target - cur;
        if (d > max_delta) d = max_delta;
        else if (d < -max_delta) d = -max_delta;
        return cur + d;
    }

    void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(m_);
        in_vx_ = LimitNumber(static_cast<float>(msg->linear.x), max_vx_);
        in_vy_ = LimitNumber(static_cast<float>(msg->linear.y), max_vy_);
        in_wz_ = LimitNumber(static_cast<float>(msg->angular.z), max_wz_);
        last_cmd_ms_ = NowMs();
    }

    /// 策略选择开关。只置一个标志，真正的切换（含 last_action 交接）在
    /// RLControlState::PolicyRunner 里做。
    ///   0 = 通用策略   1 = 专家策略（翻坎）   2 = 坑内策略（S10_POLICY_PIT）
    /// 之前这里写死 `msg->data ? 1 : 0`，把 2 也压成了 1；现在原值透传。
    void OnClimb(const std_msgs::msg::UInt8::SharedPtr msg) {
        const float f = static_cast<float>(msg->data);
        if (climb_flag_.exchange(f) != f) {
            // 相位从**进入专家区那一刻**起算 —— 训练侧也是从"离墙 2.5m"起算，
            // 两边起点必须一致，否则学到的时序对不上部署。
            // 只有 1（专家）带相位；2 是普通盲策略，不需要时序对齐。
            climb_t0_.store((f > 0.5f && f < 1.5f) ? NowMs() : -1.0);
            const char* name = (f > 1.5f) ? "坑内" : (f > 0.5f ? "专家" : "通用");
            RCLCPP_INFO(node_->get_logger(), "[CLIMB] 切到%s策略%s", name,
                        climb_clock_ ? "（相位时钟已开）" : "");
        }
    }

    /// 高程图。存一份带时间戳的副本，取用时判新鲜度。
    /// 不做插值、不做滤波 —— 训练侧喂给策略的就是原始栅格值，
    /// 部署侧多做一步平滑就是引入训练时没有的分布差异。
    void OnHeightScan(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        const int n = static_cast<int>(msg->data.size());
        if (n != kHeightScanDim) {
            static long bad = 0;
            if ((bad++ % 100) == 0)
                RCLCPP_WARN(node_->get_logger(),
                            "/height_scan 长度 %d ≠ %d，丢弃（第 %ld 次）",
                            n, kHeightScanDim, bad);
            return;
        }
        std::lock_guard<std::mutex> lk(hscan_m_);
        std::copy(msg->data.begin(), msg->data.end(), hscan_.begin());
        hscan_ms_ = NowMs();
        ++hscan_n_;
    }

    /**
     * 状态切换。这里刻意复刻 KeyboardInterface::process_mode_command 的门条件，
     * 不放宽——放宽了就不是同一个状态机行为，基线对比会失真。
     */
    void OnMode(const std_msgs::msg::UInt8::SharedPtr msg) {
        const uint8_t want = msg->data;
        const uint8_t cur = msfb_ ? msfb_->GetCurrentState() : 0;

        if (want == uint8_t(RobotMotionState::JointDamping)) {
            usr_cmd_->target_mode = want;
            RCLCPP_INFO(node_->get_logger(), "[MODE] Joint Damping");
        } else if (want == uint8_t(RobotMotionState::StandingUp) &&
                   (cur == uint8_t(RobotMotionState::WaitingForStand) ||
                    cur == uint8_t(RobotMotionState::LieDown))) {
            usr_cmd_->target_mode = want;
            RCLCPP_INFO(node_->get_logger(), "[MODE] Standing Up");
        } else if (want == uint8_t(RobotMotionState::RLControlMode) &&
                   cur == uint8_t(RobotMotionState::StandingUp)) {
            usr_cmd_->target_mode = want;
            RCLCPP_INFO(node_->get_logger(), "[MODE] RL Control");
        } else if (want == uint8_t(RobotMotionState::LieDown) &&
                   (cur == uint8_t(RobotMotionState::StandingUp) ||
                    cur == uint8_t(RobotMotionState::RLControlMode))) {
            usr_cmd_->target_mode = want;
            RCLCPP_INFO(node_->get_logger(), "[MODE] Lie Down");
        } else {
            RCLCPP_WARN(node_->get_logger(),
                        "[MODE] 拒绝切到 %u：当前状态 %u 不允许该跳转", want, cur);
        }
    }

    /// 200 Hz：看门狗 + 限斜率，然后写进 usr_cmd_。频率与状态机主循环一致。
    void ShapeLoop() {
        const double dt = 0.005;
        const auto period = std::chrono::microseconds(5000);
        auto next = std::chrono::steady_clock::now();

        while (running_) {
            float tvx, tvy, twz;
            bool expert_dropped = false;
            {
                std::lock_guard<std::mutex> lk(m_);
                bool stale = timeout_ms_ > 0.0 &&
                             (last_cmd_ms_ < 0.0 || (NowMs() - last_cmd_ms_) > timeout_ms_);
                tvx = stale ? 0.f : in_vx_;
                tvy = stale ? 0.f : in_vy_;
                twz = stale ? 0.f : in_wz_;
                // 指令断了就撤专家、回通用策略，并锁存（要回专家必须重新收到 /climb_mode=1）。
                // 专家训练时 rel_zero_vel_envs = rel_standing_envs = 0，从没见过零指令，
                // 零指令会原地打转；上面三行恰好把指令归零 —— 只归零不撤专家，等于看门狗
                // 亲手把狗送进最危险的状态。通用策略训练含 20% 零指令回合，零指令能站住。
                if (stale && climb_flag_.load() > 0.5f) {
                    climb_flag_.store(0.f);
                    climb_t0_.store(-1.0);
                    expert_dropped = true;
                }
            }
            if (expert_dropped) {
                RCLCPP_WARN(node_->get_logger(),
                            "[CLIMB] 指令超时 %.0f ms，撤专家回通用策略（锁存，需重新发 /climb_mode=1）",
                            timeout_ms_);
            }

            out_vx_ = Approach(out_vx_, tvx, slew_v_ * dt);
            out_vy_ = Approach(out_vy_, tvy, slew_v_ * dt);
            out_wz_ = Approach(out_wz_, twz, slew_w_ * dt);

            usr_cmd_->forward_vel_scale  = out_vx_;
            // 相位时钟开启且正在专家区时，这一维载的是相位而不是侧向速度。
            //
            // ⚠ 两者互斥，不能同时用。原注释写的"本项目从不发侧向速度指令
            // （out_vy_ 恒为 0），所以不冲突"**已经不成立** —— eval_driver 的
            // 卡死恢复动作用 vy 做横向纠偏（机体不转、纵横解耦，用转向纠偏
            // 试过两版都把机器人掀翻了）。开了 S10_CLIMB_CLOCK 的话，
            // 横向纠偏会**静默失效**，后撤永远退不回助跑起点。
            if (climb_clock_) {
                const double t0 = climb_t0_.load();
                usr_cmd_->side_vel_scale =
                    (t0 < 0.0) ? 0.f
                               : float(std::min((NowMs() - t0) / 4000.0, 1.5));
            } else {
                usr_cmd_->side_vel_scale = out_vy_;
            }
            usr_cmd_->turnning_vel_scale = out_wz_;
            // 借用结构体里本来就空着的 reserved_scale 传专家策略开关，
            // 免得改 UserCommand 定义、连带动通信序列化那一串。
            usr_cmd_->reserved_scale     = climb_flag_.load();
            usr_cmd_->time_stamp         = NowMs() / 1000.0;

            next += period;
            std::this_thread::sleep_until(next);
        }
    }

public:
    explicit RosCommandInterface(RobotName robot_name) : UserCommandInterface(robot_name) {
        max_vx_     = EnvF("S10_CMD_MAX_VX", 1.0f);
        max_vy_     = EnvF("S10_CMD_MAX_VY", 0.6f);
        max_wz_     = EnvF("S10_CMD_MAX_WZ", 1.0f);
        slew_v_     = EnvF("S10_CMD_SLEW_V", 0.0f);
        slew_w_     = EnvF("S10_CMD_SLEW_W", 0.0f);
        timeout_ms_ = EnvF("S10_CMD_TIMEOUT_MS", 500.0f);
        climb_clock_ = EnvF("S10_CLIMB_CLOCK", 0.0f) > 0.5f;
    }

    ~RosCommandInterface() override { Stop(); }

    void Start() override {
        if (running_) return;
        running_ = true;

        node_ = std::make_shared<rclcpp::Node>("s10_user_command");
        cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&RosCommandInterface::OnCmdVel, this, std::placeholders::_1));
        mode_sub_ = node_->create_subscription<std_msgs::msg::UInt8>(
            "/robot_mode", 10,
            std::bind(&RosCommandInterface::OnMode, this, std::placeholders::_1));
        climb_sub_ = node_->create_subscription<std_msgs::msg::UInt8>(
            "/climb_mode", 10,
            std::bind(&RosCommandInterface::OnClimb, this, std::placeholders::_1));
        // 高程图。感知侧 50Hz 发，策略 50Hz 取，队列深 2 就够 ——
        // 深队列在这里是负担：宁可丢旧帧也不能把过期地形排到策略眼前。
        hscan_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/height_scan", 2,
            std::bind(&RosCommandInterface::OnHeightScan, this, std::placeholders::_1));

        exec_.add_node(node_);
        spin_thread_  = std::thread([this] { exec_.spin(); });
        shape_thread_ = std::thread(&RosCommandInterface::ShapeLoop, this);

        RCLCPP_INFO(node_->get_logger(),
                    "ROS 指令接口已启动 | 上限 vx=%.2f vy=%.2f wz=%.2f | 斜率 v=%.1f w=%.1f | 看门狗 %.0fms",
                    max_vx_, max_vy_, max_wz_, slew_v_, slew_w_, timeout_ms_);
    }

    /// 取一帧新鲜的高程图。太旧或从没收到就返回 false，让调用方用兜底值。
    /// **过期不返回旧值**：感知挂了之后拿着几百毫秒前的地形做决策，
    /// 比明确地"没有地形信息、按平地走"更危险 —— 前者策略会自信地
    /// 去跨一个已经不在那里的坎。
    bool GetHeightScan(float *out, int len) const override {
        if (out == nullptr || len != kHeightScanDim) return false;
        std::lock_guard<std::mutex> lk(hscan_m_);
        if (hscan_ms_ < 0.0) return false;                       // 从没收到过
        if (NowMs() - hscan_ms_ > hscan_max_age_ms_) return false;  // 过期
        std::copy(hscan_.begin(), hscan_.end(), out);
        return true;
    }

    void Stop() override {
        if (!running_) return;
        running_ = false;
        exec_.cancel();
        if (spin_thread_.joinable()) spin_thread_.join();
        if (shape_thread_.joinable()) shape_thread_.join();
        exec_.remove_node(node_);
    }

    UserCommand *GetUserCommand() override { return usr_cmd_; }
};
