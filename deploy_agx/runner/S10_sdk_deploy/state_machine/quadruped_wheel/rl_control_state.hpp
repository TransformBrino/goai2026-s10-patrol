/**
 * @file rl_control_state.hpp
 * @brief rl policy runnning state for quadruped-wheel robot
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once
#include "state_base.h"
#include "policy_runner_base.hpp"
#include "s10_policy_runner.hpp"
#include "robot_interface.h"
#include "user_command_interface.h"
#include "json.hpp"
#include "basic_function.hpp"

namespace qw {
    class RLControlState : public StateBase {
    private:
        RobotBasicState rbs_[2];
        std::atomic<int> rbs_write_index_{0};
        int getrbsReadIndex() const { return 1 - rbs_write_index_.load(std::memory_order_acquire); }

        int state_run_cnt_;

        std::shared_ptr<PolicyRunnerBase> policy_ptr_;
        std::shared_ptr<S10PolicyRunner> s10_policy_;
        // 可选的专家策略（S10_POLICY_CLIMB），以及当前实际生效的那份。
        // 不设置环境变量时 s10_policy_climb_ 为空，active_policy_ 恒等于
        // s10_policy_，整条路径与改动前逐字节一致。
        std::shared_ptr<S10PolicyRunner> s10_policy_climb_;
        // 第三份「坑内」策略（S10_POLICY_PIT），reserved_scale > 1.5 时接管。
        // 为什么要它：坑里的重试要倒车，而专家策略不会倒车，所以导航层在
        // 重试期间会切回通用策略。实测 ab_best（通用=出厂盲策略）在 wp15
        // 翻越 8/15，而通用换成 v6 后 0/3 —— 决定成败的正是这几十秒倒车摆位。
        // 这一槽让「全程用带感知的 v6」和「坑里那几十秒用出厂」可以并存。
        // 留空则完全不启用，行为与原来逐字节一致。
        std::shared_ptr<S10PolicyRunner> s10_policy_pit_;
        bool pit_missing_warned_ = false;   // 0911f：楼梯槽没加载时收到 2 的警告，每次按下只打一次

        // 高程图收发计数。缺帧率要能看见 —— 静默地一直用兜底值，
        // 等于让一个按"看得见地形"训出来的策略全程当自己在平地上。
        long hscan_ok_ = 0, hscan_miss_ = 0;
        std::shared_ptr<S10PolicyRunner> active_policy_;

        std::thread run_policy_thread_;
        bool start_flag_ = true;

        float policy_cost_time_ = 1;
        // 0911g：RL 状态的指令发布/策略周期停顿监视。09-11 真机 AGX 上 DDS publish 内部阻塞过 333 ms（Codex 复现）。
        // 默认只告警、计数：RL 策略每拍按当前观测重算，停顿后不会像起立轨迹那样一步追赶；而在楼梯上锁阻尼 = 原地瘫下，更危险。
        // S10_RL_STALL_MS 阈值（默认 50）；S10_RL_STALL_DAMP=1 时超过阈值就锁存阻尼（safe_control_mode=2）。
        const double rl_stall_s_ = [](){ const char* s = std::getenv("S10_RL_STALL_MS"); const double v = (s && *s) ? std::atof(s) : 50.;
                                          return (std::isfinite(v) && v >= 10. && v <= 2000.) ? v / 1000. : .05; }();
        const bool rl_stall_damp_ = [](){ const char* s = std::getenv("S10_RL_STALL_DAMP"); return s && std::string(s) == "1"; }();
        // 测试开关（默认关）：S10_TEST_RL_STALL_MS=N → 进 RL 3 s 后在下发前阻塞 N ms 一次，用来验证上面的监视。真机不要设。
        const double test_rl_stall_s_ = [](){ const char* s = std::getenv("S10_TEST_RL_STALL_MS"); const double v = (s && *s) ? std::atof(s) : 0.;
                                               return (std::isfinite(v) && v > 0. && v <= 2000.) ? v / 1000. : 0.; }();
        bool test_rl_stall_done_ = false;
        std::chrono::steady_clock::time_point rl_last_pub_{}, rl_entered_{}, rl_stall_print_{};
        long rl_stall_n_ = 0;
        void RlStall(const char* what, double ms) {
            ++rl_stall_n_;
            const auto now = std::chrono::steady_clock::now();
            if (rl_stall_print_ == std::chrono::steady_clock::time_point{} || now - rl_stall_print_ > std::chrono::seconds(1)) {
                rl_stall_print_ = now;
                std::cout << "!! [RL_STALL] " << what << " " << ms << " ms（阈值 " << rl_stall_s_ * 1e3 << " ms，本次 RL 累计 " << rl_stall_n_ << " 次）"
                          << (rl_stall_damp_ ? "，S10_RL_STALL_DAMP=1 → 转阻尼并锁存" : "，只告警（S10_RL_STALL_DAMP=1 才锁阻尼）") << "（0911g）" << std::endl;
            }
            if (rl_stall_damp_) uc_ptr_->GetUserCommand()->safe_control_mode = 2;
        }
        // 0911h：RL 期间关节反馈"停更"检查（作者 09-11 晚定）。状态机按 5 ms 定时器走、不等关节数据：/JOINTS_DATA 断了，
        // 策略照样每拍出指令，只是用的关节位置/速度是冻住的 —— 上面的 RL_STALL 看不到这种情况。这里用本机单调时钟量
        // "关节时间戳多久没变"（不和机器人时钟比：机器人时钟比 AGX 快约 44 s 也不受影响）。
        // S10_RL_FB_STALE_MS 阈值（默认 100 ms，可设 20~2000）；默认只告警、计数；S10_RL_STALL_DAMP=1（与 RL_STALL 同一开关）时
        // 超过阈值就锁存阻尼（safe_control_mode=2）。每次停更打一行（1 s 内最多一行），恢复时再打一行这次停更的总时长。
        const double fb_stale_s_ = [](){ const char* s = std::getenv("S10_RL_FB_STALE_MS"); const double v = (s && *s) ? std::atof(s) : 100.;
                                          return (std::isfinite(v) && v >= 20. && v <= 2000.) ? v / 1000. : .1; }();
        double fb_last_stamp_ = 0.;
        std::chrono::steady_clock::time_point fb_last_change_{}, fb_print_{};
        bool fb_stale_on_ = false, fb_stale_said_ = false;
        long fb_stale_n_ = 0;
        void CheckJointFeedback() {
            const auto now = std::chrono::steady_clock::now();
            const double ts = ri_ptr_->GetInterfaceTimeStamp();
            if (fb_last_change_ == std::chrono::steady_clock::time_point{} || ts != fb_last_stamp_) {
                if (fb_stale_on_ && fb_stale_said_)
                    std::cout << "[RL_FB_STALE] 关节反馈恢复：这次停更共 " << std::chrono::duration<double, std::milli>(now - fb_last_change_).count()
                              << " ms（0911h）" << std::endl;
                fb_stale_on_ = false; fb_last_stamp_ = ts; fb_last_change_ = now;
                return;
            }
            const double age = std::chrono::duration<double>(now - fb_last_change_).count();
            if (fb_stale_on_ || age <= fb_stale_s_) return;
            fb_stale_on_ = true; ++fb_stale_n_;
            fb_stale_said_ = (fb_print_ == std::chrono::steady_clock::time_point{} || now - fb_print_ > std::chrono::seconds(1));
            if (fb_stale_said_) {
                fb_print_ = now;
                std::cout << "!! [RL_FB_STALE] 关节反馈 " << age * 1e3 << " ms 没更新（阈值 " << fb_stale_s_ * 1e3 << " ms，本次 RL 累计 " << fb_stale_n_ << " 次）"
                          << (rl_stall_damp_ ? "，S10_RL_STALL_DAMP=1 → 转阻尼并锁存" : "，只告警（S10_RL_STALL_DAMP=1 才锁阻尼）") << "（0911h）" << std::endl;
            }
            if (rl_stall_damp_) uc_ptr_->GetUserCommand()->safe_control_mode = 2;
        }

        Eigen::MatrixXf acc_rot = Eigen::MatrixXf::Zero(20, 3);
        int acc_rot_count = 0;

        void UpdateRobotObservation() {
            int write_idx = rbs_write_index_.load(std::memory_order_relaxed);
            RobotBasicState& buffer = rbs_[write_idx];

            buffer.base_rpy = ri_ptr_->GetImuRpy();
            buffer.base_rot_mat = RpyToRm(buffer.base_rpy);
            buffer.base_omega = ri_ptr_->GetImuOmega();
            buffer.base_acc = ri_ptr_->GetImuAcc();
            buffer.joint_pos = ri_ptr_->GetJointPosition();
            buffer.joint_vel = ri_ptr_->GetJointVelocity();
            buffer.joint_tau = ri_ptr_->GetJointTorque();

            // 储存
            buffer.flt_base_acc_mat.row(acc_rot_count) = buffer.base_acc.transpose();
            acc_rot_count += 1;
            acc_rot_count = acc_rot_count % 20;

            rbs_write_index_.store(1 - write_idx,  std::memory_order_release);
        }

        void PolicyRunner() {
            int run_cnt_record = -1;
            while (start_flag_) {
                if (state_run_cnt_ % policy_ptr_->decimation_ == 0 && state_run_cnt_ != run_cnt_record) {
                    timespec start_timestamp, end_timestamp;
                    clock_gettime(CLOCK_MONOTONIC, &start_timestamp);

                    const UserCommand *uc = uc_ptr_->GetUserCommand();
                    // 专家策略切换。导航层（知道机器人位置和航点）把标志写进
                    // reserved_scale —— 用这个已有的空闲字段，省得改 UserCommand
                    // 结构体、连带动一堆通信序列化。
                    {
                        const float sel = uc->reserved_scale;
                        auto want = s10_policy_;
                        if (sel > 1.5f) {
                            // 0911f（作者 09-11 拍板）：楼梯槽（S10_POLICY_PIT）没加载时收到 /climb_mode 2，
                            // 留在主策略，不再落到上墙专家（原来的 else-if 会切到 S10_POLICY_CLIMB），并打印醒目警告供控制台显示。
                            if (s10_policy_pit_) want = s10_policy_pit_;
                            else if (!pit_missing_warned_) {
                                std::cout << "!! [CLIMB] 收到楼梯专家请求（/climb_mode 2），但楼梯策略 S10_POLICY_PIT 没加载，留在主策略（0911f）" << std::endl;
                                pit_missing_warned_ = true;
                            }
                        } else {
                            pit_missing_warned_ = false;
                            if (sel > 0.5f && s10_policy_climb_) want = s10_policy_climb_;
                        }
                        if (want != active_policy_) {
                            // 观测里含 last_action。两份策略各自维护历史，直接切
                            // 过去会让观测突然跳变，新策略在一个没见过的状态上起步。
                            // 把实际下发过的动作和步数计数一并交接。
                            want->AdoptFrom(active_policy_->GetLastAction(),
                                            active_policy_->GetRunCnt());
                            std::cout << "[policy] 切换到 "
                                      << (want == s10_policy_climb_ ? "专家"
                                          : want == s10_policy_pit_ ? "坑内" : "通用")
                                      << "策略" << std::endl;
                            active_policy_ = want;
                        }
                    }
                    // 高程图（策略观测第 58~244 维）。每个控制周期取一次新鲜值；
                    // 取不到（感知没起、话题断了、数据过期）就回落到"按平地走"。
                    // 只在策略确实需要它的时候做 —— 57 维盲策略走这里是纯浪费。
                    if (active_policy_->NeedsHeightScan()) {
                        static std::vector<float> hs(S10PolicyRunner::kHeightScanDim);
                        if (uc_ptr_->GetHeightScan(hs.data(), (int)hs.size())) {
                            active_policy_->SetHeightScan(hs.data(), (int)hs.size());
                            ++hscan_ok_;
                        } else {
                            active_policy_->SetHeightScanFallback();
                            ++hscan_miss_;
                        }
                        // 缺帧率是这条链路的健康指标，必须能看见。
                        // 静默地一直用兜底值 = 策略全程当自己在平地上走，
                        // 而它是按"看得见地形"训出来的 —— 那是最坏的一种失效。
                        const long tot = hscan_ok_ + hscan_miss_;
                        if (tot == 50 || tot == 500 || (tot % 5000) == 0) {
                            std::cout << "[高程图] 收到 " << hscan_ok_ << " / 缺 "
                                      << hscan_miss_ << "（缺帧率 "
                                      << (100.0 * hscan_miss_ / tot) << "%）" << std::endl;
                        }
                    }

                    auto ra = active_policy_->getRobotAction(rbs_[getrbsReadIndex()], *uc);
                    
                    MatXf res = ra.ConvertToMat();

                    {   // 0911g：发布与周期停顿监视（见成员说明）
                        const auto t0 = std::chrono::steady_clock::now();
                        if (rl_last_pub_ != std::chrono::steady_clock::time_point{}) {
                            const double per = std::chrono::duration<double>(t0 - rl_last_pub_).count();
                            if (per > rl_stall_s_ + 0.02) RlStall("策略周期（相邻两次下发间隔）", per * 1e3);
                        }
                        if (test_rl_stall_s_ > 0. && !test_rl_stall_done_ && t0 - rl_entered_ > std::chrono::seconds(3)) {
                            test_rl_stall_done_ = true;   // 测试开关：模拟 publish 内部阻塞（只发生一次）
                            std::this_thread::sleep_for(std::chrono::duration<double>(test_rl_stall_s_));
                        }
                        ri_ptr_->SetJointCommand(res);
                        const auto t1 = std::chrono::steady_clock::now();
                        const double pub = std::chrono::duration<double>(t1 - t0).count();
                        if (pub > rl_stall_s_) RlStall("指令发布阻塞", pub * 1e3);
                        rl_last_pub_ = t1;
                    }
                    run_cnt_record = state_run_cnt_;
                    clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
                    policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                                        + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;

                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

    public:
        RLControlState(const RobotName &robot_name, const std::string &state_name,
                       std::shared_ptr<ControllerData> data_ptr) : StateBase(robot_name, state_name, data_ptr) {
            if (robot_name_ == RobotName::S10) {
                namespace fs = std::filesystem;
                fs::path base = fs::path(__FILE__).parent_path();
                // 策略文件路径可由 S10_POLICY_PATH 覆盖，便于 A/B 对比微调产物，
                // 不用来回改动官方的 policy.onnx。留空即用原厂那份。
                std::string model_path;
                if (const char* p = std::getenv("S10_POLICY_PATH"); p && *p) {
                    model_path = p;
                    std::cout << "[policy] 使用外部策略 " << model_path << std::endl;
                } else {
                    model_path = fs::canonical(base / ".." / ".." / "policy" / "policy.onnx").string();
                    std::cout << "[policy] 使用原厂策略 " << model_path << std::endl;
                }
                s10_policy_ = std::make_shared<S10PolicyRunner>("s10_policy", model_path);

                // 可选的第二份「专家」策略，只在赛道那道 37.7cm 坎前接管。
                // 为什么要两份：盲策略的观测里没有外部感知，分不清「面前是 38cm
                // 墙」和「面前是普通地形」，而翻墙需要的抬头前伸在别处就是摔倒
                // 动作。实测过三种折中（全锚定/全放开/0.25 倍锚定），分别得到
                // 「爬不动」「通用能力全毁（赛道 14/33 -> 1/33）」「两者间振荡」
                // —— 这是结构性矛盾，不是调参能解决的。而那道坎的位置是已知的，
                // 所以按位置切换比强求一个网络两头都要更靠谱。
                // 留空则完全不启用，行为与原来逐字节一致。
                if (const char* q = std::getenv("S10_POLICY_CLIMB"); q && *q) {
                    s10_policy_climb_ = std::make_shared<S10PolicyRunner>("s10_climb", q);
                    std::cout << "[policy] 已加载专家策略 " << q
                              << "（由 UserCommand.reserved_scale > 0.5 触发）" << std::endl;
                }
                if (const char* q = std::getenv("S10_POLICY_PIT"); q && *q) {
                    s10_policy_pit_ = std::make_shared<S10PolicyRunner>("s10_pit", q);
                    std::cout << "[policy] 已加载坑内策略 " << q
                              << "（由 UserCommand.reserved_scale > 1.5 触发）" << std::endl;
                }
            }

            policy_ptr_ = s10_policy_;
            active_policy_ = s10_policy_;      // 起步永远用通用策略
            if (!policy_ptr_) {
                std::cerr << "error policy" << std::endl;
                exit(0);
            }
            policy_ptr_->DisplayPolicyInfo();
        }

        ~RLControlState() {}

        virtual void OnEnter() {
            state_run_cnt_ = -1;
            rl_last_pub_ = std::chrono::steady_clock::time_point{}; rl_entered_ = std::chrono::steady_clock::now(); rl_stall_n_ = 0;   // 0911g
            fb_last_change_ = std::chrono::steady_clock::time_point{}; fb_stale_on_ = false; fb_stale_said_ = false; fb_stale_n_ = 0;   // 0911h
            std::cout << "[RL_FB_STALE] 开：关节反馈停更阈值 " << fb_stale_s_ * 1e3 << " ms，"
                      << (rl_stall_damp_ ? "超过就转阻尼并锁存（S10_RL_STALL_DAMP=1）" : "只告警、计数（S10_RL_STALL_DAMP=1 才锁阻尼）") << "（0911h）" << std::endl;
            start_flag_ = true;
            run_policy_thread_ = std::thread(std::bind(&RLControlState::PolicyRunner, this));
            policy_ptr_->OnEnter();
            // 专家那份也要初始化：它可能在本次状态里被切进来，
            // 带着上一次状态残留的内部量启动会立刻发散。
            if (s10_policy_climb_) s10_policy_climb_->OnEnter();
            if (s10_policy_pit_)   s10_policy_pit_->OnEnter();
            active_policy_ = s10_policy_;
            pit_missing_warned_ = false;   // 0911f：重新进 RL 时若仍按着 2，警告会再打一次
            StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        };

        virtual void OnExit() {
            start_flag_ = false;
            run_policy_thread_.join();
            state_run_cnt_ = -1;
        }

        virtual void Run() {
            UpdateRobotObservation();
            CheckJointFeedback();   // 0911h
            state_run_cnt_++;
        }

        virtual bool LoseControlJudge() {
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
            return PostureUnsafeCheck();
        }

        bool PostureUnsafeCheck() {
            // Vec3f rpy = ri_ptr_->GetImuRpy();
            // if(rpy(0) > 30./180*M_PI || rpy(1) > 45./180*M_PI){
            //     std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            //     return true;
            // }
            return false;
        }

        virtual StateName GetNextStateName() {
            if (uc_ptr_->GetUserCommand()->safe_control_mode != 0) 
                return StateName::kJointDamping;
            if (uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown))
                return StateName::kLieDown;
            
            return StateName::kRLControl;
        }
    };
};