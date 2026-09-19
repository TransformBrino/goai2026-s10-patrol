/**
 * @file dds_hardware_interface.hpp
 * @brief hardware interface for dds
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */

#pragma once

#include <cstdlib>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>


#include "common_types.h"
#include "robot_interface.h"
#include "dds_types.h"

#include "drdds/msg/joints_data.hpp"
#include "drdds/msg/joints_data_cmd.hpp"
#include "drdds/msg/imu_data.hpp"
#include "drdds/msg/battery_data.hpp"

using namespace dds;

struct JointConfig {
    float dir;
    float offset;
};


class DdsInterface : public RobotInterface{
protected:
    double ri_ts_ = 0.0, imu_ts_ = 0.0;
    Vec3f omega_body_, rpy_, acc_;
    VecXf joint_pos_, joint_vel_, joint_tau_;
    VecXf motor_temperture_, driver_temperture_;
    float *pos_offset_;
    float *joint_dir_;
    bool *data_updated_;
    uint16_t *control_word;

    JointConfig *joint_config_;
    std::vector<uint16_t> driver_status_;
    std::vector<uint16_t> joint_data_id_;
    BatteryInfo_dds battery_info_[2];
    std::vector<uint16_t> battery_data_;

    double time_stamp_;
    float remain_bat;

    rclcpp::Publisher<drdds::msg::JointsDataCmd>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<drdds::msg::JointsData>::SharedPtr joint_data_sub_;
    rclcpp::Subscription<drdds::msg::ImuData>::SharedPtr imu_data_sub_;
    rclcpp::Subscription<drdds::msg::BatteryData>::SharedPtr health_data_sub_;


    bool IsDataUpdatedFinished() {
        bool res = data_updated_[0];
        for (int i = 0; i < dof_num_; ++i) {
            res = res & data_updated_[i];
        }
        return res;
    }

    void ResetPositionOffset() {
        memset(data_updated_, 0, dof_num_ * sizeof(bool));
        this->SetJointCommand(MatXf::Zero(dof_num_, 5));
        usleep(100 * 1000);
        VecXf last_joint_pos = this->GetJointPosition();
        VecXf current_joint_pos = this->GetJointPosition();
        int cnt = 0;
        while (!IsDataUpdatedFinished()) {
            ++cnt;
            this->RefreshRobotData();
            rclcpp::spin_some(this->get_node());
            current_joint_pos = this->GetJointPosition();

            for (int i = 0; i < dof_num_; ++i) {
                if (!data_updated_[i] && current_joint_pos(i) != last_joint_pos(i)) {
                    data_updated_[i] = true;
                }
            }
            last_joint_pos = current_joint_pos;
            usleep(1000);
            if (cnt == 10000) {
                std::cout << data_updated_ << std::endl;
                std::cout << "joint data update is not finished\n";
            }
        }
        for (int i = 1; i < dof_num_; i += 4) {
            if (current_joint_pos(i) < -210. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] + 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            } else if (current_joint_pos(i) > 90. / 180. * M_PI) {
                pos_offset_[i] = pos_offset_[i] - 360.;
                std::cout << "joint " << i << " offset is changed to " << pos_offset_[i] << "\n";
            }
        }
        for (int i = 0; i < dof_num_; ++i) {
            joint_config_[i].dir = joint_dir_[i];
            joint_config_[i].offset = Deg2Rad(pos_offset_[i]);
        }
    }

public:
    int run_cnt_ = 0;

    DdsInterface(const std::string &robot_name, const int &dof_num) : RobotInterface(robot_name, dof_num) {
        std::cout << robot_name << " is using Ecan Hardware Interface" << std::endl;
        joint_pos_ = VecXf::Zero(dof_num_);
        joint_vel_ = VecXf::Zero(dof_num_);
        joint_tau_ = VecXf::Zero(dof_num_);
        motor_temperture_ = VecXf::Zero(dof_num_);
        driver_temperture_ = VecXf::Zero(dof_num_);

        pos_offset_ = new float[dof_num_];
        joint_dir_ = new float[dof_num_];
        data_updated_ = new bool[dof_num_];
        joint_config_ = new JointConfig[dof_num_];
        control_word = new uint16_t[4];

        driver_status_.resize(dof_num_);
        joint_data_id_.resize(dof_num_);
        battery_data_.resize(BATTERY_DATA_SIZE);

        joint_cmd_pub_ = node_->create_publisher<drdds::msg::JointsDataCmd>("/JOINTS_CMD", 10);
        joint_data_sub_ = node_->create_subscription<drdds::msg::JointsData>("/JOINTS_DATA", 10,
                                    std::bind(&DdsInterface::Handler, this,std::placeholders::_1));
        // 0911d：机器人 /IMU_DATA 的发布者是 BEST_EFFORT，RELIABLE 订阅与之不兼容、一帧都收不到（09-11 AGX 查实）。
        // BEST_EFFORT 订阅对 RELIABLE / BEST_EFFORT 两种发布者都兼容（真机、仿真通用）。
        imu_data_sub_ = node_->create_subscription<drdds::msg::ImuData>("/IMU_DATA", rclcpp::SensorDataQoS(),
                                    std::bind(&DdsInterface::HandlerIMU, this, std::placeholders::_1));
        health_data_sub_ = node_->create_subscription<drdds::msg::BatteryData>("/BATTERY_DATA", 10,
                                    std::bind(&DdsInterface::HandlerHealth, this, std::placeholders::_1));
        
        sleep(1);

        control_word[0] = kIndexDisable;
        control_word[1] = kIndexErrorReset;
        control_word[2] = kIndexEnable;
        control_word[3] = kIndexGetStatusWord;
        ResetJointError();
    }

    virtual ~DdsInterface() {
        delete[] pos_offset_;
        delete[] joint_dir_;
        delete[] data_updated_;
        delete[] joint_config_;
    }

    virtual void Start() {
    }

    virtual void Stop() {
    }

    virtual double GetInterfaceTimeStamp() {
        return ri_ts_;
    }

    virtual VecXf GetJointPosition() {
        return joint_pos_;
    }

    virtual VecXf GetJointVelocity() {
        return joint_vel_;
    }

    virtual VecXf GetJointTorque() {
        return joint_tau_;
    }

    virtual Vec3f GetImuRpy() {
        return rpy_;
    }

    virtual Vec3f GetImuAcc() {
        return acc_;
    }

    virtual Vec3f GetImuOmega() {
        return omega_body_;
    }

    virtual VecXf GetContactForce() {
        return VecXf::Zero(4);
    }

    virtual void ResetJointError() {
        auto msg = drdds::msg::JointsDataCmd();

        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < dof_num_; ++i) {
                msg.data.joints_data[i].position = 0;
                msg.data.joints_data[i].velocity = 0;
                msg.data.joints_data[i].torque = 0;
                msg.data.joints_data[i].kp = 0;
                msg.data.joints_data[i].kd = 0;
                msg.data.joints_data[i].control_word = control_word[j];
            }
            std::cout << "ResetJointError publish!! \n" << std::endl;
            joint_cmd_pub_->publish(msg);
            usleep(10 * 1000);
        }
    }

    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) {
        joint_cmd_ = input;
        auto msg = drdds::msg::JointsDataCmd();

        for (int i = 0; i < dof_num_; ++i) {
            msg.data.joints_data[i].position =
                    (input(i, 1) - joint_config_[i].offset) * joint_config_[i].dir;
            msg.data.joints_data[i].velocity = input(i, 3) * joint_dir_[i];
            msg.data.joints_data[i].torque = input(i, 4) * joint_dir_[i];
            msg.data.joints_data[i].kp = input(i, 0);
            msg.data.joints_data[i].kd = input(i, 2);
            msg.data.joints_data[i].control_word = kIndexMotorControl;
        }
        std::cout << "SetJointCommand publish!! \n" << std::endl;
        joint_cmd_pub_->publish(msg);
    }

    virtual VecXf GetMotorTemperture() {
        return motor_temperture_;
    }

    virtual VecXf GetDriverTemperture() {
        return driver_temperture_;
    }

    virtual double GetImuTimestamp() {
        return imu_ts_;
    }
    // 0911d：距本机最近一次收下 /IMU_DATA 的秒数（本机单调时钟；从没收到过返回 1e9）。
    virtual double GetImuRxAgeSec() {
        const double t = imu_rx_steady_.load();
        return t < 0.0 ? 1e9 : SteadyNow() - t;
    }

    virtual std::vector<uint16_t> GetBatteryData() {
        return battery_data_;
    }

    virtual std::vector<uint16_t> GetDriverStatusWord() {
        return driver_status_;
    }

    virtual std::vector<uint16_t> GetJointDataID() {
        return joint_data_id_;
    }

    virtual void RefreshRobotData() {
    }

    virtual void Handler(const drdds::msg::JointsData::SharedPtr msg) {
        ++run_cnt_;
        for (int i = 0; i < dof_num_; ++i) {
            joint_pos_(i) = msg->data.joints_data[i].position * joint_dir_[i] + pos_offset_[i] / 180. * M_PI;
            joint_vel_(i) = msg->data.joints_data[i].velocity * joint_dir_[i];
            joint_tau_(i) = msg->data.joints_data[i].torque * joint_dir_[i];
            motor_temperture_(i) = float(msg->data.joints_data[i].motion_temp);
            driver_temperture_(i) = float(msg->data.joints_data[i].driver_temp);
            driver_status_[i] = msg->data.joints_data[i].status_word;
            joint_data_id_[i] = uint16_t(run_cnt_);
        }
        ri_ts_ = rclcpp::Time(msg->header.stamp).seconds();
    }

    // 09-11c：真机 /IMU_DATA 的 roll/pitch/yaw 是【弧度】（REP-103：俯仰正=低头，横滚正=右侧低）。
    // 证据：09-10 真机包里原始横滚对加速度计推出的横滚之比 0.998/1.001（翻车到 -2.43 = -139°）；厂家开发指南
    // 身体姿态角单位 rad；AGX 控制台也按弧度用。厂家 SDK 原写法按"度"读、先 Deg2Rad，真机上把倾角缩小 57.3 倍，
    // 策略的重力投影几乎不随机身倾斜变化。仿真桥原来发"度"与之配对，所以 MuJoCo 里从未暴露。
    // 默认按弧度读；S10_IMU_RPY_UNIT=deg 才按度（只为兼容按度发布的旧仿真）。
    bool imu_rpy_deg_ = [](){ const char* s = std::getenv("S10_IMU_RPY_UNIT"); return s && std::string(s) == "deg"; }();
    bool imu_unit_said_ = false;
    bool imu_chk_done_ = false;
    int imu_chk_n_ = 0;
    double imu_chk_sum_ = 0.0;
    int imu_chk_phase_ = 0;
    double imu_chk_tilt_sum_ = 0.0;
    int imu_chk_streak_ = 0;
    // 0911d：接收统计与去重（见 HandlerIMU 开头）。
    std::atomic<double> imu_rx_steady_{-1.0};
    uint64_t imu_rx_n_ = 0, imu_drop_n_ = 0, imu_stat_n0_ = 0, imu_stat_d0_ = 0;
    double imu_stat_t0_ = -1.0;
    bool imu_rate_said_ = false;
    static double SteadyNow() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void HandlerIMU(const drdds::msg::ImuData::SharedPtr msg) {
        const double rx_now = SteadyNow();
        const double st = rclcpp::Time(msg->header.stamp).seconds();
        {   // 0911d 去重/乱序：AGX 侧 09-11 看到 /IMU_DATA 有两个发布者；09-10 真机包里只见到一路，时间戳严格递增
            // （每帧 +5 ms，但只按实际时间的一半在走，且与 AGX 时钟差约 43 s——所以新鲜度一律按本机接收时刻算）。
            // 时间戳不大于上一帧就丢；若已 0.5 s 没收下任何帧则无条件接收（防机器人时钟跳变后锁死）。时间戳为 0 不去重。
            const double last_rx = imu_rx_steady_.load();
            if (st > 0.0 && imu_ts_ > 0.0 && st <= imu_ts_ && last_rx >= 0.0 && rx_now - last_rx < 0.5) {
                if (imu_drop_n_++ == 0)
                    std::cout << "[IMU] 丢弃一帧重复/乱序的 /IMU_DATA（时间戳 " << std::fixed << st << " ≤ 上一帧 " << imu_ts_
                              << std::defaultfloat << "），之后只在统计行里计数（0911d）" << std::endl;
                return;
            }
        }
        if (!imu_unit_said_) {
            imu_unit_said_ = true;
            std::cout << "[IMU] /IMU_DATA 姿态角按" << (imu_rpy_deg_ ? "【度】" : "【弧度】")
                      << "读（S10_IMU_RPY_UNIT=" << (imu_rpy_deg_ ? "deg" : "rad，默认") << "）；订阅 QoS=BEST_EFFORT（0911d）" << std::endl;
        }
        if (imu_rpy_deg_)
            rpy_ = Vec3f(Deg2Rad(msg->data.roll), Deg2Rad(msg->data.pitch), Deg2Rad(msg->data.yaw));
        else
            rpy_ = Vec3f(msg->data.roll, msg->data.pitch, msg->data.yaw);
        acc_ << msg->data.acc_x, msg->data.acc_y, msg->data.acc_z;
        omega_body_ << msg->data.omega_x, msg->data.omega_y, msg->data.omega_z;
        imu_ts_ = st;
        imu_rx_steady_.store(rx_now);
        ++imu_rx_n_;
        if (imu_stat_t0_ < 0.0) {
            imu_stat_t0_ = rx_now; imu_stat_n0_ = imu_rx_n_; imu_stat_d0_ = imu_drop_n_;
        } else if (rx_now - imu_stat_t0_ >= (imu_rate_said_ ? 60.0 : 2.0)) {   // 首次 2 s 后报一次，之后每分钟
            const double dt = rx_now - imu_stat_t0_;
            std::cout << "[IMU] 近 " << int(dt + 0.5) << " s 收下 /IMU_DATA " << (imu_rx_n_ - imu_stat_n0_) << " 帧（约 "
                      << int((imu_rx_n_ - imu_stat_n0_) / dt + 0.5) << " Hz），丢弃重复/乱序 " << (imu_drop_n_ - imu_stat_d0_)
                      << " 帧；当前 rpy(rad) " << rpy_.transpose() << "（0911d）" << std::endl;
            imu_rate_said_ = true; imu_stat_t0_ = rx_now; imu_stat_n0_ = imu_rx_n_; imu_stat_d0_ = imu_drop_n_;
        }
        // 开机自检（只打印，不改控制）：严格静止时（|ω|<0.05 rad/s、||a|−g|<0.3 m/s² 且已连续 0.25 s），
        // 加速度计比力方向应与姿态角推出的"世界竖直向上"（机体系下 R^T·ez）一致。
        // 09-10 真机包实测（严格静止帧）：按弧度读误差中位 0.1~0.3°、p95 ≤1°；按度读（旧 runner）误差 ≈ 倾角本身
        // （倾角 8~30° 时约 14°，侧翻后 90~136°），倾角 <3° 时只有 1~2°，分辨不出。所以阈值取 3°：
        // 先看开机 200 帧：误差 ≥3° 判失败；误差小且平均倾角 ≥8° 判通过；倾角太小就继续等倾斜 ≥8° 的严格静止帧，攒 100 帧再判。
        // 起 runner 时狗趴着（真机约 24° 仰），通常第一阶段就能判。
        if (!imu_chk_done_) {
            const float an = acc_.norm();
            const bool still = omega_body_.norm() < 0.05f && std::fabs(an - 9.8f) < 0.3f;
            imu_chk_streak_ = still ? imu_chk_streak_ + 1 : 0;
            if (imu_chk_streak_ >= 50) {
                const float tilt = std::acos(std::max(-1.f, std::min(1.f, acc_(2) / an))) * 180.f / float(M_PI);
                if (imu_chk_phase_ == 0 || tilt >= 8.f) {
                    Eigen::AngleAxisf ya(rpy_(2), Vec3f::UnitZ()), pa(rpy_(1), Vec3f::UnitY()), ra(rpy_(0), Vec3f::UnitX());
                    const Mat3f R = (ya * pa * ra).matrix();
                    const Vec3f up_rpy = R.transpose() * Vec3f(0.f, 0.f, 1.f);
                    const float c = std::max(-1.f, std::min(1.f, up_rpy.dot(acc_ / an)));
                    imu_chk_sum_ += std::acos(c) * 180.0 / M_PI;
                    imu_chk_tilt_sum_ += tilt;
                    ++imu_chk_n_;
                }
                const int need = (imu_chk_phase_ == 0) ? 200 : 100;
                if (imu_chk_n_ >= need) {
                    const double err = imu_chk_sum_ / imu_chk_n_, tavg = imu_chk_tilt_sum_ / imu_chk_n_;
                    if (err >= 3.0) {
                        imu_chk_done_ = true;
                        std::cerr << "!! [IMU] 开机自检失败：静止时姿态角与加速度计的竖直方向平均相差 " << err
                                  << "°（≥3°），机身倾角约 " << tavg << "°。/IMU_DATA 单位或符号不对，"
                                  << "策略的重力观测会是错的！先别进 RL，检查 S10_IMU_RPY_UNIT。" << std::endl;
                    } else if (tavg >= 8.0) {
                        imu_chk_done_ = true;
                        std::cout << "[IMU] 开机自检通过：机身倾角约 " << tavg << "° 时，静止姿态角与加速度计的竖直方向平均只差 "
                                  << err << "°，单位与符号正确" << std::endl;
                    } else if (imu_chk_phase_ == 0) {
                        imu_chk_phase_ = 1; imu_chk_n_ = 0; imu_chk_sum_ = 0.0; imu_chk_tilt_sum_ = 0.0;
                        std::cout << "[IMU] 自检暂不能下结论：机身倾角只有约 " << tavg << "°（平均相差 " << err
                                  << "°），分辨不了单位；机身倾斜 ≥8° 静止时会自动再判" << std::endl;
                    }
                }
            }
        }
    }

    void HandlerHealth(const drdds::msg::BatteryData::SharedPtr msg) {
        battery_data_[0] = uint8_t(msg->data[0].battery_level);
        remain_bat = battery_data_[0] / 100;
        battery_info_[0].battery_level = msg->data[0].battery_level;
    }
};
