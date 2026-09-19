/**
 * @file standup_state.hpp
 * @brief from sit state to stand state
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once

#include "state_base.h"
#include "../../include/utils/imu_gate.hpp"   // 0911d
#include <array>
#include <chrono>
#include <thread>

namespace qw{
class StandUpState : public StateBase{
        std::chrono::steady_clock::time_point imu_gate_print_{};   // 0911d：闸门上次打印时刻
private:
    VecXf init_joint_pos_, init_joint_vel_, current_joint_pos_, current_joint_vel_;
    double time_stamp_record_, run_time_;
    VecXf goal_joint_pos_, kp_, kd_;
    MatXf joint_cmd_;
    double stand_duration_ = 2.;
    using GuardClock = std::chrono::steady_clock;
    GuardClock::time_point last_cycle_{}, last_joint_change_{};
    std::array<GuardClock::time_point, 16> tracking_bad_since_{};
    double last_joint_stamp_ = 0.;
    bool stand_fault_latched_ = false;
    GuardClock::time_point entered_at_{};
    // 0911g 测试开关（默认关）：S10_TEST_STANDUP_STALL_MS=N → 起立开始 0.5 s 后，在下发指令前阻塞 N ms 一次，
    // 在仿真里复现 09-11 真机 DDS publish 内部阻塞 333 ms 的情形，用来验证 STAND_GUARD。真机不要设。
    const double test_stall_s_ = [](){
        const char* s=std::getenv("S10_TEST_STANDUP_STALL_MS");
        if (!s || !*s) return 0.;
        const double ms=std::atof(s);
        return (std::isfinite(ms) && ms>0. && ms<=2000.) ? ms/1000. : 0.;
    }();
    bool test_stall_done_ = false;
    // 0911g：防护阈值可用环境变量调（默认 = Codex 真机排查用的保守值）；S10_STAND_GUARD=0 整个关掉（不建议）。
    static double EnvMs(const char* k, double dflt) { const char* s = std::getenv(k); const double v = (s && *s) ? std::atof(s) : dflt;
                                                     return (std::isfinite(v) && v > 0.) ? v : dflt; }
    const bool guard_on_ = [](){ const char* s = std::getenv("S10_STAND_GUARD"); return !(s && std::string(s) == "0"); }();
    const double g_dt_s_    = EnvMs("S10_STAND_GUARD_DT_MS", 50.) / 1000.;      // 控制周期 / 反馈时间跳变
    const double g_stale_s_ = EnvMs("S10_STAND_GUARD_STALE_MS", 100.) / 1000.;  // 关节反馈停更
    const double g_block_s_ = EnvMs("S10_STAND_GUARD_BLOCK_MS", 50.) / 1000.;   // 单次计算+发布阻塞
    const float  g_track_rad_ = float(EnvMs("S10_STAND_GUARD_TRACK_RAD", .35));  // 腿关节跟踪误差
    const double g_track_s_ = EnvMs("S10_STAND_GUARD_TRACK_MS", 150.) / 1000.;  // 跟踪误差持续时间

    // 0911g（合入 Codex 09-11 真机排查的修复）：起立轨迹按时间推进，停顿 334/338 ms 后会一步追赶 44° 的目标。
    // 下面这些检查都在下发下一条位置指令之前做；任何一条命中就锁存阻尼（safe_control_mode=2），重启 runner 才恢复。
    void DampingCommand() {
        MatXf stop = MatXf::Zero(16, 5);
        for (int i=0; i<16; ++i) stop(i,2) = (i%4==3) ? 1.f : 2.f;
        ri_ptr_->SetJointCommand(stop);
    }
    void StandFault(const char* reason) {
        const bool first = !stand_fault_latched_;
        stand_fault_latched_ = true;
        uc_ptr_->GetUserCommand()->safe_control_mode = 2;
        DampingCommand();
        if (first)
            std::cerr << "!! [STAND_GUARD] " << reason
                      << "，已转阻尼并锁存；请重启 runner 并排查（0911g）" << std::endl;
    }

    const float init_hipx_pos_ = Deg2Rad(0.);
    const float set_wheel_kd_ = 1.;

    void GetRobotJointValue(){
        current_joint_pos_ = ri_ptr_->GetJointPosition();
        current_joint_vel_ = ri_ptr_->GetJointVelocity();
        run_time_ = ri_ptr_->GetInterfaceTimeStamp();
    }

    void RecordJointData(){
        init_joint_pos_ = current_joint_pos_;
        init_joint_vel_ = current_joint_vel_;
        time_stamp_record_ = run_time_;
    }

    float GetHipYPosByHeight(float h){
        float l1 = cp_ptr_->thigh_len_;
        float l2 = cp_ptr_->shank_len_;
        float default_pos = (cp_ptr_->fl_joint_lower_(1)+cp_ptr_->fl_joint_upper_(1)) / 2.;
        if(fabs(h) >= l1 + l2) {
            std::cerr << "error height input" << std::endl;
            return 0;
        }
        float theta = -acos((l1*l1+h*h-l2*l2)/(2.*h*l1));
        theta = LimitNumber(theta, cp_ptr_->fl_joint_lower_(1), cp_ptr_->fl_joint_upper_(1));
        return theta;
    }

    float GetKneePosByHeight(float h){
        float l1 = cp_ptr_->thigh_len_;
        float l2 = cp_ptr_->shank_len_;
        float default_pos = (cp_ptr_->fl_joint_lower_(2)+cp_ptr_->fl_joint_upper_(2)) / 2.;
        if(fabs(h) >= l1 + l2) {
            std::cerr << "error height input" << std::endl;
            return 0;
        }
        float theta = M_PI-acos((l1*l1+l2*l2-h*h)/(2*l1*l2));
        theta = LimitNumber(theta, cp_ptr_->fl_joint_lower_(2), cp_ptr_->fl_joint_upper_(2));
        return theta;
    }

public:
    StandUpState(const RobotName& robot_name, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_name, state_name, data_ptr){
            goal_joint_pos_ = Vec4f(init_hipx_pos_, GetHipYPosByHeight(cp_ptr_->pre_height_), GetKneePosByHeight(cp_ptr_->pre_height_), 0).replicate(4, 1);
            goal_joint_pos_(4) = -init_hipx_pos_; goal_joint_pos_(12) = -init_hipx_pos_;
            // 后髋膝取反
            goal_joint_pos_(9) = -goal_joint_pos_(9);
            goal_joint_pos_(10) = -goal_joint_pos_(10);
            goal_joint_pos_(13) = -goal_joint_pos_(13);
            goal_joint_pos_(14) = -goal_joint_pos_(14);

            Vec4f one_leg_kp, one_leg_kd;
            one_leg_kp << cp_ptr_->swing_leg_kp_, 0;
            one_leg_kd << cp_ptr_->swing_leg_kd_, 0;
            kp_ = one_leg_kp.replicate(4, 1);
            kd_ = one_leg_kd.replicate(4, 1);
            joint_cmd_ = MatXf::Zero(16, 5);
            joint_cmd_.col(0) = kp_;
            joint_cmd_.col(2) = kd_;
            stand_duration_ = cp_ptr_->stand_duration_;
        }
    ~StandUpState(){}

    virtual void OnEnter() {
        GetRobotJointValue();
        RecordJointData();
        entered_at_ = last_cycle_ = last_joint_change_ = GuardClock::now();
        std::cout << "[STAND_GUARD] " << (guard_on_ ? "开" : "关（S10_STAND_GUARD=0）") << "：周期/反馈跳变 " << g_dt_s_ * 1e3 << " ms、反馈停更 " << g_stale_s_ * 1e3
                  << " ms、阻塞 " << g_block_s_ * 1e3 << " ms、跟踪误差 " << g_track_rad_ << " rad 持续 " << g_track_s_ * 1e3 << " ms（0911g）" << std::endl;
        last_joint_stamp_ = run_time_;
        tracking_bad_since_.fill(GuardClock::time_point{});
        StateBase::msfb_.UpdateCurrentState(RobotMotionState::StandingUp);
    };
    virtual void OnExit() {
    }
    virtual void Run() {
        if (stand_fault_latched_) { DampingCommand(); return; }
        if (uc_ptr_->GetUserCommand()->safe_control_mode != 0 ||
            uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) {
            DampingCommand(); return;
        }
        GetRobotJointValue();
        const auto now = GuardClock::now();
        const double dt = std::chrono::duration<double>(now-last_cycle_).count();
        const double sensor_dt = run_time_-last_joint_stamp_;
        last_cycle_ = now;
        if (!std::isfinite(run_time_) || !current_joint_pos_.allFinite() || !current_joint_vel_.allFinite()) {
            StandFault("关节数据非有限值"); return;
        }
        if (guard_on_ && (dt > g_dt_s_ || sensor_dt > g_dt_s_ + 1e-6 || sensor_dt < -1e-6)) {
            StandFault("控制周期或关节反馈时间跳变超过 50 ms"); return;
        }
        if (sensor_dt > 0.) last_joint_change_ = now;
        last_joint_stamp_ = run_time_;
        if (guard_on_ && std::chrono::duration<double>(now-last_joint_change_).count() > g_stale_s_) {
            StandFault("关节反馈 100 ms 没有更新"); return;
        }
        VecXf planning_joint_pos(current_joint_pos_.rows());
        VecXf planning_joint_vel(current_joint_pos_.rows());
        if(run_time_ - time_stamp_record_ <= stand_duration_){
            for(int i=0;i<current_joint_pos_.rows();++i){
                planning_joint_pos(i) = GetCubicSplinePos(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0, 
                                                run_time_ - time_stamp_record_, stand_duration_);
                planning_joint_vel(i) = GetCubicSplineVel(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0, 
                                                run_time_ - time_stamp_record_, stand_duration_);
                if(i%4==3){
                    kd_(i) = 0;
                }
            }
            
        }else{
            float new_time = run_time_ - time_stamp_record_ - stand_duration_;
            float dt = 0.001;
            float plan_height = GetCubicSplinePos(cp_ptr_->pre_height_, 0, cp_ptr_->stand_height_, 0, 
                                                new_time, stand_duration_);
            float plan_height_next = GetCubicSplinePos(cp_ptr_->pre_height_, 0, cp_ptr_->stand_height_, 0, 
                                                new_time+dt, stand_duration_);
            float hipy_pos = GetHipYPosByHeight(plan_height);
            float hipy_vel = (GetHipYPosByHeight(plan_height_next) - hipy_pos) / dt;
            float knee_pos = GetKneePosByHeight(plan_height);
            float knee_vel = (GetKneePosByHeight(plan_height_next) - knee_pos) / dt;
            planning_joint_pos = Vec4f(init_hipx_pos_, hipy_pos, knee_pos, 0).replicate(4, 1);
            planning_joint_pos(4) = -init_hipx_pos_; planning_joint_pos(12) = -init_hipx_pos_;

            planning_joint_vel = Vec4f(0, hipy_vel, knee_vel, 0).replicate(4, 1);

            // 后髋膝取反
            planning_joint_pos(9) = -planning_joint_pos(9);
            planning_joint_pos(10) = -planning_joint_pos(10);
            planning_joint_pos(13) = -planning_joint_pos(13);
            planning_joint_pos(14) = -planning_joint_pos(14);
            planning_joint_vel(9) = -planning_joint_vel(9);
            planning_joint_vel(10) = -planning_joint_vel(10);
            planning_joint_vel(13) = -planning_joint_vel(13);
            planning_joint_vel(14) = -planning_joint_vel(14);

            for(int i=3;i<16;i+=4){
                kd_(i) = set_wheel_kd_;
            }
        }

        // 0911g：轮子的编码角是连续累计的，不是姿态目标。原来把累计圈数也插值回 0，实测轮速目标约 71 rad/s；
        // 现在轮子保持当前角度、速度目标 0。
        for (int i=3; i<16; i+=4) {
            planning_joint_pos(i) = current_joint_pos_(i);
            planning_joint_vel(i) = 0.f;
        }
        if (!planning_joint_pos.allFinite() || !planning_joint_vel.allFinite()) {
            StandFault("起立目标非有限值"); return;
        }
        for (int i=0; i<16; ++i) {
            if (i%4==3) continue;
            if (guard_on_ && std::fabs(planning_joint_pos(i)-current_joint_pos_(i)) > g_track_rad_) {
                if (tracking_bad_since_[i] == GuardClock::time_point{}) tracking_bad_since_[i] = now;
                if (std::chrono::duration<double>(now-tracking_bad_since_[i]).count() >= g_track_s_) {
                    StandFault("腿关节跟踪误差超过 0.35 rad（20°）持续 150 ms"); return;
                }
            } else tracking_bad_since_[i] = GuardClock::time_point{};
        }
        joint_cmd_.col(1) = planning_joint_pos;
        joint_cmd_.col(3) = planning_joint_vel;
        joint_cmd_.col(2) = kd_;
        if (test_stall_s_>0. && !test_stall_done_ && std::chrono::duration<double>(now-entered_at_).count()>=.5) {
            test_stall_done_ = true;   // 测试开关：模拟 publish 内部阻塞（只发生一次）
            std::this_thread::sleep_for(std::chrono::duration<double>(test_stall_s_));
        }
        ri_ptr_->SetJointCommand(joint_cmd_);
        if (guard_on_ && std::chrono::duration<double>(GuardClock::now()-now).count() > g_block_s_)
            StandFault("起立计算或指令发布阻塞超过 50 ms");
    }
    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::JointDamping)) return true;
        return false;
    }
    virtual StateName GetNextStateName() {
        if (stand_fault_latched_) return StateName::kJointDamping;
        if(uc_ptr_->GetUserCommand()->safe_control_mode!=0) return StateName::kJointDamping;
        if(run_time_ - time_stamp_record_ <= 2.*stand_duration_){
            return StateName::kStandUp;
        }else{
            if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::RLControlMode)){
                if(!imu_gate::Ok(&*ri_ptr_, "进 RL", imu_gate_print_)) return StateName::kStandUp;   // 0911d
                return StateName::kRLControl;
            }else if(uc_ptr_->GetUserCommand()->target_mode == uint8_t(RobotMotionState::LieDown)){
                return StateName::kLieDown;
            }
        }
        return StateName::kStandUp;
    }
};

};
