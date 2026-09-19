#include "control_parameters.h"

void ControlParameters::GenerateS10Parameters(){
    body_len_x_ = 0.3095*2;
    body_len_y_ = 0.065*2;
    hip_len_ = 0.104;
    thigh_len_ = 0.25;
    shank_len_ = 0.25;
    pre_height_ = 0.12;
    stand_height_ = 0.48;
    swing_leg_kp_ << 120., 120., 120.;
    swing_leg_kd_ << 2., 2., 2.;

    // 0911h：膝限位改成 S10 URDF / 官方 09-04 参数的 ±2.7227（原 ±2.758 比机械限位多约 2°：0911g 起趴下时膝目标会被夹到 2.758、顶在止挡上）。
    // hipy 仍比 URDF 窄（±2.443），保持 0911g 不动；hipx 见下一条（09-12 改成对称）。
    // 09-12（作者拍板，AGX 侧 0911g-hipx 同一改法）：hipx 下限 -0.4363 → -0.6109，与训练模型一致的对称 ±0.6109（fr/hr 镜像后同样对称）。
    // 只影响 idle_state 的 JointDataNormalCheck（待命态准入：外劈 36.5° → 46.5°），standup/liedown 只用 (1)(2)，策略动作范围不受影响。
    fl_joint_lower_ << -0.6109, -2.443, -2.7227;
    fl_joint_upper_ << 0.6109, 2.443, 2.7227;
    joint_vel_limit_ << 45, 22.4, 22.4;
    torque_limit_ << 32.4, 76.4, 76.4;
}