/**
 * @file imu_gate.hpp
 * @brief 0911d IMU 闸门。
 *
 * 机器人 /IMU_DATA 的两个发布者都是 BEST_EFFORT，0911c 及以前的 runner 用 RELIABLE 订阅（QoS 不兼容），
 * 真机上从没收到过 IMU：策略一直在"角速度恒 0、重力恒竖直"下跑，却没有任何报错（09-11 AGX 侧查实）。
 * 这里在"起立""进 RL"两个入口拦住：按本机接收时刻计的 IMU 年龄超过 S10_IMU_MAX_AGE_MS（默认 100 ms），
 * 或读数非法（NaN、|姿态角|>π、|角速度|>π rad/s、加速度模长不在 0.1g~3g），就不放行，每秒打印一次原因。
 * 0911e：默认阈值 100 → 200 ms（09-11 真机 AGX 负载约 20 时，100 ms 每 1~2 分钟擦边一次；断流照样拦得住）。
 * 已经在 RL 里时不强制切状态（safe_controller 只打印断流告警）。
 */
#pragma once
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace imu_gate {

inline double MaxAgeSec() {
    static const double v = []() {
        const char* s = std::getenv("S10_IMU_MAX_AGE_MS");
        const double ms = (s && *s) ? std::atof(s) : 200.0;
        return (ms > 0.0 ? ms : 200.0) / 1000.0;
    }();
    return v;
}

// 返回空串 = 放行；否则是不放行的原因。
template <class RI>
inline std::string Reason(RI* ri) {
    char buf[200];
    const double age = ri->GetImuRxAgeSec();
    if (age > 1e8) return "从没收到过 /IMU_DATA";
    if (age > MaxAgeSec()) {
        std::snprintf(buf, sizeof(buf), "已 %.0f ms 没收到 /IMU_DATA（上限 %.0f ms，S10_IMU_MAX_AGE_MS）", age * 1000.0, MaxAgeSec() * 1000.0);
        return buf;
    }
    const auto rpy = ri->GetImuRpy();
    const auto omg = ri->GetImuOmega();
    const auto acc = ri->GetImuAcc();
    for (int i = 0; i < 3; ++i) {
        if (std::isnan(rpy(i)) || std::fabs(rpy(i)) > M_PI) return "IMU 姿态角非法（NaN 或 |角|>π）";
        if (std::isnan(omg(i)) || std::fabs(omg(i)) > M_PI) return "IMU 角速度非法（NaN 或 >π rad/s；机身正被快速转动？）";
    }
    const float an = acc.norm();
    if (std::isnan(an) || an < 0.1f * 9.81f || an > 3.0f * 9.81f) {
        std::snprintf(buf, sizeof(buf), "IMU 加速度模长 %.2f m/s² 不在 0.1g~3g", double(an));
        return buf;
    }
    return "";
}

// what：被拦的动作（"起立" / "进 RL"）；last：调用方各自的上次打印时刻。
template <class RI>
inline bool Ok(RI* ri, const char* what, std::chrono::steady_clock::time_point& last) {
    const std::string r = Reason(ri);
    if (r.empty()) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now - last > std::chrono::seconds(1)) {
        last = now;
        std::cerr << "!! [IMU] 不" << what << "：" << r << "。（0911d 闸门：收到新鲜 IMU 之前不起立、不进 RL）" << std::endl;
    }
    return false;
}

}  // namespace imu_gate
