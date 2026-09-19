/**
 * @file user_command_interface.h
 * @brief this file is used for robot's user command input
 * @author DeepRobotics
 * @version 1.0
 * @date 2025-11-07
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */
#pragma once

#include "common_types.h"
#include "custom_types.h"
#include "motion_state_feedback.hpp"

using namespace types;

namespace interface{

class UserCommandInterface{
private:
    /* data */
public:
    UserCommandInterface(RobotName robot_name){
        robot_name_ = robot_name;
        usr_cmd_ = new UserCommand();
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
    }
    virtual ~UserCommandInterface(){
        delete usr_cmd_;
    }

    /**
     * @brief start the thread to process user command
     */
    virtual void Start() = 0;

    /**
     * @brief stop the thread 
     */
    virtual void Stop() = 0;

    /**
     * @brief return your user command
     * @return UserCommand ptr
     */
    virtual UserCommand* GetUserCommand() = 0; 

    /**
     * @brief set the motion state feedback
     * @param  msfb         motion state feedback
     */
    virtual void SetMotionStateFeedback(MotionStateFeedback* msfb){
        msfb_ = msfb;
    }

    /**
     * @brief 取一帧高程图（策略观测的第 58~244 维）。
     *
     * 感知侧（订阅雷达、跑累积地图）算好之后经 ROS 送进来。
     * 放在这个基类上是为了让 RLControlState 能**多态**地取，
     * 不必向下转型成 RosCommandInterface —— 键盘/手柄那些接口
     * 没有感知，用默认实现返回 false 即可。
     *
     * @param out  长度 len 的缓冲，成功时写满
     * @param len  期望长度（当前是 187）
     * @return     true = 拿到了新鲜数据；false = 没有，调用方应该用兜底值。
     *             **返回 false 时不要保留上一帧** —— 感知挂了之后
     *             拿着几秒前的地形做决策比没有地形更危险。
     */
    virtual bool GetHeightScan(float* /*out*/, int /*len*/) const { return false; }

    MotionStateFeedback *msfb_;
    RobotName robot_name_;
    UserCommand *usr_cmd_;
};
};
