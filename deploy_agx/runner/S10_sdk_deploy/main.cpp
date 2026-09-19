#include "quadruped_wheel/qw_state_machine.hpp"
#include <cstdlib>
#include <string>

#ifdef USE_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward{
        backward::SignalHandling sh;
    }
#endif

using namespace types;
MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

// 指令来源改成运行时可选，免得为了换一个输入源重新编译。
// S10_CMD_SOURCE = keyboard(默认) | gamepad | ros
static RemoteCommandType PickCommandSource(){
    const char* s = std::getenv("S10_CMD_SOURCE");
    if(!s || *s == '\0') return RemoteCommandType::kKeyBoard;
    std::string v(s);
    if(v == "ros")      return RemoteCommandType::kRosTopic;
    if(v == "gamepad")  return RemoteCommandType::kGamepad;
    if(v == "keyboard") return RemoteCommandType::kKeyBoard;
    std::cerr << "未知的 S10_CMD_SOURCE=\"" << v << "\"，回退到 keyboard" << std::endl;
    return RemoteCommandType::kKeyBoard;
}

int main(){
    std::cout << "State Machine Start Running" << std::endl;
    rclcpp::init(0, 0);
    std::shared_ptr<StateMachineBase> fsm = std::make_shared<qw::QwStateMachine>(RobotName::S10, PickCommandSource());

    fsm->Start();
    fsm->Run();
    fsm->Stop();

    rclcpp::shutdown();
    return 0;
}
