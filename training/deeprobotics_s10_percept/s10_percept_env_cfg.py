"""S10 感知策略环境配置（2026-09-08 新机合并）。

S10-V0 = 交付包 V16 配方（climb_up / approach_riser 低→高课程 + step_down）原样，只换三处（作者 09-08 拍板）：
  1. 机器人 → DEEPROBOTICS_S10_DEPLOY_CFG（几何/质量/默认姿态/PD/力矩与部署 runner 逐位一致）
  2. 课程立面上限 0.40 → 0.34（石笼卷尺 33 cm，S10 腿长 0.36）
  3. 机身附加质量 (−1, 3) → (−0.5, 3.0) kg（AGX + 深度相机 + 支架的实际载荷）
其余（观测 244、奖励、噪声、域随机化、地形占比）一律不动，一次只改一个变量的原则下，V1（地形表）/V2（湿地摩擦）另起类。
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from rl_training.assets.deeprobotics import DEEPROBOTICS_S10_DEPLOY_CFG
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.percept_env_cfg import (
    DeeproboticsM20PerceptV16EnvCfg,
)

S10_RISER_MAX = 0.34


@configclass
class DeeproboticsS10PerceptV0EnvCfg(DeeproboticsM20PerceptV16EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 1) 机器人
        self.scene.robot = DEEPROBOTICS_S10_DEPLOY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 2) 课程上限
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        if "climb_up" in st:
            st["climb_up"].step_height_range = (0.06, S10_RISER_MAX)
        if "approach_riser" in st:
            st["approach_riser"].riser_height_range = (0.06, S10_RISER_MAX)
        # 3) 载荷
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 3.0)
        r = self.scene.robot
        def _lim(a, k):      # 09-09：隐式执行器只填 *_limit_sim
            v = getattr(a, k, None)
            return v if v is not None else getattr(a, k + "_sim", float("nan"))
        print("[s10] 机器人=%s  init z=%.2f  执行器=%s  腿 effort/vel=%.0f/%.2f kp/kd=%.0f/%.0f  轮 effort/vel=%.0f/%.1f kd=%.1f armature=%s"
              % (r.spawn.usd_path.split("/deep_robotics_model/")[-1], r.init_state.pos[2], type(r.actuators["joint"]).__name__,
                 _lim(r.actuators["joint"], "effort_limit"), _lim(r.actuators["joint"], "velocity_limit"),
                 r.actuators["joint"].stiffness, r.actuators["joint"].damping,
                 _lim(r.actuators["wheel"], "effort_limit"), _lim(r.actuators["wheel"], "velocity_limit"),
                 r.actuators["wheel"].damping, r.actuators["wheel"].armature))
        print("[s10] 默认姿态=%s" % r.init_state.joint_pos)
        print("[s10] 课程上限 climb_up/approach_riser=%.2f  载荷 base +(%.1f,%.1f) kg  —— 以本行为准，上面的 [v5]~[v16] 横幅是 M20 配方的继承打印"
              % (S10_RISER_MAX, *self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"]))


@configclass
class DeeproboticsS10PerceptV1EnvCfg(DeeproboticsS10PerceptV0EnvCfg):
    """S10-V1 = V0 + 上台面奖励（mdp/climb_rewards.py，依据厂家高台慢动作逐帧拆解，2026-09-08 深夜）。

    只加不改：新增 逐轮里程碑（势函数差分，只算触地轮）、禁腾空、翻越相位内三轮支撑软罚；
    翻越相位内豁免 停滞惩罚 / 轮撞立面 / 对角镜像；俯仰终止线翻越相位内 44°→60°（横滚不放）。
    其余奖励、地形课程、域随机化与 V0 逐位相同。
    """

    # 课程（作者 2026-09-08 深夜拍板：「直接从 31 开始」）：
    #   立面范围 (0.31, 0.345)：IsaacLab 第 k 行难度 = (k+η)/10，η~U(0,1) → 10 行几乎都在 31~34.5 cm，等于不做由易到难，
    #   靠 wheel_height_progress 的势函数差分给部分进展（前轮沿立面升高）打分来替代课程梯度。
    #   我方原提案 (0.15, 0.35)（起步行为盲策略已会的 15~21 cm、顶行 33~35）保留为回退项：
    #   若 500 迭代内 wheel_height_progress 无增长、课程无晋级，改回 (0.15, 0.35) 重训。
    #   V0 的 (0.06, 0.34) 顶行只到 [0.312, 0.34)，前 5 行 0.06~0.20 全是能滚过去的高度，练的不是上墙。
    # 09-08 21:45 作者：「31 不行可以从 25 开始」→ (0.25, 0.345)，跑 700 迭代无进展。
    # 09-08 22:50 实测：热启动盲策略在 PhysX 里 15/20/25 cm 立面全部 0/64（走到墙根停住），课程最低行必须是能滚过去的高度 →
    #   (0.06, 0.345)：第 k 行 [0.06+0.0285k, 0.0885+0.0285k)，第 0 行 6~9 cm，顶行 [0.3165, 0.345)。这是 V16 的课程思路。
    RISER_RANGE: tuple[float, float] = (0.06, 0.345)
    HONEST_PROMOTION: bool = True     # 立面地形上：走远 且 升高 >0.10 m 才晋级（堵住沿车道走 4 m 白晋级的漏洞）

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import curriculums as curr

        # 课程
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        if "climb_up" in st:
            st["climb_up"].step_height_range = tuple(self.RISER_RANGE)
        if "approach_riser" in st:
            st["approach_riser"].riser_height_range = tuple(self.RISER_RANGE)
        if self.HONEST_PROMOTION:
            self.curriculum.terrain_levels.func = curr.terrain_levels_climb_honest
            self.curriculum.terrain_levels.params = {"z_gain_min": 0.03, "z_gain_frac": 0.6}
        print("[s10-v1] 课程：climb_up/approach_riser 立面 %.2f~%.2f（10 行，每行 %.3f m，顶行 [%.2f,%.2f)）  诚实晋级=%s"
              % (*self.RISER_RANGE, (self.RISER_RANGE[1]-self.RISER_RANGE[0])/10,
                 self.RISER_RANGE[0]+0.9*(self.RISER_RANGE[1]-self.RISER_RANGE[0]), self.RISER_RANGE[1], self.HONEST_PROMOTION))

        R = self.rewards
        # 09-08 首跑塌成趴低爬行后的三处修正（见 climb_rewards.base_height_flat_l2 注释）
        R.base_height_flat = RewTerm(func=climb.base_height_flat_l2, weight=-20.0, params={"target_height": 0.43})
        R.joint_acc_wheel_l2.weight = -1.0e-8      # M20 配方 -1e-7 在 S10 轮子上一度到 -16/s，比速度奖励大 7 倍，降 10 倍
        R.wheel_height_progress = RewTerm(func=climb.WheelHeightProgress, weight=300.0, params={"max_rise": 0.5})
        R.wheel_height_record = RewTerm(func=climb.WheelHeightRecord, weight=200.0, params={"max_rise": 0.5})
        R.airborne = RewTerm(func=climb.airborne_penalty, weight=-2.0)
        R.support_deficit_climb = RewTerm(func=climb.support_deficit_climb, weight=-0.5, params={"min_contacts": 3})
        # 厂家八段动作的相位奖励（作者 09-08 深夜确认：提/蓄力/右前搭/左前跟/右后蹬/左后爬/右后跨/落平；左右不规定）
        R.st_crouch = RewTerm(func=climb.StageRecord, weight=100.0, params={"quantity": "crouch", "cap": 0.05})
        R.st_hind_flex = RewTerm(func=climb.StageRecord, weight=5.0, params={"quantity": "hind_flex", "cap": 0.6})
        R.st_front_lift = RewTerm(func=climb.StageRecord, weight=60.0, params={"quantity": "front_lift", "cap": 0.45})
        R.st_pitch_up = RewTerm(func=climb.StageRecord, weight=10.0, params={"quantity": "pitch_up", "cap": 0.71})
        R.st_hind_push = RewTerm(func=climb.StageRecord, weight=40.0, params={"quantity": "hind_push", "cap": 1.0})
        # 01:00 夜间调整（model_1000：前轮搭上后停在墙前不蹬）：里程碑 ×2、蹬地力 ×2、推进姿态俯仰窗 10°~55° 且 w1.0
        # 作者 09-09 00:10 看 MuJoCo 回放后的调整：慢速抵上、一只一只上（front_lift 门控）、整段 ≥3 轮支撑加重
        R.st_approach_speed = RewTerm(func=climb.approach_speed_penalty, weight=-2.0, params={"v_max": 0.6})
        R.st_hoist = RewTerm(func=climb.hoist_penalty, weight=-20.0, params={"z_max": 0.46})   # 作者 09-09：前轮没搭上前别把身子架起来
        R.support_deficit_climb.weight = -1.5
        R.st_hind_extend = RewTerm(func=climb.StageRecord, weight=5.0, params={"quantity": "hind_extend", "cap": 0.6})
        R.st_hind_lift = RewTerm(func=climb.StageRecord, weight=60.0, params={"quantity": "hind_lift", "cap": 0.45})
        R.st_push_posture = RewTerm(func=climb.push_posture, weight=1.0, params={"pitch_lo": 0.17, "pitch_hi": 0.82, "vx_min": 0.05})
        R.st_level_drag = RewTerm(func=climb.level_drag, weight=0.5, params={"vx_min": 0.05})
        R.lin_vel_z_l2.func = climb.lin_vel_z_l2_climb_aware
        R.joint_torques_l2.func = climb.joint_torques_l2_climb_aware
        R.encounter_frac = RewTerm(func=climb.encounter_frac, weight=0.001)
        R.climb_phase_frac = RewTerm(func=climb.climb_phase_frac, weight=0.001)
        # 出生行覆盖全部课程行：高墙前的蓄力/蹬地/抬头从第一迭代就有人练，低行提供完整上去的样本
        self.scene.terrain.max_init_terrain_level = 9
        print("[s10-v1] 相位纪录奖励：crouch %.0f hind_flex %.0f front_lift %.0f pitch_up %.0f hind_push %.0f hind_extend %.0f hind_lift %.0f | 每步 push_posture %.1f level_drag %.1f | lin_vel_z/torques 翻越内关 | 出生行 0~%d"
              % (R.st_crouch.weight, R.st_hind_flex.weight, R.st_front_lift.weight, R.st_pitch_up.weight, R.st_hind_push.weight,
                 R.st_hind_extend.weight, R.st_hind_lift.weight, R.st_push_posture.weight, R.st_level_drag.weight, self.scene.terrain.max_init_terrain_level))
        R.wall_ahead_frac = RewTerm(func=climb.wall_ahead_frac, weight=0.001)
        # 翻越相位内豁免（函数换成相位感知版，参数与权重原样）
        R.stall_penalty.func = climb.stall_penalty_climb_aware
        R.feet_stumble.func = climb.feet_stumble_climb_aware
        R.joint_mirror.func = climb.joint_mirror_climb_aware
        # 姿态终止/惩罚：翻越相位内俯仰放宽到 60°
        ori = {"limit": 0.7, "pitch_limit_climb": 0.87}
        R.bad_orientation_penalty.func = climb.bad_orientation_penalty_climb_aware
        R.bad_orientation_penalty.params = dict(ori, encounter_scale=0.3)      # 11:55：上墙中翻倒罚 −300，平地仍 −1000
        R.st_push_drive = RewTerm(func=climb.push_drive, weight=6.0, params={"v_max": 0.5})
        R.st_hind_splay = RewTerm(func=climb.hind_splay_penalty, weight=-10.0, params={"back_max": 0.12})   # 12:25：后轮别拖在髋后面
        # 作者 13:20：右前轮只搭一点、身子留在后面、蓄力后右后腿蹬起整机
        R.st_too_close = RewTerm(func=climb.too_close_penalty, weight=-15.0, params={"d_min": 0.22, "pitch_free": 0.5})
        R.st_launch = RewTerm(func=climb.launch_reward, weight=4.0, params={"vz_max": 0.8})
        R.flexed_frac = RewTerm(func=climb.flexed_frac, weight=0.001)
        # 14:00 墙前出生：一半立面地形 episode 直接从离墙 0.5 m、后腿蓄力姿态开始（RSI 简化版）
        self.events.spawn_at_wall = EventTerm(func=climb.spawn_at_wall, mode="reset",
                                              params={"asset_cfg": SceneEntityCfg("robot"), "frac": 0.5, "dist": 0.5})   # 11:55：前轮已上、后轮着地 → 按前进速度给分
        self.terminations.bad_orientation_2 = DoneTerm(func=climb.bad_orientation_climb_aware, params=dict(ori))
        print("[s10-v1] 站高惩罚 base_height_flat w=%.0f(target %.2f, 翻越相位外)  joint_acc_wheel_l2 w=%.1e" % (R.base_height_flat.weight, R.base_height_flat.params["target_height"], R.joint_acc_wheel_l2.weight))
        print("[s10-v1] 上台面奖励：wheel_height_progress w=%.0f + wheel_height_record w=%.0f (max_rise %.2f)  airborne w=%.1f  support_deficit_climb w=%.1f(min %d)  "
              "豁免：stall/support 只在轮子分层时，stumble/mirror 在面前有墙或分层时  俯仰终止 %.2f→%.2f(翻越内)"
              % (R.wheel_height_progress.weight, R.wheel_height_record.weight, R.wheel_height_progress.params["max_rise"], R.airborne.weight,
                 R.support_deficit_climb.weight, R.support_deficit_climb.params["min_contacts"], ori["limit"], ori["pitch_limit_climb"]))


@configclass
class DeeproboticsS10ClimbExpertEnvCfg(DeeproboticsS10PerceptV1EnvCfg):
    """上墙专家策略（作者 09-09 14:30 拍板：V1 通用环境里上墙样本只占 1~2%，改成专训）。

    与 V1 相同：机器人、244 维观测、八段相位奖励与约束。不同：
      地形只有 approach_riser（立面 0.15~0.34，10 行课程，只按升高晋降级）；每局都从离墙 0.4~1.4 m 处出生、正对墙（+x），
      一半带蓄力姿态；指令固定前进 0.3~0.6、不转向；10 s 一局；四轮上顶给一次性完成奖励。
    部署：导航层在墙前切 /climb_mode 进入本策略，上去后切回通用策略（fork runner 已支持热切）。
    """

    RISER_RANGE: tuple[float, float] = (0.15, 0.34)

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import curriculums as curr
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.approach_riser import MeshApproachRiserTerrainCfg
        # 地形：只留立面
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {"approach_riser": MeshApproachRiserTerrainCfg(proportion=1.0, riser_height_range=tuple(self.RISER_RANGE), lane_width=4.0)}
        gen.curriculum = True
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels.func = curr.terrain_levels_climb_expert
        self.curriculum.terrain_levels.params = {"up_frac": 0.6, "down_frac": 0.15}
        # 指令：固定慢速前进、不转向、航向锁 0（出生只朝 +x）
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (0.3, 0.6); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0); c.ranges.heading = (0.0, 0.0)
        for attr in ("rel_standing_envs", "rel_zero_vel_envs", "rel_only_ang_z_envs", "rel_only_lin_y_envs", "rel_only_lin_x_envs"):
            if hasattr(c, attr):
                setattr(c, attr, 0.0)
        c.resampling_time_range = (10.0, 10.0)
        self.episode_length_s = 10.0
        # 出生：每局都在墙前
        self.events.spawn_at_wall.params = {"asset_cfg": SceneEntityCfg("robot"), "frac": 1.0, "dist_range": (0.4, 1.4), "crouch_prob": 0.5, "both_sides": False,
                                            "hooked_prob": 0.35}   # 15:10：35% 从"前轮已搭上、后腿蓄力、机身抬头"开始（倒序课程）
        self.rewards.support_deficit_climb.weight = -0.5           # 15:10：三轮支撑只作软约束，不抹掉动态蹬起
        self.rewards.st_too_close.params = {"d_min": 0.15, "pitch_free": 0.35}   # 15:05：hooked 出生态本身离墙 0.15~0.30、抬头 ~29°，原 0.22/30° 会罚到它
        # 完成奖励
        self.rewards.climb_complete = RewTerm(func=climb.ClimbCompleteBonus, weight=100.0)
        self.rewards.ground_complete = RewTerm(func=climb.ground_complete_frac, weight=0.001)   # 日志：从地面起步的完成
        self.rewards.st_hind_flex.weight = 40.0     # 15:45：蓄力屈膝 5→40，前轮抬起要求已蓄力
        # 17:05：遭遇中关掉速度跟踪与停滞惩罚（它们把机身往墙里拉，与趴墙惩罚对冲；厂家在墙前不追速度、蓄力时是停着的）
        self.rewards.track_lin_vel_xy_exp.func = climb.track_lin_vel_xy_exp_climb_aware
        self.rewards.stall_penalty.func = climb.stall_penalty_encounter_free
        print("[s10-climb] 专家环境：approach_riser 立面 %.2f~%.2f，出生离墙 0.4~1.4 m 正对 +x，蓄力概率 0.5，指令 0.3~0.6 前进，10 s/局，完成奖励 100" % self.RISER_RANGE)


@configclass
class DeeproboticsS10ClimbExpertCEnvCfg(DeeproboticsS10ClimbExpertEnvCfg):
    """09-09 22:10 C 版：出生态加"蹬完态"（pushed，脚本控制器实测几何）练单腿抬起+落平；遭遇段加偏航保持罚（堵"转 90° 沿墙跑"）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        self.events.spawn_at_wall.params["hooked_prob"] = 0.3
        self.events.spawn_at_wall.params["pushed_prob"] = 0.3
        self.rewards.st_yaw_keep = RewTerm(func=climb.yaw_keep_penalty, weight=-10.0, params={"always": True})   # 23:40：全程算，−10
        print("[s10-climb-C] hooked_prob=0.3 pushed_prob=0.3  st_yaw_keep w=-10 always")


class DeeproboticsS10ClimbExpertDEnvCfg(DeeproboticsS10ClimbExpertCEnvCfg):
    """09-10 00:50 D 版：真机参考（climb33_1/2/3，34 cm）——参考态出生 ref_prob 0.4 + 压墙后相位跟踪奖励 st_ref_track。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_ref
        self.events.spawn_at_wall.params.update({"hooked_prob": 0.15, "pushed_prob": 0.15, "ref_prob": 0.4})
        self.rewards.st_ref_track = RewTerm(func=climb_ref.RefTrack, weight=5.0, params={"sigma_q": 0.35, "sigma_p": 0.35, "press_dist": 0.35, "press_w": 5.0})
        print("[s10-climb-D] hooked 0.15 pushed 0.15 ref 0.4  st_ref_track w=5 (真机参考 %s)" % climb_ref.REF_PATH)


class DeeproboticsS10ClimbExpertEEnvCfg(DeeproboticsS10ClimbExpertDEnvCfg):
    """09-10 E 版：D + 遭遇段内向墙前进奖励 st_wall_approach（堵"停在遭遇段边界不靠近"）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        self.rewards.st_wall_approach = RewTerm(func=climb.wall_approach_reward, weight=4.0, params={"v_max": 0.5, "d_touch": 0.30})
        print("[s10-climb-E] + st_wall_approach w=4 (遭遇段内向墙前进，前轮触墙前)")


class DeeproboticsS10ClimbExpertFEnvCfg(DeeproboticsS10ClimbExpertEEnvCfg):
    """09-10 F 版：E + 速度跟踪在遭遇段内保留到前轮触墙（去掉遭遇边界的奖励悬崖）+ 向墙奖励 w=8。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.st_wall_approach.weight = 8.0
        print("[s10-climb-F] 跟踪奖励保留到触墙(d_touch 0.30)，st_wall_approach w=8")


class DeeproboticsS10ClimbExpertGEnvCfg(DeeproboticsS10ClimbExpertFEnvCfg):
    """09-10 07:30 G 版（加固）：F + 墙高课程上限 0.38 + 执行器阻尼随机 0.7~1.15 倍（覆盖 runner 轮 kd 0.6 与训练 0.8 的口径差）。"""
    RISER_RANGE: tuple[float, float] = (0.15, 0.38)

    def __post_init__(self):
        super().__post_init__()
        ev = getattr(self.events, "randomize_actuator_gains", None)
        if ev is not None:
            ev.params["damping_distribution_params"] = (0.7, 1.15)
        print("[s10-climb-G] RISER_RANGE=%s  阻尼随机 0.7~1.15" % (self.RISER_RANGE,))


class DeeproboticsS10PerceptV1GEnvCfg(DeeproboticsS10PerceptV1EnvCfg):
    """09-10 07:30 V1G（加固）：V1 + 执行器阻尼随机 0.7~1.15 倍。"""

    def __post_init__(self):
        super().__post_init__()
        ev = getattr(self.events, "randomize_actuator_gains", None)
        if ev is not None:
            ev.params["damping_distribution_params"] = (0.7, 1.15)
        print("[s10-v1-G] 阻尼随机 0.7~1.15")


class DeeproboticsS10TrotEnvCfg(DeeproboticsS10PerceptV1GEnvCfg):
    """09-10 踏步模式（厂家踢踏舞步态：平地快跑 + 越野）。V1G 之上：地形改为路面/草地/坡/矮箱，速度指令到 2.3 m/s，
    对角步态 + 抬腿 9 cm + 按速度的腾空时长 + 不踏步惩罚，站高 0.35，轮力矩上限 20（真机实测尖峰 19.7）+ 轮力矩小罚。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_gait as trot
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        keep = {}
        for k, v in st.items():
            kl = k.lower()
            if "random_rough" in kl or "hf_random" in kl or kl in ("rough",):
                v.proportion = 0.30; keep[k] = v
            elif "slope" in kl:
                v.proportion = 0.15; keep[k] = v
            elif "boxes" in kl:
                v.proportion = 0.10; keep[k] = v
            elif "real_patch" in kl:
                v.proportion = 0.15; keep[k] = v
            elif kl in ("flat", "plane", "flat_ground"):
                v.proportion = 0.30; keep[k] = v
        if not any(("flat" in k.lower()) for k in keep):
            import isaaclab.terrains as terrain_gen
            keep["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.30)
        gen.sub_terrains = keep
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.5, 2.3); c.ranges.lin_vel_y = (-0.5, 0.5); c.ranges.ang_vel_z = (-1.0, 1.0)
        R = self.rewards
        R.st_diag = RewTerm(func=trot.diag_sync_reward, weight=0.5, params={"v_min": 0.3})
        R.st_clearance = RewTerm(func=trot.swing_clearance_reward, weight=0.5, params={"h_target": 0.09, "sigma": 0.04, "v_min": 0.3})
        R.st_air_time = RewTerm(func=trot.air_time_target_reward, weight=2.0, params={"v_min": 0.3, "sigma": 0.08})
        R.st_no_step = RewTerm(func=trot.no_step_penalty, weight=-0.5, params={"v_min": 0.5, "t_max": 1.0})
        if hasattr(R, "base_height_flat"):
            R.base_height_flat.params["target_height"] = 0.35
        if hasattr(R, "joint_torques_wheel_l2"):
            R.joint_torques_wheel_l2.weight = -2e-4
        self.scene.robot.actuators["wheel"].effort_limit_sim = 20.0
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 42.0      # 真机踏步推满轮速约 40 rad/s（3.2 m/s），URDF 的 65.5 真机做不到
        print("[s10-trot] 地形=%s  指令 vx %s  站高 0.35  轮力矩上限 20  奖励 diag 0.5 / clearance 0.5 / air_time 2 / no_step -0.5" % (
            {k: round(v.proportion, 2) for k, v in keep.items()}, c.ranges.lin_vel_x))


class DeeproboticsS10TrotBEnvCfg(DeeproboticsS10TrotEnvCfg):
    """B 版：蹲低纯滚为主，步态奖励/不踏步罚只在粗糙子地形生效（平地允许纯滚）；轮速上限压到真机 42 rad/s。"""

    def __post_init__(self):
        super().__post_init__()
        for nm in ("st_diag", "st_clearance", "st_air_time", "st_no_step"):
            getattr(self.rewards, nm).params["rough_only"] = True
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 42.0
        print("[s10-trot-B] 步态奖励只在粗糙地生效；轮速上限 42 rad/s")


class DeeproboticsS10FlatFastEnvCfg(DeeproboticsS10PerceptV1GEnvCfg):
    """09-10 平地快跑模式（作者拍板：两模式分训，按路段切）：蹲低纯滚，平地 + 缓坡，速度到 2.5 m/s，
    指令 2–5 s 重采样（起步/急停/转弯多练），一成零指令局，湿地摩擦 0.4 起，站高 0.35，轮力矩 20、轮速 42（真机口径）。不给任何踏步奖励。"""

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        keep = {}
        for k, v in st.items():
            kl = k.lower()
            if kl in ("flat", "plane", "flat_ground"):
                v.proportion = 0.6; keep[k] = v
            elif "slope" in kl and "stairs" not in kl:
                v.proportion = 0.2
                if hasattr(v, "slope_range"):
                    v.slope_range = (0.0, 0.15)
                keep[k] = v
        if not any(("flat" in k.lower()) for k in keep):
            import isaaclab.terrains as terrain_gen
            keep["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.6)
        gen.sub_terrains = keep
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-1.0, 2.5); c.ranges.lin_vel_y = (-0.5, 0.5); c.ranges.ang_vel_z = (-1.5, 1.5)
        c.resampling_time_range = (2.0, 5.0)
        if hasattr(c, "rel_standing_envs"):
            c.rel_standing_envs = 0.1
        R = self.rewards
        if hasattr(R, "base_height_flat"):
            R.base_height_flat.params["target_height"] = 0.35
        if hasattr(R, "joint_torques_wheel_l2"):
            R.joint_torques_wheel_l2.weight = -2e-4
        mat = getattr(self.events, "randomize_rigid_body_material", None)
        if mat is not None:
            mat.params["static_friction_range"] = (0.4, 1.0); mat.params["dynamic_friction_range"] = (0.3, 0.9)
        self.scene.robot.actuators["wheel"].effort_limit_sim = 20.0
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 42.0
        print("[s10-flatfast] 地形=%s  vx %s  重采样 %s  站高 0.35  摩擦 0.4~1.0  轮 20 N·m / 42 rad/s" % (
            {k: round(v.proportion, 2) for k, v in keep.items()}, c.ranges.lin_vel_x, c.resampling_time_range))

def _apply_crouch(cfg, hipy: float = 0.72, knee: float = 1.44, z: float = 0.37):
    """09-10 作者拍板：蹲姿默认位形（零动作 = 蹲）。hipy 0.72 / knee 1.44 → 站高 0.35、腿长 0.27（厂家踏步实测 0.24–0.28）。
    部署时 runner 设 S10_DEF_HIPY=0.72 S10_DEF_KNEE=1.44。deepcopy 以免改到模块级 DEEPROBOTICS_S10_DEPLOY_CFG。"""
    import copy
    cfg.scene.robot.init_state = copy.deepcopy(cfg.scene.robot.init_state)
    jp = cfg.scene.robot.init_state.joint_pos
    jp["f[l,r]_hipy_joint"] = -hipy; jp["h[l,r]_hipy_joint"] = hipy
    jp["f[l,r]_knee_joint"] = knee; jp["h[l,r]_knee_joint"] = -knee
    cfg.scene.robot.init_state.pos = (0.0, 0.0, z)
    print("[s10-crouch] 默认位形 hipy ∓%.2f knee ±%.2f 出生 z %.2f（站高≈0.35）" % (hipy, knee, z))


class DeeproboticsS10FlatFast14EnvCfg(DeeproboticsS10FlatFastEnvCfg):
    """FlatFast 冲极速微调版（09-10 作者拍板"按高了改再微调"）：轮力矩 20 → 14（手册值）；指令上限 2.5 → 4.5 m/s；
    轮速上限 42 → 62 rad/s（纯滚 5.0 m/s，留在硬件空载 65.45 之下）；偏航指令收到 ±1.0（4.5 m/s 配 1.5 rad/s 的侧向加速度超出摩擦，是无效指令）。"""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-1.0, 4.5); c.ranges.ang_vel_z = (-1.0, 1.0)
        self.scene.robot.actuators["wheel"].effort_limit_sim = 14.0
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 62.0
        _apply_crouch(self)
        print("[s10-flatfast14] 冲极速：vx %s wz %s  轮 14 N·m / 62 rad/s" % (c.ranges.lin_vel_x, c.ranges.ang_vel_z))


class DeeproboticsS10TrotCEnvCfg(DeeproboticsS10TrotEnvCfg):
    """C 版 = 越野专用：只有粗糙地/坡/矮箱（无平地），速度到 1.6 m/s；真机踏步周期做参考态出生（50%）+ 周期跟踪奖励（w 3）；
    步态奖励保留；轮速上限 42。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains
        for k in list(st):
            if k.lower() in ("flat", "plane", "flat_ground"):
                del st[k]
        tot = sum(v.proportion for v in st.values())
        for v in st.values():
            v.proportion = v.proportion / tot
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.5, 1.6)
        self.events.spawn_trot_ref = EventTerm(func=trot_ref.spawn_trot_ref, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "ref_prob": 0.5, "z": 0.36})
        self.rewards.st_trot_ref = RewTerm(func=trot_ref.TrotRefTrack, weight=3.0, params={"sigma_q": 0.3, "v_min": 0.3})
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 42.0
        self.scene.robot.actuators["wheel"].effort_limit_sim = 14.0          # 09-10 作者拍板：按硬件手册 14 N·m（20 是实测打滑瞬间的数）
        _apply_crouch(self)
        print("[s10-trot-C] 越野专用 地形=%s vx %s  真机踏步参考出生 0.5 + 周期跟踪 w3  轮 14 N·m / 42 rad/s" % ({k: round(v.proportion, 2) for k, v in st.items()}, c.ranges.lin_vel_x))


class DeeproboticsS10ClimbExpertBEnvCfg(DeeproboticsS10ClimbExpertEnvCfg):
    """专家变体 B（09-09 14:50，与 A 并行）：前轮已搭上出生态 35%→60%（多练主蹬），蹬起奖励 4→8。"""

    def __post_init__(self):
        super().__post_init__()
        self.events.spawn_at_wall.params["hooked_prob"] = 0.6
        self.rewards.st_launch.weight = 8.0
        print("[s10-climb-B] hooked_prob=0.6  st_launch w=8")


def _apply_realmap_and_heading(cfg, hd_w: float = 3.0, drift_w: float = -8.0, ang_std: float = 0.30,
                               degrade: bool = False):
    """09-10 作者拍板的两刀，平地/越野两条线共用：
      ① 高程图退化（几何遮挡 + 空洞 + 0~3 步延迟 + ±1 格错位 + 20% 回合整张图 = 常数 −0.10），
         让策略见到真机 雷达→位姿→hmap_node 那条链路会给出的图，而不是俯视真值图。退化项内部只给
         有效格加噪声，所以 ObsTerm 上的 noise 摘掉：真实填洞是精确常数，这个特征要留给策略去识别。
      ② 走歪惩罚：直接罚 当前航向 − 指令航向目标，并把偏航跟踪核 0.707 收紧到 ang_std。
         机理见 mdp/heading_hold.py：策略观测里没有 yaw，但漂移是它自己积出来的，压住角速度误差就不漂；
         外力造成的偏差它纠正不了，那一半要靠部署侧把航向外环补回来。"""
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import percept_degrade, heading_hold
    if degrade:                      # 09-10 作者拍板：这一轮先不开（雷达接通后就不会全瞎），模块留着备用
        cfg.observations.policy.height_scan = ObsTerm(
            func=percept_degrade.HeightScanDegraded,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5, "noise": 0.1,
                    "latency_max": 3, "mode_probs": (0.25, 0.30, 0.25, 0.20)},
            noise=None, clip=(-1.0, 1.0), scale=1.0,
        )
    cfg.rewards.st_heading_hold = RewTerm(func=heading_hold.heading_hold_reward, weight=hd_w,
                                          params={"command_name": "base_velocity", "sigma": 0.4})
    cfg.rewards.st_heading_drift = RewTerm(func=heading_hold.heading_drift_penalty, weight=drift_w,
                                           params={"command_name": "base_velocity", "dead": 0.15})
    cfg.rewards.track_ang_vel_z_exp.params["std"] = ang_std
    print("[s10-realmap] 高程图退化=%s；走歪 hold w%.1f / drift w%.1f；偏航跟踪 std 0.707→%.2f" % ("开" if degrade else "关", hd_w, drift_w, ang_std))


@configclass
class DeeproboticsS10FlatFast15EnvCfg(DeeproboticsS10FlatFast14EnvCfg):
    """平地极速 · 真机化版 = FlatFast14 + 高程图退化 + 走歪惩罚。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_realmap_and_heading(self)
        print("[s10-flatfast15] = FlatFast14 + 走歪惩罚（只这一刀）")


@configclass
class DeeproboticsS10TrotDEnvCfg(DeeproboticsS10TrotCEnvCfg):
    """越野 · 真机化版 = TrotC + 高程图退化 + 走歪惩罚。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_realmap_and_heading(self)
        print("[s10-trot-D] = TrotC + 走歪惩罚（只这一刀）")


def _apply_real_dynamics(cfg, delay: int = 3, zero_off: float = 0.02, wheel_gain: float = 0.03):
    """09-11：把真机上有、仿真里没有的两样东西补进训练 —— 回路延迟与左右不对称。
    依据：真机 bag 1 偏航角速度在 1.41 Hz 自激振荡（占 49% 能量、std 31°/s），仿真同策略只有 0.07~0.11°/s 且无此峰；
    厂家踏步的峰恰等于其步频（1.10/1.10/1.98 Hz）属步态节奏。训练侧隐式执行器 0 ms 延迟，真机 ≥20 ms 策略周期 + DDS。
    另：真机四腿 hipx 左右不对称约 0.6°（厂家控制器下也有），左右实际轮速差约 2.5%。
    做法见 mdp/real_actions.py：动作延迟 0~delay 步、腿零位偏置 ±zero_off rad、轮增益 1±wheel_gain。"""
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import real_actions as ra
    a = cfg.actions
    jp, jv = a.joint_pos, a.joint_vel
    a.joint_pos = ra.RealJointPositionActionCfg(
        asset_name=jp.asset_name, joint_names=jp.joint_names, scale=jp.scale,
        use_default_offset=jp.use_default_offset, clip=jp.clip, preserve_order=jp.preserve_order,
        delay_steps=delay, zero_offset=zero_off)
    a.joint_vel = ra.RealJointVelocityActionCfg(
        asset_name=jv.asset_name, joint_names=jv.joint_names, scale=jv.scale,
        use_default_offset=jv.use_default_offset, clip=jv.clip, preserve_order=jv.preserve_order,
        delay_steps=delay, gain_range=wheel_gain)
    print("[s10-realdyn] 动作延迟 0~%d 步(=0~%d ms)｜腿零位偏置 ±%.3f rad｜轮增益 1±%.0f%%" % (
        delay, delay * 20, zero_off, 100 * wheel_gain))


@configclass
class DeeproboticsS10FlatFast16EnvCfg(DeeproboticsS10FlatFastEnvCfg):
    """平地极速 · 真机动力学版 = FlatFast14（蹲姿 + vx 到 4.5 + 轮 14N·m/62rad·s⁻¹）+ 延迟与左右不对称随机化。
    不含走歪惩罚：09-11 实测那一刀在仿真里只值 0.4°/s，却掉 8% 极速，根因是延迟不是奖励。"""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-1.0, 4.5); c.ranges.ang_vel_z = (-1.0, 1.0)
        self.scene.robot.actuators["wheel"].effort_limit_sim = 14.0
        self.scene.robot.actuators["wheel"].velocity_limit_sim = 62.0
        _apply_crouch(self)
        _apply_real_dynamics(self)
        print("[s10-flatfast16] = FlatFast14 + 真机动力学随机化")


@configclass
class DeeproboticsS10TrotEEnvCfg(DeeproboticsS10TrotCEnvCfg):
    """越野 · 真机动力学版 = TrotC + 延迟与左右不对称随机化。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_real_dynamics(self)
        print("[s10-trot-E] = TrotC + 真机动力学随机化")


@configclass
class DeeproboticsS10PerceptV1HEnvCfg(DeeproboticsS10PerceptV1GEnvCfg):
    """通用线 · 真机动力学版 = V1G + 延迟与左右不对称随机化（站高仍 0.42，不蹲）。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_real_dynamics(self)
        print("[s10-v1h] = V1G + 真机动力学随机化")


@configclass
class DeeproboticsS10ClimbExpertHEnvCfg(DeeproboticsS10ClimbExpertGEnvCfg):
    """上台面专家 · 真机动力学版 = ClimbG + 延迟与左右不对称随机化。
    上墙靠的是贴墙推进与相位动作，对延迟更敏感，本轮延迟上限同样取 0~3 步。"""

    def __post_init__(self):
        super().__post_init__()
        _apply_real_dynamics(self)
        print("[s10-climb-H] = ClimbG + 真机动力学随机化")


@configclass
class DeeproboticsS10PerceptV1DownEnvCfg(DeeproboticsS10PerceptV1HEnvCfg):
    """通用线 · 下坎强化版 = V1H（V1G + 真机动力学）+ 大幅加重下行地形。

    09-11 彩排实测（MuJoCo 部署栈，真值高程图）：三条线在 ≤1.4 m/s 时遇到 ≥25 cm 下坎都会**在坎沿收住**，
    要 1.8 m/s 的动量才过；31 cm 路缘 V1G 三档全下不去，FlatFast16 三档姿态全崩。而决赛路线含
    台面到路 −0.25 与 下路缘 −0.31 两处，是真实阻塞点。收住时策略收到的高程图是完全正确的
    （脚下 −0.08、前方 +0.17），所以是行为选择不是感知问题 —— 要靠地形配比和停滞惩罚改。

    三处改动：
      ① step_down 占比 0.05 → 0.30，立面范围 (0.28,0.40) → (0.12,0.40)（原来根本没练过 25 cm 这一档）；
      ② pyramid_stairs（中心在顶、向外下行）0.20 → 0.30；
      ③ stall_penalty −2 → −8：坎沿收住就是"有前进指令却不前进"，正是这一项该罚的。
    """

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        if "step_down" in st:
            st["step_down"].proportion = 0.30
            st["step_down"].step_height_range = (0.12, 0.40)
        if "pyramid_stairs" in st:
            st["pyramid_stairs"].proportion = 0.30
        if getattr(self.rewards, "stall_penalty", None) is not None:
            self.rewards.stall_penalty.weight = -8.0
        print("[s10-v1down] 下坎强化：step_down 占比 %.2f 立面 %s｜pyramid_stairs 占比 %.2f｜stall_penalty %.1f" % (
            st["step_down"].proportion if "step_down" in st else -1,
            st["step_down"].step_height_range if "step_down" in st else None,
            st["pyramid_stairs"].proportion if "pyramid_stairs" in st else -1,
            self.rewards.stall_penalty.weight))


@configclass
class DeeproboticsS10PerceptV1StairEnvCfg(DeeproboticsS10PerceptV1DownEnvCfg):
    """通用线 · 上台阶强化版 = V1Down（真机动力学 + 下坎强化）+ 上行楼梯加重。

    09-11 依据：
      · MuJoCo 部署栈 bench_park（12/15/18/20 cm，踏面 0.8 m）：V1Down 从 3 m 外助跑上了第一级 12 cm，
        站在 0.8 m 踏面上**顶第二级 15 cm 15 s 上不去**（抬头 −15~−17°），最后滑回地面。
      · 训练地形里上行楼梯（pyramid_stairs_inv，坑底往外爬）只占 0.1、踏面只有 0.3 m；下行占 0.3。
        从没练过"站在宽踏面上、无助跑、逐级往上"。决赛楼梯 0.12 m 立面 / 0.65 m 踏面，每一级都是这种情形。
      · 厂家上台阶（s10_hmap_check_1）：下蹲 0.405→0.34、持续抬头约 14°、前膝深屈 1.0~1.45、后轮推，
        每级约 1.9 s 且节奏不规整（事件驱动），故不做相位参考，只靠地形与课程。
    改动：
      ① pyramid_stairs_inv 占比 0.10 → 0.40（窄踏面 0.3 m 上行）；
      ② 新增 stairs_up_wide：宽踏面 0.65 m 上行，立面 0.08~0.22 m（覆盖决赛楼梯 0.12 与园区台阶 0.15~0.20）。
    代价预期：climb_up / approach_riser 占比被稀释，33 cm 上墙成功率可能下降，训练后必须复测。"""

    def __post_init__(self):
        super().__post_init__()
        import isaaclab.terrains as terrain_gen          # 局部导入兜底：不依赖模块层是否已有此别名
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains
        if "pyramid_stairs_inv" in st:
            st["pyramid_stairs_inv"].proportion = 0.40
        st["stairs_up_wide"] = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.08, 0.22), step_width=0.65,
            platform_width=2.0, border_width=1.0, holes=False)
        tot = sum(v.proportion for v in st.values())
        print("[s10-v1stair] 上台阶强化：pyramid_stairs_inv %.2f（踏面 0.3）｜stairs_up_wide %.2f（踏面 0.65，立面 0.08~0.22）｜上行合计占 %.0f%%" % (
            st["pyramid_stairs_inv"].proportion, st["stairs_up_wide"].proportion,
            100 * (st["pyramid_stairs_inv"].proportion + st["stairs_up_wide"].proportion) / tot))


@configclass
class DeeproboticsS10StairExpertEnvCfg(DeeproboticsS10PerceptV1StairEnvCfg):
    """窄踏面上楼梯专家（09-11 16:30）：V1Stair（真机动力学 + 下坎强化 + 宽踏面楼梯）只换地形与指令。

    09-11 依据：
      · 厂家上楼梯包 s10_hmap_check_1（关节正运动学 + IMU 反推，包里没有高程图/里程计）：立面约 14~15 cm、
        踏面约 0.35~0.40 m（坡约 20°），每 3~5 级一个平台；左右交替抬腿，约 1.9 s 一级，机身抬头约 18°。
      · MuJoCo 部署栈（顶上带 40 m 平台）：V1H 在踏面 0.30 上 8 cm 只到 6/8 级、10 cm 5/8、12 cm 2/8；
        0.17/0.30 所有策略最多到第 3 级。宽踏面 0.65 / 12 cm：V1Stair 2/2 上顶。
      · 训练日志：V1Stair 末期 pyramid_stairs_inv（踏面 0.3 上行）课程只到 0.87 行（约 7.5 cm）。V1 指令含后退/横移/转向，
        坑里只有朝前能出去 → 等级被非前进的局拖住，朝前上窄楼梯几乎没练到高行。
    改动：
      ① 地形只留上行楼梯：踏面 0.30 / 0.35 / 0.40（立面 0.05~0.18，10 行课程）+ 宽踏面 0.65 少量 + random_rough 少量；
        平台 2.0 m（出生离第一级约 1 m，每侧 5~6 级）。
      ② 指令只朝前 0.2~0.6、横移 0、航向锁 +x（出生朝向 ±0.3 rad），一局一个指令。
      ③ 轮高里程碑 max_rise 0.5 → 1.0（坑深最多约 6×0.18 m）。
    部署设想：导航层在楼梯前切 /climb_mode 2（runner 的 S10_POLICY_PIT 槽位，超时锁存对它同样生效），上完切回 0。"""

    def __post_init__(self):
        super().__post_init__()
        import isaaclab.terrains as terrain_gen
        gen = self.scene.terrain.terrain_generator
        old = gen.sub_terrains
        subs = {}
        for w, p in ((0.30, 0.30), (0.35, 0.25), (0.40, 0.20)):
            subs["stairs_n%02d" % round(w * 100)] = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=p, step_height_range=(0.05, 0.18), step_width=w, platform_width=2.0, border_width=1.0, holes=False)
        subs["stairs_up_wide"] = old["stairs_up_wide"]; subs["stairs_up_wide"].proportion = 0.15
        if "random_rough" in old:
            subs["random_rough"] = old["random_rough"]; subs["random_rough"].proportion = 0.10
        gen.sub_terrains = subs
        gen.curriculum = True
        self.scene.terrain.max_init_terrain_level = 3
        c = self.commands.base_velocity
        c.heading_command = True
        if hasattr(c, "rel_heading_envs"):
            c.rel_heading_envs = 1.0
        c.ranges.lin_vel_x = (0.2, 0.6); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (-0.6, 0.6); c.ranges.heading = (0.0, 0.0)
        for attr in ("rel_standing_envs", "rel_zero_vel_envs", "rel_only_ang_z_envs", "rel_only_lin_y_envs", "rel_only_lin_x_envs"):
            if hasattr(c, attr):
                setattr(c, attr, 0.0)
        c.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.events.randomize_reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        for nm in ("wheel_height_progress", "wheel_height_record"):
            t = getattr(self.rewards, nm, None)
            if t is not None:
                t.params["max_rise"] = 1.0
        print("[s10-stairN] 窄踏面上楼梯专家：地形 %s｜指令 vx %s 航向锁 +x｜出生偏航 ±0.3｜max_rise 1.0｜每局 %.0f s" % (
            {k: round(v.proportion, 2) for k, v in subs.items()}, c.ranges.lin_vel_x, self.episode_length_s))


@configclass
class DeeproboticsS10PerceptV1DownBEnvCfg(DeeproboticsS10PerceptV1DownEnvCfg):
    """V1Down-B（09-12 真机反馈续训，从 V1Down 10695 续）：只加两处，其它不动。
      ① 四轮驱动均衡惩罚（mdp/wheel_balance.py）：真机后轮不发力、前轮出力大（仿真 0.3 m/s 前轮 −7/+10、后轮 0.7）。
      ② 转向：原来 heading_command + rel_heading_envs=1.0，策略见到的转速指令都是"朝向误差 × 0.5"，原地转 0.6 以上跟不上
         （仿真：0.6 → 0.35、1.0 → 0.46 rad/s；V1H 同样设置却能到 0.68/0.85，说明是 V1Down 续训时丢的）。改成一半环境直接给转速，
         范围 ±1.2（部署 wz 上限 0.6～1.0）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        self.rewards.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-0.5, params={"lr_only": False})
        self.rewards.rear_wheel_share = RewTerm(func=wb.rear_wheel_share, weight=0.0)   # 只看曲线
        c = self.commands.base_velocity
        if hasattr(c, "rel_heading_envs"):
            c.rel_heading_envs = 0.5
        c.ranges.ang_vel_z = (-1.2, 1.2)
        print("[s10-v1down-B] 四轮均衡 −0.5（前后 + 左右）｜rel_heading_envs 0.5｜ang_vel_z ±1.2")


@configclass
class DeeproboticsS10StairExpertBEnvCfg(DeeproboticsS10StairExpertEnvCfg):
    """StairN-B（09-12 真机楼梯翻车后续训，从 StairN-C 9300 续）：只加左右轮驱动均衡惩罚，其它不动。
    9300 切入后左前轮动作 −13、其余 −3，真机石材上左前轮空转侧翻；所有 StairN-C 检查点都这样，是训练线通病。
    上楼梯前后轮出力本来不同，所以只罚左右不均衡（lr_only）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        self.rewards.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-0.5, params={"lr_only": True})
        self.rewards.rear_wheel_share = RewTerm(func=wb.rear_wheel_share, weight=0.0)
        print("[s10-stairN-B] 左右轮均衡 −0.5")


@configclass
class DeeproboticsS10PerceptV1HTrotEnvCfg(DeeproboticsS10PerceptV1HEnvCfg):
    """V1H-Trot（09-12 19:2x，作者："直线行走要学会踏步和降低重心：运动时轮子会往外撇，踏步可以调整姿态"；作者拍板"按你的来"）。
    从 V1H_9196 续训，默认站姿仍是 0.42 那组（与 ClimbH / 楼梯专家共用一个 runner、赛段切换不受影响），地形与指令范围同 V1H。加的东西：
      ① 踏步奖励（trot_gait：对角同步 / 抬腿 7 cm / 按速度的腾空时长 / 该踏不踏罚），只在前进 ≥0.5 m/s 时算，低速与原地转仍是纯轮式；
      ② 站高目标 0.43 → 0.36（动作范围内蹲得下去，不换默认姿态）；
      ③ 髋外展惩罚 −3 → −6（行进中腿往外撇要收回来），横向推力 ±0.5 → ±1.0 且 4～8 s 一次，摩擦下界 0.35 → 0.30——让仿真里也出现"腿被顶开"，学会用踏步收腿；
      ④ 四轮驱动均衡惩罚（前后 + 左右，mdp/wheel_balance.py）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_gait as trot
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        R = self.rewards
        R.st_diag = RewTerm(func=trot.diag_sync_reward, weight=0.5, params={"v_min": 0.5})
        R.st_clearance = RewTerm(func=trot.swing_clearance_reward, weight=0.5, params={"h_target": 0.07, "sigma": 0.04, "v_min": 0.5})
        R.st_air_time = RewTerm(func=trot.air_time_target_reward, weight=2.0, params={"v_min": 0.5, "sigma": 0.08})
        R.st_no_step = RewTerm(func=trot.no_step_penalty, weight=-0.5, params={"v_min": 0.5, "t_max": 1.0})
        if hasattr(R, "base_height_flat"):
            R.base_height_flat.params["target_height"] = 0.36
        if getattr(R, "hipx_joint_pos_penalty", None) is not None:
            R.hipx_joint_pos_penalty.weight = -6.0
        R.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-0.5, params={"lr_only": False})
        R.rear_wheel_share = RewTerm(func=wb.rear_wheel_share, weight=1e-6)   # 只看曲线（权重 0 会显示 0）
        ev = getattr(self.events, "randomize_push_robot", None)
        if ev is not None:
            ev.params["velocity_range"]["y"] = (-1.0, 1.0)
            ev.interval_range_s = (4.0, 8.0)
        mat = getattr(self.events, "randomize_rigid_body_material", None)
        if mat is not None:
            for k in ("static_friction_range", "dynamic_friction_range"):
                if k in mat.params:
                    lo, hi = mat.params[k]; mat.params[k] = (min(lo, 0.30), hi)
        print("[s10-v1h-trot] 踏步（≥0.5 m/s）｜站高 0.36｜hipx 罚 −6｜横向推 ±1.0 每 4~8 s｜摩擦下界 0.30｜四轮均衡 −0.5")


@configclass
class DeeproboticsS10StairExpertB2EnvCfg(DeeproboticsS10StairExpertEnvCfg):
    """StairN-B2（09-12 20:1x）：StairN-B（力矩左右均衡 −0.5，训 1000 步）没改动左前轮独大 → 加重到 −5，并直接罚左右轮动作差 −0.2。其它同 9300。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        self.rewards.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-5.0, params={"lr_only": True})
        self.rewards.wheel_action_lr = RewTerm(func=wb.wheel_action_lr_imbalance, weight=-0.2)
        print("[s10-stairN-B2] 左右轮力矩均衡 −5 + 左右轮动作差 −0.2")


@configclass
class DeeproboticsS10StairExpertDEnvCfg(DeeproboticsS10StairExpertEnvCfg):
    """StairN-D（09-12 21:2x，从 StairN-C 9300 续训）：作者看了真机 1139/1140 录像后要的"左右前轮轮流抬腿踩上一级、两后轮跟上"（官方步态）。
    9300 现在是"两前轮一起跳上一级"（仿真回放按帧算接触：两前轮同时离地占 38%），真机右前落上、左前没落上 → 侧倾 32° 往左翻。
    加的东西（mdp/stair_step.py）：两前轮同时离地罚 −0.5；轮子顶立面罚 −0.2/只；单前轮摆到高一级 +0.5（只在翻越相位）；指令 0.15～0.40 + 超速罚 −1（作者：像官方一样慢）；
    并入 B2 的左右轮力矩均衡 −5 与左右轮动作差 −0.2（D 包含 B2）。其它同 9300。会比 9300 慢，换的是三点着地。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-5.0, params={"lr_only": True})
        R.wheel_action_lr = RewTerm(func=wb.wheel_action_lr_imbalance, weight=-0.2)
        R.ss_pair_flight = RewTerm(func=ss.front_pair_flight_penalty, weight=-0.5)
        R.ss_riser_push = RewTerm(func=ss.wheel_riser_push_penalty, weight=-0.2)
        R.ss_single_swing = RewTerm(func=ss.front_single_swing_reward, weight=0.5, params={"h_target": 0.18, "sigma": 0.08})
        # 作者 21:4x："上台阶继续放慢会好点，官方运控很慢"。9300 不听速度指令（仿真指令 0.2/0.3/0.5 实际都 0.5 m/s），
        # 这里指令范围压到 0.15～0.40（官方约 1.9 s 一级 ≈ 0.29 m/s）并罚超速，让操作员的速度旋钮在楼梯上真的管用。
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (0.15, 0.40)
        R.ss_overspeed = RewTerm(func=ss.overspeed_penalty, weight=-1.0, params={"margin": 0.10})
        # 作者 21:5x："还需要可以在途中调整前进方向"。9300 在楼梯上不听转向（仿真航向保持目标 10~30° 实际只 5~9°）：
        # 基类把航向锁死 +x（heading (0,0)、全部环境走朝向误差）。这里目标朝向放开 ±20°，一半环境直接给转速 ±0.5（V1Down-B 同法，转向跟踪 0.35→0.85）。
        c.ranges.heading = (-0.35, 0.35)
        c.ranges.ang_vel_z = (-0.5, 0.5)
        if hasattr(c, "rel_heading_envs"):
            c.rel_heading_envs = 0.5
        print("[s10-stairN-D] 两前轮同时离地 −0.5｜顶立面 −0.2/只｜单前轮摆高一级 +0.5（翻越相位）｜左右轮力矩均衡 −5 + 动作差 −0.2｜指令 0.15~0.40 + 超速罚 −1｜途中可转向：目标朝向 ±20°、半数环境直接转速 ±0.5")


@configclass
class DeeproboticsS10StairExpertEEnvCfg(DeeproboticsS10StairExpertDEnvCfg):
    """StairN-E（09-12 深夜，官方四拍步行参考版）：D 的全部项之上，加官方运控 200 Hz 录包（gfstj1）做成的周期参考
    `~/s10_logs/ref/ref_stairwalk_official.npz`（对角配对步行 1.17 Hz @0.2 m/s：左前 0.08 → 右后 0.14 → 右前 0.64 → 左后 0.78，两前轮从不同时腾空）：
    参考态出生 50%（楼梯前平台上按参考相位放，机身 z 0.40）+ 周期跟踪 w 2（σ 0.3，vx ≥0.1 时算）。
    接触项加重：两前轮同时腾空 −2、任意 ≥2 轮离地 −1/轮（视频：任何时刻 ≥3 轮着地）、顶立面 −1/只、单前轮摆高一级（0.18，视频上级那步抬到腹部）+1；轮力矩超 5 N·m 罚 −0.05/N·m（官方 p95 2.5）。
    地形加决赛踏面 0.55（占 0.3，立面 0.08～0.16）；摩擦下界 0.4。速度/转向同 D（0.15～0.40 + 超速罚；朝向 ±20°、半数环境 ±0.5）。"""

    def __post_init__(self):
        super().__post_init__()
        import os
        import isaaclab.terrains as terrain_gen
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        trot_ref.REF_PATH = os.path.expanduser("~/s10_logs/ref/ref_stairwalk_official.npz")
        R = self.rewards
        # 00:0x 第一版权重（跟踪 2、两前轮 −2、≥2 轮 −1、顶立面 −1）跑 200 步各项全平（两前轮腾空 6.5%、顶立面 0.55 只/步、跟踪 0.21），
        # 和 D 一样拉不动 9300 的"一起跳"：一次跳上一级拿 wheel_height 进度+纪录约 65 分，跳 0.3 s 只罚 30。改成让跳不划算：
        self.events.spawn_trot_ref = EventTerm(func=trot_ref.spawn_trot_ref, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "ref_prob": 0.7, "z": 0.40})
        R.st_stair_ref = RewTerm(func=trot_ref.TrotRefTrack, weight=6.0, params={"sigma_q": 0.25, "v_min": 0.1})
        R.ss_pair_flight.weight = -8.0
        R.ss_riser_push.weight = -2.0
        R.ss_single_swing.weight = 3.0; R.ss_single_swing.params = {"h_target": 0.18, "sigma": 0.06}   # 240 fps 视频：前轮过台沿余量 4～6 cm → 立面 0.13 + 0.05
        R.ss_wheel_torque = RewTerm(func=ss.wheel_torque_excess_penalty, weight=-0.1, params={"tau_max": 5.0})
        R.ss_multi_flight = RewTerm(func=ss.multi_flight_penalty, weight=-4.0, params={"v_min": 0.1})   # 视频：任何时刻 ≥3 轮着地
        gen = self.scene.terrain.terrain_generator; subs = gen.sub_terrains
        subs["stairs_n55"] = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.08, 0.16), step_width=0.55, platform_width=2.0, border_width=1.0, holes=False)
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values():
            v.proportion = v.proportion / tot
        mat = getattr(self.events, "randomize_rigid_body_material", None)
        if mat is not None:
            for k in ("static_friction_range", "dynamic_friction_range"):
                if k in mat.params:
                    lo, hi = mat.params[k]; mat.params[k] = (max(min(lo, 0.4), 0.4) if lo > 0.4 else lo, hi)
        print("[s10-stairN-E2] 官方步行参考出生 0.7 + 周期跟踪 w6/σ0.25｜两前轮同时腾空 −8｜≥2 轮离地 −4｜顶立面 −2｜单前轮摆高 0.18 +3｜轮力矩>5 −0.1｜地形 %s｜摩擦 %s" % (
            {k: round(v.proportion, 2) for k, v in subs.items()}, mat.params.get("static_friction_range") if mat is not None else "?"))


@configclass
class DeeproboticsS10PerceptV1HTrotVEnvCfg(DeeproboticsS10PerceptV1HEnvCfg):
    """Trot-V（09-12 深夜，官方踏步参考版，替代没学会踏步的 V1H-Trot）：从 V1H_9196 续训，指令范围、地形、默认姿态都不动（主策略要兼顾停/倒/转、与 runner 各槽共用姿态）。
    官方踏步 200 Hz 录包（gftbzx1 直线 + gftb1 转圈）做成参考 `~/s10_logs/ref/ref_step_official.npz`（对角小跑三档 0.41/0.69/1.03 m/s → 1.11/1.18/1.49 Hz，抬前 10～12 cm 后 7～9 cm，站高 0.34）：
    参考态出生 50%（z 0.35）+ 周期跟踪 w 3（σ 0.3，vx ≥0.3）；步态项权重比 V1H-Trot 加大（对角同步 1、抬腿 0.10 +1、腾空时长 +2、该踏不踏 −2）；
    站高目标 0.34；hipx 罚 −6；横向推 ±1.0 每 4～8 s；摩擦下界 0.30；四轮均衡 −0.5。V1H-Trot 失败原因：没有相位参考，步态项 0.5～2 拉不出纯滚的局部最优。"""

    def __post_init__(self):
        super().__post_init__()
        import os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_gait as trot
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        trot_ref.REF_PATH = os.path.expanduser("~/s10_logs/ref/ref_step_official.npz")
        R = self.rewards
        self.events.spawn_trot_ref = EventTerm(func=trot_ref.spawn_trot_ref, mode="reset", params={"asset_cfg": SceneEntityCfg("robot"), "ref_prob": 0.5, "z": 0.35})
        R.st_trot_ref = RewTerm(func=trot_ref.TrotRefTrack, weight=3.0, params={"sigma_q": 0.3, "v_min": 0.3})
        R.st_diag = RewTerm(func=trot.diag_sync_reward, weight=1.0, params={"v_min": 0.3})
        R.st_clearance = RewTerm(func=trot.swing_clearance_reward, weight=1.0, params={"h_target": 0.10, "sigma": 0.04, "v_min": 0.3})
        R.st_air_time = RewTerm(func=trot.air_time_target_reward, weight=2.0, params={"v_min": 0.3, "sigma": 0.08})
        R.st_no_step = RewTerm(func=trot.no_step_penalty, weight=-2.0, params={"v_min": 0.4, "t_max": 0.8})
        if hasattr(R, "base_height_flat"):
            R.base_height_flat.params["target_height"] = 0.34
        if getattr(R, "hipx_joint_pos_penalty", None) is not None:
            R.hipx_joint_pos_penalty.weight = -6.0
        R.wheel_drive_balance = RewTerm(func=wb.wheel_drive_balance, weight=-0.5, params={"lr_only": False})
        ev = getattr(self.events, "randomize_push_robot", None)
        if ev is not None:
            ev.params["velocity_range"]["y"] = (-1.0, 1.0)
            ev.interval_range_s = (4.0, 8.0)
        mat = getattr(self.events, "randomize_rigid_body_material", None)
        if mat is not None:
            for k in ("static_friction_range", "dynamic_friction_range"):
                if k in mat.params:
                    lo, hi = mat.params[k]; mat.params[k] = (min(lo, 0.30), hi)
        print("[s10-v1h-trotV] 官方踏步参考出生 0.5 + 周期跟踪 w3｜对角 1 抬腿 0.10 +1 腾空 +2 不踏 −2｜站高 0.34｜hipx −6｜横向推 ±1.0｜摩擦下界 0.30")


@configclass
class DeeproboticsS10PerceptV1HTrotV2EnvCfg(DeeproboticsS10PerceptV1HTrotVEnvCfg):
    """Trot-V2（09-13 上午）：Trot-V 10699 在 MuJoCo 里仍不踏步（站高 0.415、抬轮 1～2 mm），只学会"抬一只轮子躲过四轮全触地罚"（右前离地 19～35%）。
    改法照 E2 的经验——让"纯滚"不划算：按轮罚（任一轮触地 >0.8 s 罚 1/轮，−2）、参考跟踪 w3 → 6（σ 0.25）、参考出生 0.5 → 0.7、对角同步 +2、
    该踏不踏（四轮）−2 → −4；其它同 Trot-V。从 Trot-V 10699 续。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_gait as trot
        R = self.rewards
        # Trot-V 的 Isaac 日志里 st_trot_ref 全程 ≈0.02（w3 → 跟踪值 0.007）：官方踏步姿态（hipy ±0.78、knee ±1.55）离 V1H 默认姿态（∓0.35/±0.65）0.4～0.9 rad，
        # σ0.3 的 exp 核在这个距离上≈0、没有梯度；E2 能成是因为官方楼梯姿态离 9300 近。这里 σ 放宽到 0.5，让它从远处也有拉力。
        R.st_trot_ref.weight = 6.0; R.st_trot_ref.params["sigma_q"] = 0.5
        self.events.spawn_trot_ref.params["ref_prob"] = 0.7
        R.st_diag.weight = 2.0
        R.st_no_step.weight = -4.0
        R.st_wheel_no_step = RewTerm(func=trot.wheel_no_step_penalty, weight=-2.0, params={"v_min": 0.4, "t_max": 0.8})
        # 作者 09-13 10:2x："默认策略不需要会爬楼梯，让它踏步以及快速做好就行，爬楼梯交给专家"：
        #   地形去掉全部台阶/楼梯，只留平地/粗糙/坡/矮箱（同 TrotEnvCfg）；指令到 2.0 m/s；踏步项只在 0.3～1.2 m/s 要求，更快时允许纯滚（快靠轮子）。
        gen = self.scene.terrain.terrain_generator
        st = gen.sub_terrains if gen is not None else {}
        keep = {}
        for k, v in st.items():
            kl = k.lower()
            if "random_rough" in kl or "hf_random" in kl or kl in ("rough",):
                v.proportion = 0.25; keep[k] = v
            elif "slope" in kl:
                v.proportion = 0.12; keep[k] = v
            elif "boxes" in kl:
                v.proportion = 0.08; keep[k] = v
            elif "real_patch" in kl:
                v.proportion = 0.10; keep[k] = v
            elif kl in ("flat", "plane", "flat_ground"):
                v.proportion = 0.45; keep[k] = v    # 作者 10:3x："踏步我只需要平地踏步快速，越野稳定性可以慢一点"→ 平地占大头
        if not any(("flat" in k.lower()) for k in keep):
            import isaaclab.terrains as terrain_gen
            keep["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.45)
        tot = sum(v.proportion for v in keep.values())
        for v in keep.values():
            v.proportion = v.proportion / tot
        gen.sub_terrains = keep
        c = self.commands.base_velocity
        # 作者 10:3x："你看官方踏步的速度有多快，你按那个来"：09-09 真机踏步数据 trot_fast 档 1.86 m/s / 2.01 Hz、平地推满 2.2 m/s → 指令到 2.3，踏步全速段都要求。
        # 参考换成合并版 ref_step_merged.npz：今晚三档（0.41/0.69/1.03 m/s）+ 09-09 高速档（1.86 m/s，2.01 Hz）。
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as _tr
        _tr.REF_PATH = _os.path.expanduser("~/s10_logs/ref/ref_step_merged.npz")
        c.ranges.lin_vel_x = (-0.5, 2.3)
        for nm in ("st_trot_ref", "st_diag", "st_clearance", "st_air_time", "st_no_step", "st_wheel_no_step"):
            getattr(R, nm).params["v_max"] = 2.4
        print("[s10-v1h-trotV2] 参考跟踪 w6/σ0.5 出生 0.7｜对角 +2｜四轮不踏 −4｜按轮不踏（>0.8 s）−2/轮｜踏步区 0.3～2.3 m/s（官方 2.2）｜指令 vx %s｜地形 %s" % (
            c.ranges.lin_vel_x, {k: round(v.proportion, 2) for k, v in keep.items()}))


@configclass
class DeeproboticsS10StairExpertE3EnvCfg(DeeproboticsS10StairExpertEEnvCfg):
    """StairN-E3（09-13 10:4x，从 E2_10300 续）：E2 逐级核对是"左前抬（悬空 80%、净空 3～4 cm）、右前贴台沿滚（悬空 30%、净空≈0）、后轮滚上"，
    不是两只前轮都抬。改：单前轮抬高奖励改成"每只前轮相对自己起抬点"各自算（front_lift_each，两只都能拿到）+3；
    前轮顶踢面罚加到 −4（只针对前轮：用 pressed 的前两列）；其余同 E2。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.ss_single_swing.weight = 0.0
        R.ss_front_lift = RewTerm(func=ss.front_lift_each_reward, weight=3.0, params={"h_target": 0.18, "sigma": 0.06})
        R.ss_riser_push.weight = -3.0
        print("[s10-stairN-E3] 两前轮各自抬高（相对起抬点 0.18）+3｜顶踢面 −3｜其余同 E2")


@configclass
class DeeproboticsS10PerceptV1HTrotHEnvCfg(DeeproboticsS10PerceptV1HTrotV2EnvCfg):
    """Trot-H（09-13 下午）：Trot-V2 的地形/指令/参考不变，把全部"接触阈值"步态项换成几何净空版（mdp/step_gait.py：轮底比四轮最低点高 ≥4 cm 才算摆动）。
    根因：V1H-Trot / Trot-V / Trot-V2 / 09-10 TrotE 在 MuJoCo 里全部只抬 1～2 mm——接触传感器 1 N 阈值判离地，瞬间卸载就算，策略从没真抬。
    项：对角同步 +2、抬高 0.09（σ0.04）+2、摆动时长目标 +2、按轮 0.9 s 没真抬 −3；参考跟踪 w6/σ0.5、出生 0.7 保留；从 V1H_9196 重新起（不从学了卸载技巧的检查点续）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        R = self.rewards
        for nm in ("st_diag", "st_clearance", "st_air_time", "st_no_step", "st_wheel_no_step"):
            getattr(R, nm).weight = 0.0
        R.sg_diag = RewTerm(func=sg.sg_diag_sync, weight=2.0, params={"v_min": 0.3, "v_max": 2.4})
        R.sg_clear = RewTerm(func=sg.sg_clearance, weight=2.0, params={"h_target": 0.09, "sigma": 0.04, "v_min": 0.3, "v_max": 2.4})
        R.sg_swing = RewTerm(func=sg.sg_swing_time, weight=2.0, params={"v_min": 0.3, "v_max": 2.4})
        R.sg_idle = RewTerm(func=sg.sg_no_swing, weight=-3.0, params={"t_max": 0.9, "v_min": 0.3, "v_max": 2.4})
        print("[s10-v1h-trotH] 几何净空步态项（≥4 cm 才算摆动）：对角 +2｜抬高 0.09 +2｜摆动时长 +2｜按轮 0.9 s 不抬 −3｜参考跟踪 w6 保留")


@configclass
class DeeproboticsS10PerceptV1HTrotH2EnvCfg(DeeproboticsS10PerceptV1HTrotHEnvCfg):
    """Trot-H2（09-13 13:4x，从 Trot-H model_9800 续，探索噪声重置 0.5）：Trot-H 500 步学会真抬（右前 11 cm、左后 7～10 cm）但"举着不放"（离地 82～95%），
    没有落地→没有周期。改：抬高分只给摆动前 0.5 s（举着不给）、举着超过 0.6 s 罚 −3/轮、落地时按摆动时长给分的权重 2 → 6、按轮 0.9 s → 0.7 s 不抬罚。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        R = self.rewards
        R.sg_clear.params["max_dur"] = 0.5
        R.sg_hold = RewTerm(func=sg.sg_hold_penalty, weight=-3.0, params={"max_dur": 0.6, "v_min": 0.3, "v_max": 2.4})
        R.sg_swing.weight = 6.0
        R.sg_idle.params["t_max"] = 0.7
        print("[s10-v1h-trotH2] 抬高分只给前 0.5 s｜举着 >0.6 s −3/轮｜落地摆动时长 +6｜按轮 0.7 s 不抬 −3")


@configclass
class DeeproboticsS10PerceptV1HTrotH3EnvCfg(DeeproboticsS10PerceptV1HTrotH2EnvCfg):
    """Trot-H3（09-13 14:3x，从 Trot-H2 续）：H2 在 1.0 m/s 已是标准对角小跑（四轮各离地 34～72%、抬 8～10 cm、对角同步 86%），
    但 0.5 和 1.8 m/s 仍"举着一对对角腿不放"（离地 84～100%）。加重：举着 >0.5 s 罚 −3 → −8/轮；落地摆动时长 +6 → +8；按轮 0.7 → 0.6 s 不抬罚；
    抬高分只给摆动前 0.4 s。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.sg_hold.weight = -8.0; R.sg_hold.params["max_dur"] = 0.5
        R.sg_swing.weight = 8.0
        R.sg_idle.params["t_max"] = 0.6
        R.sg_clear.params["max_dur"] = 0.4
        print("[s10-v1h-trotH3] 举着 >0.5 s −8/轮｜落地摆动时长 +8｜按轮 0.6 s 不抬 −3｜抬高分只给前 0.4 s")


@configclass
class DeeproboticsS10PerceptV1HTrotH4EnvCfg(DeeproboticsS10PerceptV1HTrotH3EnvCfg):
    """Trot-H4（09-13 16:3x，从 Trot-H3 续）：H3 在 0.5～1.0 m/s 已是对角小跑（对角同步 80～93%、抬 7～13 cm），转向双向可跟，
    但**速度上不去**（指令 1.1 实际只 0.53～0.74，1.8 到不了）——步态项压过了速度跟踪。加：速度跟踪权重 5 → 12，转速跟踪 3 → 4；
    直行左漂（+0.07～0.11 rad/s）由转速跟踪加权顺带压。其余同 H3。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.track_lin_vel_xy_exp.weight = 12.0
        R.track_ang_vel_z_exp.weight = 4.0
        print("[s10-v1h-trotH4] 速度跟踪 12｜转速跟踪 4｜其余同 H3")


@configclass
class DeeproboticsS10PerceptV1HTrotH5EnvCfg(DeeproboticsS10PerceptV1HTrotH4EnvCfg):
    """Trot-H5（09-13 17:1x，从 H4 续）：H4 速度到 2.25、1.0～1.8 m/s 标准小跑，但 ① 倒退指令 −0.5 实际 +0.10 m/s 且打转（步态项/参考按速度模长触发，倒退也被逼踏步）；
    ② 0.5 m/s 低速仍举着右前轮。改：步态项和参考跟踪只按**前向指令 vx** 门控（倒退时全关，只剩速度跟踪）；低速档 0.3～0.7 的举着罚加倍（sg_hold −8 → −12）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        sg.FORWARD_ONLY[0] = True; tr.FORWARD_ONLY[0] = True
        self.rewards.sg_hold.weight = -12.0
        print("[s10-v1h-trotH5] 步态项/参考只按前向 vx 门控（倒退全关）｜举着罚 −12")


@configclass
class DeeproboticsS10PerceptV1HTrotH6EnvCfg(DeeproboticsS10PerceptV1HTrotH4EnvCfg):
    """Trot-H6（09-13 18:3x，从 H4 model_11500 续）：H4 2.3 指令跑 2.29、切爬台/切楼梯 2/2，但 ① 倒退 −0.5 → +0.10 且打转；② 0.5 m/s 三轮走（右前悬空 92%，点地即抬）。
    V1H_9196 倒退正常（−0.51），毛病是踏步系列带进来的：V2 把指令范围 ±2.0 改成 −0.5～2.3（倒退样本 50%→18%）、参考跟踪/步态项按速度模长门控（倒退时也奖励前进步态的关节轨迹）。
    H5 只改门控 + 举着罚 −12，300 步没纠回来且速度掉到 1.46。
    H6 改：步态项/参考只按前向 vx 门控（同 H5）；倒退指令范围放到 −1.0（样本 30%）；新增按轮"1 s 摆动占比 >0.6"罚 −4（点地躲不掉）；举着罚保持 H4 的 −8。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        sg.FORWARD_ONLY[0] = True; tr.FORWARD_ONLY[0] = True
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.3)
        self.rewards.sg_air_frac = RewTerm(func=sg.sg_air_frac_penalty, weight=-4.0, params={"frac_max": 0.6, "v_min": 0.3, "v_max": 2.4})
        print("[s10-v1h-trotH6] 步态项/参考只按前向 vx 门控｜指令 vx (-1.0, 2.3)｜按轮 1 s 摆动占比 >0.6 罚 −4｜举着罚 −8（H4）")


@configclass
class DeeproboticsS10PerceptV1HTrotH7EnvCfg(DeeproboticsS10PerceptV1HTrotH6EnvCfg):
    """Trot-H7（09-13 19:3x，从 H4 model_11500 续）：按机体系重算探针后，H4/H5/H6 各检查点前进速度都跟得上（2.3 指令 2.4～2.6），真正的毛病是
    ① 低速（0.5～1.0）直行时偏航 +0.1～+0.4 rad/s（左转漂）；② 倒退指令原地打转或不动。
    根因判断：训练一直开着航向指令（heading_command，wz = 0.5×航向误差），策略始终在外环纠偏之下，自身有偏航偏置也不吃亏，从没学过"wz=0 就走直"。
    H7 改：关航向指令，wz 直接采样 (−1.5, 1.5)，转速跟踪 w4 每步直接罚偏置；其余同 H6（前向门控、指令 vx (−1.0, 2.3)、摆动占比罚 −4）。"""
    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.heading_command = False
        c.ranges.ang_vel_z = (-1.5, 1.5)
        self.rewards.track_ang_vel_z_exp.weight = 8.0     # Isaac 里 H6 在 0.5/倒退时偏航 0.4～0.6 rad/s 连航向外环都拉不住 → 转速跟踪 4 → 8
        print("[s10-v1h-trotH7] 关航向指令（wz 直接采样 ±1.5，直行 wz=0 由转速跟踪硬约束）｜转速跟踪 8｜其余同 H6")


@configclass
class DeeproboticsS10StairExpertE6EnvCfg(DeeproboticsS10StairExpertE3EnvCfg):
    """StairN-E6（09-13 19:4x，从候选 D = E5_11200 续）：作者要求"后轮也抬着上楼梯"。D 的后轮是贴踢面滚上（越沿净空 0～2 cm，过沿瞬间顶踢面 hl 24%）。
    改：① 后轮各自抬高奖励（同前轮口径：相对自己起抬点 0.18，σ0.06）+3；② 参考跟踪只管前腿 6 关节（官方参考后轮是滚上去的，跟它就学不会抬）；
    ③ 其余不动：顶踢面 −3（四轮都算）、≥3 轮着地（多轮腾空 −4、支撑不足 −1.5）保证一次只抬一只、两前轮同时腾空 −8。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.ss_hind_lift = RewTerm(func=ss.front_lift_each_reward, weight=3.0, params={"h_target": 0.18, "sigma": 0.06, "wheels": (2, 3)})
        R.st_stair_ref.params["legs"] = "front"
        print("[s10-stairN-E6] 后轮各自抬高（相对起抬点 0.18）+3｜参考只跟前腿｜其余同 E3/E5")


@configclass
class DeeproboticsS10StairExpertE7EnvCfg(DeeproboticsS10StairExpertE3EnvCfg):
    """StairN-E7（09-13 22:xx，从候选 D = E5_11200 续）：作者（经 AGX 转达）"楼梯模式要能自己前进、左右走、后退、停下"——专家本身完整听指令，给 0 要原地站住。
    此前 E 系列指令只有 vx 0.15～0.40 + 航向 ±0.35，没有 0、没有 vy、没有倒退；真机 21:24 全程专家切入后给 0 它自己走（前轮 −1.0 rad/s）是直接诱因之一。
    改：① 指令 vx (−0.3, 0.5)、vy (−0.3, 0.3)、wz (−0.6, 0.6)，关航向外环（H7 教训），25% 环境零指令，5～10 s 重采样；
    ② 零指令时机体 |v_xy| + 0.5|wz| 罚 −3；速度/转速跟踪 5/3 → 8/6；超速罚改按模长（指令 0 时 |v|>0.1 也罚）；
    ③ 所有"没指令也奖励爬"的项按指令门控（climb_rewards.CMD_GATE_V=0.1：翻越相位/遇墙/爬高进展/StageRecord；前轮抬高奖励 v_min 0.1）；
    ④ 地形加平地 15%（在专家模式下平地横移/倒退也要会）；参考态出生 0.7 → 0.5。其余同 E3/E5。"""
    def __post_init__(self):
        super().__post_init__()
        import isaaclab.terrains as terrain_gen
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.CMD_GATE_V[0] = 0.1
        c = self.commands.base_velocity
        c.heading_command = False
        c.ranges.lin_vel_x = (-0.3, 0.5); c.ranges.lin_vel_y = (-0.3, 0.3); c.ranges.ang_vel_z = (-0.6, 0.6)
        c.rel_standing_envs = 0.25; c.resampling_time_range = (5.0, 10.0)
        R = self.rewards
        R.track_lin_vel_xy_exp.weight = 8.0; R.track_ang_vel_z_exp.weight = 6.0
        R.ss_zero_hold = RewTerm(func=ss.zero_cmd_motion_penalty, weight=-3.0, params={"v_cmd_max": 0.05, "w_ang": 0.5})
        R.ss_front_lift.params["v_min"] = 0.1
        gen = self.scene.terrain.terrain_generator; subs = gen.sub_terrains
        subs["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.15)
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values():
            v.proportion = v.proportion / tot
        self.events.spawn_trot_ref.params["ref_prob"] = 0.5
        print("[s10-stairN-E7] 指令 vx(-0.3,0.5) vy±0.3 wz±0.6 零指令 25%%｜零指令动罚 −3｜跟踪 8/6｜爬高/相位/抬腿按指令门控｜平地 15%%｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertE8EnvCfg(DeeproboticsS10StairExpertE7EnvCfg):
    """StairN-E8（09-13 22:1x，从 E7 末检查点续）：E7 300 步给 0 在楼梯上仍走 0.3～0.7 m、倒退只 −0.11、横移/转向不跟——零指令罚 −3 太弱（每局 −0.2），
    参考跟踪按 xy 模长门控把横移/倒退也拽向前进步态，轮子均衡罚不允许转向时左右不均。
    改：零指令动罚 −3 → −15；参考跟踪只按前向 vx 门控（倒退/横移/原地转不跟参考）；轮子均衡罚/左右动作差在 |wz|>0.1 时不算；速度/转速跟踪 8/6 → 10/8；
    参考态出生 0.5 → 0.3；平地 15% → 30%。"""
    def __post_init__(self):
        super().__post_init__()
        import isaaclab.terrains as terrain_gen
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        tr.FORWARD_ONLY[0] = True
        R = self.rewards
        R.ss_zero_hold.weight = -15.0
        R.track_lin_vel_xy_exp.weight = 10.0; R.track_ang_vel_z_exp.weight = 8.0
        R.wheel_drive_balance.params["wz_max"] = 0.1
        if getattr(R, "wheel_action_lr", None) is not None:
            R.wheel_action_lr.params["wz_max"] = 0.1
        self.events.spawn_trot_ref.params["ref_prob"] = 0.3
        subs = self.scene.terrain.terrain_generator.sub_terrains
        subs["flat"].proportion = 0.30 / 0.70 * sum(v.proportion for k, v in subs.items() if k != "flat")
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values():
            v.proportion = v.proportion / tot
        print("[s10-stairN-E8] 零指令动罚 −15｜参考只按前向门控｜转向时不罚轮子不均｜跟踪 10/8｜参考出生 0.3｜平地 30%%｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertS1EnvCfg(DeeproboticsS10StairExpertE8EnvCfg):
    """StairN-S1（09-14 00:3x，作者"按你的来"；从 E8_12398 续，训练计划 v2 §2/§6）：
    ① 速度旋钮真正管用：爬高进展/纪录 300/200 → 100/50 且只在 |机体 vx − 指令| < 0.15 时给；参考钟频率 = 2.2 × 指令（0.5 ↔ 1.10 Hz）；超速罚 −1 → −10、margin 0.05；速度/转速跟踪 σ 0.707 → 0.3；max_rise 3.0。
    ② 平地全向：顶踢面罚只在翻越相位算；参考回到全 12 关节（官方参考里后腿本来就是低抬）。
    ③ 后腿低抬：后轮抬高奖励改线性核（从 0 起有梯度）、目标 0.12（踢面高 + 3 cm）、按指令门控不按相位；爬高势函数用最后触地高（抬腿不掉分）。
    ④ 动作质量：腾空轮速 >9 rad/s 罚 −0.05/(rad/s)；侧倾罚 −5（只罚 roll）。
    ⑤ 感知鲁棒：高程图整幅偏置 ±0.10（5%）、前方假沟 −0.30（2%）。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        cr.SPEED_MATCH_TOL[0] = 0.15; cr.HOLD_LAST_CONTACT[0] = True; tr.FREQ_K[0] = 2.2
        R = self.rewards
        R.wheel_height_progress.weight = 100.0; R.wheel_height_progress.params["max_rise"] = 3.0
        R.wheel_height_record.weight = 50.0; R.wheel_height_record.params["max_rise"] = 3.0
        R.ss_overspeed.weight = -10.0; R.ss_overspeed.params["margin"] = 0.05
        R.track_lin_vel_xy_exp.params["std"] = 0.3; R.track_ang_vel_z_exp.params["std"] = 0.3
        R.ss_riser_push.params["phase_only"] = True
        R.st_stair_ref.params["legs"] = "all"
        R.ss_hind_lift = RewTerm(func=ss.front_lift_each_reward, weight=3.0, params={"h_target": 0.12, "sigma": 0.06, "wheels": (2, 3), "v_min": 0.1, "phase_only": False, "kernel": "linear"})
        R.ss_air_spin = RewTerm(func=ss.air_wheel_spin_penalty, weight=-0.05, params={"w_max": 9.0})
        R.ss_roll = RewTerm(func=ss.roll_penalty, weight=-5.0)
        hs = self.observations.policy.height_scan
        if getattr(hs, "noise", None) is not None:
            hs.noise.p_offset = 0.05; hs.noise.offset_max = 0.10; hs.noise.p_trench = 0.02; hs.noise.trench_val = -0.30
        print("[s10-stairN-S1] 爬高 100/50 按速度贴指令门控｜钟频 2.2×指令｜超速 −10｜σ0.3｜顶踢面只在相位｜后轮线性抬高 0.12 +3｜腾空轮速罚｜roll −5｜高程假图噪声")


@configclass
class DeeproboticsS10PerceptV1HTrotT1EnvCfg(DeeproboticsS10PerceptV1HTrotH7EnvCfg):
    """Trot-T1（09-14 00:3x，作者"按你的来"；从 H7_11800 续，训练计划 v2 §1）：目标官方的步频与姿态。
    ① 参考换 ref_step_official_v2（速度标签更正 0.5/1.0/1.4/1.9/2.5，频率 1.10 Hz 恒定，2.5 档 1.54）；
    ② 步态项：按轮不抬罚 t_max 0.6 → 1.0 s（去掉 1.67 Hz 步频下限）、摆动时长目标恒 0.35 s 且权重 8 → 3、举着罚 0.5 → 0.6、抬高分前 0.4 → 0.45 s；摆动高度按该轮自己触地高算（坡上不误判）；
    ③ 姿态：膝/髋回默认位罚减半、静止倍率 5 → 1；站高目标 0.345（有指令）/ 0.376（静止）；
    ④ 主策略不爬楼：爬高进展/纪录与上墙八段全部归零（坡上冲刺的来源）；
    ⑤ 指令 vx (−0.8, 2.0)、vy ±0.5、wz ±1.5；地形 平地 35 / 粗糙 20 / 坡 30（0.1～0.4）/ 箱 10 / 实测块 5；高程假图噪声同 S1。"""
    def __post_init__(self):
        super().__post_init__()
        import os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        tr.REF_PATH = os.path.expanduser("~/s10_logs/ref/ref_step_official_v2.npz")
        sg.OWN_BASE[0] = True
        R = self.rewards
        R.sg_idle.params["t_max"] = 1.0
        R.sg_swing.weight = 3.0; R.sg_swing.params["T_fixed"] = 0.35
        R.sg_hold.params["max_dur"] = 0.6; R.sg_clear.params["max_dur"] = 0.45
        for nm in ("knee_joint_pos_penalty", "hipy_joint_pos_penalty"):
            t = getattr(R, nm, None)
            if t is not None:
                t.weight = t.weight * 0.5
                if "stand_still_scale" in t.params: t.params["stand_still_scale"] = 1.0
        R.base_height_flat.params["target_height"] = 0.345; R.base_height_flat.params["target_still"] = 0.376
        for nm in ("wheel_height_progress", "wheel_height_record", "st_crouch", "st_hind_flex", "st_front_lift", "st_pitch_up", "st_hind_push",
                   "st_approach_speed", "st_hoist", "st_hind_extend", "st_hind_lift", "st_push_posture", "st_level_drag", "st_push_drive",
                   "st_hind_splay", "st_too_close", "st_launch"):
            t = getattr(R, nm, None)
            if t is not None: t.weight = 0.0
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.8, 2.0); c.ranges.lin_vel_y = (-0.5, 0.5); c.ranges.ang_vel_z = (-1.5, 1.5)
        for nm in ("sg_diag", "sg_clear", "sg_swing", "sg_idle", "sg_hold", "sg_air_frac", "st_trot_ref"):
            t = getattr(R, nm, None)
            if t is not None and "v_max" in t.params: t.params["v_max"] = 2.2
        gen = self.scene.terrain.terrain_generator; subs = gen.sub_terrains
        want = {}
        for k, v in subs.items():
            kl = k.lower()
            if "slope" in kl:
                want[k] = 0.15
                if hasattr(v, "slope_range"): v.slope_range = (0.1, 0.4)
            elif "flat" in kl or "plane" in kl: want[k] = 0.35
            elif "rough" in kl or "hf_random" in kl: want[k] = 0.20
            elif "boxes" in kl: want[k] = 0.10
            elif "real_patch" in kl: want[k] = 0.05
            else: want[k] = v.proportion
        tot = sum(want.values())
        for k, v in subs.items(): v.proportion = want[k] / tot
        hs = self.observations.policy.height_scan
        if getattr(hs, "noise", None) is not None:
            hs.noise.p_offset = 0.05; hs.noise.offset_max = 0.10; hs.noise.p_trench = 0.02; hs.noise.trench_val = -0.30
        print("[s10-v1h-trotT1] 参考 v2（1.10 Hz）｜不抬罚 1.0 s｜摆动 0.35 s w3｜站高 0.345/静止 0.376｜膝髋罚减半｜爬高/上墙归零｜vx(-0.8,2.0)｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertS2EnvCfg(DeeproboticsS10StairExpertS1EnvCfg):
    """StairN-S2（09-14 08:3x，从 S1 末检查点续）：S1 结果——平地停住 ✓、旋钮 0.25～0.5 管用（0.2 停滞）、侧倾 7～8°；横移仍 0、后轮净空 0、前腿摆动仍 0.81 s。
    改：① 前腿摆动 >0.4 s 罚 −3（sg_hold_penalty，摆动按该轮自己触地高算），给后腿腾出四拍里的抬腿窗口；② 参考钟频率下限 0.6 Hz（0.2 档不再停滞）；
    ③ 横移：单独加 vy 跟踪奖励 +5（σ 0.15，仅 |cmd_vy|>0.1），hipx 回默认位罚 −3 → −1；④ 其余同 S1。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        sg.OWN_BASE[0] = True
        R = self.rewards
        R.ss_front_hold = RewTerm(func=sg.sg_hold_penalty, weight=-3.0, params={"max_dur": 0.4, "v_min": 0.1, "v_max": 9.0})
        R.ss_vy_track = RewTerm(func=ss.vy_track_reward, weight=5.0, params={"sigma": 0.15, "vy_min": 0.1})
        if getattr(R, "hipx_joint_pos_penalty", None) is not None: R.hipx_joint_pos_penalty.weight = -1.0
        tr.FREQ_MIN[0] = 0.6
        print("[s10-stairN-S2] 前腿摆动 >0.4 s 罚 −3｜钟频下限 0.6 Hz｜vy 跟踪 +5｜hipx 罚 −1｜其余同 S1")


@configclass
class DeeproboticsS10PerceptV1HTrotT2EnvCfg(DeeproboticsS10PerceptV1HTrotT1EnvCfg):
    """Trot-T2（09-14 08:3x，从 T1 末检查点续）：T1 结果——1.0 m/s 步频 1.4 Hz ✓、踏步站高 0.33/膝 1.63 ✓、10° 坡走直 ✓；1.8 m/s 步频仍 3.6 Hz、静止站高仍 0.42、15° 坡 1.0 指令冲到 1.5。
    改：① 起抬频率 >1.3 Hz 罚 −5/Hz；② 主策略加超速罚 −5（|v|−|指令|−0.15）；③ 静止时膝/髋回默认位罚归零（stand_still_scale 0），站高罚 −20 → −40；④ 其余同 T1。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.sg_rate = RewTerm(func=sg.step_rate_penalty, weight=-5.0, params={"f_max": 1.3, "v_min": 0.3, "v_max": 2.2})
        R.tt_overspeed = RewTerm(func=ss.overspeed_penalty, weight=-5.0, params={"margin": 0.15})
        for nm in ("knee_joint_pos_penalty", "hipy_joint_pos_penalty"):
            t = getattr(R, nm, None)
            if t is not None and "stand_still_scale" in t.params: t.params["stand_still_scale"] = 0.0
        R.base_height_flat.weight = -40.0
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.CLIMB_SPLIT = 0.5; cr.WALL_AHEAD_H = 0.5; cr.TALL_WALL_H = 0.6   # 计划 v2 §1.4：主策略不再在坡/台阶上触发"上墙相位"（15° 坡曾触发，关掉站高/竖直速度/停滞罚）
        print("[s10-v1h-trotT2] 起抬频率 >1.3 Hz 罚 −5｜超速罚 −5｜静止不拉回默认位｜站高罚 −40｜其余同 T1")


@configclass
class DeeproboticsS10StairExpertS3EnvCfg(DeeproboticsS10StairExpertS2EnvCfg):
    """StairN-S3（09-14 09:5x，从 S2_13199 续）：S2 末点——5/5、侧倾 9～11°、0.2 档 70 s/0.4 档 31 s、平地停住、**左后轮开始抬（越沿净空 5.2 cm）**、右后仍滚；转向只到 0.2/0.5；横移指令把它带偏。
    作者确认"左右"= 转向，不要横移。改：vy 指令归零、横移奖励去掉；转速跟踪 8 → 12；爬高分的速度贴合容差 0.15 → 0.25（0.2 档时走时停的根源）；其余同 S2，继续让右后轮跟上。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        R = self.rewards
        R.ss_vy_track.weight = 0.0
        R.track_ang_vel_z_exp.weight = 12.0
        cr.SPEED_MATCH_TOL[0] = 0.25
        # 作者 09-14 09:5x："之前楼梯不是左右前轮都会抬腿吗"——听指令线右前退化成贴沿滚（D 6.5 cm → S2 2.5 cm）。前轮抬高奖励改线性核（0～2 cm 也有梯度），加左右对称罚
        R.ss_front_lift.params["kernel"] = "linear"; R.ss_front_lift.params["h_target"] = 0.15; R.ss_front_lift.weight = 4.0
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R.ss_lr_sym = RewTerm(func=ss.lift_lr_symmetry_penalty, weight=-5.0, params={"v_min": 0.1})
        print("[s10-stairN-S3] 无横移｜转速跟踪 12｜爬高分速度容差 0.25｜前轮抬高线性核 0.15 +4｜左右抬高对称罚 −5｜其余同 S2")


@configclass
class DeeproboticsS10PerceptV1HTrotT3EnvCfg(DeeproboticsS10PerceptV1HTrotT2EnvCfg):
    """Trot-T3（09-14 10:2x，从 T2_12998 续）。T2 探针分析：
    ① 起抬频率罚把 1.8 m/s 的步频从 3.6 压到 2.1 Hz，但方式是"举着一对对角轮不放"（fl/hr 悬空 81～88%，fr/hl 只 15%），少起抬=少罚——投机；roll p95 也因此到 6～7°。
       根因：节拍没有被钟锁住（参考跟踪 w6/σ0.5 太松），而每轮悬空占比罚 −4/上限 0.6 太弱（0.85 只扣 1/步）。改：参考跟踪 6 → 10、σ 0.5 → 0.35；悬空占比上限 0.55、权重 −4 → −15；保留起抬频率罚。
    ② 静止站高仍 0.42：站高罚是平方误差，4 cm 差 × 40 = 0.06/步，形同没有。改：−40 → −500（1 cm 差 0.05/步，4 cm 差 0.8/步）。
    ③ 15° 坡 1.0 指令 1.53 → 1.28（超速罚起效但不够）：超速罚 −5 → −10。10° 坡已贴指令。
    ④ 粗糙地对角同步只 43～49%、轮子悬空 56～91%（作者"越野踏步乱"复现）：训练粗糙地形起伏 0.05～0.10 太大且是连绵起伏，换成 0.01～0.04、步长 0.01（贴近 0.5～3 cm 碎石），占比 20 → 25%。"""
    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.st_trot_ref.weight = 10.0; R.st_trot_ref.params["sigma_q"] = 0.35
        R.sg_air_frac.weight = -15.0; R.sg_air_frac.params["frac_max"] = 0.55
        R.base_height_flat.weight = -500.0
        R.tt_overspeed.weight = -10.0
        subs = self.scene.terrain.terrain_generator.sub_terrains
        for k, v in subs.items():
            if "rough" in k.lower() and hasattr(v, "noise_range"):
                v.noise_range = (0.01, 0.04); v.noise_step = 0.01; v.proportion = v.proportion * 1.25
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values(): v.proportion = v.proportion / tot
        print("[s10-v1h-trotT3] 参考 10/σ0.35｜悬空占比 >0.55 罚 −15｜站高罚 −500｜超速 −10｜粗糙地 0.01～0.04｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertS4EnvCfg(DeeproboticsS10StairExpertS3EnvCfg):
    """StairN-S4（09-14 10:5x）。S3 三个扫描点分析：右前/右后始终贴沿滚（净空 1～2 cm）而左前越抬越高（17 cm）——"左抬右滚"的不对称步态被爬高收益锁死，
    对称罚 −5 每步只扣 <1 分撼不动；容差放宽到 0.25 后 0.5 档实际跑到 0.66（超速侧也放宽了）；侧倾 13600 回到 12°；0.2 档仍停滞。
    改：① 速度贴合门控只卡超速一侧（慢于指令也给爬高分）→ 0.2 档不再停滞、0.5 档不再超到 0.66；容差回 0.15；
    ② 右前、右后单独抬高奖励 +6（左侧保留 +4/+3），左右对称罚 −5 → −20；③ 空中轮速罚 −0.05 → −0.3、roll 罚 −5 → −10；④ 指令下限 0.25（档位口径 0.25～0.5）。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        cr.SPEED_MATCH_TOL[0] = 0.15; cr.SPEED_MATCH_UPPER_ONLY[0] = True
        R = self.rewards
        R.ss_front_lift.params["wheels"] = (0,); R.ss_hind_lift.params["wheels"] = (2,)
        R.ss_front_lift_r = RewTerm(func=ss.front_lift_each_reward, weight=6.0, params={"h_target": 0.15, "sigma": 0.06, "wheels": (1,), "v_min": 0.1, "phase_only": True, "kernel": "linear"})
        R.ss_hind_lift_r = RewTerm(func=ss.front_lift_each_reward, weight=6.0, params={"h_target": 0.12, "sigma": 0.06, "wheels": (3,), "v_min": 0.1, "phase_only": False, "kernel": "linear"})
        R.ss_lr_sym.weight = -20.0
        R.ss_air_spin.weight = -0.3; R.ss_roll.weight = -10.0
        c = self.commands.base_velocity; lo, hi = c.ranges.lin_vel_x; c.ranges.lin_vel_x = (lo, hi)   # 平地仍允许 −0.3；楼梯档位口径 0.25～0.5 由控制台保证
        if getattr(R, "stall_penalty", None) is not None: R.stall_penalty.params["cmd_min"] = 0.15   # 作者 09-14 11:0x：0.2 档很重要——指令 0.2 时原地不动也要罚（原门槛 0.3 使 0.2 档"停住"零成本）
        print("[s10-stairN-S4] 门控只卡超速(容差 0.15)｜右前/右后抬高 +6 单独计｜对称罚 −20｜空中轮速 −0.3｜roll −10｜停滞罚门槛 0.15")


@configclass
class DeeproboticsS10StairExpertS5EnvCfg(DeeproboticsS10StairExpertS4EnvCfg):
    """StairN-S5（09-14 11:4x，从 S4_14200 续）。S4 两个点：0.25～0.4 档全登顶、侧倾 6～7.6°、空中轮速 8～13；但 0.2 档连楼梯脚都不走（停在起点）——
    超速罚 −10/余量 0.05 让 0.2 平均速度下过一级的瞬时冲刺（0.4～0.5）每步扣 2 分以上，停着只挨停滞罚，"走"比"停"亏；
    对称罚是绝对差，把高的一侧往下拉（左前 11→4、左后 7→1），右侧没起来。
    改：① 超速罚 −10 → −3、余量 0.05 → 0.10；② 对称罚 −20 → −5，新增"按对取低者"抬高奖励 +8（只奖励把低的抬起来）；③ 其余同 S4。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.ss_overspeed.weight = -3.0; R.ss_overspeed.params["margin"] = 0.10
        R.ss_lr_sym.weight = -5.0
        R.ss_pair_min = RewTerm(func=ss.lift_pair_min_reward, weight=8.0, params={"h_cap": 0.15, "v_min": 0.1})
        print("[s10-stairN-S5] 超速 −3/余量 0.1｜对称罚 −5 + 按对取低者抬高 +8｜其余同 S4")


@configclass
class DeeproboticsS10StairExpertS6EnvCfg(DeeproboticsS10StairExpertS5EnvCfg):
    """StairN-S6（09-14 12:0x，从 S5_14400 续）。S5 首点：两后轮都抬了（5.0/3.7 cm）、0.25～0.4 全登顶；0.2 档以 0.024 m/s 蠕行（120 s 走 2.7 m，没到楼梯脚）——
    不是"不会走"，是 0.2 指令下"蠕行"几乎不吃亏：速度跟踪 σ0.3 时 0.02 vs 0.2 只差 3 分/步，而走起来要挨步态项的罚；停滞罚 vel_max 0.1 只对完全不动生效。
    改：① 速度跟踪 σ 0.3 → 0.18（0.02 对 0.2 差 7.5 分/步）；② 新增低速欠速罚：指令 >0.1 时 max(0, 指令 − 机体 vx − 0.05) × −8；③ 其余同 S5。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.track_lin_vel_xy_exp.params["std"] = 0.18
        R.ss_underspeed = RewTerm(func=ss.underspeed_penalty, weight=-8.0, params={"margin": 0.05, "cmd_min": 0.1})
        print("[s10-stairN-S6] 速度跟踪 σ0.18｜欠速罚 −8（指令−vx−0.05）｜其余同 S5")


@configclass
class DeeproboticsS10StairExpertS7EnvCfg(DeeproboticsS10StairExpertS6EnvCfg):
    """StairN-S7（09-14 12:3x）。0.2 档真相：以 0.12 m/s 走到第一级踢面前停住 90 s 不迈步——爬高分门控（v<指令+0.15）与超速罚都按瞬时速度算，
    迈上一级需要 0.4～0.5 的瞬时冲刺，在 0.2 档既拿不到爬高分又挨超速罚，站着最划算（0.25 档允许到 0.40，刚够）。
    改：两项都改按 1 s 滑动平均速度算（只管平均不超档位）；其余同 S6。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.SPEED_EMA_TAU[0] = 1.0
        self.rewards.ss_overspeed.params["tau"] = 1.0
        # S6 的 σ0.18 让侧倾从 7.6～9.7 涨到 10～12、后轮净空从 5.0/3.7 掉到 2.4/2.4，且没治 0.2（根因是瞬时门控）：回 σ0.3，欠速罚减到 −3
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3; self.rewards.ss_underspeed.weight = -3.0
        print("[s10-stairN-S7] 爬高分门控/超速罚按 1 s 滑动平均速度｜跟踪 σ 回 0.3、欠速罚 −3｜其余同 S5")


@configclass
class DeeproboticsS10StairExpertS8EnvCfg(DeeproboticsS10StairExpertS7EnvCfg):
    """StairN-S8（09-14 12:5x，从 S5_14400 续）。训练日志对比：爬高进展奖励每局 E5 约 1.25 → E8 0.17 → S1～S7 0.02～0.04——S1 加的"速度贴指令才给爬高分"
    把爬楼的主要激励几乎关掉了，S 系列靠参考跟踪和速度跟踪在爬；0.2 档在第一级前停住、右前不抬，都与"没有爬的激励"有关。
    改：去掉速度贴合门控（只保留"有指令才给"）；爬高 100/50 保留；超速罚 −10、余量 0.05、按 1 s 滑动平均（旋钮靠它）；其余同 S7。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.SPEED_MATCH_TOL[0] = 0.0; cr.SPEED_MATCH_UPPER_ONLY[0] = False
        self.rewards.ss_overspeed.weight = -10.0; self.rewards.ss_overspeed.params["margin"] = 0.05
        print("[s10-stairN-S8] 爬高分只按有指令门控（去速度贴合）｜超速 −10/余量 0.05/1 s 平均｜其余同 S7")


@configclass
class DeeproboticsS10StairExpertS9EnvCfg(DeeproboticsS10StairExpertS8EnvCfg):
    """StairN-S9（09-14 13:4x，从 S8_14800 续，LR 5e-5 巩固）。S8 14800/15000 0.2 档能上（两前轮对称抬 6.9/5.8），末点 15199 又停在第一级前——慢档能力没守住。
    改：学习率减半巩固；右前单独抬高奖励 6 → 10；其余同 S8。慢档指令占比无法在采样器里直接加权，先靠低学习率守住。"""
    def __post_init__(self):
        super().__post_init__()
        self.rewards.ss_front_lift_r.weight = 10.0
        print("[s10-stairN-S9] 右前抬高 +10｜其余同 S8（LR 5e-5 由链给）")


@configclass
class DeeproboticsS10PerceptV1HTrotT4EnvCfg(DeeproboticsS10PerceptV1HTrotT3EnvCfg):
    """Trot-T4（09-14 14:1x，从 T3 最好的点续）。T3 13400：四档都是干净对角小跑（离地 26～63%、对角 80～87%、侧倾 5～6°），速度全贴指令，倒退/转向好；
    但步频仍 2.3 Hz（参考跟踪 10/σ0.35 约束关节角相似度，管不住节拍）、静止站高 0.415（stand_still 关节回位罚 −1 顶着站高罚）、上坡 1.0 指令跑偏 −21～−45°、粗糙地对角同步 46～62%。
    改：① 触地相位跟钟 +6（按官方模板：此刻该腾空/该触地）；起抬频率上限 1.3 → 1.2、罚 −5 → −8；② stand_still 归零；③ 转速跟踪 12 → 16；④ 对角同步 2 → 4。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import trot_ref as tr
        R = self.rewards
        R.tr_phase = RewTerm(func=tr.phase_contact_reward, weight=6.0, params={"v_min": 0.3, "v_max": 2.2})
        R.sg_rate.weight = -3.0; R.sg_rate.params["f_max"] = 1.2   # T3 末点教训：起抬频率罚 −5 让它"举着一只轮不起抬"（右前 99% 悬空、其余三轮滚），罚太重会奖励不迈步；节拍交给触地相位跟钟
        if getattr(R, "stand_still", None) is not None: R.stand_still.weight = 0.0
        R.track_ang_vel_z_exp.weight = 16.0
        R.sg_diag.weight = 4.0
        print("[s10-v1h-trotT4] 触地相位跟钟 +6｜起抬 >1.2 Hz 罚 −8｜stand_still 0｜转速跟踪 16｜对角 4｜其余同 T3")


@configclass
class DeeproboticsS10StairExpertS10EnvCfg(DeeproboticsS10StairExpertS8EnvCfg):
    """StairN-S10（09-14 14:5x，从 F=S8_14800 续）。S9（右前加权 10、LR 减半）把右前抬到 13.7 cm 但一侧高抬把机身撬歪，侧倾 13～17°、0.2 档又停，弃。
    改（相对 S8）：右前/右后抬高奖励目标 0.15/0.12 → 0.10（不奖励抬过头），权重 6；左右对称罚 −5 → −20 但差值 <5 cm 不罚（死区）；其余同 S8。"""
    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.ss_front_lift_r.params["h_target"] = 0.10; R.ss_hind_lift_r.params["h_target"] = 0.10
        R.ss_front_lift.params["h_target"] = 0.10; R.ss_hind_lift.params["h_target"] = 0.10
        R.ss_lr_sym.weight = -20.0; R.ss_lr_sym.params["deadband"] = 0.05
        print("[s10-stairN-S10] 抬高目标 0.10 封顶｜对称罚 −20 死区 5 cm｜其余同 S8")


@configclass
class DeeproboticsS10PerceptV1HTrotT5EnvCfg(DeeproboticsS10PerceptV1HTrotT4EnvCfg):
    """Trot-T5（09-14 15:0x，从 T3_13400 续）。T3 末点与 T4 首点都塌成"右前举着 99%、三轮滚"，而训练里悬空占比罚≈0——
    说明策略在 Isaac 里把右前悬在 <4 cm（不算"摆动"，起抬频率罚归零），到 MuJoCo 里这个悬停变成 6～7 cm 才被看见。起抬频率罚是根源。
    改：① 起抬频率罚归零；② 悬空占比罚/举着罚/触地相位跟钟的"离地"门槛 4 cm → 1 cm（悬停 1 cm 也算）；③ 其余同 T4。"""
    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.sg_rate.weight = 0.0
        for nm in ("sg_air_frac", "sg_hold", "tr_phase"):
            t = getattr(R, nm, None)
            if t is not None: t.params["h_min"] = 0.01
        print("[s10-v1h-trotT5] 起抬频率罚 0｜悬空/举着/相位门槛 1 cm｜其余同 T4")


@configclass
class DeeproboticsS10StairExpertS11EnvCfg(DeeproboticsS10StairExpertS10EnvCfg):
    """StairN-S11（09-14 16:xx，小步慢调）：0.2 档能力在 S8/S10 里都是"隔一个检查点出现一次"的瞬态；从最好的 0.2 可用点续，LR 3e-5、只训 200 步、每 100 步扫，目的是守住而不是再改。配方同 S10。"""
    def __post_init__(self):
        super().__post_init__()
        print("[s10-stairN-S11] 同 S10，小步慢调（LR 3e-5 由链给）")


@configclass
class DeeproboticsS10ClimbExpertC1EnvCfg(DeeproboticsS10ClimbExpertHEnvCfg):
    """Climb-C1（09-14 16:1x，从 ClimbH_8797 续；训练计划 v2 §3 第一版）：作者要求高台模式也能在平地停/前后/转向，后腿抬着跨沿而不是贴面拖上。
    ① 平地全向：指令 vx (−0.3, 1.0)、wz ±0.6、25% 零指令、关航向外环；零指令动罚 −15；速度/转速跟踪 5/3 → 8/6（σ0.3）；没指令不给爬高/相位（CMD_GATE_V）；地形加平地 30%。
    ② 后腿跨沿：两后轮各自"相对起抬点抬高"线性奖励（目标 0.15）+6，只在翻越相位；顶立面罚 −2（相位内）；爬高势函数用最后触地高（抬腿不掉分）；
       去掉鼓励"贴面拖上"的旧目标：st_hind_extend、st_level_drag、st_push_posture 归零。
    ③ 高程假图噪声同楼梯线。官方 1 s 折腿翻越模板跟踪放 C2。"""
    def __post_init__(self):
        super().__post_init__()
        import isaaclab.terrains as terrain_gen
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.CMD_GATE_V[0] = 0.1; cr.HOLD_LAST_CONTACT[0] = True; cr.SPEED_MATCH_TOL[0] = 0.0
        c = self.commands.base_velocity
        c.heading_command = False
        c.ranges.lin_vel_x = (-0.3, 1.0); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (-0.6, 0.6)
        c.rel_standing_envs = 0.25; c.resampling_time_range = (5.0, 10.0)
        R = self.rewards
        R.track_lin_vel_xy_exp.weight = 8.0; R.track_lin_vel_xy_exp.params["std"] = 0.3
        R.track_ang_vel_z_exp.weight = 6.0; R.track_ang_vel_z_exp.params["std"] = 0.3
        R.cc_zero_hold = RewTerm(func=ss.zero_cmd_motion_penalty, weight=-15.0, params={"v_cmd_max": 0.05, "w_ang": 0.5})
        R.cc_hind_lift = RewTerm(func=ss.front_lift_each_reward, weight=6.0, params={"h_target": 0.15, "sigma": 0.06, "wheels": (2, 3), "v_min": 0.1, "phase_only": True, "kernel": "linear"})
        R.cc_riser_push = RewTerm(func=ss.wheel_riser_push_penalty, weight=-2.0, params={"phase_only": True})
        for nm in ("st_hind_extend", "st_level_drag", "st_push_posture"):
            t = getattr(R, nm, None)
            if t is not None: t.weight = 0.0
        gen = self.scene.terrain.terrain_generator; subs = gen.sub_terrains
        subs["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=0.3)
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values(): v.proportion = v.proportion / tot
        hs = self.observations.policy.height_scan
        if getattr(hs, "noise", None) is not None:
            hs.noise.p_offset = 0.05; hs.noise.offset_max = 0.10; hs.noise.p_trench = 0.02; hs.noise.trench_val = -0.30
        print("[s10-climb-C1] 平地全向(vx -0.3~1.0, wz ±0.6, 零指令 25%%)｜零指令动罚 −15｜跟踪 8/6｜后轮抬高 +6｜顶立面 −2｜拖上目标归零｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10ClimbExpertC2EnvCfg(DeeproboticsS10ClimbExpertC1EnvCfg):
    """Climb-C2（09-14 16:3x，从 ClimbH_8797 续，温和版）：C1 一次把指令分布/门控/零指令罚全换掉，200 步爬台能力就丢了（离墙 1 m 切入后站在墙前，平均奖励从上百掉到 10）。
    改回温和：指令 vx (−0.2, 0.8)、零指令 10%；零指令动罚 −15 → −5；不按指令门控爬高/相位（CMD_GATE_V 0，爬台激励原样）；平地 30% → 15%；后轮抬高 +6、顶立面 −2、拖上目标归零保留；LR 5e-5。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.CMD_GATE_V[0] = 0.0
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.2, 0.8); c.rel_standing_envs = 0.10
        R = self.rewards
        R.cc_zero_hold.weight = -5.0
        subs = self.scene.terrain.terrain_generator.sub_terrains
        subs["flat"].proportion = 0.15 / 0.85 * sum(v.proportion for k, v in subs.items() if k != "flat")
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values(): v.proportion = v.proportion / tot
        print("[s10-climb-C2] 温和版：vx(-0.2,0.8) 零指令 10%%｜零指令动罚 −5｜爬高不按指令门控｜平地 15%%｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertS12EnvCfg(DeeproboticsS10StairExpertS8EnvCfg):
    """StairN-S12（09-14 17:0x，从 F=S8_14800 续）：真机 16:23 F 在真实楼梯（里程计 Δz/Δx≈0.6，约 31°，比仿真验收的 0.11×0.55 陡得多）爬 13 级后卡住——
    两前轮在上两级、两后轮在踢面底、仰 19°，后轮滚不上去；再给 0.3 后翻车。仿真复现：0.15×0.30 停在第 2～3 级、0.13×0.30 爬 11 级后翻。
    训练分布里 0.30/0.35 踏面的踢面 0.05～0.18 由课程决定，平均难度只到 0.49（≈踢面 0.11），陡窄楼梯几乎没练过。
    改：① 地形难度区间 0.6～1.0（踢面 0.13～0.18 为主），0.30/0.35 踏面占比 0.16/0.13 → 0.30/0.20，平地 0.30 → 0.20；② 后轮抬高奖励 +6 → +10（真楼梯卡住的直接原因是后轮不抬）；③ 其余同 S8；验收换成 0.13×0.30 / 0.15×0.30 楼梯。"""
    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator; subs = gen.sub_terrains
        gen.difficulty_range = (0.6, 1.0)
        want = {"stairs_n30": 0.30, "stairs_n35": 0.20, "stairs_n40": 0.10, "stairs_n55": 0.10, "stairs_up_wide": 0.05, "random_rough": 0.05, "flat": 0.20}
        for k, v in subs.items():
            if k in want: v.proportion = want[k]
            if k in ("stairs_n30", "stairs_n35", "stairs_n40") and hasattr(v, "step_height_range"): v.step_height_range = (0.10, 0.18)
        tot = sum(v.proportion for v in subs.values())
        for v in subs.values(): v.proportion = v.proportion / tot
        R = self.rewards
        R.ss_hind_lift.weight = 10.0; R.ss_hind_lift_r.weight = 10.0
        print("[s10-stairN-S12] 难度 0.6～1.0｜窄踏面为主｜后轮抬高 +10｜地形 %s" % {k: round(v.proportion, 2) for k, v in subs.items()})


@configclass
class DeeproboticsS10StairExpertS13EnvCfg(DeeproboticsS10StairExpertS12EnvCfg):
    """StairN-S13（09-14 17:3x，作者："不需要在楼梯上停住，专注楼梯稳定性和转向"）：
    ① 楼梯上不再为"停"花算力：零指令环境 25%→10%、零指令动罚 −15→−3（平地停已经会了，靠它保住即可）；
    ② 稳：侧倾罚 −10→−25，新增侧倾角速度罚 −0.5（翻越相位内），机身角速度 xy 罚 −0.02→−0.10；
    ③ 转：楼梯转向专项奖励 +8（翻越相位内按 wz 误差 σ0.25），转速跟踪 12→16，指令 wz ±0.6→±0.8；
    ④ 地形/后轮抬高沿用 S12（陡窄 0.10～0.18 × 0.30/0.35 为主、后轮 +10）。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        c = self.commands.base_velocity
        c.rel_standing_envs = 0.10; c.ranges.ang_vel_z = (-0.8, 0.8)
        R.ss_zero_hold.weight = -3.0
        R.ss_roll.weight = -25.0
        R.ss_roll_rate = RewTerm(func=ss.roll_rate_penalty, weight=-0.5, params={"phase_only": True})
        if getattr(R, "ang_vel_xy_l2", None) is not None: R.ang_vel_xy_l2.weight = -0.10
        R.ss_yaw_stairs = RewTerm(func=ss.yaw_track_stairs_reward, weight=8.0, params={"sigma": 0.25, "phase_only": True})
        R.track_ang_vel_z_exp.weight = 16.0
        print("[s10-stairN-S13] 零指令 10%%/动罚 −3｜roll −25 + roll 速率 −0.5 + 角速度 xy −0.10｜楼梯转向 +8、转速跟踪 16、wz ±0.8｜地形同 S12")


@configclass
class DeeproboticsS10PerceptV1HTrotT6EnvCfg(DeeproboticsS10PerceptV1HTrotT3EnvCfg):
    """Trot-T6（09-14 17:4x，踏步鲁棒版；从 T3_13400 续）。真机 16:16 T3 平地退化成"举一只轮、三轮滚"（左前悬空 55～71%、roll p95 8.8°），
    MuJoCo 里 T3 之后的检查点同样退化，而 Isaac 里没有——小跑是个脆弱极限环，换动力学就熄火。
    改（只加鲁棒性与反投机，不动步态目标）：
    ① 随机化加宽：动作延迟 0～3 → 0～5 步（100 ms）、零位偏置 0.02 → 0.03 rad、轮增益偏差 3% → 6%、
       PD 刚度 0.85～1.15 → 0.7～1.3、阻尼 0.7～1.15 → 0.6～1.35、摩擦下界 0.3 → 0.25、外扰力 ±10 → ±15 N；
    ② 反"举一只轮"：悬空占比罚阈值 0.55 → 0.50、权重 −15 → −25；新增左右悬空占比不对称罚 −10（同对轮 1 s 平均悬空占比之差）；
    ③ 其余同 T3（参考 1.10 Hz、站高 0.345/静止 0.376、爬高归零、坡 30%、粗糙 0.01～0.04）。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        A = self.actions
        if getattr(A, "joint_pos", None) is not None:
            A.joint_pos.delay_steps = 5; A.joint_pos.zero_offset = 0.03
        jv = getattr(A, "joint_vel", None) or getattr(A, "wheel_vel", None)
        if jv is not None:
            jv.delay_steps = 5
            if hasattr(jv, "gain_range"): jv.gain_range = 0.06
        E = self.events
        g = getattr(E, "randomize_actuator_gains", None)
        if g is not None:
            g.params["stiffness_distribution_params"] = (0.7, 1.3); g.params["damping_distribution_params"] = (0.6, 1.35)
        mat = getattr(E, "randomize_rigid_body_material", None)
        if mat is not None:
            for k in ("static_friction_range", "dynamic_friction_range"):
                if k in mat.params: mat.params[k] = (0.25, mat.params[k][1])
        f = getattr(E, "randomize_apply_external_force_torque", None)
        if f is not None:
            f.params["force_range"] = (-15.0, 15.0); f.params["torque_range"] = (-15.0, 15.0)
        R = self.rewards
        R.sg_air_frac.weight = -25.0; R.sg_air_frac.params["frac_max"] = 0.50
        R.sg_air_sym = RewTerm(func=sg.air_frac_lr_penalty, weight=-10.0, params={"v_min": 0.3, "v_max": 2.2})
        print("[s10-v1h-trotT6] 延迟 5 步/零位 0.03/轮增益 6%%/PD 0.7～1.3｜悬空占比 >0.50 罚 −25 + 左右不对称 −10｜其余同 T3")


@configclass
class DeeproboticsS10ClimbExpertC3EnvCfg(DeeproboticsS10ClimbExpertHEnvCfg):
    """Climb-C3（09-14 17:4x，翻台阶稳定版；从 ClimbH_8797 续）。C1 激进换分布把翻越冲掉、C2 温和版爬台 3/3 但侧倾从 15～21° 涨到 26～34°、后轮仍贴面拖上。
    C3 只做三件事，不动指令分布（保持 0.3～0.6 前进，翻越能力优先）：
    ① 稳：侧倾罚 −15、侧倾角速度罚 −0.5（翻越相位内）；
    ② 后腿跨沿：两后轮相对起抬点线性抬高奖励 +8（目标 0.15，翻越相位内）、顶立面罚 −2；去掉"贴面拖上"的三项旧目标；
    ③ 高程假图噪声（LIO z 漂鲁棒）。"""
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.HOLD_LAST_CONTACT[0] = True
        R = self.rewards
        R.cc_roll = RewTerm(func=ss.roll_penalty, weight=-15.0)
        R.cc_roll_rate = RewTerm(func=ss.roll_rate_penalty, weight=-0.5, params={"phase_only": True})
        R.cc_hind_lift = RewTerm(func=ss.front_lift_each_reward, weight=8.0, params={"h_target": 0.15, "sigma": 0.06, "wheels": (2, 3), "v_min": 0.1, "phase_only": True, "kernel": "linear"})
        R.cc_riser_push = RewTerm(func=ss.wheel_riser_push_penalty, weight=-2.0, params={"phase_only": True})
        for nm in ("st_hind_extend", "st_level_drag", "st_push_posture"):
            t = getattr(R, nm, None)
            if t is not None: t.weight = 0.0
        hs = self.observations.policy.height_scan
        if getattr(hs, "noise", None) is not None:
            hs.noise.p_offset = 0.05; hs.noise.offset_max = 0.10; hs.noise.p_trench = 0.02; hs.noise.trench_val = -0.30
        print("[s10-climb-C3] roll −15 + roll 速率 −0.5｜后轮线性抬高 +8｜顶立面 −2｜拖上目标归零｜假图噪声｜指令分布不动")


@configclass
class DeeproboticsS10ClimbExpertC4EnvCfg(DeeproboticsS10ClimbExpertC3EnvCfg):
    """Climb-C4（09-14 19:5x，作者："C3 也要稳定能翻台阶，照着官方继续训"；从 C3_9596 续）。

    C3 结果：bench_top33 登顶 3/3、用时 6.1～7.2 s（比 8797 的 7.4 s 还快），但两条没治好——
      ① 后轮还是"贴着立面拖上来"：越沿净空 hl 0.001～0.004 m、hr 0.004～0.064 m（前轮 fr 0.21～0.28 m），
         hl 顶踢面 11～19%（过沿窗口内 24%）。官方是「两条后腿折叠跨过沿、左后伸展主推、不贴面拖」。
      ② 侧倾从 13.6° 涨到 20.8°，加的 roll −15 没起作用。

    查训练日志实测读数，两条都能解释：
      cc_hind_lift 原始值只有 0.037——因为它量的是"相对自己起抬点"的抬高，后轮蹭立面时不断轻触，
        起抬点一路被刷新，抬高量永远≈0，这一项根本读不出"拖"和"跨"的区别；
      cc_roll 原始值 0.0042——翻越那 1 s 的 20° 被 30 s 平地段稀释掉了，全程罚给不出相位内的梯度；
      而 wheel_height_progress（w=100，原始值 0.0065）**对贴着立面蹭上去照样给分**（源码注释原话：
        "轮子沿立面贴着滚上去时一路有分"）——那是早期的起步奖励，现在正好在奖励官方不做的动作。

    C4 四改（不动指令分布与课程）：
      ① 断掉拖上来的奖励源：爬高进展/纪录只算「触地且没顶在立面上」的轮（PROGRESS_SKIP_PRESSED）；
      ② 换度量：新增越沿净空奖励 +12——后轮腾空时触地点高出「记住的台沿高度」的部分（封顶 8 cm），
         这就是 MuJoCo 逐级核对表里的"越沿净空"，与官方动作直接对应；
      ③ 后轮顶立面单独重罚 −10（前轮蹬立面是正常上墙动作，仍按原来的 −2 全轮罚）；
      ④ 稳：侧倾罚改成翻越相位内 −60（全程那份保留 −15 不动），侧倾角速度 −0.5 → −1.5；
      ⑤ 平地零指令别漂：C3 零指令 7.9 s 位移 1.9～2.6 m，加 zero_cmd_motion_penalty −10。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.PROGRESS_SKIP_PRESSED[0] = True                                   # ①
        R = self.rewards
        R.cc_edge_clr = RewTerm(func=ss.edge_clearance_reward, weight=12.0,  # ②
                                params={"wheels": (2, 3), "cap": 0.08, "phase_only": True})
        R.cc_hind_push = RewTerm(func=ss.wheel_riser_push_penalty, weight=-10.0,   # ③
                                 params={"phase_only": True, "wheels": (2, 3)})
        R.cc_roll_ph = RewTerm(func=ss.roll_penalty, weight=-60.0, params={"phase_only": True})   # ④
        R.cc_roll_rate.weight = -1.5
        R.cc_zero_hold = RewTerm(func=ss.zero_cmd_motion_penalty, weight=-10.0,    # ⑤
                                 params={"v_cmd_max": 0.05, "w_ang": 0.5})
        print("[s10-climb-C4] 爬高进展剔除顶立面轮｜越沿净空 +12（封顶 8 cm）｜后轮顶立面 −10｜相位内侧倾 −60/角速度 −1.5｜零指令动罚 −10")


@configclass
class DeeproboticsS10PerceptV1HTrotT7EnvCfg(DeeproboticsS10PerceptV1HTrotT6EnvCfg):
    """Trot-T7（09-14 19:5x，踏步反冻结版；从 T3_13400 续，不从 T6 续）。

    T6 失败复盘（Isaac↔MuJoCo 244 维逐列对拍，09-14 19:4x）：
      两边观测**均值完全一致**（接口没有偏差，这条排除了），差的是波动——
        T3_13400（两边都好）：Isaac 膝关节位置 σ 0.065~0.138 / MuJoCo 0.032~0.071，动作 σ 0.24~0.44；
        T6_13999（坏）      ：Isaac 0.024~0.043 / MuJoCo **0.0008~0.0011**，关节速度 σ 精确为 0，动作 σ 0.003。
      也就是 T6 在 MuJoCo 里**四条腿完全锁死、纯轮子滚**；接触统计报的"右前悬空 91%"是姿势歪了那只轮
      碰不到地，不是举腿。而它在 Isaac 里就已经退化了（膝摆幅只剩 T3 的 1/3），只是没完全冻死。
      机理：不抬腿 → 起抬频率罚 sg_rate 从 3.22 清到 0.65（净赚 2.56），而所有**基于接触判定**的反作弊项
      （sg_air_frac 悬空占比 −25、T6 新加的 sg_air_sym 左右不对称 −10）读数都是 0——没有摆动就没有悬空占比
      可言，这些项根本管不到；唯一拦它的 sg_no_swing 只有 −3，压不住。

    T7 三改（随机化沿用 T6 的，那部分对真机有益、与本次退化无关）：
      ① 新增膝摆幅下限罚 −20（knee_amp_penalty，amp_min 0.06）——不依赖接触判定，直接量关节动没动，
         Isaac/MuJoCo 同口径；锁死一条腿的代价 4×0.06×20 = 4.8/s，远大于清掉 sg_rate 的 2.56；
      ② sg_no_swing −3 → −12（同一件事的接触侧备份）；
      ③ 起抬频率罚松绑：权重 −5 → −2、f_max 1.3 → 1.5——它是这次退化的诱因，而官方 1.10 Hz 的口径
         由参考时钟 st_trot_ref 管着，不需要再用重罚去压。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        R = self.rewards
        R.sg_knee_amp = RewTerm(func=sg.knee_amp_penalty, weight=-40.0,
                                params={"amp_min": 0.10, "tau": 0.5, "v_min": 0.3, "v_max": 2.2})
        R.sg_idle.weight = -12.0
        R.sg_rate.weight = -2.0; R.sg_rate.params["f_max"] = 1.5
        print("[s10-v1h-trotT7] 膝摆幅下限 0.10 罚 −40｜不摆动罚 −12｜起抬频率罚 −2/f_max 1.5｜随机化同 T6")


@configclass
class DeeproboticsS10ClimbExpertC5EnvCfg(DeeproboticsS10ClimbExpertC4EnvCfg):
    """Climb-C5（09-14 20:3x，从 C4_9599 续）。作者 20:3x："平地要学会站住，和楼梯一样，要能正常的前进后退左右停止。"

    C4_9599 的平地实测：倒退 −0.3 → 实际 vx **+0.08**（反着走）、转 ±0.5 → wz **−0.05**（完全不转）、
    零指令 8 s 漂 0.99 m。查 C4 实际生效的 env.yaml 找到根子——高台专家的指令分布是
    `heading_command: true` + `rel_heading_envs: 1.0` + `ang_vel_z: (0, 0)` + `lin_vel_x: (0.3, 0.6)`
    + `rel_standing_envs: 0.0`：偏航指令全部由航向外环产生再被 (0,0) 夹成 0，航向目标也恒为 0，
    **它从训练第一步起就只见过"往前走"这一种指令**，倒退/转向/静止一个样本都没有。
    C4 里加的零指令动罚 cc_zero_hold 读数恒为 0，也是因为没有零指令样本可罚。

    C5 把指令分布改成与楼梯专家 S13 同一套（作者："和楼梯一样"），转向收一档：
      heading_command False、rel_heading_envs 0.5、rel_standing_envs 0.10、重采样 5~10 s、
      vx (−0.3, 0.6)、wz (−0.5, 0.5)（S13 是 ±0.8，这里收一档给翻越留裕度）、heading (−0.35, 0.35)；
      速度/转速跟踪 5.0/3.0 σ0.707 → 8.0/6.0 σ0.4（σ 0.707 太平，倒退和转向学不出梯度）。
    爬台侧只加码已验证有效的两项：越沿净空 12 → 30、后轮顶立面 −10 → −20。

    **不重蹈 C1 的覆辙**：C1（16:1x）一次性把指令分布、门控、零指令罚 −15、平地比例 30% 全换掉，
    200 步就把翻越能力冲没了。C5 只动指令分布 + 跟踪权重 + 那两项爬台权重，
    **地形比例、课程、CMD_GATE_V、HOLD_LAST_CONTACT 一律不动**；链上带登顶守卫：
    任一扫描点 bench_top33 登顶 < 3/3 就停训、回退到上一个点。

    侧倾仍不加码：C4 全程 cc_roll_ph 持平，MuJoCo roll 峰 15.9~17.3° 与 C3 相当；
    官方翻高台本身是"两条后腿折叠跨沿、左后主推"的不对称动作，罚过头会把动作本身压掉。"""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.heading_command = False
        c.rel_heading_envs = 0.5
        c.rel_standing_envs = 0.10
        c.resampling_time_range = (5.0, 10.0)
        c.ranges.lin_vel_x = (-0.3, 0.6)
        c.ranges.lin_vel_y = (0.0, 0.0)
        c.ranges.ang_vel_z = (-0.5, 0.5)
        c.ranges.heading = (-0.35, 0.35)
        R = self.rewards
        R.track_lin_vel_xy_exp.weight = 8.0; R.track_lin_vel_xy_exp.params["std"] = 0.4
        R.track_ang_vel_z_exp.weight = 6.0; R.track_ang_vel_z_exp.params["std"] = 0.4
        R.cc_edge_clr.weight = 30.0
        R.cc_hind_push.weight = -20.0
        print("[s10-climb-C5] 平地全向：vx(-0.3,0.6) wz±0.5 零指令 10% 重采样 5~10s｜跟踪 8/6 σ0.4｜越沿净空 +30｜后轮顶立面 −20｜地形与课程不动")


@configclass
class DeeproboticsS10PerceptV1HTrotT8EnvCfg(DeeproboticsS10PerceptV1HTrotT7EnvCfg):
    """Trot-T8（09-14 21:0x，从 T7_13999 续 300 步，小步找回高速）。

    T7 成功把"腿锁死、纯轮子滚"治掉了——0.5 档四条腿 36/75/66/46（T3 的左后只有 23%）、
    膝 σ 0.246~0.335（跨环境门槛是 0.03，**裕度 8 倍**）。代价是高速跟踪：
    1.8 → 1.44、2.3 → 1.49（T3 是 1.74 / 2.23，掉 17~33%）；腿摆大了推进效率就低。

    既然膝摆幅的裕度有 8 倍，就不需要 −40 这么大的压力维持了。T8 只松这一处：
      ① knee_amp 权重 −40 → −12、阈值 0.10 → 0.07（仍是门槛 0.03 的 2.3 倍，退不回锁死）；
      ② 速度跟踪权重上调一档，把高速段找回来。
    其余（sg_idle −12、起抬频率罚 −2/f_max 1.5、T6 的随机化）一律不动。

    判据：膝 σ 最小值必须仍 ≥ 0.10（掉到 0.03 附近就是在往锁死滑，立刻停）；
    1.8 档实际速度要回到 ≥ 1.6；0.5 档四条腿离地占比都要 ≥ 20%。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.sg_knee_amp.weight = -12.0; R.sg_knee_amp.params["amp_min"] = 0.07
        R.track_lin_vel_xy_exp.weight = float(R.track_lin_vel_xy_exp.weight) * 1.5
        print("[s10-v1h-trotT8] 膝摆幅罚 −40 → −12（阈值 0.07）｜速度跟踪 ×1.5｜其余同 T7")


@configclass
class DeeproboticsS10ClimbExpertC6EnvCfg(DeeproboticsS10ClimbExpertC4EnvCfg):
    """Climb-C6（09-14 21:2x，从 C4_9599 续）。作者 21:2x 定死范围："那就专门翻台阶，会往前走、停止、拐弯就行，不要求后退。"

    为什么改范围——C5 用实测把代价摆出来了。C5 给了全向指令（含倒退），200 迭代后：
      倒退 −0.3：+0.08（反着走）→ **−0.29 ✓**，但
      左后轮越沿净空：**0.095/0.086/0.131 → 0.014/0.000/0.000 m**（塌回 C3 水平），
      侧倾峰 15~17° → 6~10°（退回左右对称的"贴着拖上去"，不折腿跨沿自然就不侧倾），
      转向仍然没学会（wz +0.01）。
    即"倒退"和"后腿跨沿"在抢同一份容量，而登顶一直是 3/3——只看登顶的守卫抓不到这种退化。

    C6 按作者定的范围重来（仍从 C4_9599 起，不从被侵蚀的 C5 起）：
      ① 指令去掉倒退：vx (−0.3, 0.6) → **(0.2, 0.6)**；保留 10% 零指令（停止）与 wz ±0.5（拐弯）；
      ② 跟踪权重收回来：C5 抬到 8.0/6.0 抬猛了 → **6.0/5.0**（C4 是 5.0/3.0）；
      ③ 爬台激励加码对冲稀释：越沿净空 30 → **80**、后轮顶立面 −20 → **−35**。
    侧倾维持不动：官方六次高台事件实测 |roll| ≤15°（max 14.8）、pitch 峰 −38°，
    而 C4_9599 是 roll 15~17°、pitch 36~37°——**俯仰已在官方带内、侧倾只超 0~2°，本就基本达标**，
    再压就是压掉"两条后腿折叠跨沿"这个动作本身（C5 侧倾降到 6~10° 正是退化的伴随现象，不是进步）。"""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.heading_command = False
        c.rel_heading_envs = 0.5
        c.rel_standing_envs = 0.10
        c.resampling_time_range = (5.0, 10.0)
        c.ranges.lin_vel_x = (0.2, 0.6)
        c.ranges.lin_vel_y = (0.0, 0.0)
        c.ranges.ang_vel_z = (-0.5, 0.5)
        c.ranges.heading = (-0.35, 0.35)
        R = self.rewards
        R.track_lin_vel_xy_exp.weight = 6.0; R.track_lin_vel_xy_exp.params["std"] = 0.4
        R.track_ang_vel_z_exp.weight = 5.0; R.track_ang_vel_z_exp.params["std"] = 0.4
        R.cc_edge_clr.weight = 80.0
        R.cc_hind_push.weight = -35.0
        print("[s10-climb-C6] 翻台阶专用：vx(0.2,0.6) 无倒退｜零指令 10%｜wz±0.5｜跟踪 6/5 σ0.4｜越沿净空 +80｜后轮顶立面 −35")


@configclass
class DeeproboticsS10ClimbExpertC7EnvCfg(DeeproboticsS10ClimbExpertC6EnvCfg):
    """Climb-C7（09-14 21:3x，从 C4_9599 续）＝ C6 ＋ 翻越限速。作者 21:3x："限制翻越速度，按官方 1 秒来。"

    同口径实测（起抬→落平，以 |pitch| 回到 5° 以内为界）：
      官方 1.0 s（0.9~1.2）、|pitch|>30° 持续 ≤0.45 s、隐含平均上升 ≈0.33 m/s；
      C4_9599 **0.48/0.48/0.50 s**、|pitch|>30° 只 0.08 s、vz 峰 1.39~1.57 m/s、均 0.55~0.58 m/s。
    俯仰峰值本来就对（−36.6~−37.4 vs 官方 −38），差的是**持续时间**——抬头只是个 0.08 s 的尖峰，
    后腿来不及折叠跨沿，只能贴着立面蹭。这与"越沿净空上不去"是同一件事的两面。

    C7 = C6（翻台阶专用指令：前进 0.2~0.6 / 零指令 10% / wz ±0.5，无倒退；越沿净空 +80、后轮顶立面 −35）
         ＋ 新增 `climb_rate_penalty` −60（v_max 0.6 m/s，翻越相位内，超出部分平方）。
    门限取 0.4（官方均值 0.33 略放宽）。0.6 只罚得到 1.5 m/s 那个尖峰，罚不到 0.55~0.58 m/s 的
    平均上升速度，而整段都太快才是后腿来不及折叠的原因；实测 0.6/-60 只读到 -0.22。

    判据（扫描双守卫之外自己看）：起抬→落平应从 0.48 s 拉长到 0.7 s 以上；
    后轮越沿净空中位数应从 0.066 往 0.10 走；登顶必须保持 3/3，用时可以从 7.9 s 涨到 9 s 左右。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        self.rewards.cc_rate = RewTerm(func=ss.climb_rate_penalty, weight=-250.0,
                                       params={"v_max": 0.4, "phase_only": True})
        print("[s10-climb-C7] C6 + 翻越上升限速 vz>0.4 罚 −250（官方 1 s / 均 0.33 m/s；我方 0.48 s / 均 0.58 / 峰 1.5 m/s）")


@configclass
class DeeproboticsS10ClimbExpertC8EnvCfg(DeeproboticsS10ClimbExpertC7EnvCfg):
    """Climb-C8（09-14 22:5x）＝ C7 ＋ 抬头保持正向塑形 ＋ 转向加权。起点等 C7 跑完按扫描结果定。

    C7 在 9800 点的实测（每 200 步逐点判的结论）：
      有效：后轮越沿净空中位数 0.083 → **0.103 m**、登顶稳 3/3、平地零指令 7.3 s 只漂 0.08 m。
      无效：**翻越用时 0.48 → 0.39~0.51 s，限速罚 −250 一点没拉长动作**；|pitch|>30° 仍只 0.08 s。
      缺口：**平地转向 ±0.5 给出 wz +0.01 / −0.02，等于不会转**（作者要求的三项里唯一没兑现的）。
      副作用：侧倾峰 16~18° → 19~21°（官方 ≤15），需盯住。

    为什么 C7 的限速无效、C8 换正向塑形：一次翻越冲刺的限速罚约 25 分
    （(1.4−0.4)² × 0.1 s × 250），而爬高进展给 130+ 分；更根本的是"慢慢翻"需要
    折腿→蹬伸→跨沿一整套新控制策略，冲上去是现成的局部最优——**罚"冲得快"不等于教会"慢慢翻"**。
    官方 |pitch|>30° 保持 0.45 s，我方 0.08 s：抬头是尖峰不是平台。

    C8 三改：
      ① 新增 `pitch_hold_reward` +25：翻越相位内抬头角在 25~45° **且机身在上升**的每一步给分，
         把抬头从尖峰拉成平台（要求上升是防止它停在台前仰着骗分）；
      ② 限速罚 −250 → **−150**（真正降为辅助——开训前实测 −500 时读数 −3.54，是抬头奖励的 50 倍，那还是罚主导，正是 C7 失败的原因）；
      ③ 转速跟踪 5.0 → 9.0（200 步学不出来，权重不够）。
    侧倾仍不加码，但若 C8 扫描点涨过 23° 就单独处理。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        R = self.rewards
        R.cc_pitch_hold = RewTerm(func=ss.pitch_hold_reward, weight=150.0,
                                  params={"lo_deg": 25.0, "hi_deg": 45.0, "vz_min": 0.05, "phase_only": True})
        R.cc_rate.weight = -150.0
        R.track_ang_vel_z_exp.weight = 9.0
        print("[s10-climb-C8] 抬头保持 25~45° 且上升 +150（把尖峰拉成平台）｜限速罚 −150（辅助）｜转速跟踪 9.0")


@configclass
class DeeproboticsS10ClimbExpertC9EnvCfg(DeeproboticsS10ClimbExpertC8EnvCfg):
    """Climb-C9（09-15 00:0x）。作者 23:5x 拍板的整合版计划第 1 轮，从 C8_10000 续。

    ① **相位门加停滞退出**（ENCOUNTER_STALL_S=2.0）——这是第 0 步，不修则必然第三次重演：
       C7 的 cc_edge_clr 刷到 170 倍、C8 的 cc_pitch_hold 刷到 1000 倍，两次都是同期真·爬高回落、
       扫描点登顶 0/3。根因是 encounter 靠"墙高记忆 + 轮高差"维持，在墙前反复仰起即可长期满足。
    ② **行驶姿态**（今晚补出来的缺口，此前完全没训）：官方高台模式行驶 站高 0.344、膝 |q| 1.48；
       我方 C4_9599 是 0.417 / 0.72 —— 高 5 cm、只弯一半，几乎直腿走。作者 09-14 早上提的
       "重心高 10 cm、膝不够弯"在踏步线已解决，这条线原封未动。起手没蹲下去，折腿行程被吃光，
       很可能就是"翻越时来不及折腿只能蹭"的上游原因。
    ③ **净空目标下调**：慢动作视频（240 fps）量出官方后轮跨沿净空只有**几厘米**，靠折腿避开立面
       而非举高。我方 C4 已是 0.066~0.092，反而更高 —— 之前逼它做官方不做的动作，一加码就崩。
       h_target 0.08 → 0.05。
    ④ **重罚后轮顶立面**：官方≈0，我方左后 19~33%。cc_hind_push −35 → −60。
    ⑤ 抬头保持权重回调 150 → 60（门修好后不需要这么大，且它正是被刷的那一项）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as rw
        cr.ENCOUNTER_STALL_S[0] = 2.0                                   # ①
        R = self.rewards
        # ② 行驶姿态：站高目标 0.43 → 0.37（官方高台模式行驶 ①最低轮 0.367 / ②四轮均 0.344；
        #    C4_9599 实测 0.417，正是被 0.43 这个目标训出来的）。膝角随站高下降自然变弯，不单独加项。
        R.base_height_flat.params["target_height"] = 0.37
        R.base_height_flat.weight = -30.0
        R.cc_edge_clr.params["cap"] = 0.05                               # ③
        R.cc_hind_push.weight = -60.0                                    # ④
        R.cc_pitch_hold.weight = 60.0                                    # ⑤
        print("[s10-climb-C9] 相位门停滞 2 s 退出｜站高目标 0.344 罚 −30｜净空 cap 0.05｜后轮顶立面 −60｜抬头保持 60")


@configclass
class DeeproboticsS10PerceptV1HTrotT9EnvCfg(DeeproboticsS10PerceptV1HTrotT7EnvCfg):
    """Trot-T9（09-15 00:2x，整合版计划第 2 轮；从 T7_13999 续）。**取代原 T8**——T8 只想松开膝摆幅罚找回高速，
    方向不对：踏步线真正的问题不是幅度也不是速度，是**相位**。

    09-14 夜用不依赖接触判定的判据比三方（官方录制 / 真机转储 / 两个仿真）：
      官方 gftb1 / gftbzx1：左右反相 −0.53 / −0.75，hipy σ 0.172~0.197
      真机 T3_13400（作者现场"踏步走不稳"）：**+0.05 / +0.23**，hipy σ 0.065~0.120
      MuJoCo T3 / T6：−0.11/+0.51、+0.88/+0.67
    → **我们的"踏步"从来不是官方那种交替迈腿**，是四条腿同相小幅抖 + 轮子驱动。
      旧判据全漏了：接触统计的"对角同步"按触地算（轮子贴地时怎么抖都算）；膝 σ 只量幅度；
      Isaac 侧 sg_diag 读数只有 +0.076、权重 2.0，从头到尾没拉动过相位。

    T9 四改：
      ① 新增 `lr_antiphase_reward` +8（前后各一对）。离线用同一公式验过区分度：
         官方 gftbzx1 +0.959、gftb1 +0.626；我方 T3(MuJoCo) −0.105、真机 T3 −0.182、T6 锁死 −0.337。
         防刷分：归一化符号积（不动腿 p→0 拿不到分）× 幅度门（必须"既在动又反相"）。
      ② `sg_diag` 2.0 → 8.0（它才是原本该管相位的项，权重一直太小）。
      ③ **新增 `yaw_ref_penalty` −8**（参考航向按指令角速度积分）。查出 T7 是 `heading_command: false`，
         于是 heading_hold 里原有的两项**恒返回 0、从来没生效过**；踏步线只罚瞬时角速度误差，
         不罚累计航向偏差 —— 这就是作者现场"上斜坡或越野不够稳定"的机理：
         MuJoCo 实测 15° 坡 1.0 m/s 偏航 −26°、越野 +17~+20°。
         注：训练地形本来就有 30% 坡 + 35% 粗糙，**不是见得少，是没有纠偏信号**，所以不动地形占比。
      ④ 膝摆幅罚 −40 → −12、阈值 0.10 → 0.07，速度跟踪 ×1.5（原 T8 的意图，用来找回高速 1.8→1.44）。

    步频维持约 1.6 Hz（作者 09-15 00:0x"按你的来"）：先训出正确的左右交替，再回头压向官方 1.10 Hz（T10）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import step_gait as sg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import heading_hold as hh
        R = self.rewards
        R.sg_lr_anti = RewTerm(func=sg.lr_antiphase_reward, weight=8.0,
                               params={"tau": 1.0, "amp0": 0.05, "v_min": 0.3, "v_max": 2.2})
        R.sg_diag.weight = 8.0
        R.st_yaw_ref = RewTerm(func=hh.yaw_ref_penalty, weight=-8.0, params={"dead_deg": 6.0})
        R.sg_knee_amp.weight = -12.0; R.sg_knee_amp.params["amp_min"] = 0.07
        R.track_lin_vel_xy_exp.weight = float(R.track_lin_vel_xy_exp.weight) * 1.5
        print("[s10-v1h-trotT9] 左右反相 +8（官方 +0.63~0.96 / 我方 −0.1~−0.34）｜对角同步 8.0｜参考航向罚 −8（死区 6°）｜膝摆幅 −12/0.07｜速度跟踪 ×1.5")


@configclass
class DeeproboticsS10ClimbExpertC10EnvCfg(DeeproboticsS10ClimbExpertC9EnvCfg):
    """Climb-C10（09-15 00:3x，从 C9_10400 续）。C9 的 10200/10400 两个检查点闭环后的调整。

    C9 拿到的（**翻台阶线第一次在动作形态上追平官方**，三条高度一致）：
      起抬→落平 0.48 → **0.83/0.85/0.84 s**（官方 1.0）；|pitch|>30° 0.08 → **0.65 s**（官方 ≤0.45）；
      pitch 峰 −39.7（官方 −38）；侧倾 11.5~12.8（官方 ≤15）；登顶 3/3；相位门守住了
      （C7/C8 在 10200 同位置都崩成 0/3）。

    C9 丢掉的（10200 → 10400 是**趋势不是瞬态**，所以按预写触发条件停训调整）：
      平地零指令 7.9 s 漂 **3.51 → 4.69 m**、偏航 −57°、前进 0.5 给出 **1.09**（超速 118%）、
      转向 wz +0.04/+0.02（完全不转）；落平过冲 **+15.9~17.6°** 超标（官方 ≤+12）；
      后轮 hl 过沿顶踢面仍 19%。

    C10 五改：
      ① **加 `yaw_ref_penalty` −8**（死区 6°）。这是今晚查出的**跨线共性缺陷**：高台专家与踏步策略
         都只罚瞬时角速度、不罚累计航向偏差，`heading_hold` 里原有两项因 `heading_command:false` 恒为 0。
         高台平地偏 −86°/−57°、踏步坡上偏 −26°，机理同源。
      ② `cc_zero_hold` −10 → **−25**（平地停不住）。
      ③ 站高目标 0.37 → **0.40**（改两步走）。C9 一刀从 0.43 砍到 0.37 正是平地重平衡失控的可疑源头；
         先落到 0.40 站稳，再谈 0.37。
      ④ `cc_pitch_hold` 60 → **30**。已达标且超了（0.65 s vs 官方 ≤0.45），继续加码只会让动作更夸张，
         落平过冲 +16~17.6° 就是夸张的代价。
      ⑤ `cc_hind_push` −60 → **−90**（hl 过沿顶踢面 19%，目标 <5%）。
      速度跟踪 ×1.3 治超速 118%。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import heading_hold as hh
        R = self.rewards
        R.cc_yaw_ref = RewTerm(func=hh.yaw_ref_penalty, weight=-8.0, params={"dead_deg": 6.0})   # ①
        R.cc_zero_hold.weight = -25.0                                                            # ②
        R.base_height_flat.params["target_height"] = 0.40                                        # ③
        R.cc_pitch_hold.weight = 30.0                                                            # ④
        R.cc_hind_push.weight = -90.0                                                            # ⑤
        R.track_lin_vel_xy_exp.weight = float(R.track_lin_vel_xy_exp.weight) * 1.3
        print("[s10-climb-C10] 参考航向罚 −8｜零指令动罚 −25｜站高目标 0.40（两步走）｜抬头保持 30｜后轮顶立面 −90｜速度跟踪 ×1.3")


@configclass
class DeeproboticsS10ClimbExpertC11EnvCfg(DeeproboticsS10ClimbExpertC9EnvCfg):
    """Climb-C11（09-15 00:5x，从 **C9_10400** 续 400 步）。C10 失败后按预写触发条件回退。

    C10 为什么失败（10600 点登顶 0/3）：**是我把惩罚量纲搞炸了**，不是方向错。
      `yaw_ref_penalty` 用弧度平方且不封顶，偏航 72° 时读到 **−12.70/s**——
      比速度跟踪(5.28)大一倍、比真·爬高进展(0.56)大 23 倍，整个奖励被它主导，
      策略放弃爬台去修航向，结果登顶 0/3 而平地照样漂 5.02 m，两头没做好。
      已改 Huber：死区 6° 内不罚、25° 以内二次、超出转线性（硬封顶不行——40° 和 72° 会读到同一个值，
      策略没有往回修的动力）。现在 72° → 2.44/s、40° → 0.98/s、15° → 0.07/s，与其它项同一档。

    C11 **严格只加平地项，爬台项一个不动**（含 C9 的站高目标 0.37 原样保留）——
    这是 10400 点写死的触发条件，目的是隔离变量：若平地恢复，说明站高 0.37 不是元凶；
    若仍不恢复，下一轮才动站高（→0.40）。
      ① `yaw_ref_penalty` −3（死区 6°、拐点 25°）
      ② `cc_zero_hold` −10 → −25
      ③ 速度跟踪 ×1.3（治前进 0.5 给出 1.09 的超速 118%）"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import heading_hold as hh
        R = self.rewards
        R.cc_yaw_ref = RewTerm(func=hh.yaw_ref_penalty, weight=-3.0,
                               params={"dead_deg": 6.0, "max_deg": 25.0})
        R.cc_zero_hold.weight = -25.0
        R.track_lin_vel_xy_exp.weight = float(R.track_lin_vel_xy_exp.weight) * 1.3
        print("[s10-climb-C11] 只加平地项：参考航向罚 −3（Huber 6°/25°）｜零指令动罚 −25｜速度跟踪 ×1.3｜爬台项全部沿用 C9")


@configclass
class DeeproboticsS10ClimbExpertC12EnvCfg(DeeproboticsS10ClimbExpertC11EnvCfg):
    """Climb-C12（09-15 01:2x，从 **C9_10400** 续）。C11 隔离实验结论后的修正。

    C11 做的是隔离实验（从 C9_10400 续，**只加平地项、爬台项一个不动**，含 C9 的站高目标 0.37 原样保留）：
      10600 点 登顶 3/3 ✓ 但平地照样垮（前进 0.5 给出 0.66、偏航 −65°）；10799 点 登顶 0/3。
      `cc_yaw_ref` 读 −1.84（Huber 修对了，不再像 C10 那样炸到 −12.7）。
    → **加平地项救不回平地**，按 10400 点写死的分支判定：**站高目标那一刀（0.43 → 0.37，一次 6 cm）就是元凶**。

    C12 两改：
      ① **站高目标 0.37 → 0.43**（退回 C8 的值）。先把平地拿回来；确认恢复后，后续再用
         0.43 → 0.42 → 0.41 的小步下压去追官方的 0.367，不再一刀切。
      ② `cc_pitch_hold` 60 → **30**。C11_10600 的翻越用时已经反向超标到 **1.39 s**（官方 0.9~1.2 太慢）、
         落平过冲 +18.3°，该项读到 +3.49 —— 抬头保持已经过量，继续给分只会让动作更夸张。
      平地三项（yaw_ref −3 Huber、zero_hold −25、速度跟踪 ×1.3）沿用 C11。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.base_height_flat.params["target_height"] = 0.43        # ①
        R.cc_pitch_hold.weight = 30.0                            # ②
        print("[s10-climb-C12] 站高目标退回 0.43（隔离实验证明 0.37 是平地垮掉的元凶）｜抬头保持 60→30（用时已反向超标到 1.39 s）｜平地三项沿用 C11")


@configclass
class DeeproboticsS10StairExpertS14EnvCfg(DeeproboticsS10StairExpertS8EnvCfg):
    """StairN-S14（09-15 01:3x，整合版计划第 3 轮；从 **F = S8_14800** 续）。

    前置纠正（09-14 夜）：作者给出决赛楼梯实测 **踢面 11 cm（两级 15）、踏面 52~60 cm、宽 4 m、S 型弯弧度不大**
    → 坡度 11°，而 F 拿 30/30 的验收场景 `bench_stairs_site_t55_airy` 立面正是 15/11、踏面 55，
    **就是决赛几何**。09-14 16:23 我从里程计 Δz/Δx≈0.6 推的"31°"是**另一处楼梯**（AGX 侧已确认），
    据此建的陡窄场景把 S12/S13 带偏了两轮。**楼梯线的基线其实是好的。**
    23:4x 新增验收：决赛几何 + 小幅转向（起步偏 ±8°、航向目标偏 ±8°、0.25 慢档）**5/6 上楼成功**。

    S14 只补 F 实测出来的四个缺口（**不训横移、不训楼梯上停车**——作者已定）：
      ① **机身跟坡**：官方爬楼段 pitch −15.4°（p5/p95 −20.5/−9.7），F 只有 **−2.4°**（几乎水平）。
         用 `pitch_hold_reward` 的角度带版本（8~20°、要求在上升）+10。
      ② **爬楼中航向保持**：F 起步偏 −8° 时爬楼中漂到 **+20.8°**。加 `yaw_ref_penalty` −3
         （Huber 死区 6°/拐点 25°，量纲已在 C10 事故后修正）。
      ③ **后腿抬腿做透**：官方"后轮抬高≈踢面、允许擦踢面"，F 的**左后过台沿顶踢面 21%**、
         两后轮悬空 28% vs 46% 不对称。加后轮专用顶立面罚 −12，`ss_lr_sym` −5 → −10。
      ④ **平地欠速**：倒退给 −0.3 实际 −0.19（欠 37%）、前进给 0.4 实际 0.30（欠 25%）。速度跟踪 ×1.3。
    **0.2 档为必保项**（F 实测 67.9 s 上楼成功），每个扫描点都要跑，掉了即回退。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import heading_hold as hh
        R = self.rewards
        R.ss_pitch_slope = RewTerm(func=ss.pitch_hold_reward, weight=10.0,                  # ①
                                   params={"lo_deg": 8.0, "hi_deg": 20.0, "vz_min": 0.02, "phase_only": True})
        R.ss_yaw_ref = RewTerm(func=hh.yaw_ref_penalty, weight=-3.0,                        # ②
                               params={"dead_deg": 6.0, "max_deg": 25.0})
        R.ss_hind_push = RewTerm(func=ss.wheel_riser_push_penalty, weight=-12.0,            # ③
                                 params={"phase_only": True, "wheels": (2, 3)})
        R.ss_lr_sym.weight = -10.0
        R.track_lin_vel_xy_exp.weight = float(R.track_lin_vel_xy_exp.weight) * 1.3          # ④
        print("[s10-stairN-S14] 机身跟坡 8~20° +10｜爬楼航向保持 −3（Huber）｜后轮顶立面 −12｜左右对称 −10｜速度跟踪 ×1.3｜0.2 档必保")


@configclass
class DeeproboticsS10ClimbExpertC13EnvCfg(DeeproboticsS10ClimbExpertC12EnvCfg):
    """Climb-C13（09-15 03:0x，从 C12_10799 续）。专治"平地转不动"——作者要求的"前进/停止/拐弯"里唯一没兑现的。

    **先诊断再动手**（§7 里给自己留的提醒：`track_ang_vel_z_exp` 读数一直有 +5，
    说明不是没信号、是转不动，不能继续加权重）。C12_10799 平地探针实测：
      转 +0.5 → 实际 wz **+0.046**，左轮均 −1.79 / 右轮均 −32.62，**左右差 +30.83 rad/s**
      转 −0.5 → wz −0.094，左右差 +13.99
      **零指令 → wz +0.001，右轮仍以 −14.21 rad/s 空转**（轮面速度 1.15 m/s，机身只走 0.1 m/s）
    差速转向几何：轮距 0.362 / 轮半径 0.081 → **转 0.5 rad/s 只需左右差 2.2 rad/s**。
    实测差速是需要量的 6~14 倍却一点不转 → **轮子在贴地打滑，差速没转化成转动，执行端废掉了。**

    对照楼梯专家 F（转 ±0.5 能给出 +0.31/−0.56）：F 有 `wheel_drive_balance −5` 和 `ss_air_spin −0.3`，
    **高台专家这两项一个都没有**，于是学出了这个既不推进也不转向的无效空转。

    C13 三改（不动爬台项——C12 的爬台成绩是今晚最好，不能赔进去）：
      ① 新增 `wheel_slip_penalty` −3（触地轮"轮面速度 vs 机身速度"之差超 0.30 m/s 的部分）。
         `air_wheel_spin_penalty` 只罚腾空轮，治不了贴地打滑，故另写。
      ② 新增 `wheel_drive_balance` −5（照抄 F 的配置，罚左右/前后驱动力矩不平衡）。
      ③ `ss_air_spin`（腾空轮空转）−0.3，同样照抄 F。
    转向奖励权重**不动**（信号已足够，加了也没用）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as wb
        R = self.rewards
        # 权重由开训前实测定，不照抄 F：F 没有这个病、原始值小，同样权重在这里会炸
        #   实测（C12_10799 在本环境）：slip 原始 0.42、bal 原始 1.47、air_spin 原始 31.3
        #   照抄 F 的 −3/−5/−0.3 会读到 −1.27/−7.36/−9.39，后两项都超过速度跟踪(6.08) → 必然重演 C10
        #   现按"每项 ≤ 主项 30%、合计 ≤ 40%"配：目标读数约 −1.3 / −1.5 / −1.6，合计 4.4（主项 11.2）
        R.cc_wheel_slip = RewTerm(func=ss.wheel_slip_penalty, weight=-3.0, params={"tol": 0.30})
        R.cc_wheel_bal = RewTerm(func=wb.wheel_drive_balance, weight=-1.0, params={"cmd_threshold": 0.15})
        R.cc_air_spin = RewTerm(func=ss.air_wheel_spin_penalty, weight=-0.05, params={"w_max": 9.0})
        print("[s10-climb-C13] 治转不动：贴地打滑罚 −3（>0.30 m/s）｜驱动平衡 −5（照抄 F）｜腾空空转 −0.3｜转向奖励权重不动（诊断表明信号够、执行端废）")


@configclass
class DeeproboticsS10ClimbExpertC14EnvCfg(DeeproboticsS10ClimbExpertC13EnvCfg):
    """Climb-C14（09-15 03:3x，从 C13_11198 续 800 步）。C13 用 400 迭代证明"罚无效动作"建立不了转向模态。

    C13 实测左右轮差速（左 fl,hl 均 − 右 fr,hr 均；转 0.5 rad/s 只需 ±2.2）：
      零指令  +13.81 → −2.27 → −3.22   **无效差速治好了**（三个轮子惩罚有效，这是 C13 的真实成果）
      转 +0.5 +30.83 → +25.95 → +18.08  符号对、大小错 8 倍，wz 只到 +0.078
      转 −0.5 +13.99 → +8.27  → **+8.28**  **符号自始至终是反的**
    → 策略没有"反转差速"这个概念。查证：高台专家从 C4 起转向指令范围就是 (0,0)，
      **直到 C5 才第一次见到转向指令** —— 这条线从来没学过差速转向，是从零建立新模态。

    **今晚第三次验证同一条规律**：罚"别做错的"教不会"做对的"——
      限速罚 −250 对翻越用时完全无效 → 换抬头保持正向塑形才生效；
      踏步线加左右反相**奖励**才把 −0.12 做到 −0.84；
      这里罚轮子无效动作 400 迭代，右转符号纹丝不动 → 换正向塑形。

    C14 两改：
      ① 新增 `wheel_diff_sign_reward` —— 只教"差速方向跟指令一致"，大小交给已有的
         `track_ang_vel_z_exp` 与 `wheel_drive_balance`。用与踏步线反相奖励同一套路数
         （归一化符号积 × 幅度门，防"不动差速骗分"）。
      ② 步长 400 → **800**：从零建立新运动模态，400 步不够（C13 已证）。
    爬台项与 C13 的三个轮子惩罚全部保留不动（C13 的爬台成绩 3/3、翻越 0.76 s、侧倾 11.3° 要守住）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        self.rewards.cc_wheel_dir = RewTerm(func=ss.wheel_diff_sign_reward, weight=6.0,
                                            params={"wz_min": 0.05, "amp0": 1.0})
        print("[s10-climb-C14] 新增差速方向奖励 +6（只教方向，大小交给转速跟踪）｜C13 的三个轮子惩罚保留｜800 步")


@configclass
class DeeproboticsS10ClimbExpertC15EnvCfg(DeeproboticsS10ClimbExpertC14EnvCfg):
    """Climb-C15（09-15 04:2x，从 C14_11600 续 600 步）。C14 把差速**方向**学会了，C15 收**大小**。

    C14 成果：右转差速 +8.28 → **−11.21（符号翻转）**，零指令差速 −3.22 → −1.41，登顶 3/3，
    零指令漂 0.27 m。`cc_wheel_dir` 从 −0.975 涨到 +3.20。
    C14 遗留：差速大小错 5~7 倍（±10~20 vs 需要 ±2.2），差速过大 → 轮子打滑 → 实际 wz 只到 +0.198/−0.102。

    C15 加 `wheel_diff_excess_penalty`（差速超过指令所需的部分，线性）。
    **为什么不用 exp 跟踪**：设计时先试过，σ=4 在当前误差 9~18 rad/s 处奖励≈0、**没有梯度**
    （和限速罚 −250 失效同一个毛病：罚/奖在远处没梯度）；放宽 σ 又失去靠近目标时的分辨力。
    线性超量罚处处有梯度，且 |d| 接近所需时自动归零；tol=2 保证**不干扰已治好的零指令行为**
    （零指令差速 −1.41 落在容差内，罚 0）。
    实测罚值：转 +0.5 → 2.40/s、转 −0.5 → 1.05/s、零指令 → 0。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        self.rewards.cc_wheel_ex = RewTerm(func=ss.wheel_diff_excess_penalty, weight=-0.15,
                                           params={"tol": 2.0})
        print("[s10-climb-C15] 新增差速超量罚 −0.15（线性、处处有梯度、tol 2 不扰零指令）｜C14 方向奖励与全部爬台项保留")


@configclass
class DeeproboticsS10ClimbExpertC16EnvCfg(DeeproboticsS10ClimbExpertC15EnvCfg):
    """Climb-C16（09-15 05:2x，从 C15_12000 续 600 步）。单变量推进：超量罚 −0.15 → −0.30。

    C15 结果：左转差速 20.23 → **6.98**（收 3 倍，超量罚有效）、右转 −11.21 → −15.93（反而变大）；
    转向 wz +0.186 / **−0.232**（这条线至今最好，右转 2.3 倍于 C14）；登顶全程 3/3。

    **量化出真正的瓶颈**：差速 6.98 rad/s 按运动学应产生 wz = 6.98 × 0.081/0.362 = **1.56 rad/s**，
    实测只有 0.186 → **88% 被打滑吃掉**。方向和大小都在改善，但抓地跟不上。

    C16 只动一个量（超量罚加倍），观察：
      ①左转差速能否从 6.98 继续收向 2.2 ②右转差速能否止住增长 ③wz 能否随差速收敛而上升
      （若差速收到 ~3 而 wz 仍 <0.25，则确认瓶颈在抓地，下一轮转向治打滑/配重，而不是继续收差速）
    `cc_wheel_slip` 暂不加码：它比较的是轮面速度与机身**前进**速度，纯转向时两侧本就应有差异，
    加码会误伤正当的差速转向（tol 0.3 m/s ≈ 差速 ±3.7 rad/s，目标 2.2 在容差内、当前 7 已越界）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_wheel_ex.weight = -0.30
        print("[s10-climb-C16] 差速超量罚 −0.15 → −0.30（单变量）｜瓶颈已量化：差速 6.98 应给 wz 1.56，实测 0.186，88% 被打滑吃掉")


@configclass
class DeeproboticsS10ClimbExpertC17EnvCfg(DeeproboticsS10ClimbExpertC15EnvCfg):
    """Climb-C17（09-15 05:4x，从 **C15_12000** 续 600 步。C16 判无收益已停，不做起点）。

    **C16 结论（超量罚 −0.15 → −0.30）：无收益，方向终止。**
      差速没收敛：转 +0.5 侧 6.98 → **10.05**（目标 2.2）；转 −0.5 侧 15.93 → 9.72。
      wz 没涨：+0.186/−0.232 → **+0.151/−0.197**；打滑损失 88% → **93%**。
      翻越相位内 roll 7.4/9.5/9.4°（C15 是 9.5/10.1/9.3），姿态没坏——之前把"全程 roll 峰"
      与"翻越相位内 roll 峰"混读了，汇总里每个检查点有六个 roll 值，前三全程后三相位内。

    **查出了 C13～C16 四代转向不动的真因：不是抓地，是第三次刷分。**
      四轮转速（rad/s，转 ±0.5）：
        C13_11198 转+0.5  fl −0.52  fr −1.05  hl −0.74  **hr −36.37**
        C15_12000 转−0.5  fl −13.49 fr +0.89  hl −2.89  **hr +14.59**
        C16_12200 转+0.5  fl +0.27  fr −3.00  hl −1.98  **hr −18.82**
      三个轮子 |ω|<3，整个"差速"是 hr 一个轮子。转向时 hr hipy +1.55 / 膝 −1.90（零指令 +0.57/−1.27）
      → **折起右后腿、空转悬空轮**，用假差速满足 `cc_wheel_dir` 的符号与幅度门。
      对照：前进 0.5 时四轮 10.1/9.9/9.7/9.7 均匀 → 直行健康，只有转向指令触发退化解。
      代价比：farming +4.2（cc_wheel_dir w=6.0）vs cc_air_spin −0.37（w=−0.05）= **8 倍**，必然被刷。
      连带：`wheel_diff_excess_penalty` 量的是"左右均值之差"，被这一个疯转轮主导 →
      **C15/C16 两代罚的是个假量**，这是它调权重没用的原因。

    C17 只做一件事：**把差速类奖励的口子堵上**（改定义，不改权重；与 cc_edge_clr、cc_pitch_hold 两次同源）。
      ① `cc_wheel_dir` 加 `contact_only=True`：左右均值只对触地轮取，整项再乘触地轮比例。
      ② `cc_air_spin` −0.05 → **−0.5**（把刷分代价从收益的 1/8 抬到同量级；w_max 9 不变）。
      ③ **删 `cc_wheel_ex`**：它测的是被单轮主导的假量，两代已证无效，留着只会继续误导。
    观察（检查点 12200/12400/12600）：
      ① 转向时四轮是否变成"两侧成对反向"（判据：|ω| 最小的轮 ≥ 最大轮的 30%，而不是现在的 <10%）
      ② wz 能否越过 0.35（当前 0.19/0.23 的天花板）
      ③ 登顶保持 3/3、翻越内 |roll| ≤ 12°
    下一轮触发条件：
      · 四轮成对反向出现 **且** wz > 0.35 → 转向解决，下一轮转去治**落平过冲**（见下）。
      · 四轮成对反向出现 **但** wz 仍 < 0.25 → 这才是真·抓地瓶颈，按 ±0.2 交付转向能力，
        转去治落平过冲；不再在转向上投迭代。
      · 仍是单轮空转 → `cc_wheel_dir` 本身要换成"每侧轮速跟踪各自目标"的形式，而不是差值形式。
    **另记（C17 不动，留作下一轮）**：落平过冲 C13 的 14~16° → C14_11997 起 **25~28°**（官方 ≤+12），
    退化点在 C14（引入转向那轮），不是 C16。这是翻台阶本身的质量缺口，优先级高于转向。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_wheel_dir.params["contact_only"] = True          # ①
        self.rewards.cc_air_spin.weight = -0.5                           # ②
        self.rewards.cc_wheel_ex = None                                  # ③
        print("[s10-climb-C17] 差速只看触地轮 + 腾空空转罚 −0.05→−0.5 + 删超量罚｜真因：转向是折起右后腿空转单轮刷分（hr 19~36 rad/s，另三轮<3）")


@configclass
class DeeproboticsS10PerceptV1HTrotT10EnvCfg(DeeproboticsS10PerceptV1HTrotT9EnvCfg):
    """Trot-T10（09-15 06:0x 写好待训；从 T9 末点续 600 步）。**单变量：姿态惩罚改按坡面法向算。**

    T9 成果：左右反相 −0.12 → **−0.84**（官方 −0.53/−0.75），踏步相位问题解决；
      `yaw_ref_penalty` 让平地偏航从 +16° 收到 ±5°。
    T9 遗留（作者现场原话"踏步模式上斜坡或者越野不够稳定，踏步会乱"）——
      T9 四检查点在 10° 坡 1.0 m/s：
        14000 偏航 −3°  / |roll| p95 **11.2°**      14200 偏航 **+30°** / |roll| p95 5.4°
        14400 偏航 +4°  / |roll| p95 **14.8°**      14598 偏航 **+27°** / |roll| p95 6.1°
      **强反相关**：侧倾大 → 偏航不漂；侧倾小 → 偏航漂 27~30°。不是随机双峰，是两种姿态模式。
      机理：坡有横向分量时，机身跟坡侧倾 → 两侧轮子都吃上载荷 → 抓得住；
            机身保持水平 → 下坡侧载荷不足 → 打滑 → 整车往下坡方向偏转。
      根因（读源码确认）：`flat_orientation_l2_terrain_aware` 名为地形感知，实际只按**地形高度**
      整体打折（阈 0.06 m → ×0.1），惩罚本身仍是 `projected_gravity_b[:,:2]` 的平方
      = "机身相对**重力**保持水平"，**完全不看坡面朝向**。10° 坡的高度差多数区域在 0.06 门槛之下
      → 坡上这项**全额生效**，一直把机身往水平拽。链条：罚水平 → 不敢跟坡倾 → 打滑 → 偏航漂。

    T10 只换这一个函数为 `flat_orientation_normal_aware`（高程扫描最小二乘拟合局部平面 → 法向 →
    罚"机身 z 轴 vs 地面法向"）。离线八情形自检（scratchpad/t10_test.py，纯 torch）：
      平地逐位等价（0.0000/0.0000、0.0302/0.0302）；
      **10°横坡+机身水平 0.0302（新罚）、跟坡侧倾 0.0000（不再罚，这就是打滑的解）**；
      15 cm 台阶（残差 **0.0429**）与 33 cm 墙（残差 **0.0845**）被**残差门 0.03 拦下 → 逐位退化成原式，
      爬台/楼梯行为一个字不变**。残差门必须 0.03 不能 0.05，0.05 会放台阶过去。
      自检已按真实扫描窗 1.6×1.0 m / 187 点复跑，八条结论不变。
    **口径：离线通过 ≠ 训练有效，更 ≠ 真机验收。**

    判据（每 200 点）：
      ① 10° / 15° 坡 1.0 m/s **偏航变化 ≤ ±10°**（当前 −3 / +30 / +4 / +27）
      ② 平地 |roll| p95 **不劣于 7.7°**（保证没把平地弄晃）
      ③ 左右反相不劣于 −0.70（T9 成果不能丢）
    触发条件：
      · ① 达标且 ②③ 不劣 → 收，转去压步频 1.6 → 官方 1.10 Hz（作者已批"先交替再压步频"）。
      · ① 没动但坡上 |roll| p95 升到 10° 以上 → 姿态跟坡了但打滑没好，说明还有别的因素（轮子侧向刚度），
        转去查坡上轮子侧滑，不再动姿态项。
      · ② 变差（平地晃） → 残差门 0.03 太松，收到 0.02 重训。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import parkour_rewards as pk
        R = self.rewards
        R.flat_orientation_l2.func = pk.flat_orientation_normal_aware
        R.flat_orientation_l2.params = {
            # 必须用主 height_scanner（1.6×1.0 m / 0.1 → 17×11 = 187 点），**不能用 height_scanner_base**：
            # 后者是 0.1×0.1 m 的 3×3 网格，只有 9 个点、10° 坡在窗内只有 1.8 cm 高差，
            # 拟合坡度会被噪声主导，且 cnt>8 的门刚好卡边。
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "max_slope_deg": 20.0,
            "max_residual": 0.03,
            "terrain_height_threshold": 0.06,
            "high_terrain_penalty_scale": 0.1,
        }
        print("[s10-v1h-trotT10] 姿态惩罚改按坡面法向（残差门 0.03，台阶/墙自动退化成原式）｜治 T9 坡上偏航漂 27~30°")


@configclass
class DeeproboticsS10ClimbExpertC18EnvCfg(DeeproboticsS10ClimbExpertC17EnvCfg):
    """Climb-C18（09-15 06:2x 写好待训）。**单变量：姿态惩罚的豁免门从"地形高"换成"正在翻越"。**

    **治的是"落平过冲 +28°"—— 09-15 06:1x 查出它根本不是过冲，是稳态行驶姿态。**
    登顶后稳态段实测（|wy|≈0.01，持续 >1 s，三条一致）：

    | 检查点 | 稳态 pitch | 膝 fl/fr/hl/hr | 膝顶限位占比 |
    |---|---|---|---|
    | C13_11198 | **−2.2~−2.5°** | +1.00/+0.37/−0.71/−0.11 | **0%** |
    | C15_12000 | **+22~+26°** | +1.65/+0.83/+0.27/**−2.69** | **25%** |
    | C16_12200 | +23~+26° | +1.75/+1.07/−0.42/**−2.69** | 25% |

    hr 膝 −2.69 顶在限位（±2.7227）整整一秒不动。机器人就这么**低着头 26° 在台面上开**。
    "落平过冲"指标量的是翻越后的最大 pitch，所以它测到的是这个姿态，不是落地瞬态。
    退化点在 **C14**（引入转向那轮），C13 还是好的。

    **结构缺陷（读源码确认）**：四个姿态惩罚（hipx/hipy/knee `joint_pos_penalty_except_turn_side_cmd`
    + `flat_orientation_l2_terrain_aware`）的豁免门都是 `terrain_height > 0.06 → ×0.1`。
    **33 cm 台面本身就是"高地形"** → 机器人登顶后在台面上行驶的**全程**只挨 1/10 的姿态罚。
    给 1 秒翻越设计的豁免，变成了在台面上的长期免罚。
    同时 `joint_pos_limits` 权重是 **0.0** —— 没有任何项罚"关节顶限位"。
    C13 在同样的门下姿态是好的，所以这个门是**使能条件不是起因**；但正是它让坏姿态白住、
    C14 学坏之后再也没有回正的压力。

    C18 只改门：`phase_exempt=True` → 豁免改用 `climb_exempt_mask`（= 翻越相位 | 遭遇相位），
    **落平即恢复全额姿态罚**。不新增任何惩罚项（三次刷分的教训：改定义，不是加罚）。

    判据（每 200 点）：
      ① 登顶后稳态 pitch **|p| ≤ 10°**（当前 +22~+26，C13 曾是 −2.2）
      ② 膝顶限位（|膝|>2.6）占比 **≤ 5%**（当前 25%）
      ③ 登顶保持 3/3、翻越内 |roll| ≤12°、翻越用时 0.9~1.2 s（不能为了姿态把翻越弄坏）
    触发条件：
      · ①② 达标且 ③ 不劣 → 收，"落平过冲"指标应同步回到 ≤+12。
      · ③ 变差（登顶掉条或翻越超时） → 说明翻越相位掩码盖不住真正需要豁免的窗口，
        把 `climb_exempt_mask` 往后延 0.3 s（落平缓冲）再训，而不是退回高度门。
      · ①② 没动 → 姿态罚权重本身太小，那时才谈加码（且要先量出它在台面段的实际读数）。

    **起点**：取 C17 的最优点（等 C17 三个检查点扫完再定），不预先写死。"""

    def __post_init__(self):
        super().__post_init__()
        import inspect
        R = self.rewards
        n = 0
        for name in ("hipx_joint_pos_penalty", "hipy_joint_pos_penalty",
                     "knee_joint_pos_penalty", "flat_orientation_l2"):
            term = getattr(R, name, None)
            if term is None or not isinstance(getattr(term, "params", None), dict):
                continue
            # 只给**确实接受 phase_exempt 的函数**注参数：管理器按签名校验，
            # 给一个没这个形参的函数塞 phase_exempt 会在建环境时直接报错。
            try:
                sig = inspect.signature(term.func).parameters
            except (TypeError, ValueError):
                print(f"[s10-climb-C18] !! {name} 的 func 取不到签名，跳过"); continue
            if "phase_exempt" not in sig:
                print(f"[s10-climb-C18] !! {name} 用的是 {getattr(term.func,'__name__',term.func)}，"
                      f"不接受 phase_exempt，跳过（它没走高地形豁免门）"); continue
            if "sensor_cfg" not in term.params or term.params.get("sensor_cfg") is None:
                print(f"[s10-climb-C18] !! {name} 没配 sensor_cfg，高地形门本来就没生效，跳过"); continue
            term.params["phase_exempt"] = True
            n += 1
        assert n >= 1, "C18: 一项都没改到，配置没生效，别白训"
        print(f"[s10-climb-C18] 姿态豁免门 地形高 → 正在翻越，已改 {n}/4 项｜"
              f"治登顶后稳态低头 26°、hr 膝顶限位 25%（C13 曾是 −2.2°/0%）")


@configclass
class DeeproboticsS10ClimbExpertC19EnvCfg(DeeproboticsS10ClimbExpertC18EnvCfg):
    """Climb-C19（09-15 07:3x，从 **C17_12200** 续 600 步。C18 判无效，不做起点）。

    **C18 判决：门关上了，但关错了地方 —— 600 迭代台面姿态一步没动。**
      C18_12799 台面稳态 pitch **+24.3/+24.5/+24.4°**、膝顶限位 **25%**、hr 膝 **−2.69**
      —— 与 C17_12200（+26.2°、25%、−2.69）**实质相同**。
      训练侧姿态罚确实全线涨 2~3 倍（flat_orientation −0.41→−0.86、hipx −0.47→−1.21、
      hipy −0.54→−1.23、膝 −0.29→−0.86），而且**还在继续涨**——策略宁可挨罚也不改姿态。

    **根因：自我维持回路（第四次"量本身可被伪造"）。**
      C18 的掩码是 `phase | encounter`，而 `phase = split | wall_ahead`，
      `split = 四轮触地高差 > 0.12 m`（CLIMB_SPLIT）。
      轴距 0.4554 m → **机身 pitch 24.4° 本身就产生 0.4554×sin(24.4°) = 0.188 m 的高差**，
      已越过 0.12 阈值；折起的右后腿再加一截。
      → **坏姿态自己满足"正在翻越"，于是把自己豁免掉。** 罚值涨的那部分来自接近段和其它地形，
      **台面段依旧免罚**，所以那里的姿态零压力。

    C19 只改掩码本身（`climb_exempt_mask` v2）：触发**只看前方地形**
    （`wall_ahead | tall_wall`，来自高程扫描），触发后维持 `EXEMPT_TAIL_S = 2.0 s`
    （翻越 起抬→落平 实测 0.82~0.95 s，留足余量）。**落平后墙不在前方，计时到点豁免自动关，
    机器人摆成什么姿势都续不上。** 配置层不动（C18 的 4/4 项注入保留）。

    判据与 C18 同：① 台面稳态 pitch |p| ≤10°　② 膝顶限位 ≤5%　③ 登顶 3/3、翻越内 |roll| ≤12°、用时 0.9~1.2 s
    触发条件：
      · ①② 达标 → 收，"落平过冲"应同步回到 ≤+12。
      · ③ 变差（登顶掉条/翻越超时） → `EXEMPT_TAIL_S` 2.0 → 2.5，**不要回头加 split**。
      · ①② 仍不动，且已确认台面段豁免确实关了 → 才是"姿态罚权重不够"，那时加码；
        **在确认豁免关闭之前不准调权重**（C18 就是没确认就往下走）。
    **验证豁免真关了的办法**：台面稳态段 pitch 24° 时 `wall_ahead/tall_wall` 应为 False，
    掩码在落平 2 s 后必然为 False —— 这条由掩码定义保证，不依赖训练结果。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.EXEMPT_TAIL_S[0] = 2.0
        print(f"[s10-climb-C19] 豁免掩码改成纯地形触发 + 限时 {cr.EXEMPT_TAIL_S[0]} s｜"
              f"C18 失效根因：pitch 24° 自带 0.188 m 四轮高差 > CLIMB_SPLIT 0.12，坏姿态把自己豁免了")


@configclass
class DeeproboticsS10PerceptV1HTrotT11EnvCfg(DeeproboticsS10PerceptV1HTrotT10EnvCfg):
    """Trot-T11（09-15 07:5x，从 **T10_15197** 续 600 步）。**单变量：摆动时长奖励换 tent 核。**

    T10 成果（三点全达标，今晚踏步线最干净一轮）：10°/15° 坡偏航 +7.0/−1.2°（起点 +26.5/+14.8 不达标）、
    平地 |roll| p95 8.1 → **6.2**、反相 −0.84 → **−0.91**、速度跟踪 0.54/0.99/1.92（很准）。

    **T11 治的是步频。先订正一个记录错误**：计划里写的"我方约 1.6 Hz"是**我自己定的设计目标，
    从来没量过**。实测（`ours_flat.py`，同腿再抬间隔）：

    | 指令 vx | T9_14598 | T10_15197 | 官方 |
    |---|---|---|---|
    | 0.5 | 2.63 Hz | **3.23 Hz** | **1.10 Hz** |
    | 1.0 | 3.39 Hz | 3.39 Hz | |
    | 1.8 | 3.45 Hz | **3.77 Hz** | |

    摆动只有 **0.12~0.13 s**。差官方 **3 倍**，差我自己定的目标也有 2 倍。
    作者 09-14 现场原话是「**踏得太快不稳**」——在 1.6 Hz 的错误认知下这像是对齐外观的可选项，
    在 3.4 Hz 的真实值下，**这是作者亲口报的第三个缺陷**，与 T9/T10 已治的两个同级。

    **根因（数值可对上）**：`sg_swing`（摆动时长奖励，T_fixed=0.35、σ=0.08、权重 3.0）**是死的**。
    误差 0.22 处 `exp(−(0.22/0.08)²) = 5.2e−4`，×3.0 = **0.0016**；训练日志读数 **+0.0015**，吻合。
    于是这条线只有 `sg_rate`（−5.0 / f_max 1.3，读数 **−3.27**，一直在狠罚）这个**罚**，
    **没有任何可用的正向梯度**告诉它"每步该摆 0.35 s"。
    而这条线早有定论：**光靠罚建立不起新行为**（C13 用 400 迭代证过，C14 换正向塑形才成）。

    T11 只把核换成 tent：`clamp(1 − |dur−T|/0.35, 0)` → dur=0.13 处从 5e−4 变成 **0.37（715 倍）**，
    **处处有梯度**。**目标 T 一个字没动**，`sg_rate` 一个字没动 —— 改的是"够不够得着"。

    判据（每 200 点）：
      ① 摆动 0.13 → **≥0.22 s**，步频 3.4 → **≤2.2 Hz**（第一档；官方 1.10 是终点不是本轮目标）
      ② T10 三条成果不劣：坡偏航 ≤±10°、平地 |roll| p95 ≤7.7、反相 ≤−0.70
      ③ 速度跟踪不劣（0.5/1.0/1.8 → 当前 0.54/0.99/1.92）
    触发条件：
      · ①②③ 都达标 → 下一轮把 f_max 从 1.3 收到 1.15，走向官方 1.10。
      · ① 动了但 ③ 变差（速度掉） → 说明低步频撑不住高速；**按速度分档设目标**（T(v) 本来就有分档公式，
        改回不用 T_fixed），而不是放弃压步频。
      · ① 没动 → tent 也拉不动，说明有别的项在奖励高频（首先查 `sg_clear`：它是**按落地次数**给的，
        步频越高事件越多、总分越高 —— 这是结构性高频激励，那时改它为按时间归一）。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.sg_swing.params["kernel"] = "tent"
        R.sg_swing.params["width"] = 0.35
        print("[s10-v1h-trotT11] 摆动时长奖励 高斯→tent（0.13 s 处 5e−4 → 0.37，715 倍，处处有梯度）｜"
              "目标 0.35 s 与 sg_rate 均未动｜治实测步频 3.4 Hz（官方 1.10，此前记录的 1.6 是没量过的设计目标）")


@configclass
class DeeproboticsS10ClimbExpertC20EnvCfg(DeeproboticsS10ClimbExpertC19EnvCfg):
    """Climb-C20（09-15 08:2x，从 **C19_12400** 续 600 步）。**单变量：豁免尾巴 2.0 s → 1.2 s。**

    **C19 判决：判据①② 三点全达标，掩码改对了。**

    | | 台面稳态 pitch | 膝顶限位 | 登顶 | 转+0.5 wz / 四轮比值 | 落平过冲 |
    |---|---|---|---|---|---|
    | C17_12200（起点） | +26.2° | 25% | 3/3 | +0.196 / 0.06 | +25.0 |
    | **C19_12400** | **+1.7°** ✓ | **0%** ✓ | 3/3 | +0.206 / **0.37 四轮成对** | +29.8 |
    | C19_12600 | +3.6° ✓ | 0% ✓ | 3/3 | +0.314 / 0.12 | +30.7 |
    | C19_12799 | −1.2° ✓ | 0% ✓ | 2/3 | +0.493 / 0.05 | +31.5 |

    **同时推翻了我此前一个判断**：我说过"落平过冲不是过冲、就是稳态姿态"——那是因为 C15 那次
    两者恰好相等（+27.8 vs +26）。现在稳态到了 +1.7 而过冲仍 **+30**，说明
    **存在一个真正的瞬态过冲，之前被稳态掩盖了**。

    **逐帧查清了瞬态的来源**（C19_12400 top33_1）：起抬 17.50 s（pitch −46.2）→
    17.95 s 起进入 +16~+29° 低头、hr 膝再次顶到 **−2.69**，**保持约 1.5 s** →
    19.2 s 开始回正 → 稳态 +1.7°。
    **回正时刻正好是起抬后约 1.7 s，即 `EXEMPT_TAIL_S = 2.0 s` 到期的时刻。**
    → 豁免尾巴太长，在落平之后又白送了约 1.5 s 的免罚窗口，坏姿态就在那个窗口里。

    C20 把尾巴收到 **1.2 s**（翻越 起抬→落平 实测 1.04~1.08 s，1.2 覆盖翻越本身但不再留长尾）。
    触发量仍是纯地形（`wall_ahead | tall_wall`），一个字没动。

    判据：① 落平过冲 **≤+20°**（当前 +29.8；官方 ≤+12 是终点不是本轮目标）
      ② 台面稳态 pitch ≤10°、膝顶限位 ≤5%（C19 已达标，**不能丢**）
      ③ 登顶 3/3、翻越用时 0.9~1.2 s
    触发条件：
      · ①②③ 都达标 → 尾巴再收到 1.0 s，走向官方 ≤+12。
      · ③ 变差（登顶掉条/翻越超时） → 尾巴收过头了，回 1.5 s，**不要回头加 split**。
      · ① 没动但 ②③ 保住 → 瞬态不是豁免窗口造成的，改查翻越末段的落地缓冲（vz 峰 0.66~0.69 偏高）。

    **另记（C20 不动，下一轮候选）**：翻越相位内 |roll| C19 三点是 9.7~15.5°（判据 ≤12 没达标），
    比 C17_12200 的 10.9 差；后轮越沿净空 0.076 → 0.036~0.086，12400 那点偏低。两项都要盯。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        cr.EXEMPT_TAIL_S[0] = 1.2
        print(f"[s10-climb-C20] 豁免尾巴 2.0 → {cr.EXEMPT_TAIL_S[0]} s｜治落平后 1.5 s 的低头 +29° 瞬态"
              f"（回正时刻恰好=2.0 s 尾巴到期）｜触发量仍是纯地形")


@configclass
class DeeproboticsS10PerceptV1HTrotT12EnvCfg(DeeproboticsS10PerceptV1HTrotT11EnvCfg):
    """Trot-T12（09-15 09:0x，从 **T10_15197** 续 600 步。T11 无效，不做起点）。
    **单变量：摆动时长奖励按时间归一（单次奖励 × dur），并把权重提到与步频罚同量级。**

    **T11 判决：tent 核生效了（`sg_swing` 0.0015 → 0.09，50 倍），但步频 3.23 Hz 一动不动。**
    根因是可以**精确推导**的，而且说明我的 tent"修复"比原来更糟：
      本项是**落地事件触发**，每秒总收益 = f × r(dur)。设占空比 duty = dur·f 固定，
      tent 在 dur<T 时 r = dur/width → **每秒收益 = (duty/dur)×(dur/0.35) = duty/0.35，
      与 dur、f 完全无关，梯度精确为 0**。
      实算（duty 0.42）：dur 0.13/0.20/0.26/0.30/0.35 → tent 每秒收益 **全是 1.200**；
      原高斯核是 0.002/0.062/0.456/0.947/1.200 —— **有梯度，只是起点量级太小**。
      → 我当初"高斯是死的"只对了一半：单次值 5e−4 可忽略，但每秒总量的形状是对的。

    T12：`time_norm=True`（单次再乘 dur）→ 每秒收益 = duty × r(dur)，在 dur=T 取最大、与频率无关。
    实算（tent×dur）：0.156/0.240/0.312/0.360/**0.420**，单调。
    同时把权重 3.0 → **25.0**：归一后满值每秒只有 0.42，对比 `sg_rate` −3.22、`sg_lr_anti` +4.70，
    3.0 的权重量级根本翻不动盘（这不是"先调权重再看"，是**梯度方向修正后**必须配的量级）。

    **步频口径（重要，此前写错过）**：同一策略两边对照 —— 真机 T3_13400 **1.97 Hz** vs
    MuJoCo T3_13400 **2.73 Hz** → **MuJoCo 高读 1.39 倍**。
    故 MuJoCo 侧目标应为 1.10 × 1.39 ≈ **1.53 Hz**，不是 1.10。
    T10 的 MuJoCo 3.23 Hz ≈ 真机 2.3 Hz；作者嫌"踏得太快不稳"的 T3 真机是 1.97 —— **T10 比那版还快**。
    （频谱已确认 MuJoCo 3.23 Hz 是基频，非谐波假象。）

    判据（每 200 点）：
      ① MuJoCo 步频 ≤ **2.2 Hz**、摆动 ≥0.22 s（第一档；终点 1.53）
      ② T10 三条成果不劣：坡偏航 ≤±10°、平地 |roll| p95 ≤7.7、反相 ≤−0.70
      ③ 速度跟踪不劣（0.5/1.0/1.8 → T10 是 0.54/0.99/1.92）
    触发条件：
      · ①②③ 达标 → f_max 1.3 → 1.15，走向 1.53。
      · ① 动了但 ③ 变差 → 低步频撑不住高速，改用按速度分档的 T(v)（去掉 T_fixed），不放弃压步频。
      · ① 仍不动 → 查 `sg_lr_anti`(+4.70) 的 EMA 是否随频率变化（tau=1.0 的 EMA 在低频下幅度会衰减，
        若如此它就是在结构性奖励高频），而不是继续加 `sg_swing` 权重。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.sg_swing.params["kernel"] = "tent"
        R.sg_swing.params["width"] = 0.35
        R.sg_swing.params["time_norm"] = True
        R.sg_swing.weight = 25.0
        print("[s10-v1h-trotT12] 摆动时长奖励 按时间归一（×dur）+ 权重 3→25｜"
              "T11 的 tent 单用在数学上对步频梯度精确为 0（每秒收益恒 duty/0.35），已推导并实算验证")


@configclass
class DeeproboticsS10ClimbExpertC21EnvCfg(DeeproboticsS10ClimbExpertC20EnvCfg):
    """Climb-C21（09-15 09:2x，从 **C20_12800** 续 600 步）。**单变量：重新启用差速超量罚，带触地门控。**

    **C20 判决：主目标达成，但暴露/放大了一个更严重的缺陷。**

    达成（用原始轨迹算，**不用 flip_time**，它对这批数据不可信）：

    | | 抬头峰 | 落平过冲 | 登顶 | 净空 | 台面姿态 | 翻越后站立位移幅度 |
    |---|---|---|---|---|---|---|
    | C19_12400 | −46.2° | **+29.8°** | 3/3 | 0.036 | +1.7° | 0.25 m |
    | **C20_12800** | −43.4~−43.6° | **+16.8/+15.7/+20.3°** | 3/3 | 0.054 | −1.0~−1.4° | **0.03 m** |

    翻越后站立（作者 09-15 要求"翻越之后要站得稳／接着走也可以，反正不能摔倒"）：
    C20_12800 站立 6 s **位移幅度 0.03 m**、|roll|/|pitch| 峰 2.7°/3.6°，三条全"站住 ✅"。
    切入段对齐也是 C20 最好：专家实际只控 **0.30 m / 0.6 s**，偏航漂 C17 −11.5° → C19 −8.8° → **C20 −2.9°**。

    **缺陷：台面直行绕圈。** 指令 vx=0.4、**wz=0** 时实际 wz：
      C17_12200 −0.045 → C19_12400 −0.040 → C20_12600 −0.13 → **C20_12800 −0.245 rad/s（27 s 绕一圈）**
      → 末点 12999 更糟（平地直行 0.5 m/s 偏 **+61°**，零指令偏 +52/−58°）。
    **根因是左右轮持续不对称，仍是 hr 那条腿**（台面直行四轮转速）：
      C19_12400  fl −4.67 fr −4.24 hl −4.44 **hr −1.53** → 左 −4.56 右 −2.88 差 **−1.67**
      C20_12800  **fl −7.01** fr −2.82 hl −4.04 **hr −1.03** → 左 −5.52 右 −1.93 差 **−3.60**
    训练侧看不见：`cc_yaw_ref` C19 −1.22 / C20 −1.20 几乎不变 → **又一个只在 MuJoCo 出现的现象**
    （与步频的 sim2sim 差距同类）。

    C21 重新启用 `cc_wheel_ex`（C17 删过），**这次带 `contact_only=True`**：
    当年删它是因为它量的"左右均值之差"被一个悬空空转轮主导、罚的是假量；
    触地门控后这个量是真的。`tol` 用 **1.0**（比 C15 的 2.0 紧，因为现在治的是 wz=0 时的偏置，
    实测偏差 1.67~3.60 都要罚到）。权重 **−0.20**（C15 是 −0.15，C16 加倍到 −0.30 无效那次是假量，不作参考）。
    先例：C13 用同类惩罚把零指令无效差速从 +13.81 治到 −3.22。

    判据（每 200 点）：
      ① **台面直行 |wz| ≤ 0.06 rad/s**（当前 0.245；C19 是 0.040，回到那个水平即可）
      ② C20 的成果不能丢：落平过冲 ≤+20、台面姿态 ≤10°/顶限位 ≤5%、登顶 3/3、翻越后站住 3/3
      ③ 切入段（切入→起抬 0.30 m）偏航漂 ≤±5°（C20 是 −2.9°）
    触发条件：
      · ①②③ 达标 → 回头继续压落平过冲到官方 ≤+12。
      · ① 没动 → 差速不是唯一来源，查 hipx 左右不对称（C20_12800 台面 hipx fl −0.015 fr −0.184 hl −0.065 hr +0.047，
        fr 明显外展）带来的侧向推力偏置，而不是继续加罚。
      · ② 掉了（尤其登顶或站立） → 退回 C20_12800 交付，把绕圈问题交给"登顶后切回主策略"的流程规避。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_step as ss
        self.rewards.cc_wheel_ex = RewTerm(func=ss.wheel_diff_excess_penalty, weight=-0.20,
                                           params={"tol": 1.0, "contact_only": True})
        print("[s10-climb-C21] 重启差速超量罚 −0.20（tol 1.0，**触地门控**）｜"
              "治台面直行绕圈：C20_12800 指令 wz=0 实际 −0.245 rad/s，左右轮差 −3.60（hr 只有 −1.03）")


@configclass
class DeeproboticsS10ClimbExpertC22EnvCfg(DeeproboticsS10ClimbExpertC21EnvCfg):
    """Climb-C22（09-15 10:2x，从 **C20_12800** 续 600 步。C21 已被守卫停训，不做起点）。
    **单变量：差速超量罚在翻越窗口内关闭（`skip_climb=True`）。**

    **C21 判决：偏航治好了，但把爬台掐死了 —— 登顶 0/3，守卫连续两点异常自动停训。**

    | | 登顶 | ①台面直行 wz | ③切入段偏航漂 | 翻越后站住 |
    |---|---|---|---|---|
    | C20_12800（起点） | 3/3 | **−0.245**（27 s 绕一圈） | −2.9° | 3/3，位移 0.03 m |
    | C21_13000 | **0/3** | **+0.013/+0.009/−0.049** ✓ | −0.8~−2.4° ✓ | 3/3 |
    | C21_13200 | **0/3** | −0.101~−0.104 | −2.9~−3.4° ✓ | 3/3 |

    **判据①③ 都达标了 —— 差速罚确实能治绕圈**（wz −0.245 → ±0.01~0.05）。
    **但它不分相位**：翻越那 1 s 左右轮**本来就该不对称**（一侧蹬立面、一侧在地面），
    罚下去等于禁止翻越动作，登顶直接归零。

    C22 只加相位门：`skip_climb=True` → 用 `climb_exempt_mask`（**纯地形触发 + 限时 1.2 s**）
    在翻越窗口内关掉这一项。该掩码不含任何机器人姿态量，
    **不会被"轮速差大"这个被罚量自己满足**（纪律 20 的教训）。其余参数不动（tol 1.0、权重 −0.20）。

    判据：
      ① 台面直行 |wz| ≤0.06（C21 已证明能到 0.01~0.05）
      ② 登顶 **3/3**（C21 的 0/3 是本轮要修的）、翻越后站住 3/3
      ③ 落平过冲 ≤+20、台面姿态 ≤10°/顶限位 ≤5%、切入段偏航漂 ≤±5°
    触发条件：
      · 全达标 → 收，作为翻台阶交付版本。
      · 登顶仍 <3/3 → 掩码时长 1.2 s 盖不住需要非对称轮速的全部窗口，延到 1.8 s 再试。
      · 登顶回来但 ① 退回 >0.1 → 翻越窗口外的罚力度不够，把 tol 1.0 收到 0.6（**不加权重**，
        因为权重大会再次压爬台）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_wheel_ex.params["skip_climb"] = True
        print("[s10-climb-C22] 差速超量罚加翻越相位门（skip_climb）｜"
              "C21 治好了绕圈（wz −0.245→±0.01~0.05）但登顶 0/3：翻越时左右轮本就该不对称，不能罚")


@configclass
class DeeproboticsS10StairExpertS15EnvCfg(DeeproboticsS10StairExpertS14EnvCfg):
    """Stair-S15（09-15 11:2x，从 **S14_15000** 续 600 步）。**单变量：机身跟坡奖励换斜坡核。**

    S14 成果（决赛真实几何 踢面 15/11、踏面 55）：上楼 **6/6**、0.2 档通过 56.6 s、
    最差单条偏航 +20.8° → **+12.2°**、侧倾峰 9.5~12.7°、用时 23.4~27.3 s。

    **S15 治"机身不跟坡"**：爬楼段 pitch 均只有 **−5.0°**（抬头 5°），官方 **−15.4°**。

    **先纠正计划里的一个错判**：§422 写「`ss_pitch_slope` 权重 10 太弱，原始值只到 1.8%，下一轮单独加码」。
    **加码是错的方向**。查实：该项用 `pitch_hold_reward`，档位 **band [8°, 20°]**，是**硬判据**
    （档内 1、档外 0）；而实际抬头只有 5°，**在档外**。
    训练读数 0.147~0.162 ÷ 权重 10 = 原始值 **0.015（1.5% 的步在档内）**，对比 `track_lin_vel` +10.7 差 70 倍。
    **值恒为 0 时权重乘多少还是 0，而且档外没有任何梯度指向档内。**
    （与 T11 tent 核、T12 时间归一同一类：先确认当前工作点有没有梯度，再谈权重。）

    S15 把 `ss_pitch_slope` 换成 **ramp 核**：`lo 2° → tgt 15° 线性 0→1`，`15~22°` 保持 1，
    `22°` 起在 `fade 5°` 内退回 0（不鼓励过度抬头）。离线自检：
    抬头 5° → **0.231**（原 band 给 0）、8° → 0.462、15° → 1.000、25° → 0.400、27° → 0。
    **权重 10 不动**（预计读数从 0.15 升到 ~2.3，与 `ss_yaw_ref` −1.46 同量级）。
    爬台线的 `cc_pitch_hold` 用默认 `kernel="band"`，**行为逐位不变**。

    判据（每 200 点）：
      ① 爬楼段 pitch 均 −5.0° → **≤−10°**（官方 −15.4，本轮先到一半）
      ② 爬楼段 |roll| p95 **≤6.8**（S14 是 7.9，官方 6.8）
      ③ **上楼 6/6 不能丢**、最差单条偏航 ≤+12.2° 不劣、0.2 档必须通过
    触发条件：
      · ①②③ 达标 → 继续把 tgt 推向官方 15.4（本轮已是 15，则改推 ② 的侧倾）。
      · ① 动了但 ③ 掉条 → 跟坡与上楼冲突，`tgt` 从 15 收到 10 再训。
      · ① 没动 → 查 `phase_only`/`vz_min=0.02` 的门是否把大部分爬楼时间挡在外面
        （即这一项根本没在爬楼段生效），**而不是继续调核**。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.ss_pitch_slope.params.update({"kernel": "ramp", "lo_deg": 2.0, "tgt_deg": 15.0,
                                        "hi_deg": 22.0, "fade_deg": 5.0})
        print("[s10-stairN-S15] 机身跟坡换 ramp 核（lo2→tgt15→hi22，fade5）｜"
              "S14 实测抬头仅 5°、band[8,20] 在档外恒 0（原始值 0.015）→ 加权重无效，改核才有梯度")


@configclass
class DeeproboticsS10ClimbExpertC23EnvCfg(DeeproboticsS10ClimbExpertC20EnvCfg):
    """Climb-C23（09-15 11:4x，从 **C20_12800** 续 600 步）。**单变量：俯仰保持奖励加时间预算 0.45 s。**
    （不继承 C21/C22——差速罚那条线两轮都把登顶打到 0/3，已终止。）

    **依据：官方录包实测**（`~/s10_logs/ref/ref_climb34_50hz.npz`，wall_h 0.34，3 段，含 vx；
    以「pitch 首次 >10°」为 t=0 对齐两边）：

    | 相对起抬 t | 官方 vx 均/峰 | 我方 vx 均/峰 | 官方 pitch 均/峰 | 我方 pitch 均/峰 |
    |---|---|---|---|---|
    | −0.6~−0.3 | **+0.087 / +0.132** | **+0.307 / +0.405** | +3.3 / +3.5 | +0.7 / +1.3 |
    | −0.3~0 | +0.353 / +0.821 | +0.658 / +0.882 | +5.3 / +9.2 | +3.6 / +9.8 |
    | 0~+0.3 | +0.293 / +0.991 | +0.748 / +1.405 | +28.0 / +40.1 | +23.0 / +37.5 |
    | +0.3~+0.6 | +0.546 / +0.885 | +0.740 / +1.062 | +33.4 / +40.1 | +37.8 / +39.9 |
    | **+0.6~+1.0** | +1.063 / +1.362 | +1.129 / +1.341 | **+4.5 / +16.7** | **+29.3 / +43.9** |

    **两个差距**：① 接近快 3.5 倍（0.307 vs 0.087）→ **验收/部署参数**（V_CLIMB，已加 `V_NEAR/NEAR_D` 单独测）；
    ② **落平太慢**——起抬后 0.6~1.0 s 官方已落到 +4.5°，我方还挂在 **+29.3°**。
    爬升段本身两边几乎一致（pitch 峰都 40°、vx 都 1.1~1.3）→ **爬得没问题，问题是"赖在抬头姿态上不下来"**。

    **根因**：`cc_pitch_hold` 是**按步给分**（档内每步 +1），**赖得越久拿得越多**，
    与官方"|pitch|>30° ≤0.45 s"直接冲突（我方 0.66~0.81 s）。
    （与 T11/T12 的 per-event 问题同族：**先写出"每秒/每次遭遇总量"再判方向**。）

    C23：`budget_s=0.45` —— 每次遭遇只给**前 0.45 s** 的在档时间发分，超出不再发。
    **不加对抗性惩罚**（C21/C22 的教训：与核心动作冲突的惩罚会把登顶打死），只是**不再奖励拖延**。

    判据（每 200 点）：
      ① **|pitch|>30° 持续 ≤0.45 s**（当前 0.66~0.81）
      ② **落平过冲 ≤+12°**（当前 +16~20，官方 ≤12）
      ③ 登顶 3/3、翻越后站住 3/3、台面姿态 ≤10°/顶限位 ≤5%、切入段偏航漂 ≤±5° 都不能丢
    触发条件：
      · ①②③ 达标 → 收，作为翻台阶交付版本（配合"切前停 2 s、切入 ≤2.0 m"的操作条件）。
      · ① 达标但 ③ 掉登顶 → 预算太紧，0.45 → 0.60 再训。
      · ① 没动 → 预算没生效（查 `_pitch_budget` 是否每步只累加一次、遭遇结束是否清零），
        **而不是去加惩罚**。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_pitch_hold.params["budget_s"] = 0.45
        print("[s10-climb-C23] 俯仰保持奖励加时间预算 0.45 s（官方 |pitch|>30° ≤0.45，我方 0.66~0.81）｜"
              "按步给分=赖得越久拿得越多，改成只发前 0.45 s，不加对抗惩罚")


@configclass
class DeeproboticsS10StairExpertS16EnvCfg(DeeproboticsS10StairExpertS15EnvCfg):
    """Stair-S16（09-15 11:5x，从 **S14_15000** 续 600 步。S15 几乎白跑，不做起点）。
    **单变量：机身跟坡项去掉相位门（`phase_only=False`）—— 改门不改核。**

    **S15 判决：核换对了，但加在一个几乎从不触发的项上。**

    | | ① 爬楼段 pitch 均 | ② \\|roll\\| p95 | ③ 上楼 |
    |---|---|---|---|
    | S14_15000 基线 | −3.6° | **6.8°** | 6/6 |
    | S15_15400 | −4.6° | 8.0° | 6/6、0.2 档通过 |

    ① 只挪 1°（目标 −10），② 反而劣化。**根因是相位门**：

    | 常量 | 值 | 决赛楼梯（踢面 11，两级 15） |
    |---|---|---|
    | `TALL_WALL_H` | **0.18** | 够不着 |
    | `WALL_AHEAD_H` | 0.15 | 0.11 够不着、0.15 卡线 |
    | `CLIMB_SPLIT` | **0.12** | 0.11 够不着 |

    `encounter = tall_wall | (split & mem>0)`，`mem` 只在 `tall_wall` 时累积
    → **11 cm 踢面上这个门几乎永远关闭**。训练读数从开训 0.31 掉到末期 **0.038**（原始值 0.0038），印证门是关的。
    **这些常量是为 33 cm 单级高台设的，楼梯线一直在借用** —— 结构性教训（纪律 31）。

    S16：`phase_only=False`。该项已有 `vz_min=0.02`（机身上升才给分），
    **爬楼全程满足、平地几乎不满足**，本身就是对的门；ramp 核（lo2→tgt15→hi22, fade5）保持不变。
    起点回 **S14_15000**（S15 的 roll 更差，不做起点）。

    判据（每 200 点）：
      ① 爬楼段 pitch 均 −3.6° → **≤−10°**（官方 −15.4）
      ② 爬楼段 |roll| p95 **≤6.8**（S14 基线正好 6.8，不能再劣）
      ③ 上楼 **6/6** 不丢、最差单条偏航 ≤+12.2°、**0.2 档必过**
    触发条件：
      · ①②③ 达标 → 收；若 ① 仍差官方较远，再把 tgt 推向 15.4。
      · ① 动了但 ③ 掉条 → 跟坡与上楼冲突，`tgt` 从 15 收到 10 再训。
      · ① 仍不动（读数仍 <0.1） → 门还是没开，查 `vz_min=0.02` 在爬楼段是否成立
        （楼梯是一级一级上，vz 可能在踏面段归零），**改用"高度较起点已增加"做门**。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ss_pitch_slope.params["phase_only"] = False
        print("[s10-stairN-S16] 机身跟坡去掉相位门（phase_only=False）｜"
              "S15 白跑根因：TALL_WALL_H 0.18 / CLIMB_SPLIT 0.12 都是为 33cm 高台设的，"
              "11cm 踢面够不着 → encounter 几乎恒 False，读数末期只剩 0.038")


@configclass
class DeeproboticsS10ClimbExpertC24EnvCfg(DeeproboticsS10ClimbExpertC23EnvCfg):
    """Climb-C24（09-15 13:4x，从 **C23_13600** 续 600 步）。**单变量：俯仰时间预算 0.45 → 0.35 s。**
    **有界一轮**：只冲"落平过冲 ≤+12"这最后一项；不达标就收在 C23_13600。

    **C23 收官（原始轨迹复核，三条逐条）**：

    | | 抬头峰 | 落平过冲 | \\|pitch\\|>30° | ①台面直行 wz | ③切入偏航 | 站立位移 |
    |---|---|---|---|---|---|---|
    | C20_12800 | −43.5 | +16.8/15.7/20.3 | 0.67 | **−0.245** ✗ | −2.9° | 0.03 |
    | **C23_13600** | **−41.4** | +13.6/15.1/20.5 | **0.36** ✓ | +0.009~0.049 ✓ | −0.4~−2.0 ✓ | 0.20~0.28 |

    **意外收获：台面绕圈被彻底治好**（C23 全部检查点 |wz| ≤0.05）。
    C21/C22 两轮**专门**治它（差速超量罚 ± 相位门）都把登顶打到 0/3 没做成；
    C23 治"别赖在抬头姿态上"**顺手**做成了。机理：不再长时间维持抬头 → 更快回到对称站姿 → 左右轮不对称消失。
    → **正面塑形"该做什么"，比惩罚"不该做什么"更容易带出连带改善**（纪律 36）。

    **仅剩差距**：落平过冲 +13.6~20.5，官方 ≤+12。
    C24 把预算 0.45 → **0.35**（官方 |pitch|>30° ≤0.45，我方已到 0.36，再收一档看过冲跟不跟着降）。

    判据：
      ① 落平过冲 **≤+12°**（三条都要）
      ② |pitch|>30° 仍 ≤0.45 s
      ③ 登顶 3/3、站住 3/3、台面直行 |wz| ≤0.06、切入段偏航 ≤±5°、台面姿态 ≤10°/顶限位 ≤5° 一个都不能丢
    触发条件：
      · ①②③ 全达标 → 收，作为最终交付。
      · ① 没到 +12 但 ③ 保住 → **停止投入，交付 C23_13600**（已归档 md5 b4056d7c）。
      · ③ 掉任何一项 → 预算收过头，立即回 0.45 并交付 C23_13600。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_pitch_hold.params["budget_s"] = 0.35
        print("[s10-climb-C24] 俯仰时间预算 0.45 → 0.35 s（有界一轮，只冲落平过冲 ≤+12；不成就收在 C23_13600）")


@configclass
class DeeproboticsS10PerceptV1HTrotT13EnvCfg(DeeproboticsS10PerceptV1HTrotT12EnvCfg):
    """Trot-T13（09-15 14:2x，从 **T10_15197** 续 600 步）。**单变量：摆动时长奖励权重 25 → 600。**
    **有界一轮**：不成就收在 T10_15197（md5 ebef4dd2，已归档）。

    **为什么现在加权重是合法的**（与 T12 那次"梯度方向没修好就加权重"性质不同）：
    T12 已经把**梯度方向**修对了（`time_norm=True`：每秒总量 = duty × r(dur)，在 dur=T 取最大、与频率无关，
    **不会被"多踩几步"刷分**），实测三个检查点步频仍 3.17~3.70 Hz 没动。
    **量级核算**：`sg_swing` 读数 **0.07**，而 `sg_rate` **−3.2**、`sg_lr_anti` **+4.7**、
    `track_lin_vel` **+11.0** —— 差 **45~160 倍**。**剩下唯一的解释就是量级不够。**
    权重 600 时：当前 dur 0.13 处读数约 **1.7**，满值（dur 0.35）约 **7.6** —— 与 `sg_rate` 同量级。
    **本项有上界**（每秒总量 ≤ duty × 1.0），加权重不会发散。

    判据（每 200 点）：
      ① **MuJoCo 步频 ≤2.2 Hz、摆动 ≥0.22 s**（T10 是 3.23 Hz / 0.13 s；
         MuJoCo 高读真机 1.39 倍，故 MuJoCo 目标 1.53 对应官方 1.10）
      ② **T10 三条成果一条不许掉**：10°/15° 坡偏航 ≤±10°、平地 |roll| p95 ≤7.7°、反相 ≤−0.70
      ③ 速度跟踪不劣（T10：0.5/1.0/1.8 → 0.54/0.99/1.92）
    触发条件：
      · ①②③ 全达标 → 收，换踏步交付版本。
      · ① 动了但 ②③ 掉 → 步频与稳定/速度是真实取舍，**按纪律 34 保稳定，收在 T10_15197**。
      · ① 仍不动（权重 600 都拉不动）→ **这一项不是杠杆**，停止投入；
        步频只能靠相位钟进观测解决（作者已拍板不改接口）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.sg_swing.weight = 600.0
        print("[s10-v1h-trotT13] 摆动时长奖励权重 25 → 600（有界一轮）｜"
              "T12 已修好梯度方向（时间归一、与频率无关、有上界），实测读数 0.07 vs sg_rate −3.2 差 45 倍 → 只剩量级问题")


@configclass
class DeeproboticsS10ClimbExpertC25EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C25（09-15 14:4x，从 **C24_13800** 续 600 步）。**单变量：开 `joint_vel_l2` 惩罚 −0.05。**

    **起因：作者看 MuJoCo 回放后指出「现实里肯定关节故障了」——查证属实。**

    翻越窗口实测 vs 限位 vs 官方录包 `ref_climb34`（真机 3 段）：

    | 关节 | 我方 C24_13800 | 官方 | 限位 | |
    |---|---|---|---|---|
    | hipx \\|q\\| | **0.660** | 0.430~0.592 | 0.6109 | **越限**（占比 11.9%） |
    | hipy \\|q\\| | **2.562** | 2.212~2.283 | 2.5307 | **越限** |
    | hipy \\|dq\\| | **26.21** | 14.55~15.26 | 25.76 | **越限，1.7 倍官方** |
    | **knee \\|dq\\|** | **36.24** | 17.89~22.65 | 25.76 | **超额定 1.4 倍** |

    **越限自 C17 起一直存在**（C17 35.80 / C19 39.30 / C20 35.62 / C23 36.66 / C24 36.86），整条线都有。

    **根因**：`joint_vel_l2` / `joint_vel_limits` / `joint_pos_limits` **权重全是 0，从来没改过**；
    MJCF **只有力矩 ctrlrange、无速度限**；唯一有速度限的是 Isaac 执行器（求解器会夹）
    → **Isaac 夹住 → 惩罚读数恒 0 → 训练以为没事；MuJoCo 不夹 → 36 rad/s；真机额定 25.76。**

    **统一结论**：三条线关节角速度分别是官方的 2~3（踏步）/ 6~8（楼梯）/ 1.6~1.7（爬台）倍 ——
    **同一个病：高频甩关节**。它同时解释了步频 3.3 vs 1.1、摆动 0.13 vs 0.35、关节越限。
    T11/T12/T13 三轮打的是症状（落地事件的摆动时长奖励），已停。

    **权重定法（先验读数）**：实测 12 个腿关节 dq² 之和，翻台阶均值 **32.2**（p95 191.7、峰 2004.9）
    → 取 **−0.05**，预计读数约 **−1.6**，与 `cc_pitch_hold`(1.3)、`cc_wheel_dir`(2.5) 同量级。

    判据（每 200 点）：
      ① **knee/hipy \\|dq\\| ≤25.76**（硬限，必须）；进一步目标向官方 17.9~22.65 靠
      ② **hipx \\|q\\| ≤0.6109**（当前 0.660，占比 11.9%）
      ③ C24 的成果不能丢：登顶 3/3、站住 3/3、落平过冲 ≤+13、\\|pitch\\|>30° ≤0.45、
         台面直行 \\|wz\\| ≤0.06、切入偏航 ≤±5°
    触发条件：
      · ①②③ 达标 → 收，作为真正可上真机的爬台交付。
      · ① 达标但 ③ 掉登顶 → 惩罚过强，−0.05 → −0.02 再训（**不要回到 0**）。
      · ① 没动（读数 <0.5）→ 说明 Isaac 侧速度本就被执行器夹住、该项拿不到梯度；
        那时**改在 MuJoCo 侧加速度限让验收诚实**，而不是继续加权重。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_vel_l2.weight = -0.05
        print("[s10-climb-C25] 开 joint_vel_l2 −0.05（先验读数：腿关节 dq²和 均 32.2 → 预计读数 −1.6）｜"
              "治 knee 角速度 36.2 超额定 25.76（官方 17.9~22.65）")


@configclass
class DeeproboticsS10StairExpertS17EnvCfg(DeeproboticsS10StairExpertS16EnvCfg):
    """Stair-S17（09-15 14:5x，从 **S16_15200** 续 600 步）。**单变量：开 `joint_vel_l2` 惩罚 −0.03。**

    **依据：楼梯线是三条里关节角速度偏离官方最大的**（对 `ref_stairwalk_official`，布局"按类分块"、
    `qd` 已数值求导验为 rad/s）：

    | | hipx q/dq | hipy q/dq | knee q/**dq** |
    |---|---|---|---|
    | 官方 stair_walk | 0.085 / **0.84** | 0.815 / **2.71** | 1.787 / **2.37** |
    | 我方 S16_15200 | **0.634 ✗** / **13.50** | 1.584 / **17.91** | **2.737 ✗** / **19.03** |

    **角速度是官方的 6~8 倍。** 且有个矛盾：**我们每级用时 1.45 s 比官方 0.86 s 还慢，关节却快 8 倍**
    → **大量关节运动是无用的快速抖动，不是在迈步。**

    **权重定法（先验读数）**：实测 12 个腿关节 dq² 之和，楼梯全程均值 **57.6**（p95 304.1、峰 660.4）
    → 取 **−0.03**，预计读数约 **−1.7**，与 `ss_pitch_slope`(0.8)、`ss_yaw_ref`(−1.3) 同量级。

    判据（每 200 点）：
      ① **腿关节 |dq| 峰减半**（19.0 → ≤10；官方 2.4 是终点，本轮不苛求一步到位）
      ② **hipx |q| ≤0.6109**（当前 0.634，占比 8.8%）
      ③ **S16 成果不能丢**：上楼 6/6、0.2 档通过、爬楼段 |roll| p95 ≤6.8
    触发条件：
      · ①②③ 达标 → 收；若还想更靠近官方，下一轮 −0.03 → −0.06。
      · ① 达标但 ③ 掉条 → 惩罚过强，−0.03 → −0.015（**不回到 0**）。
      · ① 没动 → 与爬台线同因：Isaac 侧速度被执行器夹住、该项无梯度；
        改在 **MuJoCo 侧加速度限让验收诚实**，而不是继续加权重。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_vel_l2.weight = -0.03
        print("[s10-stairN-S17] 开 joint_vel_l2 −0.03（先验读数：腿关节 dq²和 均 57.6 → 预计读数 −1.7）｜"
              "治关节角速度 13.5/17.9/19.0 vs 官方 0.84/2.71/2.37（6~8 倍）")


@configclass
class DeeproboticsS10ClimbExpertC26EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C26（09-15 15:3x，从 **C24_13800** 续 600 步。C25 判失败，不做起点）。
    **改用模仿：复活官方真机轨迹跟踪 `st_ref_track`，同时关掉 `joint_vel_l2`。**

    **起因**：作者看 MuJoCo 回放判「这个翻台阶一定不对，现实里肯定关节故障了」。已量化证实：
    翻越段 knee \\|dq\\| 36.24（额定 25.76、官方 22.65）、hipx/hipy 角越限；
    **且逐关节与真机官方轨迹 RMS 差 0.876 rad（约 50°）**。

    **C25（`joint_vel_l2` −0.05）判失败**：knee \\|dq\\| 36.24 → 32.75 → **33.02（反弹）**，
    落平过冲却 +12.8 → 15.3 → 15.8 → **18.3 单调劣化**，\\|pitch\\|>30° 0.40 → 0.52。
    **机理**：Isaac 执行器把速度夹在 25.76，训练里该项看到的是**已被夹过的值**，
    加权重只能压低"Isaac 里本来就不高的数"，代价是把落平动作弄慢弄糟。

    **发现：模仿机制早就齐全，只是奖励是死的。**
      · `climb_ref.py` 有 **RSI 参考态出生**（`spawn_at_wall` 的 `ref_prob=0.4`，40% 从真机帧直接落在上墙过程中）
      · `RefTrack` 相位跟踪奖励（weight 5，σ_q=σ_p=**0.35**）
      · 压墙事件门 `离墙<0.35m & 前轮前向轮速>5` —— **实测触发率 100%**（轮速 12.8~15.7），**门没问题**
      · **问题在核**：`exp(−mean(Δq²)/σ²)·exp(−Δpitch²/σ²)`，实测误差 Δq 0.876 / Δpitch 0.540
        → σ=0.35 时核值 **0.0041**；训练读数 0.0107/权重5 = **0.0021**，离线与在线对得上。
        **与 `sg_swing`、`ss_pitch_slope` 同一类：核在当前工作点为零、没有梯度。**

    **C26 改法**：
      ① σ_q / σ_p **0.35 → 0.80**（当前误差处核值 0.004 → **0.222**，有梯度且不饱和）
      ② `st_ref_track` 权重 **5 → 10**（预计读数 **2.2**，与 `cc_edge_clr` 1.6 同量级）
      ③ `joint_vel_l2` **−0.05 → 0**（C25 已证明是错杠杆）
    **为什么这条路对**：跟上官方轨迹时，**关节角、关节速度、俯仰剖面、落平时机是一起被约束的**，
    不必再逐项加惩罚互相打架。与今天反复验证的一条一致：
    **正面塑形（该做什么）比惩罚（不该做什么）更容易带出连带改善**（C23 治俯仰时长顺手治好台面绕圈）。

    判据（每 200 点）：
      ① `st_ref_track` 训练读数 **≥1.5**（当前 0.011）——说明真的在跟
      ② 12 腿关节 RMS 跟踪误差 **0.876 → ≤0.60**（`scratchpad/reftrack_err.py`，已并入扫描）
      ③ knee \\|dq\\| **36.24 → ≤25.76**（官方 22.65）；hipx \\|q\\| ≤0.6109
      ④ C24 成果不丢：登顶 3/3、站住 3/3、落平过冲 ≤+13、\\|pitch\\|>30° ≤0.45、台面 wz ≤0.06
    触发条件：
      · ①②③④ 达标 → 收，作为可上真机的爬台交付。
      · ① 达标但 ② 没动 → σ 0.8 太宽、"差不多就给分"，收到 0.6 再训。
      · ② 达标但 ④ 掉登顶 → 官方轨迹与我方初始条件不兼容，**提高 RSI 比例**（ref_prob 0.4 → 0.6），
        让策略更多从官方状态出发，而不是调核。
      · ① 没动（读数 <0.5）→ 相位钟没启动，查压墙事件触发率（本轮实测 100%，若变则先查装置）。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.st_ref_track.params.update({"sigma_q": 0.80, "sigma_p": 0.80})
        R.st_ref_track.weight = 10.0
        R.joint_vel_l2.weight = 0.0
        print("[s10-climb-C26] 参考跟踪 σ 0.35→0.80、权重 5→10；joint_vel_l2 关回 0｜"
              "实测跟官方轨迹差 0.876 rad，σ0.35 核值 0.004（死）→ σ0.80 给 0.222")


@configclass
class DeeproboticsS10ClimbExpertC27EnvCfg(DeeproboticsS10ClimbExpertC26EnvCfg):
    """Climb-C27（09-15 15:5x，从 **C24_13800** 续 600 步）。**单变量：`st_ref_track` 权重 10 → 30。**

    **C26 判失败**：跟踪 RMS 误差 0.879（起点）→ 0.846 → 0.877 → **0.865**，**无趋势、就是噪声带**；
    `st_ref_track` 读数 0.37 → 0.31 → 0.32；末点台面直行 wz **−0.07 首次跌出判据**。
    σ 0.35→0.80 把死核救活了（读数 34 倍），**但权重 10 仍拉不动动作**。

    **自纠一个错误推理**：C26 收官时我曾说"满值 5.2、实际 0.32 只有 6%，是 episode 稀释，加权重没用"。
    **错**：该项在压墙窗口外恒为 0，加权重只影响窗口内；而竞争项（`cc_edge_clr` 等）同样相位门控、
    被同样比例稀释。**两边稀释一致，关键是比值** —— 0.32 : 2.03，跟踪项弱 6 倍；
    权重 30 → 0.96 : 2.03，**有意义**。

    判据与 C26 同：① 读数 ≥1.5　② 跟踪 RMS ≤0.60　③ knee \\|dq\\| ≤25.76、hipx \\|q\\| ≤0.6109
    　④ 登顶 3/3、站住 3/3、过冲 ≤+13、\\|pitch\\|>30° ≤0.45、台面 wz ≤0.06
    触发条件：
      · ② 动了（≤0.75）→ 方向对，继续跑或再加到 60。
      · ② 仍无趋势（0.82~0.90 震荡）→ **不再加权重**。转而怀疑
        **官方轨迹从我方接近状态不可达**（官方是近停起步 0.087 m/s + 右后腿主蹬，
        我方 0.4 m/s 冲上去，初始条件不同）→ 那时**提高 RSI 比例 `ref_prob` 0.4 → 0.7**，
        让策略大部分从官方状态出发再跟踪，而不是从自己的状态硬凑。
      · ④ 掉登顶 → 权重过强压住了爬升，退回 20。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.st_ref_track.weight = 30.0
        print("[s10-climb-C27] 参考跟踪权重 10 → 30（σ 0.80 不动）｜"
              "C26 读数 0.32 vs cc_edge_clr 2.03，弱 6 倍 → 30 后约 0.96")


C28_JPL_W = -6.0   # 先验读数（512env×60it 探针）：权重 −1.0 → 读数 **−0.0672**；
                   # 主项 cc_edge_clr 读数 0.91~1.05 → 取主项 40% → −6.0（预计读数 −0.40）


@configclass
class DeeproboticsS10ClimbExpertC28EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C28（09-15 16:2x，从 **C24_13800** 续）。**单变量：开 `joint_pos_limits` 惩罚。**

    **C27 判失败 + 我自己的一个假说被测量推翻，两条一起记：**

    ① C27（`st_ref_track` 权重 10 → 30）：读数 0.32 → **1.21（4 倍）**，
       跟踪 RMS **0.835/0.855/0.884 —— 仍是噪声带，纹丝不动**。
       策略只是把该项的分多赚了点，动作一点没改。**权重路线判死，不再走。**

    ② **"超速是免费刹车换来的作弊"——错。** state.csv 直接记了 dq/tau（不用数值微分）：
       基线后膝急甩时，超额定帧里电机只在 36~40% 的帧驱动、力矩中位 **7.7~9.4 N·m**（可用 50），
       主要是**被机身甩的**。给 MuJoCo 加最严的速度-力矩降扭（额定即上限）后，
       驱动占比 → 0~20%、力矩中位 → 1.8~4.5 N·m（曲线确实咬住了），
       峰值 36.9/35.9/38.6 → 31.8/33.3/31.5（**只降 14%**），**登顶仍 3/3**。
       → **超速不承重，是机身猛甩过沿的副产品。电机曲线不是杠杆。**

    **那么真正会砸硬件的是关节位置越限**（MuJoCo 限位是软约束压得进去，真机是硬机械挡块）：
      · 接近段（**T10 踏步主策略**，16.2 s）：fl_hipx **26%**、fr_hipx **25%**、hr_hipx **25%**
      · 专家段（爬台，15.5 s）：fr_hipx 11%、hr_hipy 6%（**96% 是电机主动往外推**）
      · 峰值超出 +1.0°~+3.2°；hr_hipy 峰 2.576 比 runner 口径 ±2.443 超 **+7.6°**
      · s10_ws 源码里 RL 控制态**对关节指令不做任何限位钳**（只有起立/趴下/Idle 的样条 IK 用限位）

    `joint_pos_limits` 权重一直是 **0.0 —— 这个硬物理约束从来没有被定价过**。
    这不是"加惩罚做塑形"，是给一个真机上必然存在的约束补上代价。
    软限位系数 0.9 → hipx 阈值 0.5498 rad，实测常驻 0.61~0.67，excess 0.06~0.12 rad。

    判据：① 读数达到主项 cc_edge_clr(2.03) 的 1/3~1/2　② hipx 越限占比 25% → ≤5%
    　　　③ 登顶 3/3、站住 3/3、台面 wz ≤0.06、落平过冲 ≤+13
    触发条件：
      · ② 动了但 ③ 掉登顶 → 权重过强，减半。
      · ② 不动 → 权重太弱，翻倍；若两次翻倍仍不动 → 说明越限是被地面反力压进去的、
        策略改不了，那就**不是训练问题**，改到 runner 侧做指令钳（要通知 AGX 会话）。
      · ③ 全过且 ② ≤5% → 同一改法搬到 T10 踏步主策略（越限比爬台严重一倍）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_pos_limits.weight = C28_JPL_W
        print(f"[s10-climb-C28] 开 joint_pos_limits {C28_JPL_W}（软限位 0.9×，hipx 阈值 0.5498）｜"
              f"实测 MuJoCo 侧 hipx 越限 25%、hr_hipy 96% 电机主动外推")


@configclass
class DeeproboticsS10ClimbExpertC29EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C29（09-15 16:4x，从 **C24_13800** 续）。**单变量：RSI 参考态出生 `ref_prob` 0.4 → 0.7。**
    **有界一轮**：200 步即判，不成就收在 C24_13800（md5 1c5d079d）。

    **为什么只剩这一个杠杆**：C25/C26/C27/C28 连续四轮没有一轮改进 C24_13800 ——
      · C25 `joint_vel_l2` −0.05 → 速度只降 9% 就停、过冲劣化
      · C26 `st_ref_track` σ 0.35→0.80 + 权重 10 → 跟踪 RMS 无趋势
      · C27 权重 10→30 → 读数涨 4 倍（0.32→1.21）、**跟踪 RMS 纹丝不动**
      · C28 `joint_pos_limits` −6.0 → 惩罚足额支付（读数 −0.38）、越限不降、过冲 +12.8→+18.4 劣化
    七轮里只有两次成功（C23 俯仰时间预算、C24 收紧），**两次都是改奖励的定义**。
    加权重/加惩罚这一类已经四连败，**本轮改的是数据分布，与成功的那两次同类**。

    **口径订正（重要）**：`spawn_at_wall` 里 hooked/pushed/ref 是**顺序条件**、不是划分：
      P(hooked)=0.15，P(pushed)=0.15×0.85=0.128，P(ref)=ref_prob×0.7225。
      所以现行 ref_prob=0.4 的**实际参考态出生只有 28.9%**，不是 40%。
      改 0.7 → **50.6%**，正常出生仍留 21.7%（保住"从自己 0.4 m/s 接近状态爬"的能力，
      这是部署时的真实工况，不能训没了）。

    **为什么怀疑分布是瓶颈**：官方是**近停起步 0.087 m/s + 右后腿主蹬**，我方 0.4 m/s 冲上去，
    初始条件根本不同。让策略从自己的状态去硬凑官方轨迹，可能物理上就不可达 ——
    那样再大的跟踪权重都只是在两个不兼容的目标之间打架（C27 正是这个形态：分照拿、动作不动）。

    判据：① 跟踪 RMS ≤0.75（起点 0.879，四轮都卡在 0.84~0.88 噪声带）
    　　　② 登顶 3/3、后轮越沿净空 ≥0.03、落平过冲 ≤+13、台面 wz ≤0.06
    触发条件：
      · ① 动了且 ② 全过 → 方向对，继续跑满 600，并考虑再提到 0.85。
      · ① 动了但 ② 掉（尤其净空/过冲）→ 参考态喂多了、自己接近的能力被稀释，退回 0.55。
      · ① 仍在 0.84~0.88 → **模仿路线整体判死**，爬台线冻结在 C24_13800，
        GPU 不再投爬台，剩余时间全部转真机趟数（爬台真机趟数目前 = 0）。"""

    def __post_init__(self):
        super().__post_init__()
        self.events.spawn_at_wall.params["ref_prob"] = 0.7
        print("[s10-climb-C29] RSI ref_prob 0.4 → 0.7（实际参考态出生 28.9% → 50.6%，正常出生留 21.7%）｜"
              "有界一轮 200 步即判，不成收在 C24_13800")


@configclass
class DeeproboticsS10ClimbExpertC30EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C30（09-15 17:1x，从 **C24_13800** 续）。
    **单变量：`cc_pitch_hold` 的闸门 `encounter` → `after_press`。** 有界一轮 200 步即判。

    **这一轮是作者目视定靶 + 我逐帧对照量出来的，不是又一次指标盲猜。**

    作者看 MuJoCo 回放后的判断：C29「和官方差太多」；C3_9596「降身举轮蹭墙，和官方不一样」。
    我把官方 seg0 与 C29_13999 按同一 t 轴（压墙事件对齐）逐帧对照，pitch 曲线是**反相**的：

    | t(s) | 官方 | C29 |
    |------|------|-----|
    | −0.02（压墙） | **−3.1°** | **−32.2°** |
    | +0.14 | −3.5° | **−39.3°（我方峰值）** |
    | +0.78 | **−40.8°（官方峰值）** | +16.8° |

    **峰值幅度几乎一样（−39.3 vs −40.8），但我方早 0.64 s。** 扫时间平移：
    最优 **−0.68~−0.70 s**，RMS 0.781→0.626（降 20%）—— 相位是真的，但只占 20%，另 80% 是形状。

    **根因（我自己代码里的机理）**：`climb_rewards.py:119` `encounter = tall_wall | (split & mem>0)`，
    `tall_wall` 是**纯地形量，前方看见高墙就为真、不需要任何接触**；而 `pitch_hold_reward` 的返回是
    `r * s["encounter"]`。→ **抬头奖励在接近途中就开始付钱**，策略被教成"还没到墙就仰起来"。
    我在同文件第 64 行写过"在墙前反复仰起就能长期满足"，当时只当刷分漏洞补了停滞退出，
    **没意识到它同时在塑造动作时机**。

    改法：闸门改成**前轮真的顶在立面上**（`pressed` = 接触且水平分力 > 0.5×垂直分力）之后才开，
    并**锁存**到本次遭遇结束（官方抬头峰值在压墙后 +0.78 s，纯 pressed 会太早关）。

    判据（**新的是形态量，不是 RMS —— RMS ≤0.60 这个判据已被作者证伪作废**）：
      ① **压墙瞬间 |pitch| ≤ 10°**（官方 3.1°，C29 是 32.2°）← 本轮主判据
      ② 抬头峰值时刻落在压墙后 +0.4~+1.0 s（官方 +0.78）
      ③ 登顶 3/3、站住 3/3、后轮越沿净空 ≥0.03、台面 wz ≤0.06、落平过冲 ≤+13
    触发条件：
      · ① 动了且 ③ 全过 → 方向对，跑满 600。
      · ① 动了但 ③ 掉登顶 → 抬头开太晚来不及，改成"压墙前 0.15 s 起 ramp 开闸"。
      · ① 不动（仍 >25°）→ 说明抬头不是这个奖励驱动的，回头查 `wheel_height_progress` 等项。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_pitch_hold.params["gate"] = "after_press"
        print("[s10-climb-C30] cc_pitch_hold 闸门 encounter → after_press（前轮压上立面才开闸，锁存至遭遇结束）｜"
              "靶：压墙瞬间 |pitch| 32.2° → ≤10°（官方 3.1°）")


C31_ERP_W = -100.0  # 先验读数（512env×60it）：权重 −1.0 → **−0.0039**；竞争项 cc_pitch_hold 0.5875、
                    # cc_edge_clr 1.3263。取 −100 → 预计 −0.39（约 pitch_hold 的 2/3）。不取等量：
                    # 本项只在远处生效、是集中的，等量会强到让策略根本不敢抬头。
                    # 注：探针第一次抓到反号（函数返负 × 权重负 = 正奖励 = 奖励提前抬头），已改成返正量。


@configclass
class DeeproboticsS10ClimbExpertC31EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C31（09-15 17:2x，从 **C24_13800** 续）。**单变量：新增 `cc_early_rear` 惩罚。**
    有界一轮 200 步即判。

    **C30 是零结果，不是证伪**：把 `cc_pitch_hold` 闸门关到 `after_press` 之后，
    压墙瞬间 |pitch| 30.2~30.4° → **33.5/32.6/32.6°（没降）**、峰值时刻 +0.26 → +0.22~0.24s（没移）。
    但"撤走激励"≠"施加反向压力"——从已收敛策略上拿掉一个它本来就满足的奖励，梯度极弱，
    200 步不足以让它改掉习惯。所以**不写死"抬头与 pitch_hold 无关"**。
    C30 唯一的正面读数：后轮越沿净空 **0.048 m**（历轮最好）；负面：落平过冲 +18.7（劣于 C24 +12.8）。

    **C31 直接罚量出来的那个缺陷**（前七轮 C25~C30 罚的都不是实际缺陷，或只是加聚合项权重、撤激励）：
    离墙 > 0.30 m 时抬头超 10° 就罚，近处（官方峰值所在的 0.25 m）**完全不罚**。

    判据（形态量）：
      ① **离墙 0.60 m 处 |pitch| ≤ 10°**（官方 3.3°，C24 27.8°）← 主判据
      ② 压墙瞬间 |pitch| ≤ 15°（官方 3.1°，C24 30.4°）
      ③ 登顶 3/3、站住 3/3、后轮越沿净空 ≥0.03、台面 wz ≤0.06
    触发条件：
      · ① 动了且 ③ 全过 → 方向对，跑满 600，再看②和落平过冲。
      · ① 动了但 ③ 掉登顶 → 平着走到墙根后起不来，把 far_m 0.30 → 0.45（给更多起抬空间）。
      · ① 不动 → 权重不够，翻倍；两次翻倍仍不动 → **爬台线冻结在 C24_13800，
        GPU 转真机验收**（爬台真机趟数至今 = 0，距决赛 7 天）。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        self.rewards.cc_early_rear = RewTerm(func=_cr.early_rear_penalty, weight=C31_ERP_W,
                                             params={"far_m": 0.30, "allow_deg": 10.0})
        print(f"[s10-climb-C31] 新增 cc_early_rear {C31_ERP_W}（离墙>0.30m 抬头超 10° 才罚，近处不罚）｜"
              f"靶：离墙 0.60 m 处 |pitch| 27.8° → ≤10°（官方 3.3°）")


@configclass
class DeeproboticsS10ClimbExpertC32EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C32（09-15 17:4x，从 **C24_13800** 续 **600 步**）。
    **相对 C26 是单变量：`st_ref_track` 打开预压墙窗口 `pre_dist=0.70 / pre_max_s=0.5`。**
    （σ 0.80 + 权重 10 与 C26 完全相同 —— σ 0.35 的核在数值上是死的，C26 已验，是前提不是变量。）

    **作者 09-15：「我需要你根据那个 mujoco 里官方的作为参考训练我们的」。**
    查下来参考跟踪有一个**实打实的缺陷**，而不是权重不够：

        ph[press] = 0.0        # 相位钟只在"压墙"时置 0
        active = ph >= 0       # ph<0 时奖励恒为 0

    参考文件时间轴是 **t = −0.50 ~ +1.20 s**，**t = −0.50~0 那 0.5 s 从来没有被跟踪过**。
    而实测我方与官方最大的偏差**恰好全在这个窗口里**：

    | 离墙 | 官方 | C24 | C31 |
    |------|------|-----|-----|
    | 0.60 m | **−3.3°** | −27.8° | −27.4° |
    | 0.25 m | −38.1° | −33.4° | −32.5° |
    | 0.00 m | **−0.6°** | −42.0° | −39.9° |

    → **C26/C27 加权重为什么没用也清楚了：加的是压墙"之后"那段的权重，而那段本来就没差那么多。**

    官方那 0.5 s 的实测：x_rel −0.571 → −0.562，**只走了 0.009 m，平均 vx 0.02 m/s**
    —— 是【停在离墙 0.56 m、身子放平】的**静止保持**。姿态恒定 ⇒ 好跟，且相位精度不敏感。

    **顺带修好同一 bug 的第二个出口**：RSI 参考态出生的相位可以是负的（`apply_ref_spawn` t_lo=−0.3），
    原来 `active = ph >= 0` 让这些环境**一出生就不计跟踪**。C29（RSI 0.7）没效果，这可能是原因之一。

    **预窗口里只跟 pitch，不跟关节角**（作者问"能不能把官方那段放长、走过去"后查出来的）：
      · 源文件往前有到 t=−1.5 s，但官方整整 1.5 s 只挪了 **0.018 m** —— **它是停着的**，
        这是翻越剪辑不是接近剪辑，拉长只能得到"站得更久"。
      · 用官方踏步参考拼一段接近？**不行**：走路是周期动作、**没有相位锚点**
        （翻越能跟是因为有压墙事件锚住），逐关节比对会因无法控制的步态相位差产生巨大误差。
        （另：该文件里 xslow 与 slow 的 q **完全相同**，5 档实为 4 档，轮列是累积转角到 1884 rad。）
      · 所以接近段**只跟与相位无关、且恰好是真实差异的那个量：机身俯仰**。
        拿官方的"站立关节角"去要求一台正在往墙走的机器人自相矛盾，还会直接奖励"停在墙前不动"。

    **反刷分**（对照 [[feedback-reward-unfakeable]]）：预窗口一旦打开，策略若停在墙前不压墙就能无限领
    "平姿跟踪"的钱 —— 这正是"门的条件不会结束"那一类。所以预窗口**限时 0.5 s**
    （= 官方自己那段的长度）且**每次遭遇只给一次**（`_ref_pre_used`，遭遇结束才清）。

    **协议变更（有理由的）**：本轮跑 **600 步**，不是 200。
    C25~C31 七轮全是 200 步有界一轮，而其中多轮的失败形态是"惩罚足额支付、行为纹丝不动"
    （C28 −0.38、C31 −0.63），这与**固着**（策略在动作盆地里，跨盆要先经过更差的区域）一致。
    **200 步根本测不了固着假设。** 扫描仍每 200 步出点，坏了随时停。

    判据（形态量）：
      ① **离墙 0.60 m 处 |pitch| ≤ 10°**（官方 3.3°，C24 27.8°）← 主判据
      ② 压墙瞬间 |pitch| ≤ 15°（官方 3.1°，C24 30.4°）
      ③ 登顶 3/3、站住 3/3、越沿净空 ≥0.03、台面 wz ≤0.06、落平过冲 ≤+13
      ④ `st_ref_track` 读数应显著大于 C26 的 0.32（预窗口新增了有效帧）
    触发条件：
      · ① 在 400 步内动到 ≤15° → 方向对，跑满并考虑提权重。
      · ④ 涨了但 ① 不动 → 预窗口的分被"停在墙前"赚走了，把 pre_max_s 0.5 → 0.25 并查停车占比。
      · ① 600 步仍不动 → **模仿路线终判死**，爬台冻结 C24_13800，GPU 转真机（爬台真机趟数至今 0）。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.st_ref_track.params.update({"sigma_q": 0.80, "sigma_p": 0.80,
                                      "pre_dist": 0.70, "pre_max_s": 0.5})
        R.st_ref_track.weight = 10.0
        print("[s10-climb-C32] st_ref_track 打开预压墙窗口 pre_dist=0.70 / pre_max_s=0.5（σ0.80 权重10 同 C26）｜"
              "靶：离墙 0.60 m 抬头 27.8° → ≤10°（官方 3.3°）｜600 步")


@configclass
class DeeproboticsS10ClimbExpertC33EnvCfg(DeeproboticsS10ClimbExpertC24EnvCfg):
    """Climb-C33（09-15 18:0x，从 **C24_13800** 续 **600 步**）。
    **`st_ref_track` 改为位置驱动相位，锚在离墙 0.58 m，只用 seg0/seg1。**
    （σ 0.80 + 权重 10 同 C26 —— σ0.35 的核数值上是死的，是前提不是变量。）

    **作者规格（09-15）：「你只要学会在 0.58 米起步 走过去 上墙 就可以」。**

    **找到了真正的 bug，而且它比之前所有假设都简单：锚点对错了位置。**
      · 我方相位钟触发条件 `dist_wall < 0.35`（+ 前轮轮速>5）
      · 参考的 **t=0 在 dist 0.56**（x_rel −0.561），那是官方**起步**的一刻
      · ⇒ 把参考的"起步"对齐到我方"已经快贴墙"，**整条参考被平移 0.21 m 去比对**
      · ⇒ C26/C27 加权重为什么没用：比对本身就错位，加多少权重都对不上

    **自纠两条**（都在同一轮里）：
      ① 我先说"官方压墙前 0.5 s 几乎没动、是停着的"——对，但我据此推出"官方不走那 0.5 米"是**错的**。
         官方 t=0~1.2 s 走完 x_rel −0.561→+0.074，**0.56 m 接近 + 翻越都在里面**，平均 0.53 m/s。
      ② 我为此建的 C32「预压墙窗口」是在修一个不存在的缺口，作废（代码保留但默认 `pre_dist=0`）。

    **位置驱动**：走到哪个距离，就比官方在那个距离上的姿态。天然免疫速度差
    （我方到 0.58 m 时是 0.9 m/s 在走，官方是从静止起步；时间驱动必然越走越偏）。
    实现：按 `signed_dist`（正=机身在墙面前方）查"距离→帧"表，
    区间 [−0.35, +0.60] 共 190 格；建表时对 x_rel 取 `cummin` 消掉 ≤0.007 m 的微小回退。

    **只用 seg0/seg1**：作者看回放认可这两段"比较像官方运控"；实测两段在**位置域上高度一致**
    （0.55 m: −3.3/−4.2°；0.45: −8.6/−8.2°；0.35: −20.1/−18.2°；0.25: −38.1/−32.8°），
    而 seg2 明显是另一回事（0.35 m 处只有 −6.6°，0.05 m 处却 −15.4°）。**作者的眼睛与数据一致。**

    判据（形态量）：
      ① **离墙 0.60 m 处 |pitch| ≤ 10°**（官方 3.3°，C24 27.8°，C31 27.4°）← 主判据
      ② 离墙 0.25 m 处 |pitch| 30~40°（官方 38.1°）——**要的是"该抬的时候抬"，不是一味压低**
      ③ 过墙面（0.00 m）时 |pitch| ≤ 15°（官方 0.6°，C24 42.0°）
      ④ 登顶 3/3、站住 3/3、越沿净空 ≥0.03、台面 wz ≤0.06、落平过冲 ≤+13
    触发条件：
      · ①③ 在 400 步内动 → 方向对，跑满并考虑提权重到 20。
      · ① 动但 ④ 掉登顶 → 平着到墙根起不来，`anchor_dist` 0.58 → 0.45 缩短平走段。
      · ①③ 600 步仍不动 → **模仿路线终判死**，爬台冻结 C24_13800，GPU 转真机（爬台真机趟数至今 0）。

    **部署侧连带（未做，待作者定）**：训练按"0.58 m 起步"，部署的 switch_stair_driver
    目前是 2.0 m 切入后 0.4 m/s 一路开过去、中途不停。两边口径要对齐才有意义。"""

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        R.st_ref_track.params.update({"sigma_q": 0.80, "sigma_p": 0.80,
                                      "phase_mode": "dist", "anchor_dist": 0.58, "segs": (0, 1)})
        R.st_ref_track.weight = 10.0
        print("[s10-climb-C33] st_ref_track 位置驱动相位（锚 0.58 m，seg0/1，σ0.80 权重10）｜"
              "修的是锚点错位 0.21 m｜靶：离墙 0.60 m 抬头 27.8°→≤10°，过墙面 42.0°→≤15°｜600 步")


@configclass
class DeeproboticsS10ClimbExpertC34EnvCfg(DeeproboticsS10ClimbExpertC33EnvCfg):
    """Climb-C34（09-15 18:3x，从 **C33_14200** 续 400 步）。
    **相对 C33 单变量：`cc_pitch_hold` 闸门 `encounter` → `after_press`。**

    **C33 判决：半成半败，外加一个真退步。**
    | | 官方 | C24起点 | 14000 | 14200 | 14399 |
    |---|---|---|---|---|---|
    | ① 离墙 0.60 m | −3.3° | −27.8° | −28.0° | −28.6° | **−30.8°** ✗ 反而更仰 |
    | ② 离墙 0.25 m | −38.1° | −33.4° | −37.5° | −40.6° | −42.3° 略过 |
    | ③ 过墙面 | −0.6° | **−42.0°** | −32.6° | **−7.5°** | −10.1° **✓解决** |
    | 起抬→落平 | 1.0 s | 0.68 s | 0.93 | 0.98 | **1.00 s ✓** |
    | 越沿净空 | — | — | 0.034 | **0.066** | 0.062 ✓ |
    | 落平过冲 | — | **+12.8** | +18.1 | +22.0 | **+27.1 ✗翻倍** |
    登顶 3/3、站住 3/3（|roll|峰 1.6°）全程保持。**锚点修复是对的**：③ 一项就从 −42° 修到 −10°。

    **① 为什么不动 —— 两个奖励在同一距离上直接对冲**：
    `cc_pitch_hold` 闸门是 `encounter`（看见墙就开，读数 0.53），在离墙 0.60 m 处**正在付钱让它仰头**；
    `st_ref_track`（读数 0.38）在同一位置要求放平。**仰头那边出价更高。**
    C30 单独关这个闸门是零结果 —— 但那时**没有反向拉力**（撤激励≠施加压力）且只跑 200 步。
    **C33 已证明位置驱动的拉力真实有效（③ 动了 32°），此时拿掉对冲才有意义。**

    起点取 **C33_14200 而非末点 14399**：14200 的过墙面 −7.5°、净空 0.066、wz≈0 都是全线最好，
    且落平过冲（+18~22）比 14399（+21~27）轻。

    判据：① 离墙 0.60 m ≤15°（C33 −28.6°）　② 0.25 m 保持 30~42°　③ 过墙面 ≤15°（保住）
    　　　④ 登顶 3/3、站住 3/3、净空 ≥0.03、台面 wz ≤0.06
    　　　⑤ **落平过冲不得再涨**（C33_14200 是 +18.1/+22.0/+18.8）
    触发条件：
      · ① 动且 ③⑤ 保住 → 方向对，跑满。
      · ⑤ 再涨 → **立即停**，交付取 C33_14200，不再调。
      · ① 仍不动 → 对冲不是原因；**爬台线冻结**（候选在 C24_13800 与 C33_14200 之间按过冲/形态取舍），
        GPU 转真机（爬台真机趟数至今 0，距决赛 7 天）。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_pitch_hold.params["gate"] = "after_press"
        print("[s10-climb-C34] cc_pitch_hold 闸门 → after_press（拿掉与 st_ref_track 在远端的对冲）｜"
              "靶：离墙 0.60 m −28.6° → ≤15°；过墙面保住 ≤15°；落平过冲不得再涨")


@configclass
class DeeproboticsS10ClimbExpertC35EnvCfg(DeeproboticsS10ClimbExpertC33EnvCfg):
    """Climb-C35（09-15 18:4x，从 **C33_14200** 续 200 步，有界一轮）。
    **单变量：`cc_pitch_hold` 权重 30 → 0（整项关掉）。**

    **C34 判失败，而且砸了 C33 唯一修好的那项**：
    | | C33_14200 起点 | C34_14400 |
    |---|---|---|
    | 离墙 0.60 m | −28.6° | −29.5° ① 仍不动 |
    | **过墙面** | **−7.5°** | **−27.4°** ✗ 严重倒退 |
    | 落平过冲 | +18~22 | +20.0 |
    登顶3/3、站住3/3、净空0.059、wz✓ 其余保住。

    **我的假设错了，而且错得有信息量**：我以为对冲在**远端**，把闸门关到 `after_press`。
    结果远端照样仰，而**闸门开在压墙之后 = 把"付钱让它仰头"挪到了过墙面那一刻**，
    正好是 `st_ref_track` 要求放平的地方。**没有消除对冲，只是把它从远端搬到近端，还砸了好不容易修好的项。**

    **真问题是结构性的**：`cc_pitch_hold` 是**参考轨迹还不存在时手工设计的俯仰奖励**，
    而位置驱动的 `st_ref_track` 本身就在每个距离上规定了俯仰。
    **两者职能完全重叠，手工那个必然和官方轨迹打架 —— 闸门挪到哪儿都躲不开。**
    "按官方来"的完整含义是：**把手工那项关掉，让官方参考独自定义俯仰剖面。**

    风险：`cc_pitch_hold` 权重 30、读数 0.53，是主要奖励之一，关掉可能直接爬不上去。
    所以**有界 200 步 + 硬回退条件**。

    判据：① 离墙 0.60 m ≤15°（C33_14200 −28.6°）　② 过墙面 ≤15°（**必须保住 −7.5°**）
    　　　③ 登顶 3/3、站住 3/3、净空 ≥0.03、台面 wz ≤0.06　④ 落平过冲不得超 +22
    触发条件：
      · ③ 掉登顶 → `cc_pitch_hold` 仍是承重项，**权重改 30→5 再试一轮**，不再归零。
      · ② 破 → 关掉它反而更糟，**立即停，交付取 C33_14200**。
      · ①② 都好 → 跑满 600 并考虑把 `st_ref_track` 权重 10 → 20。

    **旁证（34 cm 重测，作者拍板口径，09-15 18:2x）**：两候选在 34 cm 上都 3/3 登顶 + 3/3 站住；
    形态优势也成立 —— 过墙面 C24_13800 **−41.4°** vs C33_14200 **−12.3°**。"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cc_pitch_hold.weight = 0.0
        print("[s10-climb-C35] cc_pitch_hold 权重 30 → 0（手工俯仰项与位置驱动参考职能重叠、必然打架）｜"
              "让官方参考独自定义俯仰剖面｜有界 200 步，掉登顶即回退到权重 5")


@configclass
class DeeproboticsS10ClimbExpertC36EnvCfg(DeeproboticsS10ClimbExpertC35EnvCfg):
    """Climb-C36（09-15 20:0x，从 **C35_14399** 续）。
    **单变量：新增 `cc_rear_sym`（蓄力段左右后腿对称罚）。**

    **作者 09-15 逐帧对照 MuJoCo 回放提的五条**（官方 0.12/0.52/0.78/1.00/1.18 ↔ 我方 16.98/17.11/17.36/17.64/18.14）：
    ① 官方接近前姿势正常，我们后腿已在奇怪动作
    ② 官方靠近墙后腿弯曲蓄力、伸**右**前轮；我们没蓄力、动作变形、伸的是**左**前轮
    ③ 官方后腿发力站起、右前轮上去后左前轮跟随；我们不好发力、右前轮乱打上去
    ④ 我们没推起身子
    ⑤ 落地不标准、几乎要摔倒

    **逐条核对数据，五条全部成立**：
    | | 官方 | 我方 C35 |
    |---|---|---|
    | ① 接近前(0.55~0.69m) | pitch −3.3° roll +0.4° 四轮贴地，hl/hr hipy 都 +0.72 | pitch −16.8° roll **−12.3°**，**hr 轮已抬 149mm**，hl +0.78 / hr **−0.29** |
    | ② 蓄力(0.32/0.60m) | **fr** 抬 381mm，两后腿同屈(膝 −2.00/−1.69) | **fl** 抬 258mm、fr 仅 +1mm，只有 hl 屈、hr hipy 卡在 −0.29 |
    | ③ 前轮上墙 | fr 381→414 平滑 | fr **+1mm → +540mm**，0.25 s 窜起 |
    | ④ 推起身子 | z **0.639**、pitch 已放平到 −13° | z 0.597、pitch 仍 **−33.6°** |
    | ⑤ 落地 | pitch **+2.1°**、z 0.714、四轮均匀 | pitch **+17.3°**、z **0.570**、后轮高差 **194mm** |

    **根因（①②，③④⑤是其下游）**：`|hl_hipy − hr_hipy|` 逐帧均值
        蓄力段 0.15~0.70m   官方 **0.098/0.206**   我方 **0.609/0.665**（3~6 倍）
        蹬地段 <0.15m       官方 **1.090/0.640**   我方 0.247/0.393
    → **官方"对称蓄力 → 不对称蹬地"，我方"不对称蓄力 → 对称蹬地"，不对称用在了相反的阶段。**
    后腿 hipy 默认 +0.35、蓄力应更正（官方到 +1.0），而**我方右后腿整个蓄力段是负值（大腿前甩）**。
    C14/C24/C33/C35 全线如此，二十多轮无人量过。

    **自纠**：我第一版判断是"`joint_mirror` 的翻越豁免开太宽，收窄它"。**错的** ——
    该项配的是**对角**对（fl↔hr、fr↔hl，trot 同步用），翻越时前后腿本就该不同
    （官方 t=0.52：fl hipy −0.82 vs hr +1.01），豁免它是**对的**，收窄只会把前后腿强行拉平。
    **左右对称（hl vs hr）这个约束压根不存在**，不是被豁免 —— 所以是新增项，不是改闸门。

    判据（按作者五条逐条量）：
      ① 接近段(0.55~0.70m) |roll| ≤5°（C35 12.3°）、hr 轮离地 ≤30mm（C35 149mm）
      ② 蓄力段 |Δhipy| 均 ≤0.25（官方 0.10~0.21，C35 0.609）；**hr_hipy 均须转正**（C35 −0.30）
      ③ fr 轮上升连续（0.25 s 内涨幅 ≤300mm，C35 539mm）
      ④ 过墙面时机身 z ≥0.62（官方 0.639，C35 0.597）
      ⑤ 落地 pitch ≤+10°（官方 +2.1，C35 +17.3）、后轮高差 ≤100mm（C35 194mm）
      ⑥ 不得掉：登顶 3/3、站住 3/3、越沿净空 ≥0.03、台面 wz ≤0.06
    触发：⑥ 掉 → 权重减半重跑；①② 动而 ③④⑤ 不动 → 说明还有别的根，再查；
    　　　②600 步不动 → 对称罚拉不动它，停，交付维持 C35_14399。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        self.rewards.cc_rear_sym = RewTerm(func=_cr.rear_lr_symmetry, weight=-15.0,
                                           params={"lo": 0.15, "hi": 0.70, "w_knee": 0.5})
        print("[s10-climb-C36] 新增 cc_rear_sym −15（蓄力段 0.15~0.70m 生效，蹬地段豁免；权重 −1 时读数仅 −0.014=主项1%，先验定到 −15）｜"
              "靶：|Δhipy| 均 0.609 → ≤0.25（官方 0.10~0.21），hr_hipy 均 −0.30 → 转正")


@configclass
class DeeproboticsS10ClimbExpertC37EnvCfg(DeeproboticsS10ClimbExpertC36EnvCfg):
    """Climb-C37（09-15 21:3x，从 **C36b_15197** 续）。
    **单变量：新增 `cc_front_early`（近墙前抬前轮罚），治作者五条里的 ③④。**

    **C36/C36b 共 800 步的结论**：`cc_rear_sym` 修的是 ①②（后腿），有效但已近收敛：
      ② |Δhipy|均 0.620 → 0.198 → 0.184（官方 0.10~0.21，**达标**）
      ② hr_hipy −0.334 → −0.102 → −0.064 → **−0.053**（收敛放缓，边际收益低）
      ① |roll|峰 12.6° → 6.7°~9.5°（震荡，整体减半）　hr轮离地 175 → 63~91mm
      **③ fr涨幅 543 → 515mm、④ 过墙面 z 0.611 → 0.585：八百步纹丝不动 → 不归后腿对称管。**

    **③④ 的根（实测，离墙 0.45~0.60 m，官方此时四轮仍全贴地）**：
      官方 seg0  fl 离地占比 **0%**（均高 +28mm）　fr 7%（+26mm）
      我方       fl 离地占比 **100%**（均高 **+334mm**）　fr 40%（+51mm）
      顺序亦反（0.15~0.60 m 均高）：官方 **fr 168 > fl 91**，我方 **fl 332 > fr 282**
      → 右前轮到 0.35 m 才动，只能 0.25 s 内窜 522mm 够墙沿（③）；
        支撑点只剩两后轮，推不起身子（④）。

    **不硬编"哪个轮先抬"**，只要求近墙前别抬 —— 左右顺序交给位置驱动的 `st_ref_track`。
    C34 已证明手工先验项（`cc_pitch_hold`）会与官方参考正面打架，不再重蹈。

    判据（承接作者五条）：
      ③ fr轮 0.25 s 涨幅 ≤300mm（C36b 515）
      ④ 过墙面机身 z ≥0.61（C36b 0.585，官方 0.639）
      ①② 不得退（|roll|峰 ≤10°、|Δhipy| ≤0.25）
      ⑥ 不得掉：登顶 3/3、站住 3/3、净空 ≥0.03、台面 wz ≤0.06、切入偏航 ≤±5°
    触发：⑥ 掉登顶 → 权重减半；③④ 400 步不动 → 前腿早抬不是根，停并重查；
    　　　③④ 动但 ①② 退 → 两项冲突，需分阶段加权而非同时全开。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        self.rewards.cc_front_early = RewTerm(func=_cr.front_early_lift, weight=-8.0,
                                              params={"lo": 0.45, "hi": 0.70, "thr": 0.05})
        print("[s10-climb-C37] 新增 cc_front_early（离墙 0.45~0.70m 罚前轮离地超 5cm）｜"
              "靶：fl 离地占比 100%→接近 0（官方 0%）、fr 涨幅 515→≤300mm、过墙面 z 0.585→≥0.61")


@configclass
class DeeproboticsS10ClimbExpertC38EnvCfg(DeeproboticsS10ClimbExpertC36EnvCfg):
    """Climb-C38（09-15 22:0x，从 **C36b_15197** 续）。**按作者五条做整条时序链。**

    作者五条（逐帧对照官方 vs 我方五张截图）+ 补充「你可以不那么急，完成蓄力动作，右前轮再开始往上」：
      ① 接近时保持正常姿势 → ② 到墙前两后腿一起蹲到位 → ② 蹲够了右前轮先伸
      → ③ 右前轮平稳上墙、左前轮跟随 → ④ 后腿蹬地推起身子 → ⑤ 放平站稳

    **前几轮拆开一个个治的结果**：① 修好了（C36 后腿对称 + 切入距离 1.0→0.5m，
    hr轮离地 175→1mm、roll 12.6→1.1°），但 ②③④⑤ 各自卡住。**这次按链条整体做。**

    **实测靶（官方 seg0 vs C36b_15197 切0.5m）**：
    | | 官方 | 我方 |
    |---|---|---|
    | 蓄力段后腿 hipy 峰 | hl **+1.08** / hr **+1.01** | +0.70 / +0.68（**对称了，深度只有七成**） |
    | 蓄力段前轮均高 | fr **+168mm** > fl +91mm（右前领先 77mm） | fl **+175mm** > fr +123mm（**左前领先 52mm，反了**） |
    → **我方是"没蹲够就急着抬左前轮"**，正是作者那句"不那么急"说的。

    三项一起上（拆开会互相打架：单独加"别抬前轮"只会把动作整体往后拖，不会去蹲）：
      A `cc_load_depth`    蓄力深度：罚 (1.0 − min(hl,hr)_hipy)²，取小的防"一条蹲够就交差"
      B `cc_front_early2`  蓄力未完成（load<0.85）就抬前轮，罚超出 5cm 的部分
      C `cc_front_order`   蓄力完成后，罚 (fl − fr) 的正值部分 —— 左前高过右前才罚，
                           右前领先不罚、不强求领先多少（避免又造一个手工先验，见 C34 教训）

    判据（承接作者五条）：
      ② 后腿 hipy 峰 ≥+0.95（现 +0.70，官方 +1.05）　② 蓄力段 fr 均高 > fl（现反）
      ③ fr 轮 0.25 s 涨幅 ≤300mm（现 509）　④ 过墙面 z ≥0.62（现 0.613）
      ⑤ 落地 pitch ≤+10°（现 +24.8）、后轮高差 ≤100mm（现 247）
      ① 不得退：hr轮离地 ≤30mm、|roll|峰 ≤5°（现 1mm / 1.1°）
      ⑥ 不得掉：登顶 3/3、站住 3/3、净空 ≥0.03、台面 wz ≤0.06
    触发：⑥ 掉登顶 → 三项权重同时减半；② 深度动而 ③④⑤ 不动 → 链条断在别处，再查；
    　　　A 读数涨但深度不动 → 罚被吸收（同 C28/C31 形态），改塑形不改惩罚。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        W = {"lo": 0.15, "hi": 0.70}
        self.rewards.cc_load_depth = RewTerm(func=_cr.climb_load_depth, weight=-6.0,
                                             params={**W, "target": 1.0})
        self.rewards.cc_front_early2 = RewTerm(func=_cr.climb_premature_front, weight=-6.0,
                                               params={**W, "ready": 0.85, "thr": 0.05})
        self.rewards.cc_front_order = RewTerm(func=_cr.climb_front_order, weight=-6.0,
                                              params={**W, "ready": 0.85})
        print("[s10-climb-C38] 时序链三项：A蓄力深度(+0.70→+1.0) B蓄力未完成不许抬前轮(load<0.85) "
              "C蓄力后右前先(现 fl+175>fr+123，官方 fr+168>fl+91)｜权重先各 −6，待先验读数调")


@configclass
class DeeproboticsS10ClimbExpertC39EnvCfg(DeeproboticsS10ClimbExpertC36EnvCfg):
    """Climb-C39（09-15 23:1x，从 **C36b_15197** 续）。**按训练计划第三十四章（Astra 复核）重建。**

    相对 C36b 的改动（一组，因为它们是同一个机制的三个面，拆开无法单独验证）：
      1. **新相位** `climb_seq._phase`：距离触发进入 + 阶段记忆 + 局部时间推进 + **只向前**
         （距离仅做 ±30% 限幅）。治旧 `RefTrack` 在离墙 0.31 m 处的相位倒退
         —— 官方自跟踪反例 seg0 t=0.74s 实际 fr膝 1.925、查表却要求 t=0.54s 的 2.710。
         自检：新相位官方自跟踪误差 **0.000**、零倒退、零停住（旧的峰值误差 0.784、倒退 2~3 帧）。
      2. **分组跟踪** `seq_track`：后腿对 / 右前膝 / 左前膝 **分开计分**，
         **Huber(δ=0.5) 不用窄高斯**（σ=0.5 在 2.58rad 处 = 2.7e-12，与 C26 的 σ=0.35 死核同错）。
         右前膝权重加倍 —— 作者明确要求"右前先上"。
      3. **准备门锁存** `seq_ready_latch`：治 Astra 3.3。旧版每步重判，蹬地时后腿伸展→hipy 下降
         →又判"未准备好"，**会罚官方动作**（官方被罚 21/23 帧、罚值和 3.02/3.49）；
         锁存后官方被罚 2/1 帧、罚值 **0.0002/0.0000**（降 15000 倍）。

    **旧 `st_ref_track` 保留但降权**（10 → 3）：它是 12 关节平均，会把单关节误差摊薄成 1/12，
    与分组项职能重叠；不直接删是为了保留 C33 已验证有效的"过墙面姿态"塑形（−42°→−10°）。

    **离线判别力已验**（六种行为，分组代价）：
      正确 0.000 < 稍偏离 0.053 < 只折膝未蓄力 0.243 < 停住 0.374 < 右前膝伸直 1.232 < 当前动作 1.539
    且每种错误在分组得分里有各自签名。**离线通过不代表动态可学**（Astra 明确指出），故本轮先做动态探针。

    **验收用 `fivechk2.py`（v2）**，阈值由官方实测标定、官方九项全过。
    C36b 基线（同一把尺子）：①接管后 roll 18.1✗ / ②后腿峰 0.69,0.71✗ / ②fr膝 1.19✗ 左前领先✗ /
    ③加加速度 461539✗（官方 1347）/ ④过墙 z 0.613✗ / ⑤落地 pitch 24.8✗ 后轮高差 247✗ —— **九项过一项**。

    首轮判据（按 Astra 第 7 节第 5 条，以**实际新增 200 迭代**为检查点，不按文件名整数倍）：
      报 **正确顺序率 / 有效蓄力率 / 后轮过沿率 / 稳定落地率 / 接管后异常 / 完整成功率**；
      总奖励与三趟登顶只作补充。按失败阶段决定下一轮，**不自动长训**。"""

    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_seq as _cs
        R = self.rewards
        # **旧 st_ref_track 关掉**（Astra P1-5）：它用的是有相位倒退 bug 的距离查表
        # （官方自跟踪反例：t=0.74s 实际 fr膝 1.925、查表要求 2.710），
        # 且把重力投影与弧度参考直接相减（量纲不一致）。
        # 其职能由 cs_all12（全身跟踪）+ cs_pitch（统一量纲）在**修正相位**上承接。
        R.st_ref_track.weight = 0.0
        P = {"enter_dist": 0.70, "corr": 0.30, "huber_d": 0.5}
        R.cs_all12 = RewTerm(func=_cs.SeqTrack, weight=-2.0, params={**P, "group": "all12"})
        R.cs_rear = RewTerm(func=_cs.SeqTrack, weight=-4.0, params={**P, "group": "rear"})
        R.cs_rear_knee = RewTerm(func=_cs.SeqTrack, weight=-2.0, params={**P, "group": "rear_knee"})
        R.cs_fr_knee = RewTerm(func=_cs.SeqTrack, weight=-8.0, params={**P, "group": "fr_knee"})
        R.cs_fl_knee = RewTerm(func=_cs.SeqTrack, weight=-4.0, params={**P, "group": "fl_knee"})
        R.cs_pitch = RewTerm(func=_cs.SeqPitch, weight=-3.0,
                             params={"enter_dist": 0.70, "corr": 0.30, "huber_d": 0.20})
        R.cs_ready = RewTerm(func=_cs.SeqReadyLatch, weight=-6.0,
                             params={"enter_dist": 0.70, "corr": 0.30, "ready": 0.85, "thr": 0.05})
        print("[s10-climb-C39] 修正相位(只向前+阶段记忆+生命周期) + 分组跟踪(全身/后腿/右前膝×2/左前膝, Huber) "
              "+ 统一量纲俯仰 + 准备门锁存｜**旧 st_ref_track 关掉**（相位倒退+量纲错）｜"
              "靶：共同蓄力 0.67→≥0.95、首抬转为右前、fr膝峰 1.18→≥2.0")


@configclass
class DeeproboticsS10ClimbBranchAEnvCfg(DeeproboticsS10ClimbExpertC39EnvCfg):
    """**分支 A：定向 RSI 课程**（09-16 夜间计划 §5）。起点 **C36b_15197**，非 C39。

    **开训前诊断（不更权重，`climb_stage_eval.py` 全字段记录）结论**：
      · 执行侧无瓶颈：右前膝 |实际−目标| 仅 **0.102 rad**，力矩均 6.9 N·m、饱和 0.4%
      · 命令侧才是问题：右前膝 |目标−参考| = **1.028 rad**（差 10 倍）→ 策略分布/课程问题
      · **动作可达**：网络原始输出 fr_knee 范围 **−3.96 ~ +5.09**，够到参考 1.66 只需 +4.04
        → 此前"动作空间不可达、PPO 学不出来"的说法**已被数据推翻并撤回**
      · 相位起点处于深屈(>1.2) 的样本**只有 12.9%**（配置 ref_prob=0.7，逐层条件采样把它稀释掉了）
      · **从深屈出生的样本，策略立刻下达伸直目标**（当前 1.68 → 目标 0.82）

    **本分支只改课程/出生分布**，不动动作缩放、学习率、探索噪声、奖励权重（计划 §5.3）：
      1. `categorical=True` —— 一次类别采样，配置比例即实际比例
         （验证：配置 ref 0.60 → 实测 0.600；旧逐层法 ref_prob=0.7 实际仅约 0.50，叠 sign 后约 0.25）
      2. `ref_t_range=(-0.10, 0.76)` —— **按事件定的关键区间**，不套绝对秒数：
         官方后腿共同最深 t=0.52(seg0)/0.38(seg1)、右前膝峰 t=0.56/0.38，两事件重合；
         区间覆盖离墙 0.56→0.31 m，即"共同承重准备 → 右前收膝"那一段。
      3. 比例 **60% 关键参考态 / 20% hooked+pushed（前轮已支撑、推进落地衔接）/ 20% 正常接近**

    **不做的事**（计划 §五、§六边界）：不改 244 维接口与动作缩放；不降低作者五项要求；
    不把"出生时膝已很弯"计成功 —— 看的是**之后 0.2~0.5 s 能否接续参考并建立支撑**。

    判决（计划 §5.4，每 200 实际新增迭代）：关键区间右前髋膝跟踪误差较**本分支基线**降 ≥20%，
    或有效右前先搭接/局部接续成功率 +20 个百分点，或后腿推进接稳定落地比例改善。
    **连续两个 200 块无改善即停，禁止自动跑满 800。**"""

    def __post_init__(self):
        super().__post_init__()
        self.events.spawn_at_wall.params.update({
            "categorical": True,
            "hooked_prob": 0.10, "pushed_prob": 0.10, "ref_prob": 0.60,   # 余 20% 普通接近
            "ref_t_range": (-0.10, 0.76),
        })
        print("[s10-climb-BranchA] 定向 RSI 课程：一次类别采样 60%关键参考态/10%hooked/10%pushed/20%正常接近｜"
              "关键区间 t∈[−0.10,+0.76]（后腿共同最深+右前膝峰，离墙 0.56→0.31m）｜"
              "起点 C36b_15197｜只改课程，不动缩放/权重/噪声")


@configclass
class DeeproboticsS10ClimbApproachEnvCfg(DeeproboticsS10ClimbBranchAEnvCfg):
    """09-16 作者：**只要模仿官方那套上台阶动作**。

    与 BranchA 的唯一差别：**出生分布**。BranchA 有 80% 的回合直接出生在爬台姿势
    （60% 关键参考态 + 10% hooked + 10% pushed），机器人几乎没机会经历
    「正常站姿 → 走过去 → 蓄力 → 上墙 → 落地」这条完整链路。

    作者五项要求的第一条正是：「官方在还没靠近台阶是正常姿势，我们的已经后腿在奇怪的动作了」。
    80% 的爬台姿势出生会直接把「爬台姿势」训成默认姿势，该要求在现课程下学不出来。

    09-16 实测依据：固定正常接近入口下，起点策略 C36b_15197 **两个种子各 32/32 全部登顶**，
    即「爬上去」这件事已不需要 RSI 加速；缺的是动作形态。

    **只改出生分布，不动**：动作缩放、奖励权重、噪声、观测 244 / 动作 16、物理与执行器限制、地形。
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.spawn_at_wall.params.update({
            "categorical": True,
            "hooked_prob": 0.0, "pushed_prob": 0.0, "ref_prob": 0.0,   # 100% 正常接近
        })
        print("[s10-climb-Approach] 全程正常接近出生（ref/hooked/pushed 全 0）｜"
              "起点 C36b_15197｜只改出生分布，不动缩放/权重/噪声/地形｜"
              "目标 = 作者五项动作要求")


@configclass
class DeeproboticsS10ClimbBlk1EnvCfg(DeeproboticsS10ClimbApproachEnvCfg):
    """块1：后腿蓄力（09-16 五阶段计划 v2，根因 R1）。

    官方蓄力 = 后腿更屈、机身几乎不升：后髋 +0.65→+1.08/+1.63、后膝 −1.47→−2.26、
    蓄力段机身只升 0.031 m。我方 C36b 实测相反：后髋 +0.78→−0.70/−0.49（反向伸展）、
    后膝只到 −1.42/−0.83、机身升 0.206 m（6.6 倍）——用后腿把身子撑起来，而不是蹲下去转起来。
    抬头量 0.303 看似达标是**用相反机制凑出来的**，故抬头量不可单独作判据。

    唯一改动：把既有 cs_rear / cs_rear_knee 权重按**先验读数**抬到与任务奖励同量级。
    实测原读数 cs_rear −0.0674、cs_rear_knee −0.0240，而 wheel_height_progress +0.4645、
    track_lin_vel_xy_exp +6.0995 —— 弱一到两个数量级。倍数由 S10_BLK1_K 给。
    不动：动作缩放、244/16 接口、物理/执行器限制、地形、出生分布、其余奖励项。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        k = float(_os.environ.get("S10_BLK1_K", "1"))
        R = self.rewards
        w0r, w0k = R.cs_rear.weight, R.cs_rear_knee.weight
        R.cs_rear.weight = w0r * k
        R.cs_rear_knee.weight = w0k * k
        print(f"[s10-climb-Blk1] 后腿蓄力加权 x{k:g}: cs_rear {w0r:g}->{R.cs_rear.weight:g}  "
              f"cs_rear_knee {w0k:g}->{R.cs_rear_knee.weight:g} | 其余继承 Approach | "
              f"目标 后髋>=+1.0/+1.4 后膝<=-2.1 蓄力段机身升高<=0.06m", flush=True)




@configclass
class DeeproboticsS10ClimbBlk2EnvCfg(DeeproboticsS10ClimbApproachEnvCfg):
    """块2：跟随前腿抬起够台面（判据 D）。09-16 第四轮复核后重建。

    **唯一经得起可信几何量检验的缺口。** 官方三段一致：领先前腿上台后，
    跟随前腿在整个「领先上台→自己上台」窗口内 **100% 离地 >0.10 m**，
    轮底抬到 **0.472~0.506 m**（远高于 0.34 台面），髋收到 **−2.21~−2.28**，
    从墙前 0.13~0.23 m 处**抬高收腿够过去**——它是纯摆动腿，**不承重**。
    我方实测跟随前腿支撑占比 **73~76%**（官方 24~28%），是一路**蹭**上去的。

    镜像后跟随前腿 = **右前**。本块只加一项：`cs_foll_front` 跟踪镜像参考的
    右前髋/膝（= 官方左前的镜像），权重由 S10_BLK2_KF 给，开训前先验读数。

    此前作废的结论不再作为依据：角色对调、蓄力反向、跟随太晚、领先轮最强占比 —— 全部已证伪。
    不动：动作缩放、244/16 接口、物理/执行器限制、地形、出生分布、其余奖励项。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_seq as _cs
        kf = float(_os.environ.get("S10_BLK2_KF", "30"))
        R = self.rewards
        P = dict(R.cs_rear.params); P.pop("group", None)
        R.cs_foll_front = RewTerm(func=_cs.SeqTrack, weight=-kf,
                                  params={**P, "group": "foll_front", "mirror": True})
        print(f"[s10-climb-Blk2] 跟随前腿(右前)抬起够台面: cs_foll_front w=-{kf:g} (镜像取列) | "
              f"目标 窗口内离地>0.10m 占比>=90%, 轮底峰>=0.40m, fr_hipy<=-2.1", flush=True)


@configclass
class DeeproboticsS10ClimbBlk3EnvCfg(DeeproboticsS10ClimbApproachEnvCfg):
    """块3（09-16）：**墙前慢速接近 → 停住 → 后退**。作者 09-16 原话点名的缺口。

    **块1、块2 都没有训这两项。** 块1 训的是后腿跟踪权重（跟的还是未镜像的参考），
    块2 训的是跟随前腿（而该项经统一窗口复算本来就已达标）。**「停」这个机制在训练侧
    从不存在**：`lin_vel_x=(0.3, 0.6)` + `resampling_time_range=(10,10)` = 整回合恒定前进指令。

    ## 实测缺口（三档一致，不是个别回合）

        墙前 0.6~0.0 m 最小 |vx|   C36b 0.721  块1 0.511  块2 0.827
        停住时长（|vx|<0.05）      C36b 0.000  块1 0.000  块2 0.000  ← 一次都没停过
        蓄力窗时长（0.6→0.15 m）   我方 0.41~0.47 s   官方 1.80~2.48 s   ← 差 5 倍

    官方 ref_climb34 三段按离墙分箱的 vx 中位数：**0.60~0.45 m 处 +0.017 / +0.035 / +0.006**
    （= 停住），0.45~0.35 m 处才爆发到 +1.077 / +1.078 / +0.229。后退同样是真的：
    seg0 最长连续后退 0.405 s、最低 −0.347 m/s、总后退 0.039 m。

    ## 单变量：只换命令项，不加奖励、不动权重

    `base_velocity` 换成 `ClimbApproachVelocityCommand`，按 signed_dist 改写 vx：
    0.75 m 起减速 → 0.62~0.54 m 后退 −0.15 → 0.54~0.45 m 停住 0 → 0.45 m 释放。
    借**已有最强奖励** `track_lin_vel_xy_exp`（读数 +6.20，全表第一）去要求策略停下来，
    不新增弱项（新项典型读数 0.01~0.3，比主项弱 20~600 倍）。

    开训前已核实的三条配合关系（见 mdp/climb_cmd.py 模块文档）：
      · `track_lin_vel_xy_exp_climb_aware` 只在 dist_wall ≤ 0.30 关闭 → 停止带里它开着；
      · `stall_penalty_encounter_free` 遭遇段豁免停滞罚 → 指令停车不会被反罚；
      · `approach_wall`（会与停车打架）在本任务族未注册，日志 grep 0 次。

    **不动**：奖励权重与项集、动作缩放、观测 244 / 动作 16、物理与执行器限制、地形、出生分布。

    ## 验收（200 步后按同一窗口复算）

        墙前最小 |vx|      0.72~0.83 → ≤0.10
        停住时长           0.000 s   → ≥0.5 s
        蓄力窗时长         0.47 s    → ≥1.5 s（官方 1.80~2.48）
        后退              无        → 出现连续后退 ≥0.1 s
        不得退化           三段登顶率、判据 D（跟随支撑 24~28%）不掉

    环境变量：S10_BLK3_BACK 改后退指令（默认 −0.15）、S10_BLK3_DBG 打开剖面占比自检。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_cmd as _cc
        old = self.commands.base_velocity
        new = _cc.ClimbApproachVelocityCommandCfg()
        for k, v in vars(old).items():
            if k == "class_type":
                continue
            if hasattr(new, k):
                setattr(new, k, v)
        new.back_v = float(_os.environ.get("S10_BLK3_BACK", "-0.15"))
        new.debug_every = int(_os.environ.get("S10_BLK3_DBG", "0"))
        self.commands.base_velocity = new
        print(f"[s10-climb-Blk3] 墙前速度剖面: 减速{new.d_slow_hi:.2f} 后退{new.d_back_hi:.2f}"
              f"~{new.d_stop_hi:.2f}@{new.back_v:+.2f} 停住{new.d_stop_hi:.2f}~{new.d_release:.2f}"
              f" 释放{new.d_release:.2f} | 只换命令项，奖励/权重/缩放/地形全不动 | "
              f"靶 停住时长>=0.5s 蓄力窗>=1.5s 墙前最小|vx|<=0.10", flush=True)


@configclass
class DeeproboticsS10ClimbBlk3cEnvCfg(DeeproboticsS10ClimbApproachEnvCfg):
    """块3c（09-16 18:5x）：**停住后才开始做上墙动作**。作者原话的直接实现。

    > 「慢速开到墙前 **停下来** 然后开始做动作」

    ## 块3（只改指令）失败，原因已查清且可复现

    两轮（首轮有 bug 作废，重跑守卫通过）都是停住 0.000 s。实测定位：

        离墙 ≥0.90 m   指令跟踪误差 **0.035**   —— 听指令，能停死（探针实测停住 100 s）
        离墙 0.75~0.45 指令跟踪误差 **0.48~0.68** —— 不听指令

    而 `climb_seq.advance_phase` 的相位进入条件正是 `sd <= enter_dist`（0.70）——
    **「见墙就冲」的起点与指令失效的起点是同一条线**。一到 0.70 m 相位时钟就起跑，
    七个 `cs_*` 跟踪项开始要求关节走参考轨迹，机身仍有 0.87 m/s，于是「边冲边做动作」。

    官方不是这样：ref_climb34 三段在离墙 0.60~0.45 m 处 vx 只有 **+0.006~+0.035**（= 停住），
    0.45 m 之后才爆发到 +1.08。**先停住，再起动作。**

    ## 单一机制：相位进入加「已停住」门

    `advance_phase` 新增可选 `ready` 门（None 时行为与加门前逐位一致）：
    进入条件变为 `未激活 & 未完成 & 遭遇 & sd<=enter_dist & **已连续停住 hold_s 秒**`，
    超时 `timeout_s` 秒仍没停住也放行（防死锁，保住登顶）。

    命令剖面同时从「按距离分带」换成**状态机**（接近→停住→后退→释放），
    与部署驱动 `switch_stair_driver.py::_wall_prof` 逐条对应。分带版实测两个死结：
      ① 停死后出不来（释放条件是更近的距离，但它停住就到不了）—— 卡死 100 s；
      ② 后退带排在停住带外侧，机器人在减速带就停住，永远进不到后退带 —— 后退 0.000 s。

    ## 部署侧已先行验证该状态机可行（零训练）

    驱动侧用同一套状态机，C36b/块3 权重不变：**三条全部在 0.91 m 停住 0.61 s、
    后退 0.37 s、登顶仍 3/3**，接近速度 0.383 → 0.103。
    能力存在，缺的是把停住位置从 0.91 m 挪进 0.45~0.60 m —— 这一块就是干这个。

    **不动**：奖励项与权重、动作缩放、观测 244 / 动作 16、物理与执行器限制、地形、出生分布。

    ## 验收（`blk3_check.py`，双窗口，口径写死）

        接近全程窗 D<=1.2   停住 >=0.50 s（驱动侧已达 0.61）　后退 >=0.10 s（已达 0.37）
        官方几何窗 D<=0.6   停住 >=0.50 s　墙前最小 |vx| <=0.10　蓄力窗 >=1.50 s
        不得退化            登顶 3/3；判据 D 跟随支撑仍在 24~28%

    **触发**：登顶 <2/3 → `timeout_s` 4.0→2.0 放宽后重跑；
    官方几何窗仍为 0 而全程窗达标 → 说明卡在 enter_dist=0.70 这条线上，
    下一块降 `enter_dist` 0.70→0.55 让停住点跟着前移。

    环境变量：S10_SEQ_STOP_HOLD_S / S10_SEQ_STOP_V / S10_SEQ_STOP_TIMEOUT。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_cmd as _cc
        _os.environ.setdefault("S10_SEQ_STOP_HOLD_S", "0.4")
        _os.environ.setdefault("S10_SEQ_STOP_V", "0.10")
        _os.environ.setdefault("S10_SEQ_STOP_TIMEOUT", "4.0")
        old = self.commands.base_velocity
        new = _cc.ClimbApproachVelocityCommandCfg()
        for k, v in vars(old).items():
            if k == "class_type":
                continue
            if hasattr(new, k):
                setattr(new, k, v)
        new.d_slow_hi = 0.85; new.d_stop_hi = 0.68
        new.v_stop = float(_os.environ["S10_SEQ_STOP_V"])
        new.hold_s = float(_os.environ["S10_SEQ_STOP_HOLD_S"])
        new.timeout_s = float(_os.environ["S10_SEQ_STOP_TIMEOUT"])
        new.back_s = float(_os.environ.get("S10_BLK3C_BACK_S", "0.4"))
        new.back_v = float(_os.environ.get("S10_BLK3_BACK", "-0.15"))
        new.debug_every = int(_os.environ.get("S10_BLK3_DBG", "0"))
        self.commands.base_velocity = new
        print(f"[s10-climb-Blk3c] 停住后才起相位: hold={new.hold_s:g}s v_stop={new.v_stop:g} "
              f"timeout={new.timeout_s:g}s | 命令状态机 接近{new.d_slow_hi:.2f}->停住{new.d_stop_hi:.2f}"
              f"->后退{new.back_s:g}s@{new.back_v:+.2f}->释放 | 相位进入距离 0.70 不动 | "
              f"靶 全程窗停住>=0.50s 后退>=0.10s，登顶不掉", flush=True)


@configclass
class DeeproboticsS10ClimbBlk5EnvCfg(DeeproboticsS10ClimbBlk3cEnvCfg):
    r"""块5（09-16 18:5x）：**停 → 退 → 才起动作**。修块3c 剩下的「后退没发生」。

    ## 块3c 成绩与剩余差距

    | 判据 | C36b | 块3c | 官方/靶 |
    |---|---|---|---|
    | 登顶 | 3/3 | **3/3** | 不掉 |
    | 墙前最小 \|vx\| | 0.721 | **0.056** ✅ | ≤0.10 |
    | 蓄力窗 | 0.470 | **0.965** | 1.80~2.48 |
    | 停住位置 | — | **0.633 m** | 0.45~0.60 |
    | 接近 \|vx\| | 0.383 | **0.109** | 慢 |
    | 后膝左右不对称 | 0.648 | **0.135** ✅ | 0.111~0.195 |
    | **后退** | 0.000 | **0.010 s** ❌ | 0.405 s |

    ## 单一机制：相位门从「已停住」改成「命令剖面已释放」

    块3c 的门是「连续停住 0.4 s」。停够就开门 → 上墙序列启动 → 机器人往前冲，
    命令里那段 −0.15 的后退被同一个「见墙就冲」反射压掉，所以后退只有 0.010 s。

    官方次序是 **停 → 退 → 才起动作**：ref_climb34 seg0 的连续后退 0.405 s 发生在
    vx≈+0.02 的停住段**之内**，动作是之后才起的。

    本块把门换成「命令状态机已走到 ST_DONE」（= 停住 hold_s + 后退 back_s 都完成），
    等于把后退段整个放到相位起跑之前，**同时把蓄力窗再拉长 back_s 秒**（0.965 → 约 1.37）。
    超时 `S10_SEQ_STOP_TIMEOUT` 仍兜底防死锁。

    时序说明：命令项在 `command_manager.compute`(278) 把状态写到 `env._climb_cmd_state`，
    奖励侧在 `reward_manager.compute`(252) 读，**滞后 1 步（20 ms）**。门只关心"是否已完成"
    这个单向事件，滞后无害，但不得当成同步。

    **不动**：奖励项与权重（含 `cc_rear_sym.w_knee` 仍为 0.5）、动作缩放、观测 244 / 动作 16、
    物理与执行器限制、地形、出生分布、命令状态机各段时长。

    ## 验收

        后退（全程窗 D<=1.2）  0.010 s → >=0.10 s
        蓄力窗                0.965 s → >=1.30 s
        不得回退              墙前最小|vx| <=0.10、后膝Δ 留在 0.111~0.195、登顶 3/3

    **触发**：登顶 <2/3 → `S10_SEQ_STOP_TIMEOUT` 4.0→2.0 重跑；
    后退仍 <0.10 s → 说明后退段本身也被反射压掉，改为在后退段加专项奖励并先验读数；
    后膝Δ 退出 0.111~0.195 → 启用块4（w_knee 0.5→2.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        _os.environ["S10_SEQ_GATE"] = "released"
        c = self.commands.base_velocity
        print(f"[s10-climb-Blk5] 相位门=命令剖面已释放（停{c.hold_s:g}s + 退{c.back_s:g}s 都完成才起动作）"
              f" timeout={c.timeout_s:g}s | 靶 后退>=0.10s 蓄力窗>=1.30s，墙前最小|vx|与后膝Δ不得回退",
              flush=True)


@configclass
class DeeproboticsS10ClimbBlk6EnvCfg(DeeproboticsS10ClimbBlk5EnvCfg):
    r"""块6（09-16 19:1x）：**把停住时间拉长**。一次同时打「蓄力窗」与「后腿对称」。

    ## 三个点显示后膝对称跟着蓄力窗长度走

    | | 蓄力窗（0.6→0.15 m） | 后膝左右不对称 | 官方 |
    |---|---|---|---|
    | C36b | 0.470 s | 0.648 | — |
    | 块3c | **0.965 s** | **0.135** ✅ | 1.80~2.48 s / 0.111~0.195 |
    | 块5 | 0.590 s | 0.648 ❌ | — |

    窗长腿就对称，窗短就回到一前一后。机理与 §39.6d 一致：不对称是**恒同号、0 次过零**
    的结构性偏差，本质是「到墙前时正卡在小跑相位的一前一后，且没时间调整」——
    给够时间它自己会蹲平，不需要额外加罚。

    **这是三点相关，不是定论**。但蓄力窗本来就是落后最多的一项（0.59 vs 官方 1.80~2.48），
    先延长它，比直接提 `cc_rear_sym.w_knee` 更治本，也不冒过冲风险。

    > 因此**改了 §39.8d 预写的触发**（原为「后膝退出区间 → 启用块4」）。
    > 块4 仍保留：若本块把蓄力窗拉长而后膝**仍**不回来，那才说明时间不是主因，再提 w_knee。

    ## 单一机制：`hold_s` 0.4 → 1.0 s

    命令状态机的停住段从 0.4 s 延长到 1.0 s（后退段 `back_s` 0.4 s 不动）。
    相位门仍是「剖面已释放」，于是相位起跑前的静止段变成 1.0 + 0.4 = **1.4 s**。
    官方在 0.60~0.45 m 处 vx 只有 +0.006~+0.035，持续约 2 s —— 本块朝这个方向走一步。

    `timeout_s` 4.0 不动（1.4 s 远小于它，不会被兜底提前打断）。

    **不动**：奖励项与权重（`cc_rear_sym.w_knee` 仍 0.5）、后退段时长与速度、动作缩放、
    观测 244 / 动作 16、物理与执行器限制、地形、出生分布。

    ## 验收

        蓄力窗        0.590 s → >=1.00 s（官方 1.80~2.48）
        后膝Δ         0.648 → 回到 0.111~0.195
        不得回退      后退 >=0.10 s、停住 >=0.40 s、墙前最小|vx| <=0.10、登顶 3/3

    **触发**：登顶 <2/3 → `hold_s` 1.0→0.7 重跑；
    蓄力窗上去但后膝Δ 仍 >0.30 → 时间不是主因，启用块4（w_knee 0.5→2.0）；
    两者都不动 → 停，等作者定方向。

    环境变量：S10_BLK6_HOLD_S（默认 1.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        hs = float(_os.environ.get("S10_BLK6_HOLD_S", "1.0"))
        c = self.commands.base_velocity
        old = c.hold_s
        c.hold_s = hs
        _os.environ["S10_SEQ_STOP_HOLD_S"] = str(hs)
        print(f"[s10-climb-Blk6] 停住段 {old:g}s -> {hs:g}s（后退 {c.back_s:g}s 不动，"
              f"相位起跑前静止 {hs + c.back_s:g}s，timeout {c.timeout_s:g}s）| "
              f"靶 蓄力窗>=1.00s 且 后膝Δ 回到 0.111~0.195，后退/停住/登顶不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk7EnvCfg(DeeproboticsS10ClimbBlk6EnvCfg):
    r"""块7（09-16 19:3x）：**把停住+后退整段挪进官方窗内**。

    ## 块6 证伪了什么、没证伪什么

    块6 把 `hold_s` 0.4→1.0，结果蓄力窗 0.590 → **0.580（没动）**、后膝Δ 0.648 → **0.638（没动）**。

    **被证伪的是「延长 hold_s 就能拉长蓄力窗」**，原因是位置：

        蓄力窗定义 = 在 D∈(0.15, 0.60] 的停留时长
        块6 的停住位置 = **0.671 m** —— 在窗外

    多停的 0.6 s 一秒都没算进窗内。**「蓄力窗长度 ↔ 后膝对称」这个假设本身并没有被检验**，
    因为这一块压根没把窗拉长。不能说它错，只能说没测到。

    ## 停住点被后退一路往外推

    | | 停住位置 | 蓄力窗 | 后膝Δ | 墙前最小 \|vx\| |
    |---|---|---|---|---|
    | 块3c（无有效后退） | **0.633 m** | **0.965** | **0.135** | **0.056** |
    | 块5（后退 0.105 s） | 0.652 m | 0.590 | 0.648 | 0.099 |
    | 块6（后退 0.410 s） | **0.671 m** | 0.580 | 0.638 | 0.491 |

    后退越长，停住点越往外推，蓄力窗与 vmin 一起垮。**在当前位置，后退和蓄力窗互相打架。**

    ## 单一机制：停住带整体前移

        d_slow_hi  0.85 -> 0.72     减速起点
        d_stop_hi  0.68 -> 0.52     停住带上沿

    目标是让机器人在 **0.52 m 附近停住、退到约 0.57 m**，**退完仍在官方窗 0.45~0.60 内**，
    于是 1.0 s 停住 + 0.4 s 后退共 1.4 s 全部计入蓄力窗（0.58 → 约 1.5~1.9，逼近官方 1.80~2.48）。

    能不能停在 0.52 m？块3c/块5/块6 已把可控区从 C36b 的 0.75 m 推进到 0.63~0.67 m
    （C36b 在 0.75 m 以内指令跟踪误差 0.48~0.68，现已能在 0.65 m 停住），再推 0.13 m 是合理的一步。

    **不动**：hold_s 1.0 / back_s 0.4 / timeout 4.0、奖励项与权重、动作缩放、
    观测 244 / 动作 16、物理与执行器限制、地形、出生分布、相位门（仍为「剖面已释放」）。

    ## 验收

        停住位置   0.671 m → 落进 0.45~0.60
        蓄力窗     0.580 s → >=1.00 s
        墙前最小|vx| 0.491 → <=0.10
        不得回退   后退 >=0.35 s（块6 已达 0.410）、登顶 3/3

    **触发**：登顶 <2/3 → `d_stop_hi` 0.52→0.58 回退半步重跑；
    停住位置进窗但后膝Δ 仍 >0.30 → **这才真正证伪「窗长↔对称」**，届时启用块4（w_knee 0.5→2.0）；
    停不到 0.52 m（位置仍 >0.62） → 说明可控区推不动了，停，等作者定方向。

    环境变量：S10_BLK7_SLOW_HI（默认 0.72）、S10_BLK7_STOP_HI（默认 0.52）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        c = self.commands.base_velocity
        o1, o2 = c.d_slow_hi, c.d_stop_hi
        c.d_slow_hi = float(_os.environ.get("S10_BLK7_SLOW_HI", "0.72"))
        c.d_stop_hi = float(_os.environ.get("S10_BLK7_STOP_HI", "0.52"))
        assert c.d_slow_hi > c.d_stop_hi, "减速起点必须大于停住带上沿"
        print(f"[s10-climb-Blk7] 停住带前移 减速{o1:.2f}->{c.d_slow_hi:.2f} 停住{o2:.2f}->{c.d_stop_hi:.2f}"
              f"（停{c.hold_s:g}s+退{c.back_s:g}s 共 {c.hold_s + c.back_s:g}s 应全部落进官方窗 0.45~0.60）| "
              f"靶 停住位置进窗、蓄力窗>=1.00s、墙前最小|vx|<=0.10，后退与登顶不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk8EnvCfg(DeeproboticsS10ClimbBlk7EnvCfg):
    r"""块8（09-16 20:0x）：**把「停住」的判定阈值对齐到验收判据**。

    ## 块7 的成绩与那一项不可复现的判据

    块7 首次让四项同时达标，但 n=3 太薄。加跑三条独立确认后（合并 n=6）：

    | | n | ①vmin | **②停住** | ③蓄力窗 | ④后退 | 后膝Δ | 后髋Δ |
    |---|---|---|---|---|---|---|---|
    | 首批 | 3 | 0.000 | **1.010** | 2.225 | 0.330 | 0.187 | 0.152 |
    | 确认批 | 3 | 0.003 | **0.330** | 1.665 | 0.285 | 0.178 | 0.162 |
    | **合并** | **6** | 0.002 ✓ | **0.333 ✗** | 1.698 ✓ | 0.302 ✓ | 0.181 ✓ | 0.152 ✓ |

    登顶 6/6、翻越后站住 6/6、停住位置 0.58 m（官方 0.45~0.60）。
    **②停住两批差 3 倍，合并后 0.333 s 不达标（靶 ≥0.50）。**

    ## 根因：机制的「停住」定义比判据松一倍

        状态机    still = |vx| < v_stop，**v_stop = 0.10**
        验收判据  停住   = |vx| < **0.05** 的最长连续时长

    机器人蹭到 0.09 m/s 就被状态机算作"停住"，满 hold_s 即放行 —— 它**没有义务停到 0.05 以下**，
    于是批次间在 0.05 这条线上下浮动，判据时有时无。这是**定义不一致**，不是训练不足。

    ## 单一机制：`v_stop` 0.10 → 0.04

    把状态机的判定阈值压到判据之下（0.04 < 0.05），机器人必须真正停到 0.04 m/s 以下
    才开始计 hold_s，否则计时清零。`timeout_s` 4.0 不动 —— 停不下来仍会兜底放行，保住登顶。

    顺带预期：停得更深 → hold 段更实 → 蓄力窗从 1.698 向官方 1.80~2.48 靠。

    **不动**：停住带位置（0.72/0.52）、hold_s 1.0 / back_s 0.4 / timeout 4.0、奖励项与权重、
    动作缩放、观测 244 / 动作 16、物理与执行器限制、地形、出生分布、相位门。

    ## 验收（**n=6**，不再用 n=3 下结论）

        ②停住     0.333 s → >=0.50 s，且两批差异 <2 倍
        ③蓄力窗   1.698 s → >=1.80 s（官方下沿）
        不得回退  ①vmin <=0.10、④后退 >=0.25 s、后膝Δ 与后髋Δ 留在官方区间、登顶 6/6

    **触发**：登顶 <5/6 → `v_stop` 0.04→0.07 回退半步；
    ②仍 <0.50 且两批差异仍 >2 倍 → 不是阈值问题，改查台面/轮子残余速度来源；
    ②达标但落地仍差（台面零指令前漂 >0.08 m）→ 转做落地专项（作者第五条，尚未训过）。

    环境变量：S10_BLK8_VSTOP（默认 0.04）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        vs = float(_os.environ.get("S10_BLK8_VSTOP", "0.04"))
        c = self.commands.base_velocity
        old = c.v_stop
        c.v_stop = vs
        _os.environ["S10_SEQ_STOP_V"] = str(vs)
        print(f"[s10-climb-Blk8] 停住判定阈值 {old:g} -> {vs:g} m/s（验收判据是 0.05，机制须严于判据）"
              f" | 带 {c.d_slow_hi:.2f}/{c.d_stop_hi:.2f} 停{c.hold_s:g}s 退{c.back_s:g}s 超时{c.timeout_s:g}s 不动 | "
              f"靶 n=6 下 ②停住>=0.50s、③蓄力窗>=1.80s，登顶 6/6 不掉", flush=True)


@configclass
class DeeproboticsS10ClimbBlk9EnvCfg(DeeproboticsS10ClimbBlk8EnvCfg):
    r"""块9（09-16 20:3x）：**停住段里让后腿真正蹲下去**。作者看回放后点名的三条。

    > 「首先对的不够近」「后退蓄力不对称，你和官方的对比啊」
    > 「应该是正常平地姿态走到墙前 然后抬头 后退弯曲蓄力」

    ## 缺口是我自己造出来的

    块3c 把相位门改成「停住后才起动作」，块5 又改成「停住+后退完成才起动作」。
    于是**停住蓄力那一整段里七个 `cs_*` 参考跟踪项全是关着的**（相位尚未起跑），
    **没有任何奖励在要求后腿弯曲、对称或抬头** —— 机器人只是站着等放行。
    官方恰恰相反：深蹲与抬头**就发生在停住段之内**。

    ## 干净窗口实测（D 0.80 → 起动瞬间，n=3）

    | | 官方 seg0/1/2 | 块7 |
    |---|---|---|
    | 起动处离墙 | **0.355~0.497 m** | **0.587~0.597 m** ❌ 远 0.1~0.24 m |
    | 窗内前进 | 0.082~0.128 m（边停边蹭近） | ≈0（停住不动） |
    | 机身俯仰 | 1.2~2.7° → **5.6~7.4°**（单调抬头） | −4.4 → −2.1°（**全程低头**）❌ |
    | **后膝深度** | **−1.43 ~ −1.47** | **−0.72 ~ −0.91** ❌ 只弯一半 |
    | \|Δhipx\| 均 | 0.036~0.073 | **0.157** ❌ |
    | \|Δknee\| 均 | 0.041~0.072 | **0.153~0.181** ❌ |
    | \|Δhipy\| 均 | 0.102~0.165 | 0.016~0.028 ✅ 我方更好 |

    > **撤回**：§39.8k 说过「后膝/后髋对称已进官方区间」。那是在 D∈(0.15,0.60] 量的，
    > 该窗口把爬墙过程也算了进去。换成作者眼睛看的那一段（起动之前），膝与 hipx 差 2~4 倍。
    > **而且 hipx 我从头到尾没量过** —— 作者一眼看出的不对称正在这一项上。
    > 见 [[feedback-visual-over-metric]]：聚合指标又一次被目视证伪。

    ## 单一机制：新增 `cc_load_pose`

    门：遭遇 & 命令剖面处于停住/后退段 & 相位尚未起跑。
    值：两条后膝到 **−1.45**（官方深度）的绝对误差之和，作为惩罚。

    **两条腿同一个目标，对称是自动的** —— 不再单独加对称罚，`cc_rear_sym` 保持 w_knee=0.5 不动。
    预期连带：后腿蹲下去 → 机身后低前高 → **抬头**（作者第二条）。

    **不动**：其余奖励项与权重、命令状态机各段、停住带位置、相位门、动作缩放、
    观测 244 / 动作 16、物理与执行器限制、地形、出生分布。

    ## 验收（n=6）

        后膝深度（干净窗口）  −0.80 → <=−1.25（官方 −1.43~−1.47）
        |Δknee| 均            0.153~0.181 → <=0.10（官方 0.041~0.072）
        机身俯仰（起动处）    −2.1° → >=+2°（官方 +5.6~7.4°）
        不得回退              ②停住、④后退、①vmin、登顶 6/6

    **触发**：登顶 <5/6 → 权重减半重跑；
    膝到位但俯仰仍为负 → 抬头另有成因，单开一块查前腿；
    膝不动 → 权重不够或门没开，先查 `Episode_Reward/cc_load_pose` 读数是否非零。

    环境变量：S10_BLK9_W（权重，**开训前必须先验读数定档**）、S10_BLK9_KNEE（默认 −1.45）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        # ---- 同时修一个致命 bug：相位门超时 4.0 s 在训练里永远攒不满 ----
        # 打点实测（1024 env）：遭遇一次只有 1.08~1.48 s，状态机全部卡在 ST_STOP
        # （st分布 [864,160,0,0]，ST_DONE 0.00%、超时 0.00%、ready 0.00%），相位门从不开，
        # 于是**块5~块8 全程 `cs_*` 参考跟踪 = 0.0000，上墙参考训练是死的**。
        # 评测仍 6/6 是因为继承了 C36b 的权重还会爬，但已不再学习/维持官方形态。
        # 超时降到 0.8 s（短于遭遇时长）后先验读数立刻恢复：
        #   cs_all12 0 -> -0.1910（基线 -0.1177）  cs_fr_knee 0 -> -0.1671（基线 -0.0890）
        #   climb_complete 0.0104 -> 0.0521（基线 0.0535）  st分布 [887,30,19,88] 四态流转
        # 蓄力姿态改由本块的 cc_load_pose 直接管，不再靠"饿死相位"来间接实现。
        # **两个超时作用不同，必须分开**（09-16 20:5x 修）。块9 首轮我把两个一起设成 0.8，
        # 而命令状态机的 hold_s = 1.0 > 0.8 → 「停够 1 秒」永远不可能先满足，每次都是超时先触发，
        # 机器人在剖面里待 0.8 s 就被放行、根本没停。实测代价：停住 1.138 → **0.000 s**、
        # 蓄力窗 2.755 → 0.590、后退 0.323 → 0.000、后膝Δ 0.150 → 0.591。
        #   · 命令状态机 timeout_s：防死锁上限，必须 > hold_s + back_s(=1.4)，保持 4.0 不动；
        #   · 相位门 S10_SEQ_STOP_TIMEOUT：相位最晚何时起跑，取 0.8（短于遭遇时长 1.08~1.48 s），
        #     让 cs_* 参考跟踪活过来。相位在停住期间起跑是**对的** —— 官方正是「停着做蓄力动作」。
        _os.environ["S10_SEQ_STOP_TIMEOUT"] = _os.environ.get("S10_BLK9_GATE_TIMEOUT", "0.8")
        assert self.commands.base_velocity.timeout_s > (
            self.commands.base_velocity.hold_s + self.commands.base_velocity.back_s), \
            "命令状态机超时必须大于 hold_s+back_s，否则停住永远完不成"
        # ---- 门的阈值必须落在**训练可达**范围内，否则是死核、无梯度 ----
        # 打点实测（1024 env，块8 权重）：停住段 |vx| 中位 **0.078~0.100 m/s**，
        # 而块8 把 v_stop 定在 0.04 → 够不到 → hold 永远清零 → 状态机困在 ST_STOP
        # （st分布 [944,80,0,0]，ST_DONE 0.00%、后退段 0.00%、已释放 0.00%）。
        # 部署侧能到 0.000 是因为那是干净台架；训练侧有推力/摩擦/质量随机化与地形变化。
        # **门的职责是排序，不是逼深度**：深度交给「命令 0 的跟踪奖励」与 cc_load_pose 连续地压。
        # 另：遭遇一次只有 0.86~1.74 s，停住+后退必须能塞进去，故 hold 1.0→0.5、back 0.4→0.3。
        c = self.commands.base_velocity
        c.v_stop = float(_os.environ.get("S10_BLK9_VSTOP", "0.12"))
        c.hold_s = float(_os.environ.get("S10_BLK9_HOLD_S", "0.5"))
        c.back_s = float(_os.environ.get("S10_BLK9_BACK_S", "0.3"))
        w = float(_os.environ.get("S10_BLK9_W", "12.0"))
        kt = float(_os.environ.get("S10_BLK9_KNEE", "-1.45"))
        self.rewards.cc_load_pose = RewTerm(func=_cr.wall_load_pose, weight=-w,
                                            params={"knee_target": kt})
        print(f"[s10-climb-Blk9] cc_load_pose w=-{w:g} 后膝目标 {kt:g} | 门阈值改为训练可达: "
              f"v_stop={c.v_stop:g} hold={c.hold_s:g}s back={c.back_s:g}s 状态机超时={c.timeout_s:g}s "
              f"相位门超时={_os.environ['S10_SEQ_STOP_TIMEOUT']}s | "
              f"靶 后膝<=-1.25、|Δknee|<=0.10、起动处俯仰>=+2°，且 cs_* 必须非零", flush=True)


@configclass
class DeeproboticsS10ClimbBlk10EnvCfg(DeeproboticsS10ClimbBlk9EnvCfg):
    r"""块10（09-16 21:4x）：**让机身仰起来**。作者三条批评里最刺眼的那一条。

    ## 官方蓄力段的真相（此前我整晚都理解错了）

    「后膝首次到 −1.2」→「后膝最深」这一段，三段一致：

    | | seg0 | seg1 | seg2 |
    |---|---|---|---|
    | 时长 | 2.16 s | 1.99 s | 1.58 s |
    | 离墙 | 0.579→**0.327** | 0.592→**0.341** | 0.482→**0.243** |
    | 推进 | 0.251 m | 0.251 m | 0.239 m |
    | vx 中位 | +0.022 | +0.036 | +0.102 |
    | 后膝 | −1.375→**−2.104** | −1.447→**−2.116** | −1.398→**−2.016** |
    | **机身俯仰** | +2.3°→**+41.6°** | +2.7°→**+41.1°** | +1.2°→**+35.9°** |

    **官方不是「停死」，是一边以 0.02~0.10 m/s 蹭近 0.25 m、一边蹲到 −2.1、同时仰起到 +41°。**
    我把它理解成「停住」，于是设计了一整套「停死」机制 —— 机器人停在 0.58 m 再也不动，
    正是作者说的「对的不够近」。

    > **撤回一次错误的撤回**：我曾把 41° 判为窗口伪影并撤回。**那次撤回是错的，41° 是真的**，
    > 它就是作者说的「抬头」。

    ## 我方差距（块8/块9，n=6）

    | | 块8 | 块9 | 官方 |
    |---|---|---|---|
    | 最深后膝 | −1.718 | −1.743 | **−2.01~−2.12** |
    | 发生在离墙 | 0.462 | 0.566 | **0.24~0.34** |
    | **该处俯仰** | **−16.1°** | **−22.7°** | **+41°** |

    **机身姿态差 57~64 度** —— 官方在仰头 41°，我方在低头 16~23°。

    ## 本块两件事

    1. **修 bug**：`cc_load_pose` 目标 −1.45 → **−2.05**。
       −1.45 取自官方 t=0 时刻的中位，**比我方已有的 −1.72 还浅**，该项一直在把膝往上拉，
       方向是反的 —— 块9 变差，这一项要负责任。
    2. **单变量**：蹭近段改为**按距离结束** —— 发 `creep_v=+0.08 m/s` 的慢速前进指令，
       一直走到离墙 `d_creep_end=0.32 m` 才起动作（旧逻辑是「停够 hold_s 就放行」，与距离无关）。

       > 放弃了先想的「提 `cs_pitch` 权重」：先验读数显示它每步只罚 0.29，而 `cc_load_pose`
       > 每步 7.9、主项 6.7，要同量级得 w≈50（17 倍跳）；且长期笔记记着 `cc_pitch_hold`
       > 曾被刷到 1000 倍，直接压俯仰易逼出「原地翘着」的假动作。
       > **+41° 是结果不是原因** —— 官方仰得起来是因为已经蹭到 0.24~0.34 m、前轮搭上墙面了。

    > **更正**：我一度说「我方停死不动、完全不蹭近」。用正确窗口量，**块8 蹭近推进 0.228 m、
    > vx 0.041 m/s，与官方 0.239~0.251 / 0.022~0.102 基本一致** —— 它会蹭。
    > 真正的缺口是**整段位置往外偏 0.12~0.20 m**（官方 0.48~0.59→0.24~0.34，块8 0.69→0.46）。

    **起点从块8 续**（其部署行为最好：停住 1.138 s、蓄力窗 2.755 s、后退 0.323 s、登顶 6/6），
    块9 的门修复与 `cc_load_pose` 机制保留。

    **不动**：命令状态机各段与阈值、相位门超时、其余奖励项与权重、动作缩放、
    观测 244 / 动作 16、物理与执行器限制、地形、出生分布。

    ## 验收（n=6）

        最深处俯仰    −16.1° → >=+10°（官方 +36~+42）
        最深后膝      −1.718 → <=−1.90（官方 −2.01~−2.12）
        最深处离墙    0.462 → <=0.40（官方 0.24~0.34）
        cs_* 非零     一票否决
        不得回退      ②停住 >=0.50、④后退 >=0.25、登顶 6/6

    **触发**：登顶 <5/6 → `cs_pitch` 权重减半重跑；
    俯仰起来但离墙仍 >0.45 → 专开一块治「蹭近」（命令改为 +0.05 m/s 慢速推进而非 0）；
    俯仰不动 → 不是权重问题，查 `SeqPitch` 的 active 占比与参考取帧。

    环境变量：S10_BLK10_PITCH_W（先验定档）、S10_BLK10_KNEE（默认 −2.05）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        # ① 修 bug：cc_load_pose 目标 −1.45 取自官方 t=0 的中位，**比我方已有的 −1.72 还浅**，
        #    该项一直在把膝往上拉、方向是反的。官方蓄力峰值是 −2.02~−2.12。
        kt = float(_os.environ.get("S10_BLK10_KNEE", "-2.05"))
        self.rewards.cc_load_pose.params["knee_target"] = kt
        # ② 单变量：蹭近段按**距离**结束，走到离墙 d_creep_end 才起动作。
        c = self.commands.base_velocity
        c.creep_v = float(_os.environ.get("S10_BLK10_CREEP_V", "0.08"))
        c.d_creep_end = float(_os.environ.get("S10_BLK10_CREEP_END", "0.32"))
        print(f"[s10-climb-Blk10] 蹭近段: 指令 {c.creep_v:+g} m/s，走到离墙 {c.d_creep_end:g} m 才起动作"
              f"（官方 vx 中位 0.022~0.102、最深处 0.243~0.341）| cc_load_pose 目标 -1.45 -> {kt:g}"
              f"（官方峰值 −2.02~−2.12，旧值比我方已有的 −1.72 还浅、方向反了）| "
              f"靶 E离墙<=0.40 F俯仰>=+10° D最深膝<=-1.90，A/B/C 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk11EnvCfg(DeeproboticsS10ClimbBlk10EnvCfg):
    r"""块11（09-16 22:1x）：**给蹭近段封顶**，别磨。

    ## 块10 拿下两条，但磨太久

    | | 官方 | 块8 | 块10(n=6) |
    |---|---|---|---|
    | **D 最深后膝** | −2.02~−2.12 | −1.718 ✗ | **−1.944 ✓** |
    | **E 最深处离墙** | 0.243~0.341 | 0.462 ✗ | **0.322 ✓** |
    | A 时长 | 1.58~2.16 | 3.29 | **7.655 ✗** |
    | B 推进 | 0.239~0.251 | 0.228 | 0.380 |
    | C vx 中位 | 0.022~0.102 | 0.041 ✓ | **0.014 ✗** |
    | F 俯仰 | +36~+42 | −16.1 | −10.6 ✗ |
    | G Δknee/Δhipx | 0.04~0.07 | 0.176/0.117 | **0.273/0.347 ✗** |
    | 登顶 | — | 6/6 | **5/6 ✗** |

    「蹭近按距离结束」把 D、E 拉进官方区间 —— 作者的「对的不够近」「弯曲蓄力」两条实现了。
    代价是**段长 7.655 s，官方的 3.5 倍**：以 0.014 m/s 龟速爬 0.38 m。
    这一拖，对称性垮（0.273/0.347）、登顶掉到 5/6。

    深蹲（后膝 −1.94）本身会让轮子难滚，所以「蹲得深」与「蹭得动」是有张力的；
    官方的做法是**只蹭约 2 s 就爆发**（vx 峰 +0.79~+1.18），不是一路磨。

    ## 单一机制：蹭近段 `timeout_s` 4.0 → 2.2 s

    段的结束条件本就是 `(sd <= d_creep_end) | (tmr >= timeout_s)`，把上限压到官方段长
    （1.58~2.16 s）的上沿。到点就放行，不再磨。断言 `timeout_s > hold_s + back_s`
    （2.2 > 0.5+0.3）仍满足。

    **不动**：`creep_v` 0.08、`d_creep_end` 0.32、`cc_load_pose` 目标 −2.05、
    减速带位置、相位门超时、其余奖励项与权重、动作缩放、观测/动作维度、物理限制、地形、出生分布。

    ## 验收（n=6）

        A 段长      7.655 → 1.2~3.0 s
        G 对称      0.273/0.347 → <=0.10/<=0.09（至少不差于块8 的 0.176/0.117）
        登顶        5/6 → 6/6
        不得回退    D 最深后膝 <=-1.90、E 最深处离墙 <=0.40（块10 刚拿下，不能丢）

    **触发**：登顶仍 <6/6 → `d_creep_end` 0.32→0.38 退半步（少蹲一点换稳）；
    A 合格但 F 俯仰仍 <0 → 抬头另有成因，单开一块查前腿有没有搭上墙面；
    G 仍差 → 查 `cc_load_pose` 是否把两腿压得不一样深（它按单腿误差求和，理论上对称）。

    环境变量：S10_BLK11_TIMEOUT（默认 2.2）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        c = self.commands.base_velocity
        old = c.timeout_s
        c.timeout_s = float(_os.environ.get("S10_BLK11_TIMEOUT", "2.2"))
        assert c.timeout_s > (c.hold_s + c.back_s), "状态机超时必须大于 hold_s+back_s"
        print(f"[s10-climb-Blk11] 蹭近段封顶 {old:g}s -> {c.timeout_s:g}s（官方段长 1.58~2.16；"
              f"块10 磨了 7.655 s，对称垮到 0.273/0.347、登顶掉到 5/6）| "
              f"靶 A段长 1.2~3.0s、G对称<=0.10/0.09、登顶 6/6，D/E 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk12EnvCfg(DeeproboticsS10ClimbBlk11EnvCfg):
    r"""块12（09-16 22:4x）：**治右后腿顶在限位上的侧向外张**。作者看回放图2 点名的关节。

    ## 实测（最深蓄力帧，n=6）

        官方   hl_hipx −0.047~−0.181   hr_hipx −0.038~−0.094
        块8    hr_hipx **+0.643**   块10 **+0.617**   块11 **+0.618**
        hipx 限位 ±0.6109 → **已顶到限位饱和**，差官方 3~13 倍

    `hind_splay_penalty` 罚的是后轮**前后**拖在髋后（用 `hind_tuck`，x 坐标），
    **侧向 hipx 此前没有任何奖励在管**。

    ## 单一机制：新增 `cc_hipx_splay`

    门：遭遇 & 后轮尚未上台。值：两条后腿 `|hipx|` 超出 `lim=0.15` 的部分之和（惩罚）。
    **两腿同一上限，不编码镜像方向**。

    ## 顺带修两处我自己定错的判据

    1. **G 改为镜像无关**：原来用 `hl−hr` 直接比，但官方最深帧两后膝本就差 0.199~0.281，
       而我方是镜像方案（左前先抬/右后发力），镜像会把深浅两腿对调 ——
       直接比会把**正确的镜像算成误差**。现改为 G1=后膝深浅差（排序后相减）、G2=max|hipx|。
       改后块8 的 dknee 从 0.176 变 **0.079**（比官方还对称），块11 为 0.524。
    2. **A/B 改双边**：原来只设下限，于是 7.66 s、0.42 m 这种「磨太久／冲太远」显示为达标，
       是虚假通过。现 A 1.2~3.0 s、B 0.15~0.35 m。

    ## 块11 现状（n=6，新判据）

        A 5.28✗  B 0.415✗  C 0.032✓  D −1.978✓  E 0.367✓  F −9.9✗  G 0.524✗/0.618✗  登顶 6/6✓

    ## 验收（n=6）

        G2 max|hipx|  0.618 → <=0.20（官方 0.094~0.181）
        不得回退      D<=-1.90、E<=0.40、C 0.02~0.20、登顶 6/6

    **触发**：登顶 <6/6 → 权重减半；
    hipx 下来但 F 俯仰仍 <0 → 抬头的直接前提是**前腿收起来搭上墙面**
    （官方领先前腿膝 +2.37~+2.62，我方仅 +1.82），下一块治前腿。

    环境变量：S10_BLK12_W（先验定档）、S10_BLK12_LIM（默认 0.15）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        w = float(_os.environ.get("S10_BLK12_W", "8.0"))   # 按每步量级定：项值≈0.47/步，×8≈3.8，与主项 6.7、cc_load_pose 7.9 同级。回合读数（w=1 时 −0.0019）只反映生效占比低，不能据此定档 —— cs_pitch 那次就是这么错的。
        lim = float(_os.environ.get("S10_BLK12_LIM", "0.15"))
        self.rewards.cc_hipx_splay = RewTerm(func=_cr.rear_hipx_splay, weight=-w,
                                             params={"lim": lim})
        print(f"[s10-climb-Blk12] 新增 cc_hipx_splay w=-{w:g} lim={lim:g}"
              f"（后腿侧向外张；hr_hipx 实测 +0.618 已顶限位 0.6109，官方 0.038~0.181）| "
              f"靶 max|hipx|<=0.20，D/E/C 与登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk13EnvCfg(DeeproboticsS10ClimbBlk12EnvCfg):
    r"""块13（09-16 23:0x）：**领先前轮搭上台面**。作者点明的因果链的关键一环。

    > 作者：「官方后退是弯曲蓄力，前腿一只先抬起上去，这样整体就抬头了　这个之前你不是分析过吗？」

    分析过 —— 官方上台顺序 fr → fl → hr → hl，领先前腿早 0.400/0.375/0.505 s，
    是支点（支撑占比 82.7/80.5/85.6%），跟随腿是纯摆动腿。我之前只盯静态膝角，
    **漏了「先抬上去」这个事件本身**。

    ## 官方实测：俯仰是被前轮撑起来的

        领先前轮过台面  t=0.53/0.38/0.25   离墙 0.316/0.322/0.259   该刻俯仰 **+27.3/+26.3/+28.3°**
        上台前 0.3 s 俯仰仅 +4.4/+4.3/+4.5°  →  上台瞬间 +27°  →  再 0.3 s 到 +34~+38°
        最深蓄力在领先轮上台**之后** 0.10~0.13 s

    **不是先抬头再上去，是前轮上去把机身撑起来。**

    ## 我方进展与唯一缺口（n=6，最深蓄力帧）

    | | 块11 | 块12 | 官方 |
    |---|---|---|---|
    | 离墙 | 0.367 | **0.307 ✓** | 0.259~0.322 |
    | 机身 z | 0.317 | **0.389 ✓** | 0.391~0.409 |
    | 领先前腿膝 | +1.819 | **+2.046** | +2.37~+2.62 |
    | 后腿蓄力 | −1.978 | −1.893 | −2.02~−2.12 |
    | **俯仰** | −9.9 | **−10.8 ✗** | +26~+28（该时刻） |

    距离、机身高、后腿蓄力都到位了，**唯独前轮没真正搭上台面**。

    ## 单一机制：新增 `cc_front_onto`

    门：遭遇 & 后腿已蓄力（两后膝均值 ≤ −1.2）& 两前轮都还没上台。
    值：**较高那只**前轮轮底距「墙高记忆 + 0.02」的差额（惩罚）。
    取 `max` → **镜像无关**，哪条腿先上都算数。轮底用 `_state` 既有的 `z`，与登顶判据同源。

    ## 放下的两条（避免继续跑偏）

    - `cc_hipx_splay`（块12）**无效且追错了目标**：读数非零（−0.0096）说明门开着、罚了但压不住，
      因为 `|hipx|=0.617` 只是**最深那一帧的瞬时尖峰**，窗内均值很小，我按尖峰估权重估错了。
      更重要的是**它不是作者图2 指的东西** —— 图2 是**翻上台阶之后**的右后腿
      （实测 hr_knee −0.842 vs hl −0.500、hr_hipy +0.179 vs 标称 +0.35），是膝和髋，不是 hipx。
      该项保留（w=8，无害），台上姿态另开一块治。
    - 直接压 `cs_pitch`：**+41° 是结果不是原因**，已由本块的因果链证据确认。

    ## 验收（n=6）

        F 俯仰（最深帧）  −10.8° → >=+10°（官方该时刻 +26~+28）
        不得回退          E 离墙 <=0.40、D 最深后膝 <=-1.90、C 0.02~0.20、登顶 6/6

    **触发**：登顶 <6/6 → 权重减半；
    前轮上去了但俯仰仍 <0 → 查是不是两只前轮同时上（官方是**一只先上**，另一只晚 0.4~0.5 s）；
    前轮仍上不去 → 不是奖励问题，查蓄力末机身离墙/高度是否够前轮够到台沿。

    环境变量：S10_BLK13_W（按**每步量级**定档，不按回合读数 —— 见块12 教训）、S10_BLK13_MARGIN。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        w = float(_os.environ.get("S10_BLK13_W", "5.0"))
        # 09-17 00:15 首轮 w=20 崩了：**登顶 1/3**，机器人在墙前**停 20.6 s** 一直试着抬前轮
        # 而不去爬（蓄力窗 30.9 s）。典型的奖励陷阱 —— 给"做到某姿态"的惩罚项过大权重，
        # 策略会选择停在那里一直做，而不是完成任务。旧判据②③④ 反而全 ✓（停 20.6 s 当然"达标"），
        # **只有登顶那道硬熔断抓住了它**。
        # 预案写的是"减半"，但 1/3 不是临界差一点、是崩掉，减半到 10 大概率还崩，故直接降到 1/4。   # 按**最坏情况**定档：轮子仍在地面时项值≈0.33，×20≈6.6/步 与主项 6.7 相当；典型时刻项值仅 0.04（前轮平均只差台面 4cm），×20=0.8/步 温和。按平均值放大会让趴地时刻冲到 47/步、压过主项。
        mg = float(_os.environ.get("S10_BLK13_MARGIN", "0.02"))
        self.rewards.cc_front_onto = RewTerm(func=_cr.lead_front_onto_top, weight=-w,
                                             params={"margin": mg, "knee_lo": -1.2})
        print(f"[s10-climb-Blk13] 新增 cc_front_onto w=-{w:g} margin={mg:g}"
              f"（领先前轮搭上台面；官方过台面那刻俯仰已 +26~+28°，我方 −10.8°）| "
              f"取较高那只轮故镜像无关 | 靶 最深帧俯仰>=+10°，离墙/蓄力/登顶不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk14EnvCfg(DeeproboticsS10ClimbBlk13EnvCfg):
    r"""块14（09-16 夜）：**蹭近段起点前移**，治 A 时长／B 推进／C 速度。

    官方蹲起点 **0.48~0.59 m**，我方 **0.70 m** —— 整段起早了 0.12~0.20 m，
    于是磨 5.27 s（官方 1.58~2.16）、冲 0.474 m（官方 0.239~0.251）、速度掉到 0.014。

    **单变量**：`d_slow_hi` 0.72 → **0.58**（减速起点对齐官方蹲起点）。
    `d_creep_end` 0.32 不动 —— 它已经把 E 拉进官方区间（0.307）。

    验收：A 1.2~3.0 s、B 0.15~0.35 m、C 0.02~0.20；**E<=0.40、D<=-1.90、登顶 6/6 不得回退**。
    触发：登顶 <6/6 → `d_slow_hi` 退到 0.65；A 仍 >3.0 → 不是起点问题，查蹲深后轮子滚不动。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        c = self.commands.base_velocity
        old = c.d_slow_hi
        c.d_slow_hi = float(_os.environ.get("S10_BLK14_SLOW_HI", "0.58"))
        assert c.d_slow_hi > c.d_creep_end, "减速起点必须大于蹭近结束距离"
        print(f"[s10-climb-Blk14] 蹭近段起点 {old:.2f} -> {c.d_slow_hi:.2f}"
              f"（官方蹲起点 0.48~0.59；我方 0.70 起太早 → 磨 5.27s／冲 0.474m）| "
              f"靶 A 1.2~3.0s、B 0.15~0.35m、C 0.02~0.20，E/D/登顶不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk15EnvCfg(DeeproboticsS10ClimbBlk14EnvCfg):
    r"""块15（09-16 夜）：**台上后腿姿态**。作者看回放图2 点名的那条腿。

    实测（块7 台上稳定段）：hl knee −0.500 / hipy +0.318；**hr knee −0.842 / hipy +0.179**
    （标称 −0.65 / +0.35）。右后腿多蹲 0.34 rad、髋少转 0.14 rad ——「站住」二值判据 6/6 通过，
    但姿态不对。

    **单变量**：新增 `cc_top_posture`（四轮已上台 & 指令近零时，罚两后腿偏离标称站姿 + 左右差）。
    标称取部署 runner 的 `dof_default`，与真机同口径。

    验收：台上 Δknee 0.342 → <=0.10、hr_hipy 0.179 → 0.30~0.40；**前面七条判据与登顶不得回退**。
    触发：登顶 <6/6 或 F 俯仰回退 → 权重减半；读数为零 → 先查门（四轮上台判定），不加权重。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        w = float(_os.environ.get("S10_BLK15_W", "1.0"))
        self.rewards.cc_top_posture = RewTerm(func=_cr.on_top_posture, weight=-w,
                                              params={"knee_nom": -0.65, "hipy_nom": 0.35, "w_sym": 1.0})
        print(f"[s10-climb-Blk15] 新增 cc_top_posture w=-{w:g}"
              f"（台上后腿姿态；实测 hr knee −0.842 vs hl −0.500、hr hipy +0.179 vs 标称 +0.35）| "
              f"靶 Δknee<=0.10、hr_hipy 0.30~0.40，其余判据与登顶不得回退", flush=True)


@configclass
class DeeproboticsS10TrotPhaseEnvCfg(DeeproboticsS10PerceptV1HTrotT11EnvCfg):
    r"""Trot-Phase（09-17，作者拍板「进观测」）：**观测 244 → 246，加步态相位钟 sin/cos**。

    ## 为什么必须进观测（09-15 记录的原话）

    > 「步频对齐官方 1.10 Hz — 关闭。**相位钟在观测里不可见 → 奖励被相位平均、无梯度**。
    > 　正规解要改 244 维接口（约 1 天 + 真机风险），作者拍板不改」

    步频机制**早就齐了**（`step_rate_penalty` / `sg_swing_time` / `sg_diag_sync` /
    `knee_amp_penalty` / `lr_antiphase_reward`），缺的不是奖励，是策略**看不见自己在周期哪一相**：
    同一观测下「该抬腿」和「该落腿」无法区分，频率类奖励一平均就没梯度。
    加两维后这些既有项立刻有梯度 —— **这就是「两个新变量」**。

    ## 差距

        官方   步频 **恒 1.10 Hz**（0.5~1.9 m/s 不变）  摆动 **恒 0.35 s**  靠**步幅**提速
        我方   09-15 交付态 **3.2~3.8 Hz**（≈真机 2.3），**2~3 倍** —— 作者看到的「踏得快、不稳」
               摆动 0.11~0.15 s，步幅只有官方 40%

    ## 两处改动

    1. **观测**：末尾追加 `gait_phase`（2 维：sin/cos，1.10 Hz 开环钟）→ **246**。
       **追加在 244 之后、不插中间** → 部署侧 runner 只需末尾补两维，前 244 维排布完全不动。
    2. **奖励**：新增 `gp_swing`（`phase_swing_track`）把对角腿摆动绑到钟上 ——
       钟只是"看得见"，还得有东西把动作绑上去，否则策略没有理由跟着走。
       摆动判定复用 `step_gait._swing`，与既有步态项**同口径**。

    ## 接口影响（**真机部分不自动推**）

        训练侧 246          本块负责
        export 246          需新增 export246（本块负责，仿真验收要用）
        runner 仿真侧 246   需在末尾补两维同频钟（本块负责）
        **真机重验**        **人工**，不在夜间自动推范围内

    ## 验收

        步频      3.2~3.8 Hz → <=1.5 Hz（官方 1.10）
        摆动时长  0.11~0.15 s → >=0.28 s（官方 0.35）
        `gp_swing` 读数必须非零（否则钟没被用上，先查门不加权重）
        不得回退  膝 σ（MuJoCo 侧 >=0.03，Isaac 侧 >=0.06）—— 真在踏步的判据
                  平地 0.5/1.0 指令跟踪、坡上不漂

    环境变量：S10_GAIT_FREQ（默认 1.10）、S10_TROTP_W（`gp_swing` 权重，先验定档）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import gait_clock as _gc
        f = float(_os.environ.get("S10_GAIT_FREQ", "1.10"))
        _gc.GAIT_FREQ_HZ[0] = f
        # 观测末尾追加两维（不插中间，保证前 244 维排布不变）
        self.observations.policy.gait_phase = ObsTerm(func=_gc.gait_phase_obs,
                                                     params={"freq": f}, scale=1.0)
        w = float(_os.environ.get("S10_TROTP_W", "1.0"))
        self.rewards.gp_swing = RewTerm(func=_gc.phase_swing_track, weight=-w,
                                        params={"freq": f, "h_min": 0.04, "v_min": 0.3})
        print(f"[s10-trot-Phase] 观测 244 -> **246**（末尾追加 sin/cos 相位钟 {f:g} Hz）| "
              f"新增 gp_swing w=-{w:g}（对角摆动绑钟）| "
              f"靶 步频 3.2~3.8 -> <=1.5 Hz、摆动 0.11~0.15 -> >=0.28 s，膝σ 不得跌破 0.06(Isaac)",
              flush=True)


@configclass
class DeeproboticsS10ClimbBlk16EnvCfg(DeeproboticsS10ClimbBlk15EnvCfg):
    r"""块16（09-17 08:0x）：**把两条后腿的蹲深拉到一起**。夜跑逐关节诊断后定位的主因。

    ## 夜跑三块（13/14/15）逐关节实测，最深蓄力帧，n=6

    | | 块14 | 块15 | 官方（镜像对应） |
    |---|---|---|---|
    | 深腿膝 hr | −2.083 | **−2.340** | −2.24 ✓ 匹配 |
    | **浅腿膝 hl** | **−1.405** | **−1.380** | −1.97 ✗ **差 0.59** |
    | **hr_hipx** | **+0.621** | **+0.619** | +0.047 ✗ **顶限位 0.6109** |
    | 领先前腿膝 fl | **+2.258** | +1.921 | +2.37 ✓ 接近 |
    | **机身 z** | **0.402** | **0.396** | 0.391~0.409 **✓ 已达标** |
    | 俯仰 | −7.3 | −4.3 | +36~+42 |

    ### 由此推翻块13 的立论

    机身高度与领先前腿**都已到位**，所以「前轮够不到台面所以仰不起来」**不成立**。
    块13（`cc_front_onto`）因此两个极端都不对：w=20 时机器人在墙前停 20.6 s 一直试着抬轮、
    登顶崩到 1/3；w=5 时几乎无作用、俯仰反而退到 −12.3。**该项立论错误，不再加码。**

    ### 真正的病灶

    **两条后腿在做完全不同的事**：一条 −2.34（比官方还深），另一条 −1.38（官方对应腿 −1.97），
    差 **0.96**；官方只差 **0.27**。机身一边高一边低，自然仰不起来。

    `cc_load_pose` 给两腿同一目标 −2.05，理论上该对称，实际一条过一条欠，
    **每步白付 ~11.5 的罚仍不改** —— 要么别处收益更大，要么那条腿压根控制不了
    （hipx 动作缩放仅 0.125，顶到限位多半是被接触力推出去的）。

    > **并撤回块12 的判断**：我当时说 `hr_hipx` 只是「最深帧的瞬时尖峰、不是作者图2 指的东西」
    > 就放掉了。**那个判断是错的** —— 它从块8 到块15 一直是 0.617~0.643、**纹丝不动的限位饱和**。

    ## 单一机制：`cc_rear_sym.w_knee` 0.5 → 2.0

    即启用此前搁置的**块4**。当时以为对称已达标才停掉，那个结论后来已撤回；
    现在 G 膝深浅差 **0.96 vs 官方 0.199~0.281**，其立论重新成立。

    `cc_rear_sym` 比的是 `hl−hr`（不含镜像），官方本身有 0.27 的差，
    所以**不追求压到 0**，靶设在官方上沿附近。

    ## 验收（n=6）

        G 膝深浅差   0.960 → <=0.35（官方 0.199~0.281）
        浅腿膝       −1.38 → <=−1.75（官方对应腿 −1.92~−1.98）
        F 俯仰       −4.3 → 继续上行（不设硬靶，看是否随对称改善）
        不得回退     E 离墙 <=0.40、机身 z 0.39~0.41、登顶 6/6

    **触发**：登顶 <6/6 → w_knee 回 1.0；
    对称好了但俯仰不动 → 俯仰另有成因，查 `hr_hipx` 限位饱和是不是被接触力推的（加探针测该关节力矩）；
    对称不动 → `cc_rear_sym` 拉不动，改用 max(单腿误差) 替代 sum 让最差那条腿主导。

    环境变量：S10_BLK16_WKNEE（默认 2.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        wk = float(_os.environ.get("S10_BLK16_WKNEE", "2.0"))
        t = self.rewards.cc_rear_sym
        old = t.params.get("w_knee")
        t.params["w_knee"] = wk
        print(f"[s10-climb-Blk16] cc_rear_sym.w_knee {old} -> {wk:g}（两后腿蹲深差 0.96，官方 0.20~0.28）| "
              f"块13 立论已推翻：机身z 0.396~0.402 与领先前腿 +2.26 都已达标，前轮够不到不成立 | "
              f"靶 G<=0.35、浅腿膝<=-1.75，E/机身z/登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk17EnvCfg(DeeproboticsS10ClimbBlk16EnvCfg):
    r"""块17（预排，块16 触发条件之一）：`cc_load_pose` 由 **sum 改 max**，让最差那条腿主导。

    块16 若把 G 拉不动，说明「两腿误差求和」这个形式有问题：
    和可以被「一条过 + 一条欠」以较低代价满足（实测 −2.34 / −1.38，和 0.96）。
    改成 `max(单腿误差)` 后，**只有把最差那条腿拉回来才能降罚**，凑不了。

    单变量：`cc_load_pose` 的值由 `|hl−t| + |hr−t|` 改为 `max(|hl−t|, |hr−t|)`。
    量级约减半，故权重同步 12 → 24 保持每步力度不变（不是新变量，是保持等效）。

    验收：G 膝深浅差 <=0.35、浅腿膝 <=−1.75；E/机身z/登顶 6/6 不得回退。
    触发：登顶 <6/6 → 权重回 12；G 仍不动 → 不是形式问题，查 hr_hipx 限位饱和的力矩来源。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        self.rewards.cc_load_pose.params["use_max"] = True
        self.rewards.cc_load_pose.weight = -float(_os.environ.get("S10_BLK17_W", "24.0"))
        print(f"[s10-climb-Blk17] cc_load_pose 改 max（最差腿主导），权重 {self.rewards.cc_load_pose.weight:g}"
              f" | 靶 G<=0.35、浅腿膝<=-1.75", flush=True)


@configclass
class DeeproboticsS10ClimbBlk18EnvCfg(DeeproboticsS10ClimbBlk17EnvCfg):
    r"""块18（预排，块17 触发条件之一）：**后腿 hipx 限位饱和**专项。

    `hr_hipx` 从块8 到块15 一直钉在 **+0.617~+0.643**（限位 ±0.6109），纹丝不动。
    块12 的 `cc_hipx_splay`（w=8）读数非零却压不动 —— 高度怀疑**不是策略选的，是被接触力推出去的**
    （hipx 动作缩放仅 0.125，策略权限有限）。

    单变量：`cc_hipx_splay` 权重 8 → 24，并把 `lim` 0.15 → 0.10（官方 0.038~0.181）。
    若仍钉在限位，则证明是被动饱和，**奖励这条路走到头**，下一步要改的是几何/姿态前提
    （例如蓄力末的机身偏航或左右重心），届时停链等作者。

    验收：max|hipx| 0.619 → <=0.35（先求脱离限位，不苛求官方 0.18）；登顶 6/6 不得回退。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        t = self.rewards.cc_hipx_splay
        t.weight = -float(_os.environ.get("S10_BLK18_W", "24.0"))
        t.params["lim"] = float(_os.environ.get("S10_BLK18_LIM", "0.10"))
        print(f"[s10-climb-Blk18] cc_hipx_splay w={t.weight:g} lim={t.params['lim']:g}"
              f"（hr_hipx 钉在限位 0.6109 已 8 块未动）| 靶 max|hipx|<=0.35，登顶 6/6", flush=True)


@configclass
class DeeproboticsS10ClimbBlk19EnvCfg(DeeproboticsS10ClimbBlk16EnvCfg):
    r"""块19（09-17 08:5x）：`cc_rear_sym.w_knee` 2.0 → 4.0。继承**块16**（不含块17 的 max）。

    ## 块17（sum→max，w=24）**登顶 0/3**，形式本身危险

    `max(最差腿误差)` 把全部压力压在偏离最大的那条腿上。但上墙过程中**必须有一条后腿抬起摆动** ——
    它一抬误差立刻最大，惩罚就全砸在**正在做正确动作的那条腿**上，机器人不敢抬腿，自然爬不上去。
    `sum` 的"容忍一条过一条欠"正是上墙需要的容忍度。**不按预案的「权重回 12」走** ——
    预案假设是权重问题，实测证明是形式问题。

    ## 走已被证明安全的路

    块16（`w_knee` 0.5→2.0）把 G 膝深浅差 **0.960 → 0.640**（改善 33%）**且登顶 6/6 全程守住**。
    有效且安全，继续加倍。

    验收：G ≤0.35（官方 0.199~0.281）；E ≤0.40、C 0.02~0.20、登顶 6/6 不得回退。
    触发：登顶 <6/6 → 回 3.0；G 仍 >0.5 → 对称这条路到头，转查 hipx 限位饱和（块18）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        wk = float(_os.environ.get("S10_BLK19_WKNEE", "4.0"))
        t = self.rewards.cc_rear_sym
        old = t.params.get("w_knee"); t.params["w_knee"] = wk
        print(f"[s10-climb-Blk19] cc_rear_sym.w_knee {old} -> {wk:g}（块16 用 2.0 把 G 0.960→0.640、"
              f"登顶 6/6 守住；块17 的 max 形式登顶 0/3 已作废）| 靶 G<=0.35，E/C/登顶不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk20EnvCfg(DeeproboticsS10ClimbBlk16EnvCfg):
    r"""块20（09-17 09:1x）：**蓄力时后轮压地**。MuJoCo 真模型 FK 查出的根因，继承块16。

    详见 `mdp/climb_rewards.py::rear_wheels_planted` 的完整实测表。一句话：
    **官方抬右前轮（0.44）、两后轮踩死（0.000~0.025）；我方从块8 到块16 一直抬右后轮，
    而且越抬越高（0.162 → 0.362）。**十几块里从来没有「后轮不许抬」这个约束。

    最讽刺的是：**后轮抬得越高，俯仰看起来越好**（0.162→−16.1°，0.362→−4.3°），
    我据此以为方向对 —— 其实是机器人用翘后腿假装抬头，而判据只看俯仰角，
    **正好奖励了这个错误动作**。

    ## 单一机制：新增 `cc_rear_plant`

    门与 `cc_load_pose` 同源（蓄力段，相位未起）；值 = 两后轮轮底超出 0.05 的部分之和。
    **相位起跑后自动关闭** —— 真正上墙时后轮该抬就抬，不受约束。

    ## 判据升级为**构型级**（`s10_dev/wheel_fk.py`，从 MJCF 真模型 FK 取轮位）

        领先前轮轮底 >=0.35   （官方 0.38~0.44）  块16: 0.219 ✗
        **后轮最高   <=0.08** （官方 0.00~0.03）  块16: 0.312 ✗  ← 以前完全没有这条
        最深处俯仰   >=+10°   （官方 +36~+42）    块16: −7.0  ✗
        最深处离墙   <=0.40   （官方 0.24~0.34）  块16: 0.313 ✓

    **以后凡判断姿态一律用真模型 FK 取轮位，不靠关节角推断**（作者 09-17 定）。

    ## 验收（n=6）

        后轮最高 0.312 → <=0.08
        不得回退：离墙 <=0.40、登顶 6/6
        观察项：领先前轮与俯仰是否随之改善（预期会，但不设硬靶）

    **触发**：登顶 <6/6 → 权重减半；后轮压下去但领先前轮仍 <0.35 → 再加「领先前轮抬起」项；
    后轮压不下去 → 查是不是被 `cc_load_pose` 的深蹲目标逼着抬（两项冲突）。

    环境变量：S10_BLK20_W（**先验读数定档**）、S10_BLK20_LIM（默认 0.05）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        w = float(_os.environ.get("S10_BLK20_W", "12.0"))   # 先验定档：最深帧项值≈0.26（后轮抬到0.312），×12≈3.1/步，与 cc_load_pose 同刻 4.8/步 同量级。不按回合读数（−0.0024）定——墙区仅占 0.3~0.6%，回合读数天然偏小，今日已两次踩此坑。
        lim = float(_os.environ.get("S10_BLK20_LIM", "0.05"))
        self.rewards.cc_rear_plant = RewTerm(func=_cr.rear_wheels_planted,
                                             weight=-w, params={"lim": lim})
        print(f"[s10-climb-Blk20] 新增 cc_rear_plant w=-{w:g} lim={lim:g}"
              f"（蓄力段后轮压地；官方后轮 0.000~0.025，我方块16 抬到 0.312）| "
              f"靶 后轮最高<=0.08，离墙/登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk21EnvCfg(DeeproboticsS10ClimbBlk16EnvCfg):
    r"""块21（09-17 09:4x）：**恢复部分参考态出生**。继承块16。

    ## 为什么换路：加惩罚这条路已被三次崩溃证伪

    | 块 | 新约束 | 权重 | 结果 |
    |---|---|---|---|
    | 13 | 罚前轮离台面高度差 | 20 | 登顶 **1/3**，墙前停 20.6 s |
    | 17 | `cc_load_pose` 改 max | 24 | 登顶 **0/3** |
    | 20 | 蓄力段后轮压地 | 12 | 登顶 **0/3**，**后膝从未到 −1.2**（完全不蹲了） |

    同样这三项在弱权重下**又都毫无作用**（块13 w=5 无效、`cc_hipx_splay` w=8 八块没动过）。

    **不是权重没调好，是路走不通。** 机器人这套上墙动作**结构上就是「抬后轮」的** ——
    深蹲与重心转移都建立在抬起一条后腿之上。禁掉抬后轮等于抽掉它唯一会的办法，
    **而没有给它新的**，所以它干脆不蹲也不爬。
    **官方是「抬前轮、后轮踩死」，两者是不同策略，不是同一策略的参数差异。**

    ## 根因：是我自己关掉了它体验正确构型的唯一途径

    `BranchA` 原本 `ref_prob=0.60`、`ref_t_range=(−0.10, +0.76)` —— 60% 的回合**直接出生在
    官方参考的中段姿态**，而该时间窗正好覆盖关键构型（领先前轮过台面 t=0.25~0.53、
    最深蓄力 t=0.35~0.66）。

    **09-16 我为了训「完整链路」把它全改成 0**（`hooked=0 pushed=0 ref=0`，100% 正常接近）。
    自此机器人**再没体验过「后轮踩地、前轮搭台、身体立起 41°」这个构型**，
    只能自行摸索 —— 摸索出来的就是抬后轮那套。这解释了后面十几块为何怎么加惩罚都掰不回来。

    ## 单一机制：出生分布 ref 0 → 0.35

        hooked 0.05 / pushed 0.05 / **ref 0.35** / 正常接近 0.55
        ref_t_range 沿用 BranchA 的 (−0.10, +0.76)

    **保留 55% 正常接近**（作者要的「墙前慢慢走→停→上墙」完整链路不能丢），
    另外 35% 让它重新体验正确构型。奖励项一律不动。

    ## 验收（n=6，含轮位判据）

        后轮最高 0.312 → <=0.08     （官方 0.00~0.03）
        领先前轮 0.219 → >=0.35     （官方 0.38~0.44）
        俯仰     −7.0 → >=+10°      （官方 +36~+42）
        离墙     <=0.40、登顶 6/6 不得回退
        接近段判据（A/B/C）不得因 RSI 比例上升而消失 —— 55% 正常接近仍应给出有效蹭近段

    **触发**：登顶 <5/6 → ref 降到 0.20；轮位判据仍不动 → RSI 也救不回来，
    说明需要的是**重新分叉**（从更早的、还没学歪的检查点重训），届时停链等作者。

    环境变量：S10_BLK21_REF（默认 0.35）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        rp = float(_os.environ.get("S10_BLK21_REF", "0.35"))
        self.events.spawn_at_wall.params.update({
            "categorical": True,
            "hooked_prob": 0.05, "pushed_prob": 0.05, "ref_prob": rp,
            "ref_t_range": (-0.10, 0.76),
        })
        print(f"[s10-climb-Blk21] 出生分布恢复参考态: hooked 0.05 / pushed 0.05 / **ref {rp:g}** / "
              f"正常接近 {1-0.1-rp:.2f}｜ref_t_range (−0.10, +0.76) 覆盖「前轮过台面 t=0.25~0.53、"
              f"最深蓄力 t=0.35~0.66」| 加惩罚三连崩（13/17/20 登顶 1/3、0/3、0/3）后换路 | "
              f"靶 后轮最高<=0.08、领先前轮>=0.35、俯仰>=+10°", flush=True)


@configclass
class DeeproboticsS10ClimbB2cEnvCfg(DeeproboticsS10ClimbBlk2EnvCfg):
    r"""**B2c：从块2 重新分叉**（09-17 10:1x，作者拍板）。块2 权重 + 蹭近机制，**不带相位门**。

    ## 为什么退回块2：抬后轮是中途长出来的，块2 之前没有

    全部 15 个历史档的轮位（MuJoCo 真模型 FK，`s10_dev/wheel_fk.py`）：

    | 块 | 领先前轮 | **后轮最高** | 俯仰 | 离墙 |
    |---|---|---|---|---|
    | blk0（C36b 起点） | 0.307 | **0.046 ✓** | −26.2 | 0.625 |
    | **blk1** | **0.344** | **0.001 ✓** | −29.5 | 0.522 |
    | **blk2** | **0.342** | **0.001 ✓** | −26.6 | 0.512 |
    | **blk3c ← 我加相位门** | 0.288 | **0.142 ✗** | −17.5 | 0.486 |
    | blk8 / blk12 | 0.273 / 0.222 | 0.171 / 0.288 | | |
    | blk15 / blk16 | 0.155 / 0.218 | **0.364 / 0.311** | −4.3 / −7.1 | 0.322 ✓ |

    **块1/块2 的后轮是踩地的（0.001，官方 0.000~0.025 几乎完全一致），领先前轮 0.342~0.344
    （靶 0.35，只差 0.006）** —— 整套动作的**构型核心起点就是对的**。

    **毛病从块3c 开始**（我加「停住后才起相位」那一块），后轮从 0.001 跳到 0.142，
    之后单调恶化到 0.364。而我十三块一直在追的离墙与俯仰**确实改善了**（0.62→0.29，−26°→−4.3°），
    **代价是把构型核心毁了，我却完全没察觉 —— 因为从来没算过轮子在哪。**

    ## 为什么不在块16 上继续修

    块20 用 w=12 的专项惩罚压后轮，实测它**反而从 0.311 涨到 0.351**，一点没压住。
    十三块反复强化出来的行为，加惩罚掰不回来（今日四次崩溃已证）。
    **块2 缺的是我会修的东西（离墙），块16 缺的是我修不动的东西（构型）。**

    ## 本块改什么

    块2 权重 + **只加蹭近机制**（`creep_v` / `d_creep_end`，块10~14 已验证能把离墙从 0.51 拉到 0.29）：

        d_slow_hi 0.72   减速起点
        creep_v   0.08   蹭近段慢速前进（官方 vx 中位 0.022~0.102）
        d_creep_end 0.32 走到此距离才起动作（官方最深处 0.243~0.341）

    **明确不带相位门**（`S10_SEQ_GATE` / `S10_SEQ_STOP_HOLD_S` 一律关闭）——
    那正是块3c 毁掉后轮压地的机制。也不带 `cc_load_pose` / `cc_front_onto` /
    `cc_rear_plant` / `cc_hipx_splay` / `cc_top_posture` 这些块12~20 的新项（全部未证明有效，且多次致崩）。

    ## 验收（n=6，**轮位判据为硬守卫**）

        后轮最高  0.001 → 必须保持 <=0.10   ← **一票否决，超了立刻熔断**
        领先前轮  0.342 → 保持 >=0.30
        离墙      0.512 → <=0.40
        登顶      3/3 → 6/6

    **触发**：后轮最高 >0.10 → 立即停，说明蹭近机制本身也会引入抬后轮；
    离墙下不来 → 加大 creep_v；登顶 <5/6 → creep 退到 0.05。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_cmd as _cc
        # 相位门明确关闭 —— 块3c 的门是毁掉后轮压地的根源
        _os.environ["S10_SEQ_GATE"] = ""
        _os.environ["S10_SEQ_STOP_HOLD_S"] = "0"
        old = self.commands.base_velocity
        new = _cc.ClimbApproachVelocityCommandCfg()
        for k, v in vars(old).items():
            if k == "class_type":
                continue
            if hasattr(new, k):
                setattr(new, k, v)
        new.d_slow_hi = float(_os.environ.get("S10_B2C_SLOW_HI", "0.72"))
        new.d_stop_hi = 0.52
        new.creep_v = float(_os.environ.get("S10_B2C_CREEP_V", "0.08"))
        new.d_creep_end = float(_os.environ.get("S10_B2C_CREEP_END", "0.32"))
        new.hold_s = 0.5; new.back_s = 0.3; new.timeout_s = 2.2; new.v_stop = 0.12
        new.debug_every = int(_os.environ.get("S10_BLK3_DBG", "0"))
        self.commands.base_velocity = new
        print(f"[s10-climb-B2c] **从块2 分叉**：块2 权重 + 蹭近（creep {new.creep_v:g} 到 {new.d_creep_end:g}m）"
              f"｜**不带相位门**、不带块12~20 的新奖励项 | "
              f"块2 轮位 后轮 0.001✓ 领先前轮 0.342；毛病从块3c（相位门）才出现 | "
              f"守卫：后轮最高 >0.10 立即熔断", flush=True)


@configclass
class DeeproboticsS10ClimbB2dEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""B2d（09-17 10:4x）：**相位门 + 后轮守卫**，一起上。继承 B2c。

    ## B2c 的结论：三条守住，但离墙没动，而且查出一个耦合

    | | 块2 | B2c(n=6) | 靶 |
    |---|---|---|---|
    | 后轮最高 | 0.001 | **−0.002 ✓** | ≤0.10 |
    | 领先前轮 | 0.342 | **0.337** | ≥0.30 |
    | 登顶/站住 | 3/3 | **6/6 / 6/6 ✓** | 不退 |
    | **离墙** | 0.512 | **0.492 ✗** | ≤0.40 |
    | 蓄力段时长 A | 0.35 s | **0.13 s** | — |

    蹭近段只有 **0.13 s**，根本没机会蹭。而块10~14 的蹭近是**有效的**（离墙 0.51→0.29）——
    差别是**那时都带相位门**：门把上墙序列拖住，蹭近才有时间起作用。

    **所以「让它靠近」和「毁掉后轮压地」是同一个机制：相位门。**
    门把机器人停在墙前，而那段时间**没有任何参考在指导它**，它就自己发明 —— 发明出来的是抬后轮。

    ## 本块同时改两处（**不是单变量，且有明确理由**）

    1. 开相位门（`S10_SEQ_GATE=released`，超时 0.8 s）—— 让蹭近有时间起作用
    2. 同时装 `cc_rear_plant`（后轮压地惩罚）—— 防止门的已知副作用长出来

    **只加门已被证明会坏**（那就是块3c：后轮 0.001 → 0.142），所以单独加门没有意义。

    ### 时机是关键

    块20 在块16 线上加同样的 `cc_rear_plant`（w=12），要压的是**已强化十三块的行为**，
    实测**完全压不住**（0.311 → 0.351，反而涨）。
    而本块起点后轮还在 **0.002** —— **装守卫是防止一个还不存在的行为长出来**，难度完全不同。
    这也是「同样的项、不同时机、结果可能完全相反」的一个实例。

    ## 验收（n=6）

        离墙      0.492 → <=0.40      ← 本块唯一要争取的
        后轮最高  0.002 → 必须保持 <=0.10   ← **一票否决**
        领先前轮  0.337 → 保持 >=0.30
        登顶      6/6 不得回退

    **触发**：后轮又涨过 0.10 → 说明门的副作用压不住，**相位门这条路彻底放弃**，
    改为不带门、直接加大 `creep_v` 去拉近距离；
    离墙仍 >0.45 → 加大 `creep_v` 到 0.15。

    环境变量：S10_B2D_PLANT_W（默认 12）、S10_B2D_GATE_TIMEOUT（默认 0.8）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as _cr
        _os.environ["S10_SEQ_GATE"] = "released"
        _os.environ["S10_SEQ_STOP_TIMEOUT"] = _os.environ.get("S10_B2D_GATE_TIMEOUT", "0.8")
        w = float(_os.environ.get("S10_B2D_PLANT_W", "12.0"))
        self.rewards.cc_rear_plant = RewTerm(func=_cr.rear_wheels_planted,
                                             weight=-w, params={"lim": 0.05})
        print(f"[s10-climb-B2d] 相位门开(released, 超时 {_os.environ['S10_SEQ_STOP_TIMEOUT']}s) + "
              f"cc_rear_plant w=-{w:g} 同时上 | 只加门=块3c（后轮 0.001→0.142）已证会坏；"
              f"起点后轮仅 0.002，守卫是**防止行为长出来**而非掰回已固化行为 | "
              f"靶 离墙<=0.40，后轮<=0.10 一票否决，登顶 6/6", flush=True)


@configclass
class DeeproboticsS10ClimbB2eEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""B2e（09-17 11:0x）：**降参考相位的进入距离** `enter_dist` 0.70 → 0.50。继承 B2c。

    ## 今日五次崩溃归纳出的规则：只动已有参数，不加新项、不加门

    | 改什么 | 块 | 结果 |
    |---|---|---|
    | **加新惩罚项**（有效权重） | 13(w20) / 17(max) / 20(w12) | 登顶 **1/3 / 0/3 / 0/3** |
    | **加相位门** | 21 / B2d | 登顶 **0/3 / 0/6**，且必然带出抬后轮 |
    | **只改已有参数** | 14(减速起点) / 16(权重) / B2c(命令剖面) | **活，6/6** |

    B2d 触发条件兑现：后轮涨到 **0.111 > 0.10** → **相位门这条路彻底放弃**。

    ## 为什么降 enter_dist 能治「离墙太远」

    `climb_seq.advance_phase` 第 130 行：`enter = ... & (sd <= enter_dist)`。
    `enter_dist = 0.70` 意味着**机器人一到离墙 0.70 m 就开始走上墙序列**，
    所以动作永远从 0.7 附近起，离墙自然停在 0.49~0.51。官方最深蓄力在 **0.243~0.341**。

    降到 0.50 → 相位晚 0.20 m 才起跑 → 上墙动作从更近处开始。
    **这是已有参数（`SeqTrack` 的 `enter_dist`），不是新机制**，符合上面的规则。
    块11 的触发条件里我就写过要降它，一直没做。

    ## 验收（n=6）

        离墙      0.492 → <=0.40      ← 本块唯一要争取的
        后轮最高  0.002 → 保持 <=0.10  ← **一票否决**
        领先前轮  0.337 → 保持 >=0.30
        登顶      6/6 不得回退

    **触发**：登顶 <5/6 → enter_dist 退到 0.60；
    离墙下来但后轮涨过 0.10 → 说明「靠得近」本身就会引出抬后轮，那是几何冲突，停链找作者；
    离墙仍 >0.45 → 继续降到 0.42。

    环境变量：S10_B2E_ENTER（默认 0.50）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        ed = float(_os.environ.get("S10_B2E_ENTER", "0.50"))
        n = 0
        for nm in ("cs_all12", "cs_rear", "cs_rear_knee", "cs_fr_knee", "cs_fl_knee",
                   "cs_pitch", "cs_ready", "cs_foll_front"):
            t = getattr(self.rewards, nm, None)
            if t is not None and "enter_dist" in t.params:
                t.params["enter_dist"] = ed; n += 1
        assert n > 0, "没有任何 cs_* 项带 enter_dist，继承链被改过"
        print(f"[s10-climb-B2e] enter_dist 0.70 -> {ed:g}（共 {n} 项）| "
              f"相位晚 {0.70-ed:.2f} m 起跑，上墙动作从更近处开始；官方最深蓄力在 0.243~0.341 | "
              f"今日规则：只动已有参数（加新项 3 次崩、加门 2 次崩） | "
              f"靶 离墙<=0.40，后轮<=0.10 一票否决，登顶 6/6", flush=True)


@configclass
class DeeproboticsS10ClimbB2fEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""B2f（09-17 14:1x）：**治落地冲击**。作者：「差最多的是落地不够稳，真实情况下容易摔倒」。

    ## 实测（B2c，n=6）

        台面站高应 ≈0.75     机身最低 **0.527~0.545**（砸下去 0.22 m）
        落地 vz              **−0.41 ~ −0.74 m/s**
        |roll| 峰            **17.6~19.9°**       恢复用时 2.1~2.6 s
        官方对照             末端 z 0.705~0.718、末端 vz +0.007~+0.031、|roll| 峰 9.4~14.5°

    **但我方 |roll| 中位 1.0°、90 分位 3.0°，都比官方（1.0~3.2 / 6.5~9.4）好** ——
    不是「一直不稳」，是**落地那一下没缓冲**。所以**不压全程横滚**（会拖慢动作、今日已因类似操作反复吃亏）。

    ## 根因：豁免范围写宽了一行

        def lin_vel_z_l2_climb_aware(...):
            # 原注释：竖直速度惩罚，翻越相位内关掉（起步那 0.3 s 是猛推）
            return v**2 * (~phase)

    豁免**本意是起步猛推**，但 `phase` 覆盖整个翻越过程、**落地也在里面** ——
    于是从起跳到落地整段**没有任何项在罚竖直冲击**。

    ## 本块两处（都是改已有项，不加新项）

    1. 门控：相位内只罚 `vz<0`（下落），`vz>0`（上推）仍免罚。先验读数已验证生效：
       `lin_vel_z_l2` −0.0156 → **−0.031~−0.036**（约 2 倍，10 个采样点稳定）。
    2. `land_k=4`：**只放大相位内的下落罚**，非翻越段（走路/踏步）完全不变。
       全局 weight=−2.0 时落地每步罚 0.98（主项 6.3，占 16%）；×4 后约 3.9/步。
       预测读数 ≈ −0.09（= −0.034 + 3×0.018），开训首批迭代核对。

    ## 验收（n=6）

        |roll| 峰    17.6~19.9° → <=14.5°（官方上限）
        落地 vz      −0.7 → >=−0.35
        机身最低 z   0.53 → >=0.65（官方末端 0.705~0.718）
        不得回退     接近段后轮峰 <=0.05、横滚峰 <=5.0（B2c 的 0.013/4.0 是唯一干净的）、登顶 6/6

    **触发**：登顶 <5/6 → land_k 降到 2；|roll| 峰不动 → 冲击不是横滚的成因，改查落地瞬间是哪条腿先触台。

    环境变量：S10_B2F_LANDK（默认 4.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        k = float(_os.environ.get("S10_B2F_LANDK", "4.0"))
        self.rewards.lin_vel_z_l2.params = {"land_k": k}
        print(f"[s10-climb-B2f] 落地冲击: lin_vel_z_l2 豁免只覆盖上推 + land_k={k:g}"
              f"（相位内下落罚 ×{k:g}，非翻越段不变）| B2c 落地 vz −0.7、机身砸到 0.53、|roll|峰 18~20° | "
              f"靶 |roll|峰<=14.5°、vz>=-0.35、最低z>=0.65，接近段干净与登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbBlk4EnvCfg(DeeproboticsS10ClimbBlk3cEnvCfg):
    """块4（09-16）：**后腿对称蓄力**。作者 09-16 点名的第二项。

    **继承块3c**（保住「停住后才起相位」的门与命令状态机）。09-16 18:44 链一度按旧继承
    （Blk3，按距离分带、无停住门）自动开过一次块4，已拦截，run 目录 2026-09-16_18-44-20 作废。

    > **09-16 18:44 起本块暂缓**：块3c 实测后膝左右不对称已从 0.648 降到 **0.135**
    > （官方 0.111~0.195，**达标**）—— 本块的靶被块3c 顺带解决了（停住把蓄力窗从 0.47 s
    > 拉到 0.965 s，两条后腿终于有时间对称下蹲）。此时再提 w_knee 有过冲风险。
    > 改为先做块5（后退与蓄力窗），后膝若回退再启用本块—— 两者不是并列关系：蓄力窗只有 0.41~0.47 s 时
    根本没有对称蓄力的时间，官方是 1.80~2.48 s。停住是对称蓄力的前提。

    ## 单变量：`cc_rear_sym.w_knee` 0.5 → 2.0（调已有项，不加新项）

    `cc_rear_sym`（−15，蓄力段 0.15~0.70 m）**已经存在**，但它主罚 hipy，膝只给一半权重。
    对上实测，分工很清楚：

        后**髋**左右不对称   官方 0.098~0.152   C36b **0.141**  ← 已达标，该项起了作用
        后**膝**左右不对称   官方 0.111~0.195   C36b **0.669**  ← 差 3~5 倍，w_knee=0.5 压不住

    **不是相位伪影**（已判别）：蓄力窗内 Δ 过零 **0 次**、恒同号 **100%**、均值恒为 −0.669，
    左后膝全程比右后膝多弯 0.5~0.67 rad；官方过零 2~3 次、恒同号 67~84%、均值 −0.004~−0.079。

    **不动**：其余奖励项与权重、命令剖面、动作缩放、观测 244 / 动作 16、物理与执行器限制、
    地形、出生分布。

    ## 验收

        后膝左右不对称   0.669 → ≤0.20（官方 0.111~0.195）
        后髋左右不对称   不得退出 0.098~0.152
        块3 四项判据     不得回退（停住时长、蓄力窗、墙前最小 |vx|、后退）
        登顶            3/3 不掉

    **触发**：后膝 200 步内不动 → 提 `cc_rear_sym` 总权重（−15 → −30）而非再提 w_knee；
    后髋被拖出区间 → w_knee 回 1.0 并改为只在「两后轮均触地」时计罚。

    环境变量：S10_BLK4_WKNEE（默认 2.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        wk = float(_os.environ.get("S10_BLK4_WKNEE", "2.0"))
        t = getattr(self.rewards, "cc_rear_sym", None)
        assert t is not None, "块4 依赖已有 cc_rear_sym，未找到该项——继承链被改过，先查 C36"
        old = t.params.get("w_knee", None)
        t.params["w_knee"] = wk
        print(f"[s10-climb-Blk4] 后腿对称蓄力: cc_rear_sym.w_knee {old} -> {wk:g}（w={t.weight:g} 不变，"
              f"蓄力段 {t.params.get('lo')}~{t.params.get('hi')} m）| 继承块3 命令剖面 | "
              f"靶 后膝Δ 0.669->≤0.20，后髋Δ 须留在 0.098~0.152", flush=True)


@configclass
class DeeproboticsS10ClimbMirrorEnvCfg(DeeproboticsS10ClimbApproachEnvCfg):
    """全身统一镜像（09-16 第五轮复核后）。作者拍板：左前领先 + 右后对角，完整镜像官方。

    **问题**：此前各块只给个别项开了 mirror，其余仍跟踪原版目标，**目标互相打架**。
    实测块2 保存配置：cs_all12/cs_rear/cs_rear_knee/cs_fr_knee/cs_fl_knee/cs_pitch/cs_ready
    全部 mirror=False，只有 cs_foll_front=True。其中权重最大的 cs_fr_knee(−8.0) 在追
    **官方右前膝 +2.72**，而镜像后右前是**跟随腿**（应 ≈+1.46）—— 与 cs_foll_front(−30) 正面冲突。

    本配置：**所有按关节跟踪的 cs_* 项统一 mirror=True**，并把左右分组按镜像重命名角色：
      领先前腿 = 左前（对应官方右前）、跟随前腿 = 右前（对应官方左前）。
    cs_pitch / cs_ready 不涉及左右分组，保持原样（俯仰与准备锁存是左右对称量）。

    不动：动作缩放、244/16 接口、物理/执行器限制、地形、出生分布、权重数值。
    """

    def __post_init__(self):
        super().__post_init__()
        R = self.rewards
        changed = []
        for name in ("cs_all12", "cs_rear", "cs_rear_knee", "cs_fr_knee", "cs_fl_knee"):
            t = getattr(R, name, None)
            if t is not None and isinstance(getattr(t, "params", None), dict):
                t.params["mirror"] = True
                changed.append(name)
        print(f"[s10-climb-Mirror] 全身统一镜像，已开 mirror=True 的项: {changed} | "
              f"cs_pitch/cs_ready 为左右对称量，不需镜像", flush=True)


@configclass
class DeeproboticsS10ClimbB2gEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""**B2g：B2c + 台上膝限位惩罚**（09-17 14:4x）。唯一改动，其余与 B2c 逐字相同。

    ## 前两次改落地都被自己的数据推翻，记在这里

    - **B2f（`land_k`）**：以为落地没被罚。实测砸最猛那帧 `phase` 为假 →
      落地一直在挨 1× 罚；该改动只把**蓄力**多罚 land_k 倍（96.6% 的罚落在上墙过程）。登顶 0/6，已回退。
    - **「前膝左右差致横滚」**：单帧读数。看分布我方对称性**不比官方差**，已撤回。

    ## 本块依据（`s10_dev/land_check.py`，同口径 n=3）

        下陷            官方 0.042   我方 **0.206**
        台上|roll|峰    官方 0.11°   我方 **18.3°**
        恢复            官方 0.03 s  我方 **1.45 s**
        台上最大下落vz  官方 −0.487  我方 −0.398   ← 我方更轻，**竖直速度不是成因**
        台上膝最大幅值  官方 2.18    我方 **2.70**（限位 2.7227）

    官方掉得更快却 0.03 s 接住；我们右后膝压到 2.70 已无行程可撑 → 沉 0.21 m、1.45 s 才回来。

    ## 改什么

        cc_top_knee = on_top_knee_limit, w=-10, lim=2.30, h_min=0.18

    门用**轮底绝对高度**，不用 `front_up`——遭遇结束后 `wall_mem` 清零会让 `front_up` 退化成
    `z>0`，平地上也成立（块15 `on_top_posture` 的坑）。先验读数：触发 1.4~1.8 s（占台上 8.6~10.6%），
    峰值 0.163，w=10 → 峰值罚 1.6 = 主项 26%。

    ## 验收（n=6）与触发条件

        登顶            6/6           <5/6 → **立即回退到 B2c**，说明台上门仍漏进了爬升段
        接近段后轮峰    <=0.05        超了同上（B2c 是 0.013，唯一干净的）
        横滚峰(接近段)  <=5.0         B2c 4.0
        **台上膝max**   2.70 → <=2.40  没降 → 该项无效，改查"到达时姿态离站姿多远"
        **下陷**        0.206 → <=0.12 膝降了但下陷没降 → 成因不在膝行程，转查到达时的支撑腿
        **恢复**        1.45 → <=0.8 s
        台上|roll|峰    18.3 → <=10°   （官方 0.11，但先要一半的改善）

    环境变量：S10_B2G_W（默认 10）、S10_B2G_LIM（默认 2.30）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B2G_W", "10.0"))
        lim = float(_os.environ.get("S10_B2G_LIM", "2.30"))
        self.rewards.cc_top_knee = RewTerm(func=climb.on_top_knee_limit, weight=-w,
                                           params={"lim": lim, "h_min": 0.18})
        print(f"[s10-climb-B2g] 台上膝限位: w=-{w:g} lim={lim:g} 门=四轮轮底>0.18m | "
              f"B2c 台上膝max 2.70(限位2.72)、下陷0.206、恢复1.45s、|roll|峰18.3° | "
              f"靶 膝max<=2.40 下陷<=0.12 恢复<=0.8s |roll|<=10° | 登顶6/6与接近段干净不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2hEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""**B2h：B2c + 把「已蓄力」的门槛提到官方水平**（09-17 15:1x）。唯一改动：`FLEX_GATE` 0.25 → 0.65。

    ## 作者目视（看 B2c 回放后）

    > 「没有走到靠近台阶，在靠近台阶之前就开始做动作，正确是靠近台阶做动作」

    ## 实测（MuJoCo 真模型逐帧，同一定义：前轮最前沿到墙面，n=6）

    | | 官方 | B2c |
    |---|---|---|
    | 后膝弯到 −1.2（蓄力开始）时离墙 | 0.204 | **0.212** ← 一样 |
    | **前轮抬起时离墙** | **0.115** | **0.225** |
    | **蓄力开始→抬前腿 Δt** | **0.94~1.78 s** | **0.02~0.04 s** |
    | **该段推进** | +0.038~0.148 m | **−0.014 m（反而退了）** |
    | **抬腿那一刻后膝** | **−1.42~−1.50** | **−1.11** |

    **动作起点没错，错在顺序**：官方先蹲实、再蹭近，最后才抬前腿；我方蹲和抬挤在 0.03 s 内。
    这正是作者五条要求里的第②条「靠近墙后腿弯曲蓄力、右前轮伸出」的先后关系。

    ## 为什么是改一个数，不是加新项

    `st_front_lift` **已经有「必须已蓄力」这个门**（`s["flexed"]`），只是门槛 `hind_flex > 0.25`
    换算成膝角只有 **−0.90**，我方 −1.11 轻松过门。**机制在，阈值设错了。**
    今天七次「加新项」七次崩，唯一没崩的是不加新项——所以本块只动这一个阈值。

    取 **0.65（膝角 −1.30）**，不取官方的 0.77（−1.42）：留余量保证可达
    （我方蓄力本来就能到 −1.74，−1.30 在路上）。`flexed` 是**闩锁**，门槛只在第一次起作用。

    ## 连带影响（已核对）

    `flexed` 被 6 个门共用（front_lift / hind_extend / push_posture / level_drag / cc_riser_push 等），
    全部是「蓄力之后」的阶段，提高门槛 = 整条序列都必须先蹲实，方向一致。
    **风险**：门槛够不到 → 六项奖励一起消失 → 崩。所以守卫是登顶。

    ## 验收（n=6）与触发条件

        登顶            6/6           <5/6 → **立即回退 B2c**，说明 0.65 够不到
        **抬腿时后膝**  −1.11 → <=−1.30  没变 → 门没起作用，查 flexed 闩锁是不是被别处提前置真
        **蓄力→抬腿Δt** 0.03 → >=0.4 s
        **抬腿时离墙**  0.225 → <=0.17
        离墙(最深)      0.492 → <=0.40
        接近段后轮峰    <=0.05        B2c 0.013，不得回退
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as cr
        g = float(_os.environ.get("S10_B2H_FLEX", "0.65"))
        cr.FLEX_GATE[0] = g
        print(f"[s10-climb-B2h] 已蓄力门槛 FLEX_GATE={g:g}（膝角 {-(g+0.65):.2f}），原 0.25（膝角 −0.90）| "
              f"B2c 抬腿时后膝 −1.11、蓄力→抬腿 0.03s、离墙 0.225 | 官方 −1.42~−1.50、0.94~1.78s、0.115 | "
              f"靶 后膝<=−1.30 Δt>=0.4s 离墙<=0.17，登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2iEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""**B2i：B2c + 刚翻上台后前腿不许折塌**（09-17 16:0x）。作者看并排回放后点名的问题。

    > 「同一个过墙之后，我们的左前腿应该做支撑，而不是塌下去，你先把这个问题解决，应该就稳一些了」

    ## 实测（MuJoCo 真模型，n=6 vs 官方 n=3）

        腿长(m)            fl      fr      hl      hr
        我方 上台那刻     0.323   0.268   0.321   0.179
        我方 机身最低     0.237   0.378   0.233   0.166
        我方 变化        −0.086  +0.110  −0.089  −0.019
        官方 全程变化     0.000   0.000   0.000   0.000

    **左前塌 8.6 cm，右前伸 11 cm 接管。官方上台后机身最低点就是刚上台那刻，之后只升不降。**

    **不是撑不动，是没在撑**（塌陷段 0.22~0.29 s 的膝力矩）：

        左前 |τ| 峰 13.5 / 18.5 / 26.1   上限 50，**饱和 0%**
        右前 |τ| 峰 49.1 / 49.1 / 50.0   **9% 时间打满**

    ## 改什么

        cc_front_hold = front_leg_hold, w=-20, lim=1.70,  门 = top_win（刚翻上台 1.5 s）

    阈值 1.70：官方台上前膝最大 1.42/1.53/1.59，我方 1.98/2.01/1.99。
    先验读数（门内 1.5 s）：**官方触发 0%**；我方触发 59%~82%、峰值均 0.086。
    w=20 → 峰值罚 1.72（主项 7.0/步的 25%），官方恒为 0。

    ## 门为什么不是高度阈值（B2g 的教训）

    `s["z"]` 是相对 **env 出生点** 的，多级课程地形上「爬过任何东西之后永久为真」，
    B2g 因此把惩罚落到抬前腿上，登顶 6/6 → 2/6。本块改用**状态转变**起计时：
    `encounter` 真→假 & 转假前记过墙高 & 四轮都高过那个旧墙高 → 起 75 步（1.5 s）窗口。

    ## 已被先验读数否掉的候选（不要再试）

    「罚前腿长度差」：爬升段（两前轮上台、后轮还在下面）官方长差 **中位 0.045~0.051、峰 0.071~0.106**，
    我方只有 **中位 0.020、峰 0.068** —— **官方比我方还不对称**，罚它等于把我们推离官方。

    ## 验收（n=6）与触发条件

        登顶            6/6            <5/6 → **立即回退 B2c**
        接近段后轮峰    <=0.05         B2c 0.013，不得回退
        **前腿塌陷**    0.092 → <=0.057（官方 0.020）
        **下陷**        0.216 → <=0.087（官方 0.042）
        **恢复**        1.71 → <=0.53 s
        台上|roll|峰    18.5 → <=12.8（官方 6.3）

    **触发**：前腿塌陷降了但下陷没降 → 成因不在前腿，转查后腿 hr（上台时腿长只有 0.179）；
    前腿塌陷没降 → 门没生效，先打印 `top_win` 的触发率再说。

    环境变量：S10_B2I_W（默认 20）、S10_B2I_LIM（默认 1.70）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B2I_W", "20.0"))
        lim = float(_os.environ.get("S10_B2I_LIM", "1.70"))
        self.rewards.cc_front_hold = RewTerm(func=climb.front_leg_hold, weight=-w, params={"lim": lim})
        print(f"[s10-climb-B2i] 台上前腿不许折塌: w=-{w:g} lim={lim:g} 门=top_win(翻上台后 1.5s) | "
              f"B2c 左前塌 0.086m（力矩只用到 13~26/50，右前却打满 50）| 官方全程 0.000 | "
              f"靶 前腿塌陷<=0.057 下陷<=0.087 恢复<=0.53s | 登顶 6/6 与接近段干净不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2jEnvCfg(DeeproboticsS10ClimbB2cEnvCfg):
    r"""**B2j：把蓄力的价钱标对**（09-17 16:5x）。唯一改动：`st_hind_flex` cap 0.6→0.90、w 5→40。

    ## 作者目视 + 全 MuJoCo 判据定位到的头号差距

    > 「没有走到靠近台阶，在靠近台阶之前就开始做动作，正确是靠近台阶做动作」

        蓄力→抬前腿   官方 **1.135~1.870 s**   B2c **0.008 s**
        抬腿时后膝    官方 **−1.46~−1.55**     B2c **−1.18**

    ## 为什么会这样：价钱标错了

        st_hind_flex（蓄力）  w=5   cap=0.60  →  单次遭遇最多 **3.0**
        st_front_lift（抬腿） w=60  cap=0.45 ×2 →  最多 **54.0**      ← **18 倍**

    而且 `hind_flex = clamp(−膝角 − 0.65, 0)`，**cap=0.6 换算膝角正好 −1.25** ——
    **蹲到 −1.25 就不再给钱**。实测我方抬腿时后膝 −1.18~−1.25，**正好卡在 cap 上**。

    机器人的行为完全合理：蹲到不给钱的地方，就立刻去拿那 18 倍的抬腿奖励。
    **不是它学坏了，是价钱标错了。**

    ## 改什么

        st_hind_flex: cap 0.60 → **0.90**（膝角 −1.55，= 官方抬腿时的深度）
                      w   5    → **40**（单次最多 36，与 front_lift 的 54 同量级）

    从 −1.25 再蹲到 −1.55 这 0.30 rad，边际收益从 **1.5** 变成 **12**。

    ## 为什么这次方向是对的（前九次全崩的反面）

    块13/17/20/21、B2d/e/f/g 是**加罚**，B2h 是**设卡**——都在**减少**爬墙路径上的奖励，
    九次里九次让登顶掉到 0~2/6。B2h 更是死锁：门设在 −1.30，而机器人在没有奖励牵引时
    根本蹲不到那里（实测 knee=−1.218），挂在 `flexed` 上的六项奖励一起归零，它干脆不爬了。

    **本块是加钱**：不动任何门、不加任何惩罚，只把一个已有奖励的上限和单价改对。
    最坏情况是它多蹲一会儿、别的不变。

    ## 不可伪造

    `hind_flex` 的门是「高墙前 & 四轮着地 & 前轮未上」，抬腿刷不到；
    且是**纪录奖励**（只给新纪录增量，一次遭遇一次），在墙前反复蹲下起来拿不到更多。
    想靠「只蹲不爬」白拿 36 分，就要放弃 front_lift 54 + hind_lift 54 + wheel_height_progress 300。

    ## 验收（n=6，`s10_dev/climb_check.py`，全部量由 MuJoCo 真模型算）

        登顶          6/6            <5/6 → 立即回退 B2c
        **抬腿时后膝** −1.18 → <=−1.40（官方 −1.46~−1.55）
        **蓄力→抬腿** 0.008 → >=0.60 s（官方 1.14~1.87）
        抬前腿时离墙   0.214 → <=0.17（官方 0.112~0.149）
        接近段后轮峰   <=0.05         B2c 0.002，不得回退
        下陷          0.216 → 看它跟不跟着降（不降说明落地与蓄力时序无关）

    **触发**：后膝蹲深了但 Δt 没拉开 → 它在蹲的同时抬腿，需要的是**次序**不是深度，
    改用「按蓄力深度**分级缩放** front_lift」（不是 B2h 那种二值门，避免死锁）。

    环境变量：S10_B2J_CAP（默认 0.90）、S10_B2J_W（默认 40）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        R = self.rewards
        cap = float(_os.environ.get("S10_B2J_CAP", "0.90"))
        w = float(_os.environ.get("S10_B2J_W", "40"))
        R.st_hind_flex.params["cap"] = cap
        R.st_hind_flex.weight = w
        print(f"[s10-climb-B2j] 蓄力价钱: st_hind_flex cap {cap:g}（膝角 {-(cap+0.65):.2f}）w={w:g} "
              f"→ 单次最多 {cap*w:.1f}（原 0.6×5=3.0，抬腿是 54）| "
              f"B2c 抬腿时后膝 −1.18、蓄力→抬腿 0.008s | 官方 −1.46~−1.55、1.14~1.87s | "
              f"靶 后膝<=−1.40 Δt>=0.60s，登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2kEnvCfg(DeeproboticsS10ClimbB2jEnvCfg):
    r"""**B2k = B2j + 把撞墙的价钱标对**（09-17 17:1x）。相对 B2j 唯一改动：`st_approach_speed` 2 → 6。

    ## 基座换成 B2j 的理由（实测 n=6）

        蓄力→抬腿   B2c 0.008 s → B2j **2.880 s**（官方 1.135~1.870）
        蓄力段推进   B2c −0.002 m → B2j **+0.122 m**（官方 0.135~0.186）—— 作者要的「边做动作边往前挪」出现了
        后腿塌陷     0.151 → **0.114** ✓      台上侧倾 18.4 → 15.5
        登顶 6/6 保持，判据 8/17 → **9/17**，已刷新最好模型

    **但落地一点没好**：下陷 0.216 → 0.218，恢复 1.71 → **2.115（更差）**。
    同时**冲墙峰值 1.35 → 1.22 m/s（几乎没动）、超 0.6 的时长 0.78 → 0.82 s**。
    → **落地的能量确实不来自蓄力时序，来自撞墙**，本块专治这条。

    ## 作者收窄目标

    > 「你先完成能翻墙站稳」

    所以只认两条：**登顶 6/6** + **落地站稳**（下陷 / 台上侧倾 / 恢复），形似官方先放一边。

    ## 落地不稳的能量是哪来的：撞墙

    B2c 一条完整轨迹（离墙为机身中心）：

        t 12.5~16.0  离墙 2.5→1.1   cmd 0.500  实际 0.41~0.42   跟得上
        t 17.5~19.0  离墙 **0.65 卡住不动**  cmd 0.25~0.27  实际 **0.046/−0.007/−0.017/0.048**
        t 19.5       离墙 0.64      cmd 0.400（剖面超时放行）  实际 0.172
        t 20.5       离墙 0.39      cmd 0.400  实际 **1.202**
        t 21.0       离墙 −0.16（已过墙）      实际 **1.233**

    **停死两秒，然后以 1.35 m/s 峰值冲过去 —— 指令 0.4 的三倍多。** 0.5 s 走 0.55 m。
    官方同段 vx 是 **0.022~0.102 m/s**。

    下陷 0.216、台上侧倾 18.4°、前腿塌陷 0.093 —— **都是撞出来的**。

    ## 为什么它会停死：低于 0.3 的指令它不动

    墙前 1 m 内，指令 → 实际（n=6）：

        0.12~0.20 → 0.028（18%）   0.20~0.30 → **0.024（10%）**
        0.30~0.38 → 0.165（48%）   0.38~0.45 → 0.278（67%）

    所以蹭近剖面（`WP_CREEP_V=0.08`）从机制上就跑不成：给 0.08 或 0.24 都等于叫它停着，
    剖面只能等 `WP_TIMEOUT_S=2.2 s` 超时放行。实测放行点是 **离墙 0.61~0.66 m**，
    蹭近段（d≤0.52）**一次都没走到过**。

    ## 改什么：现成的项，权重标对

    `approach_speed_penalty`（09-09 就写了，注释原文：「首版策略学成 **1.3 m/s 撞墙靠惯性抬头**」）
    在配置里是 `w=−2, v_max=0.6`。账：

        超速罚  2 × (1.35−0.6) × 39 步 × 0.5 ≈ **−29**
        爬高奖  wheel_height_progress 300 × 0.33 m ≈ **+99**
        → **撞墙净赚 70，所以它当然撞。**

        w=6 → −88，与 +99 **基本打平**；w=8 → −117（做成不划算，风险是干脆不爬）

    取 **w=6**，让它自己把速度降到划算的点，而不是一刀切断。

    **官方 vx 0.022~0.102 远在 v_max=0.6 以下，永远不会被这项罚到** —— 加重它不会把我们推离官方。

    ## 与前九次失败的区别

    块13/17/20/21、B2d/e/f/g 是**新增**惩罚项；B2h 是**新设**二值门（死锁）。
    本块**不新增任何项、不新增任何门**，只把一个 09-09 就存在、一直在生效但标价过低的惩罚调对。

    ## 验收（n=6，`s10_dev/climb_check.py`，全量 MuJoCo）

        登顶            6/6           <6/6 → **立即回退 B2c**（作者：先完成能翻墙站稳）
        **冲墙峰值**    1.35 → <=0.9 m/s
        **下陷**        0.216 → <=0.11（官方 0.043~0.058）
        **台上侧倾峰**  18.4 → <=14（官方 4.1~9.5）
        **恢复**        1.71 → <=0.5 s
        接近段后轮峰    <=0.05        B2c 0.002 不得回退

    **触发**：登顶掉 → 说明冲量确实是上墙的必要条件，改为**只在过墙后**限速（而不是接近段）；
    冲墙峰值降了但下陷没降 → 落地能量不来自水平速度，转查过墙瞬间的竖直抛射。

    环境变量：S10_B2K_W（默认 6）、S10_B2K_VMAX（默认 0.6）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        R = self.rewards
        w = float(_os.environ.get("S10_B2K_W", "6"))
        vmax = float(_os.environ.get("S10_B2K_VMAX", "0.6"))
        R.st_approach_speed.weight = -w
        R.st_approach_speed.params["v_max"] = vmax
        print(f"[s10-climb-B2k] 撞墙价钱: st_approach_speed w=-{w:g} v_max={vmax:g}（原 w=-2）| "
              f"B2c 冲墙峰 1.35 m/s、超速罚仅 −29 而爬高奖 +99 → 撞墙净赚 70 | "
              f"靶 冲墙峰<=0.9 下陷<=0.11 侧倾<=14 恢复<=0.5s，登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2lEnvCfg(DeeproboticsS10ClimbB2jEnvCfg):
    r"""**B2l = B2j + 给「身子上去」付钱**（09-17 17:4x）。相对 B2j 唯一改动：新增 `st_body_rise`。

    ## 先撤回两个我自己说错的结论

    1. 「落地是砸下去的」——**最高→最低落差我方 0.038~0.042，官方 0.050 还更大**。没有砸。
    2. 「多灌 40 J 变成砸地」——**机身翻越全程最高才 0.576，根本没冲高过**。

    ## 真相（全 MuJoCo，n=6 vs 官方 n=3）

        翻越全程机身最高 z    官方 **0.715**（≈台上站高 0.706，+0.008）
                             B2c 0.576 / B2j 0.572 / B2k 0.547（比应有站高 0.75 低 **0.18~0.20 m**）
        末端 z               我方 0.738~0.747 —— **最后是站起来了，只是要花 2~3 s**

    **官方是站着翻过去的，我们是趴着爬过去、过去之后再慢慢把身子撑起来。**
    判据「下陷 0.216」量的不是落地冲击，是**过墙时身子本来就低**。
    此前 B2f（罚竖直速度）、B2i（罚前膝弯曲）、B2k（罚水平速度）都在治"落地冲击"，
    **治的是一个不存在的病**，所以下陷从来没动过（0.216 → 0.218 → 0.273）。

    ## 根因：第四个价钱问题

        轮子上去   wheel_height_progress w=300 + wheel_height_record w=200   ≈ **500**
        身子上去   **没有任何一项**（`base_height_flat` 在翻越相位被 `(~phase)` 豁免）

    策略照做：把四个轮子送上台面，身子留在低处爬过去。
    这也解释了 B2k 灌了最多腿功（131 J vs 官方 53 J）却机身最低（0.547）——**功没变成高度**。

    ## 改什么

        st_body_rise = StageRecord(quantity="body_rise"), w=150, cap=0.33

        量 = clamp(机身高 − 0.43, 0, 墙高)，高度相对**出生地面**（不用脚下地形，避免过墙沿跳变）
        门 = `encounter & 两前轮都已在台面` —— 必须真爬上去才进得来，墙前抬头翘身子刷不到

    先验读数：我方该量 **0.156**、官方 **0.28**，cap 0.33。w=150 → 我方现在能拿 23，官方 42，上限 49.5，
    与 `st_front_lift` / `st_hind_lift` 的 54 同量级。**官方拿得比我们多 → 这个奖励把我们推向官方。**

    ## 为什么是加钱不是加罚

    今天加罚/设门九次，九次让登顶掉到 0~2/6；唯一既没崩又真改了行为的是 B2j（把蓄力的价钱改对）。
    本块同样是加钱：不动任何已有项、不加任何门。

    ## 验收（n=6）与触发条件

        登顶              6/6            <6/6 → 立即回退 B2j
        **翻越最高机身z**  0.572 → >=0.66（官方 0.715）
        **下陷**          0.218 → <=0.12
        **恢复**          2.115 → <=1.0 s
        台上侧倾峰        15.5 → <=14
        接近段后轮峰      <=0.05         B2j 0.002 不得回退

    **触发**：机身高上去了但下陷/恢复没好 → 说明慢是别的原因，转查台上站起来时哪条腿在拖；
    机身高没上去 → 该量拿不到（打印 `st_body_rise` 读数确认门有没有开）。

    环境变量：S10_B2L_W（默认 150）、S10_B2L_CAP（默认 0.33）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        from isaaclab.managers import RewardTermCfg as _RT
        w = float(_os.environ.get("S10_B2L_W", "150"))
        cap = float(_os.environ.get("S10_B2L_CAP", "0.33"))
        self.rewards.st_body_rise = _RT(func=climb.StageRecord, weight=w,
                                        params={"quantity": "body_rise", "cap": cap})
        print(f"[s10-climb-B2l] 给身子上去付钱: st_body_rise w={w:g} cap={cap:g} 门=两前轮已在台面 | "
              f"我方翻越最高机身 0.572（应有 0.75）、官方 0.715 | 先验读数 我方 0.156 官方 0.28 | "
              f"靶 翻越最高z>=0.66 下陷<=0.12 恢复<=1.0s，登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2mEnvCfg(DeeproboticsS10ClimbB2jEnvCfg):
    r"""**B2m = B2j + 翻上台后不许一直趴着**（09-17 18:1x）。相对 B2j 唯一改动：新增 `cc_top_low`。

    依据、先验读数、与前四轮为什么无效，全部写在 `climb_rewards.top_low_posture` 的 docstring 里。

    一句话：**「落地砸下去」这个病不存在**（落差我方 0.038~0.042 < 官方 0.050），
    真病是**过墙时身子就低、之后要 2~3 s 才撑起来**。B2l 付钱给「翻越时抬高身子」没用，
    因为那个状态从没到达过、没梯度；本块改罚「在台上停在低姿态的时长」，
    **撑起来的过程连续经过每个高度，梯度处处存在**。

    先验读数：低于 0.90×站高的时长 官方 **0.00 s** / 我方 **1.50 s**。w=30 → 每步约 1.0（主项 15%），
    整段合计约 39，爬高奖 +99 仍然划算。

    ## 验收（n=6）与触发条件

        登顶            6/6           <6/6 → 立即回退 B2j
        **恢复**        2.115 → <=1.0 s
        **下陷**        0.218 → <=0.12
        台上侧倾峰      15.5 → <=14
        接近段后轮峰    <=0.05        B2j 0.002 不得回退

    **触发**：恢复变快但下陷没降 → 下陷是"过墙时就低"，只能回到翻越段想办法（且已知无梯度，
    需要改的是出生态/课程而不是奖励）；登顶掉 → 立即回退。

    环境变量：S10_B2M_W（默认 30）、S10_B2M_RATIO（默认 0.90）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B2M_W", "30"))
        ratio = float(_os.environ.get("S10_B2M_RATIO", "0.90"))
        self.rewards.cc_top_low = RewTerm(func=climb.top_low_posture, weight=-w,
                                          params={"ratio": ratio, "nom": 0.43})
        print(f"[s10-climb-B2m] 台上不许趴着: w=-{w:g} 阈值={ratio*0.43:.3f}（{ratio:g}×0.43）门=top_win | "
              f"先验 低于阈值时长 官方 0.00s / 我方 1.50s | 靶 恢复<=1.0s 下陷<=0.12，登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2nEnvCfg(DeeproboticsS10ClimbB2jEnvCfg):
    r"""**B2n = B2j + 把课程最低行也做成真墙**（09-17 18:5x，作者拍板「改课程或出生态」）。

    ## 今天所有奖励实验都是在一个**从未训练过的场景**上测的

    训练日志每一轮都是同一个形状：

        [课程] approach_riser=**0.34**(8192) …… 开训
        [课程] approach_riser=**0.01**(8192)  ‖累计 升级 0~1% 降级 **99~100%**  平均离出生点 3.98m / 升级线 4.0m

    立面范围 (0.15, 0.38) 分 10 行 → 等级 0.01 ≈ **0.15 m**。
    **8192 个环境几乎全程在练 15 cm 的坎，而验收是 33 cm 的墙。**

    ## 为什么课程会一路塌到底：升级线和降级线是同一条

        升级：走出 > 地形尺寸/2 = **4.0 m**  **且**  四轮最低点升高 ≥ 0.6×该行墙高
        降级：走出 < |指令| × 局时长 × 0.5 = 0.8 × **10** × 0.5 = **4.0 m**

    两条线重合 → **只有「走够 4 m 且爬高达标」才不降级，其余全降**。
    实测平均 3.98 m（差 2 cm），于是每轮 99% 全降，等级钉死在第 0 行。
    再加上 `max_init_terrain_level = 2`（ClimbExpert 设的），环境从一开始就只在第 0~2 行。

    **恶性循环**：我今天把蓄力从 0.008 s 拉到 2.88 s（B2j），机器人在墙前多停 3 秒 →
    走得更短 → 降级更快。**「改对了动作」反而加速了课程塌陷。**

    ## 改什么（单一改动）

        RISER_RANGE  (0.15, 0.38) → **(0.30, 0.38)**

    最低行 0.30 m、顶行 0.38 m，**课程掉到底也是在练真墙**，等级塌不塌都无所谓。
    不动降级公式、不动 `max_init_terrain_level`、不动任何奖励 —— 一次只改一个变量。

    ## 风险与守卫

    风险：0.30 起步比 0.15 难得多，可能大量失败 → 登顶掉。
    但实测它在 **0.33 m** 上登顶 6/6，说明能力是有的；缺的是训练信号。

        登顶          6/6          <6/6 → 立即回退 B2j
        接近段后轮峰  <=0.05       B2j 0.002 不得回退
        **课程等级**  0.01 → 看它能不能站住（这本身就是本块的主要读数）
        下陷/恢复/台上侧倾  只记录 —— **这一轮的目的是把训练信号接上，不指望落地立刻好**

    **触发**：登顶掉到 <6/6 → 回退并改用 (0.25, 0.38)；
    课程仍然 99% 降级但登顶保持 → 说明降级公式也要改（降级线改成 0.5×升级线）。

    环境变量：S10_B2N_LO（默认 0.30）、S10_B2N_HI（默认 0.38）。
    """

    def __post_init__(self):
        import os as _os
        lo = float(_os.environ.get("S10_B2N_LO", "0.30"))
        hi = float(_os.environ.get("S10_B2N_HI", "0.38"))
        self.RISER_RANGE = (lo, hi)
        super().__post_init__()
        print(f"[s10-climb-B2n] 课程立面 {lo:g}~{hi:g}（原 0.15~0.38）—— 最低行也是真墙 | "
              f"根因：升级线 4.0m 与降级线 0.8×10×0.5=4.0m 重合 → 每轮 99% 降级，等级钉在第 0 行(0.15m) | "
              f"本轮主要读数是课程等级能不能站住；登顶 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2pEnvCfg(DeeproboticsS10ClimbB2nEnvCfg):
    r"""**B2p = B2n + 打开「蹬地」这条链的两个堵点**（09-17 22:1x）。

    ## 作者把目标说死了

    > 「我需要你落地稳，是腿蹬起来身子翻越过去，而不是我们现在贴边上去」

    实测正是如此：**翻越时机身最高 官方 0.715（≈台上站高）/ 我方 0.578~0.599**（该到 0.75）。

    ## 完整因果链（每一环都有实测，n=18 跨三个种子）

        以 1.17~1.43 m/s 冲墙
          → 推起段 `slow` 门（|vx|<0.5）**0% 打开**（官方 100%）
          → 唯一给「后腿蹬直」付钱的 `st_hind_extend` **一分拿不到**
          → 只能去拿「收腿抬轮」`st_hind_lift` 的 **54 分**
          → 后腿长变化 **−0.012~−0.026（收）**，官方 **+0.041（蹬直）**
          → 机身只撑到 0.578~0.599 → **贴着边爬过去**
          → 上台姿态低（下陷 0.245~0.265）、**|roll|峰 18~24°**（官方 9.4~14.5）

    ## 改什么（**有意的例外：同时改两个数**）

        st_approach_speed  w=2 → **6**    超速罚 −29 → −88，与爬高奖 +99 打平
        st_hind_extend     w=5 → **60**   单次最多 3.0 → 36，与收腿的 54 同量级

    **为什么必须一起改**：速度不降，`slow` 门就不开，蹬地加多少钱都是零；
    门开了但蹬地只值 3 分，策略仍然会去拿 54 分的收腿。**这两个数是同一个机制的两半，
    单独改任何一个都测不出这条假设。** 违反「一次一个变量」是有意的，理由记在此。

    依据：B2k 实测把 w 提到 6 之后，冲墙峰 1.23→1.09、推起段 |vx| 0.73→0.49、
    **`slow` 门从 3% 开到 54%**（但那是在旧课程下，课程塌在 15 cm，所以结论不能用）。
    我方实测单条后膝能蹬到 |q|=0.03（比官方 0.11~0.25 还直），`hind_ext` 可得量 **0.62 > cap 0.6**，
    钱是拿得到的。

    ## 判据（作者 09-17 22:0x 定的口径）

    - **漂移门限放宽**：作者「可以放宽漂，我要的是攀爬上去不摔倒」。`POST_MAX_DRIFT` 0.30 → **0.60**。
      此前「登顶 0/6 vs 6/6」的满量程噪声，实测**全部是漂移超限**，三个种子都是 **6/6 真正爬上台面**。
    - **主判据 = |roll|峰**（跨种子稳：各档均值 18.6 / 21.5 / 20.5，波动 ±1.5），靶 **≤14.5°**（官方上限）。
    - **次判据 = 翻越最高机身 z**，0.578~0.599 → 靶 **≥0.66**（官方 0.715）。
    - 守卫：**真正爬上台面 6/6**（不是"登顶"字段）、接近段后轮峰 ≤0.05。

    ## 触发条件

    - `slow` 门仍 0% → 速度没降下来，w 提到 8 再试（B2k 的账：−117 vs +99）。
    - 门开了但后腿长变化仍为负 → 蹬地的钱还是不够，`st_hind_lift` 要下调（但那是"减奖励"，
      今天已知风险，需先跑两个种子确认）。
    - 真正爬上台面 <6/6 → 立即回退 B2n。

    环境变量：S10_B2P_SPD（默认 6）、S10_B2P_EXT（默认 60）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        R = self.rewards
        spd = float(_os.environ.get("S10_B2P_SPD", "6"))
        ext = float(_os.environ.get("S10_B2P_EXT", "60"))
        R.st_approach_speed.weight = -spd
        R.st_hind_extend.weight = ext
        print(f"[s10-climb-B2p] 打开蹬地链：撞墙罚 w=-{spd:g}（原 2，罚 −29→−88 与爬高奖 +99 打平）"
              f" + 蹬地奖 w={ext:g}（原 5，单次 3.0→{ext*0.6:.0f}，收腿是 54）| "
              f"实测 推起段 |vx| 1.17~1.43、slow 门 0% 开、后腿长 −0.012~−0.026（收腿）、机身最高 0.578~0.599 | "
              f"靶 |roll|峰<=14.5 翻越最高z>=0.66，真正爬上台面 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2rEnvCfg(DeeproboticsS10ClimbB2pEnvCfg):
    r"""**B2r = B2p + 把参考态出生开回来 30%**（09-17 22:4x）。作者拍板方向：改课程或出生态。

    ## 为什么奖罚两条路都走到头了（三条独立证据）

        加罚十次（块13/17/20/21、B2d~B2i、B2k）—— 没有一次建立起新行为
        加奖（B2l 给「翻越时抬高机身」付钱 w=150）—— 翻越最高 0.572→0.579，**纹丝不动**
        把撞墙罚提到账面不划算（w=6，罚 137 > 爬高奖 99）—— 冲墙峰反而 **1.35 → 1.97**

    **奖罚只能放大已有行为。** B2l 失败是因为「翻越时机身 0.66」这个状态从没被探索到；
    这次「推起段 |vx| < 0.5 的准静态蹬地」同样从没被探索到 —— 实测 `slow` 门在推起段只开 **1%**，
    六个门条件里其余五个（前轮已上 100%、后轮触地 79%、已蓄力 58%、room 73%、能蹬直 24%）都不是瓶颈。

    ## 根因：这条链上 RSI 是**关着**的

        [s10-climb-Approach] 全程正常接近出生（ref/hooked/pushed 全 0）

    `Approach` 把参考态出生全部关掉了，**整条链上机器人从没从官方那个准静态中途姿态出生过一次**。
    它的经验里只有「从远处正常走过来」这一种开局，所以慢速蹬地那套解法根本不在它的经验分布里。

    ## 但不能开高 —— `Approach` 当初关掉它是有理由的

    BranchA 用 **80%** 爬台姿势出生（ref 60 + hooked 10 + pushed 10），
    结果把「爬台姿势」训成默认站姿 —— **正是作者五条要求第一条抱怨的
    「官方还没靠近台阶是正常姿势，我们的已经后腿在做奇怪动作」**。

    所以本块只开 **ref 30%**（hooked / pushed 保持 0），其余 70% 仍是正常接近，
    并用作者目视抓出来的那条做硬守卫。

    ## 判据与守卫

        **接近段后轮峰 <=0.05**   现在 0.005 —— **一票否决**，超了立即回退（这就是「奇怪动作」的量化）
        真正上台面 6/6           现在 6/6 —— 不得回退
        **推起段 slow 门开启比例** 1% → 靶 >=30%（这是本块真正要看的：它有没有学到慢速解法）
        **后腿长变化**           −0.02（收）→ 靶 转正（蹬）
        **|roll|峰**             20.0 → 靶 <=14.5（官方 9.4~14.5）
        翻越最高机身 z           0.58 → 靶 >=0.66（官方 0.715）

    **触发**：接近段后轮峰 >0.05 → 立即回退 B2p，ref 降到 0.15 再试；
    slow 门仍 <10% → 参考态覆盖的时间窗不对（现在是蓄力段 t∈[−0.10,+0.76]），
    要改成覆盖**推起段**；真正上台面 <6/6 → 回退。

    环境变量：S10_B2R_REF（默认 0.30）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        ref = float(_os.environ.get("S10_B2R_REF", "0.30"))
        self.events.spawn_at_wall.params["ref_prob"] = ref
        self.events.spawn_at_wall.params["hooked_prob"] = 0.0
        self.events.spawn_at_wall.params["pushed_prob"] = 0.0
        print(f"[s10-climb-B2r] 参考态出生开回 {ref:.0%}（hooked/pushed 保持 0，其余 {1-ref:.0%} 正常接近）| "
              f"依据：推起段 slow 门只开 1%，其余五个门条件都满足 → 慢速蹬地不在经验分布里 | "
              f"BranchA 开 80% 曾把爬台姿势训成默认站姿（作者第一条要求），故只开 {ref:.0%} | "
              f"硬守卫：接近段后轮峰<=0.05（现 0.005）、真正上台面 6/6", flush=True)


@configclass
class DeeproboticsS10ClimbB2sEnvCfg(DeeproboticsS10ClimbB2rEnvCfg):
    r"""**B2s = B2r + 把参考态出生的时间窗从「蓄力段」挪到「推起段」**（09-17 23:3x）。

    ## B2r（窗口 (−0.10,+0.76)）两轮实测：没用，而且第二轮开始反噬

        推起段四个量一个没动：|vx| 1.42→1.43，slow 门 1%→0%，
                              后腿长变化 −0.019→−0.026，翻越最高机身 0.578→0.582
        第二轮开始毁蓄力：**抬腿时后膝 −1.995 → −0.543**（几乎不蹲就抬腿）
                          抬前腿时离墙 0.329 → 0.632，接近段后轮峰 0.003 → 0.016

    ## 根因：窗口全落在「抬前腿之前」

    `ref_t_range` 的 t=0 锚在**离墙 0.56 m**。实测官方各阶段在该坐标下：

        | 段 | 前轮抬起 | 两前轮上台 | 四轮全上 |
        |---|---|---|---|
        | seg0 | 1.87 | 2.24 | 2.54 |
        | seg1 | 1.70 | 2.06 | 2.36 |
        | seg2 | 1.25 | 1.82 | 2.11 |

    BranchA 的 **(−0.10, +0.76)** 全在「抬前腿之前」—— 等于反复把机器人扔在
    「墙前但还没上墙」的位置，它学到的是「从这里直接冲」，**蓄力那一段被跳过**。
    而我要教的「后腿蹬直把身子顶上去」发生在 t≈1.2~2.6，**它从没从那里出生过**。

    ## 改什么（单一改动）

        ref_t_range  (−0.10, +0.76) → **(1.2, 2.6)**   覆盖 抬前腿 → 四轮全上

    ref_prob 保持 30%（B2r 实测：30% 不会把爬台姿势训成默认站姿，接近段后轮峰 0.003 仍远低于红线）。

    ## 验收与守卫

        **推起段 slow 门开启比例**  0~1% → 靶 **>=30%**（本块真正要看的）
        **后腿长变化**              −0.026（收） → 靶 **转正（蹬）**
        **翻越最高机身 z**          0.582 → 靶 **>=0.66**（官方 0.715）
        |roll|峰                    19.9 → 靶 <=14.5
        ——— 硬守卫 ———
        真正上台面 6/6              不得回退
        接近段后轮峰 <=0.05         B2r 第2轮已到 0.016，再涨就停
        **抬腿时后膝 <=−1.30**      B2r 第2轮退到 −0.543，**本块必须拉回来**；
                                    仍 >−1.30 → RSI 在毁蓄力，ref 降到 0.15

    **触发**：slow 门涨上去但横滚没降 → 慢速解法学到了但落地另有原因，回到落地端查；
    slow 门仍 <10% → 出生态这条也走不通，只剩「改动作缩放/执行器」这类更底层的改动，需作者拍板。

    环境变量：S10_B2S_TLO（默认 1.2）、S10_B2S_THI（默认 2.6）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        lo = float(_os.environ.get("S10_B2S_TLO", "1.2"))
        hi = float(_os.environ.get("S10_B2S_THI", "2.6"))
        self.events.spawn_at_wall.params["ref_t_range"] = (lo, hi)
        print(f"[s10-climb-B2s] 参考态时间窗 ({lo:g}, {hi:g}) —— 覆盖「抬前腿→四轮全上」"
              f"（官方 前轮抬起 1.25~1.87 / 两前轮上台 1.82~2.24 / 四轮全上 2.11~2.54，t=0 锚在离墙 0.56m）| "
              f"原窗口 (−0.10, 0.76) 全在抬前腿之前，两轮实测推起段四个量一个没动、第二轮还把蓄力毁到 −0.543 | "
              f"靶 slow门>=30% 后腿长转正 机身最高>=0.66；守卫 上台面6/6、接近段后轮<=0.05、抬腿时后膝<=−1.30", flush=True)


@configclass
class DeeproboticsS10ClimbB2uEnvCfg(DeeproboticsS10ClimbB2pEnvCfg):
    r"""**B2u = B2p + 把低速纳入指令分布**（09-18 00:2x）。夜间计划阶段 2。

    ## 缺口：策略从没被要求走过慢速

        训练指令分布  `c.ranges.lin_vel_x = (0.2, 0.6)`（ExpertC6 设的）＋ 10% 零指令
        蹭近剖面发的  **0.08 m/s** —— **完全在分布之外**

    实测墙前 1 m 内 指令→实际（n=6）：

        0.12~0.20 → 0.028（**18%**）   0.20~0.30 → 0.024（**10%**）
        0.30~0.38 → 0.165（48%）       0.38~0.45 → 0.278（67%）

    **低于 0.3 的指令它基本不执行。** 于是：
    蹭近剖面必然超时放行（实测放行点离墙 0.61~0.66 m，蹭近段 d≤0.52 **一次没走到**）
    → 推起段 |vx| 1.2~1.4 → `slow` 门（<0.5）开 **0~1%** → 蹬地奖一分拿不到
    → 后腿收不蹬 → 机身只撑到 0.58 → **贴边上去**（作者原话）。

    **这是整条链最上游的堵点，而且此前从没动过。**

    ## 改什么（单一改动）

        c.ranges.lin_vel_x  (0.2, 0.6) → **(0.05, 0.6)**

    不动零指令比例、不动任何奖励、不动地形。

    ## 验收

        **墙前 1m 内 指令 0.12~0.30 档的跟踪率**  10~18% → 靶 **>=60%**（本块唯一要看的）
        ——— 守卫（其它基准不许丢）———
        真正上台面 6/6      接近段后轮峰 <=0.05      接近段侧倾 <=5.0
        抬腿时后膝 <=−1.30  课程等级 >=4

    **触发**：跟踪率上去了 → 进阶段 3（看 `slow` 门与后腿长变化）；
    跟踪率没上去 → 低速执行不了是能力问题不是分布问题，需要改动作缩放/执行器（作者拍板）。

    环境变量：S10_B2U_VLO（默认 0.05）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        vlo = float(_os.environ.get("S10_B2U_VLO", "0.05"))
        c = self.commands.base_velocity
        hi = float(c.ranges.lin_vel_x[1])
        c.ranges.lin_vel_x = (vlo, hi)
        print(f"[s10-climb-B2u] 指令分布 lin_vel_x ({vlo:g}, {hi:g})（原 0.2~0.6）—— 把蹭近的 0.08 m/s 纳入分布 | "
              f"实测墙前 指令0.12~0.30 只跟到 10~18%，蹭近段一次没走到过 | "
              f"靶 跟踪率>=60%；守卫 上台面6/6、接近段后轮<=0.05、抬腿时后膝<=−1.30、课程>=4", flush=True)


@configclass
class DeeproboticsS10ClimbB2vEnvCfg(DeeproboticsS10ClimbB2uEnvCfg):
    r"""**B2v = B2u + 把「收腿抬轮」的价钱降下来**（09-18 00:2x）。夜间计划阶段 3 的备用分支。

    ## 只有在阶段 2 达标（低速能执行）之后才用

    价钱对比（单次遭遇最多能拿）：

        st_hind_extend  后腿蹬直      w=60 × cap 0.60      = **36**   （B2p 已从 5 提到 60）
        st_hind_lift    后腿收起抬轮  w=60 × cap 0.45 × 2  = **54**

    即使蹬地提到 60，收腿仍然更值钱。本块把收腿降到 **w=20 → 18**，让蹬地（36）占优。

    **这是"减奖励"，今天已知有风险**（减少爬墙路径上的奖励十次崩过），所以：
    - 只在 `slow` 门已经打开（>=30%）之后才启用 —— 那时蹬地是**拿得到**的替代路径
    - 守卫照旧：真正上台面 6/6，破了立即回退 B2u

    环境变量：S10_B2V_LIFT（默认 20）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B2V_LIFT", "20"))
        self.rewards.st_hind_lift.weight = w
        print(f"[s10-climb-B2v] 收腿抬轮 w=60→{w:g}（单次 54→{w*0.45*2:.0f}），蹬地 36 由此占优 | "
              f"仅在 slow 门已开>=30% 时启用；守卫 真正上台面 6/6 不得回退", flush=True)


@configclass
class DeeproboticsS10ClimbB2wEnvCfg(DeeproboticsS10ClimbB2pEnvCfg):
    r"""**B2w = B2p + 把「爬高奖」减半**（09-18 01:4x）。今晚唯一还没碰过的东西。

    ## 今晚所有失败指向同一个原因

        撞墙罚 2→6      冲墙峰 1.35 → **1.97**（更猛）
        低速纳入指令分布 蹭近带实际速度 1.91 → **2.00**（没慢下来）
        蹬地奖 5→60     读数 0.0001 → 0.0007（门开 0%，拿不到）
        RSI 30%×2 窗口  推起段四个量一个没动

    机器人被命令 **0.4**，实际跑 **2.0 m/s（5 倍）**。跟踪奖励在这个误差下已归零
    （std=0.4，exp(−(1.6/0.4)²)≈1e−7）—— **它宁可放弃全部跟踪奖励也要冲**。

    因为爬高那一侧是 **500 分**：

        wheel_height_progress  w=300   +   wheel_height_record  w=200

    而撞墙罚（−88）、跟踪（7/步）、蹬地（36）加起来都追不上。
    **冲量是拿到「轮子高度」最快的路，所以它一直冲。**

    ## 改什么（单一改动）

        wheel_height_progress  300 → **150**
        wheel_height_record    200 → **100**

    减半而不是砍掉：爬墙仍要划算（150×0.33 + 100×0.33 ≈ 82 分），
    只是不再压倒跟踪与限速，让「慢慢蹬上去」成为可比的选项。

    ## 守卫与验收

        **真正上台面 6/6**   —— 一票否决。减爬高奖是今天已知有风险的方向（「减少爬墙路径上的奖励」十次崩过）
        接近段后轮峰 <=0.05   接近段侧倾 <=5.0
        **冲墙峰**  2.00 → 靶 <=1.2      **蹭近带速度** 2.00 → 靶 <=0.6（先降一半，不苛求 0.25）
        slow 门 0% → 靶 >=10%（先见到门开，再谈 30%）

    **触发**：上台面 <6/6 → 立即回退 B2p，改为只减 record（200→100）保留 progress；
    冲墙峰没降 → 500 分不是主因，回头查 `climb_complete`/完成奖励 100 那一侧。

    环境变量：S10_B2W_PROG（默认 150）、S10_B2W_REC（默认 100）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        R = self.rewards
        pg = float(_os.environ.get("S10_B2W_PROG", "150"))
        rc = float(_os.environ.get("S10_B2W_REC", "100"))
        R.wheel_height_progress.weight = pg
        R.wheel_height_record.weight = rc
        print(f"[s10-climb-B2w] 爬高奖减半: progress 300→{pg:g}  record 200→{rc:g}（合计 500→{pg+rc:g}）| "
              f"依据：被命令 0.4 实际跑 2.0 m/s，跟踪奖已归零仍要冲 —— 500 分压倒一切 | "
              f"靶 冲墙峰<=1.2 蹭近带<=0.6 slow门>=10%；守卫 真正上台面 6/6 一票否决", flush=True)


@configclass
class DeeproboticsS10ClimbB2xEnvCfg(DeeproboticsS10ClimbB2wEnvCfg):
    r"""**B2x = B2w + 把「收腿抬轮」降价**（09-18 02:5x）。作者要的「后腿蹬腿翻越」最后一环。

    ## 减爬高奖解决了「身子上不上得去」，不解决「靠什么上去」

        爬高奖 500→250→125 三轮：
          翻越最高机身 z  0.579 → 0.741 → 0.745 → **0.761**   ✓ 作者要的「不贴边」已达成
          恢复            4.283 → 0.270 → 0.250 → **0.193 s**
          前/后腿塌陷     双双首次达标
          **后腿长变化   −0.043 → −0.048 → −0.061**   ← **三轮单调恶化，仍是收腿**

    ## 直接动价钱

        st_hind_lift   收腿抬轮  w=60 × cap0.45 × 2 = **54**
        st_hind_extend 后腿蹬直  w=60 × cap0.60     = **36**（B2p 已从 5 提到 60）

    **收腿比蹬地值钱，它当然收。** 本块把收腿降到 **w=20 → 18 分**，蹬地（36）由此占优。

    **这是「减奖励」，今天已知有风险**（减少爬墙路径上的奖励崩过多次），所以：
    真正上台面 6/6 **一票否决**，破了立即回退。

    ## 验收

        **后腿长变化**  −0.061 → 靶 **转正**（官方 +0.041）—— 本块唯一要看的
        守卫：真正上台面 6/6、接近段后轮峰 <=0.05、翻越最高机身 z >=0.66（不许把已得的丢掉）
        盯防：最深帧抬头 已连续三轮下滑 27.9→24.7→22.5，再掉说明爬高奖减过头

    环境变量：S10_B2X_LIFT（默认 20）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B2X_LIFT", "20"))
        self.rewards.st_hind_lift.weight = w
        print(f"[s10-climb-B2x] 收腿抬轮 w=60→{w:g}（单次 54→{w*0.45*2:.0f}），蹬地 36 由此占优 | "
              f"依据：爬高奖减半三轮后腿长变化 −0.043→−0.048→−0.061 仍收腿 | "
              f"靶 后腿长转正；守卫 真正上台面 6/6 一票否决、机身最高 >=0.66 不许丢", flush=True)


@configclass
class DeeproboticsS10ClimbB2yEnvCfg(DeeproboticsS10ClimbB2wEnvCfg):
    r"""**B2y = B2w + 把「抬头」的价钱标对**（09-18 05:0x）。作者五条要求里的第②条。

    ## 抬头整夜单调下滑，而且是价钱错

        最深帧抬头  29.4 → 27.9 → 24.7 → 22.5 → 20.5 → 23.3 → 22.1 → 21.7 → 17.7 → **16.3**
        官方 **35.9~41.6**

    R7~R11 **同一个爬高奖设置(250)** 照样从 23.3 掉到 16.3 → 不是爬高奖造成的。
    查价钱：

        st_pitch_up    抬头  w=10 × cap 0.71      = **7.1**
        st_front_lift  抬腿  w=60 × cap 0.45 × 2  = **54**      ← **7.6 倍**

    **抬头只值 7.1 分。** 今晚的突破正是靠纠正这类价钱错（爬高奖 500 压倒一切）拿到的，同一套路。

    `cap = 0.71 rad = 40.7°`，正好是官方区间（35.9~41.6）的上沿 —— **靶本来就设对了，只是没给够钱**。
    我方现在 16.3° = 0.285 rad；权重 10→60（最多 42.6，与抬腿同量级）后，
    从 16° 蹬到 30° 的边际收益由 **2.4 分** 变成 **14.7 分**。

    ## 起点与守卫

    **从 R9 的峰值检查点起**（不在 R11 这个退化点上盖楼）：
    R10/R11 纯累积已转为下降（机身最高 0.782→0.713、判据 8→7、侧倾 15.0→19.5）。

        守卫：真正上台面 6/6、**翻越最高机身 z >=0.66（今晚的成果不许丢）**、接近段后轮峰 <=0.05
        靶：最深帧抬头 16.3 → **>=25**（先追回 R3 的 27.9，不苛求官方 36）

    环境变量：S10_B2Y_PITCH（默认 60）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B2Y_PITCH", "60"))
        self.rewards.st_pitch_up.weight = w
        print(f"[s10-climb-B2y] 抬头价钱 st_pitch_up w=10→{w:g}（单次 7.1→{w*0.71:.1f}，抬腿是 54）| "
              f"抬头整夜 29.4→16.3，官方 35.9~41.6；cap 0.71rad=40.7° 正是官方上沿，靶对钱不够 | "
              f"靶 抬头>=25；守卫 上台面6/6、机身最高>=0.66 不许丢", flush=True)


@configclass
class DeeproboticsS10ClimbB2zEnvCfg(DeeproboticsS10ClimbB2wEnvCfg):
    r"""**B2z = B2w + 上台后前腿当支撑柱**（09-18 08:3x）。作者看回放点名的问题。

    > 「目前就是因为没有蹬腿以及左前轮没有上墙后支撑，导致会有个摔倒墙上的动作」

    实测（交付模型 R13，n=12 vs 官方 n=3；上台后 1 s 内腿长缩短量）：

        | | fl 左前 | fr 右前 | hl 左后 | hr 右后 | 合计 |
        | 我方 | **−0.094** | −0.029 | −0.134 | −0.052 | **0.309** |
        | 官方 | **−0.022** | −0.010 | −0.099 | **+0.000** | **0.131** |

    **官方有一条腿（右后）一点不缩 —— 那是支撑柱；我们四条腿一起软，总量 2.4 倍，前腿差 4.3 倍。**

        cc_front_support = front_leg_support, w=-300, lim=0.04, 门 = top_win（翻上台后 1.5 s）

    阈值 0.04（官方前腿上限 0.022）→ 我方超出 0.054 → 值 0.0029；**官方恰好 0**。
    w=300 → 我方每步约 0.9（主项 7 的 13%），**官方恒为 0**。

    ## 与昨晚失败的 B2i 的区别（关键）

    B2i 罚**膝角绝对值**（|knee|>1.70）—— 官方台上前膝也到 1.47~1.59，**会被一起罚**，
    且膝角大是结果不是原因。本项罚**相对落台那刻缩短了多少**，官方为 0。

    ## 已被先验读数否掉的假设（记此防再犯）

    「落台那刻腿太直所以撑不住」—— 踏台瞬间较直那条前膝 我方 0.80 / **官方 0.97**，
    差 0.17，按阈值 1.2 罚会把官方也罚进去。**不是病因，已撤回。**

    ## 验收与守卫

        **前腿塌陷** 0.095 → 靶 <=0.041（官方 0~0.041）—— 本块唯一要看的
        **台上侧倾峰** 18.2 → 靶 <=14.3
        守卫：真正上台面 6/6、**翻越最高机身 z >=0.66（昨晚的成果不许丢）**、接近段后轮峰 <=0.05

    环境变量：S10_B2Z_W（默认 300）、S10_B2Z_LIM（默认 0.04）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B2Z_W", "300"))
        lim = float(_os.environ.get("S10_B2Z_LIM", "0.04"))
        self.rewards.cc_front_support = RewTerm(func=climb.front_leg_support, weight=-w,
                                                params={"lim": lim, "w_hind": 0.0})
        print(f"[s10-climb-B2z] 上台后前腿当支撑: w=-{w:g} lim={lim:g} 门=top_win(1.5s) | "
              f"实测上台后 1s 前腿缩短 我方 0.094 / 官方 0.022（4.3 倍）；官方右后腿缩 0.000 是支撑柱 | "
              f"靶 前腿塌陷<=0.041、台上侧倾<=14.3；守卫 上台面6/6、机身最高>=0.66", flush=True)


@configclass
class DeeproboticsS10ClimbB3aEnvCfg(DeeproboticsS10ClimbB2zEnvCfg):
    r"""**B3a = B2z（前腿撑）+ 推起段后腿不许被压垮（蹬腿）**（09-18 09:0x）。作者：

    > 「蹬腿那个是不是一起做，不然等不起来、左前腿也伸不直？」

    **作者说得对，而且这是力学耦合**：后腿不蹬 → 身子起不来 → 过墙那一刻整个重量压到
    前腿 → 前腿再有力也被压软。只治前腿等于让前腿单独对抗物理。

    ## 先验读数（推起段 = 两前轮已上台 & 后轮还没全上；相对该段起点的最大缩短量）

        |            | hl 左后 | hr 右后 |
        | 我方 R13   | 0.002   | **0.173** |   n=12
        | 官方       | 0.000   | **0.049** |   n=3

    **我们的右后腿在推起过程中塌了 17.3 cm** —— 不是在蹬，是被压垮。
    阈值 **0.06**（官方上限 0.049 之上）：我方超出 0.113 → 值 0.0128；**官方恰好 0**。
    w=150 → 我方每步约 1.9（主项 7 的 27%），官方恒为 0。

    ## 为什么这次可能行（昨晚六种蹬腿办法失败的共同原因）

    撞墙罚 2→6、蹬地奖 5→60、RSI 两种窗口、低速指令、爬高奖 500→250→125、收腿降价 54→18，
    **六种全部卡在 `slow` 那道门上**：`st_hind_extend` 要求 |vx|<0.5，而推起段实测 1.2~2.1 m/s，
    门只开 **0~8%**，奖励项根本没被触发过（读数恒为 0.0001~0.0002）。

    本项的门是「两前轮已上台 & 后轮还没全上 & 遭遇中」，**与速度无关，实测 100% 命中**；
    也不要求「伸直到 |knee|<0.65」这种官方自己都未必满足的姿态，只要求**别被压垮**。

    ## 代价与守卫

    同时加两个负项（前腿 −300 + 后腿 −150），**无法分离两者各自的贡献** —— 作者点名一起做，
    记此备查。若本轮同时改善，下一轮用单项复跑才能归因。

        靶：**后腿长变化 −0.043 → 转正**（官方 +0.041）、前腿塌陷 0.095 → <=0.041
        守卫：真正上台面 6/6、**翻越最高机身 z >=0.66**、接近段后轮峰 <=0.05

    环境变量：S10_B3A_W（默认 150）、S10_B3A_LIM（默认 0.06）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B3A_W", "150"))
        lim = float(_os.environ.get("S10_B3A_LIM", "0.06"))
        self.rewards.cc_hind_push = RewTerm(func=climb.hind_leg_push, weight=-w, params={"lim": lim})
        print(f"[s10-climb-B3a] 推起段后腿不许被压垮: w=-{w:g} lim={lim:g} 门=两前轮上台&后轮未上 | "
              f"实测推起段后腿缩短 我方 0.173 / 官方 0.049；阈值 0.06 下官方恒为 0 | "
              f"与 B2z 前腿撑同时生效（作者点名一起做，本轮无法分离归因）", flush=True)


@configclass
class DeeproboticsS10ClimbB3bEnvCfg(DeeproboticsS10ClimbB2wEnvCfg):
    r"""**B3b = B2w + 前腿合计不许塌 + 推起段后腿不许被压垮**（09-18 09:2x）。

    继承 **B2w**（= R13 交付配置），不继承 B2z —— B2z 的逐腿形已被 R20 证伪，保留它只为可复现。

        cc_front_total = front_leg_support_total  w=-600  lim=0.06  门=top_win（翻上台后 1.5 s）
        cc_hind_push   = hind_leg_push            w=-150  lim=0.06  门=两前轮上台&后轮未上

    ## 为什么两个一起（作者 09-18 09:0x）

    > 「蹬腿那个是不是一起做，不然等不起来、左前腿也伸不直？」

    力学耦合：后腿不蹬 → 身子起不来 → 过墙那刻整个重量压到前腿 → 前腿再硬也被压软。
    **R20 从反面证实了这一点**：只治前腿，前腿塌陷 0.0775→0.0735 纹丝不动。

    ## 先验读数（均与 climb_check/MuJoCo 同口径）

        前腿合计塌陷峰值   官方 最差 0.054 | 我方 中位 0.111   → lim 0.06：官方 0.000、我方 0.00258
        推起段后腿缩短峰值 官方 hr 0.049   | 我方 hr **0.173** → lim 0.06：官方 0.000、我方 0.0128

    每步罚：前腿 1.55 + 后腿 1.92 ≈ 3.5（主项约 7 的一半），官方两项都恒为 0。

    ## 方法论代价

    同时上两项 + 前腿换形，**本轮无法分离归因**。作者点名一起做、力学上本就耦合，先合跑；
    若同时改善，下一轮用单项复跑归因。

    ## 验收靶与守卫

        靶：**后腿长变化 −0.043 → 转正**（官方 +0.041）；前腿塌陷 0.095 → <=0.041；
            **领先前轮 >= 0.25**（R20 塌到 0.084，本形若再塌说明和形也没堵住）
        守卫：真正上台面 6/6、翻越最高机身 z >= 0.66、接近段后轮峰 <= 0.05

    环境变量：S10_B3B_FW（600）、S10_B3B_FLIM（0.06）、S10_B3B_HW（150）、S10_B3B_HLIM（0.06）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        fw = float(_os.environ.get("S10_B3B_FW", "600")); fl = float(_os.environ.get("S10_B3B_FLIM", "0.06"))
        hw = float(_os.environ.get("S10_B3B_HW", "150")); hl = float(_os.environ.get("S10_B3B_HLIM", "0.06"))
        self.rewards.cc_front_total = RewTerm(func=climb.front_leg_support_total, weight=-fw, params={"lim": fl})
        self.rewards.cc_hind_push = RewTerm(func=climb.hind_leg_push, weight=-hw, params={"lim": hl})
        print(f"[s10-climb-B3b] 前腿合计不许塌 w=-{fw:g} lim={fl:g}（和形，对分摊中性；R20 的逐腿形被证伪）"
              f" + 推起段后腿不许被压垮 w=-{hw:g} lim={hl:g} | "
              f"先验：前腿合计 官方最差 0.054/我方中位 0.111；后腿推起段 官方 0.049/我方 0.173 | "
              f"靶 后腿长变化转正、前腿塌陷<=0.041、**领先前轮>=0.25**；守卫 上台面6/6、机身最高>=0.66", flush=True)


@configclass
class DeeproboticsS10ClimbB3cEnvCfg(DeeproboticsS10ClimbB2wEnvCfg):
    r"""**B3c = B3b 的两项，按价目表重新标定**（09-18 09:4x）。R21 证伪了 B3b 的**定价**，不是机制。

    ## R21（B3b）的硬读数：两个被罚量自己都没动

        前腿合计塌陷   R13 0.111 → R21 **0.126**（反而更差）
        推起段后腿缩短 R13 0.174 → R21 **0.169**（−3%，噪声内）

    ## 为什么没动 —— 价目表（本轮训练末，每局加权收益）

        cc_edge_clr   越沿净空      **+13.96**
        cc_hind_lift  **后轮抬起**  **+2.26**
        wheel_height_progress 爬高   +0.36
        cc_hind_push  我加的蹬腿罚  **−0.0105**

    **奖励「把后腿收上来」的项，是惩罚「后腿被压垮」的 215 倍 —— 而这两件事是同一个动作（缩腿）。**
    昨晚六种蹬腿办法失败的共同原因由此定量：**定价差两个数量级**，机制本身没问题。

    平方形让价格雪上加霜：量只有 0.0128，平方后剩 0.000074。

    ## 本轮改法：换线性形 + 按目标价位标定权重

    离线标定（窗口时均，与 Episode_Reward 成正比；后腿预测 0.0111 / 实测 0.0105，**误差 6%**）：

        线性/平方 = 后腿 12×、前腿 22×
        目标每局 ≈ −1.0（cc_hind_lift 的 44%，roll_rate/air_spin 同量级）
        → cc_hind_push  线性 **w=1200**   cc_front_total 线性 **w=6000**

    爬台类正项合计约 +17（edge_clr 14 + hind_lift 2.3 + 爬高 0.6 + complete 0.2），
    加 −2.0 只占 12%，**不会把「爬」变成净亏** —— 这是不敢加价时最该算的一笔。

    ## 撤回一条我上一轮的解释

    我说领先前轮塌陷是「逐腿平方和奖励摊薄（36 倍）」。**和形是分摊中性的，lead 仍只有 0.119**
    （R13 0.282、R20 0.084）—— 摊薄至多是部分原因。更可能是：只要罚前腿塌陷，
    「两前轮一起上墙」就真的更省塌陷量。**这条留待单项复跑，不再当成已知结论。**

    ## 验收靶与守卫

        靶：**推起段后腿缩短 0.169 → <=0.10**（官方 0.030）、前腿合计塌陷 0.126 → <=0.09
        守卫：真正上台面 6/6、翻越最高机身 z >= 0.66（R21 0.771）、接近段后轮峰 <= 0.05
        **加价专用守卫：climb_complete 不得归零**（加太狠会让策略干脆不爬）

    环境变量：S10_B3C_FW（6000）、S10_B3C_HW（1200）、S10_B3C_LIM（0.06）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        fw = float(_os.environ.get("S10_B3C_FW", "6000"))
        hw = float(_os.environ.get("S10_B3C_HW", "1200"))
        lim = float(_os.environ.get("S10_B3C_LIM", "0.06"))
        self.rewards.cc_front_total = RewTerm(func=climb.front_leg_support_total, weight=-fw,
                                              params={"lim": lim, "pw": 1.0})
        self.rewards.cc_hind_push = RewTerm(func=climb.hind_leg_push, weight=-hw,
                                            params={"lim": lim, "pw": 1.0})
        print(f"[s10-climb-B3c] 线性形重新标定：前腿合计 w=-{fw:g}、推起段后腿 w=-{hw:g}、lim={lim:g} | "
              f"病因：cc_hind_lift(奖励收腿) +2.26/局 是 cc_hind_push(罚压垮) 0.0105 的 **215 倍** | "
              f"目标每局各 ≈ -1.0（爬台正项合计 +17，占 12%）| "
              f"靶 后腿缩短<=0.10、前腿合计<=0.09；守卫 上台面6/6、机身>=0.66、climb_complete 不归零", flush=True)


@configclass
class DeeproboticsS10ClimbB3dEnvCfg(DeeproboticsS10ClimbB3bEnvCfg):
    r"""**B3d = B3b（腿的两项维持 R21 低价）+ 撞墙速度罚按价目表加价**（09-18 10:0x）。

    ## 三轮连起来的因果链

        R21：六种蹬腿杠杆失败的共同原因 = **定价差两个数量级**（cc_hind_lift +2.26 对 cc_hind_push −0.0105）
        R22：定价补上 → 被罚量首次真的降（后腿 0.169→0.139、前腿 0.126→0.089），
             **但守卫全破**（上台面 5/6、机身 0.771→0.559、侧倾 33°、恢复 nan）
        ⇒ **在 2.09 m/s 的撞速下，缩腿就是唯一的减震。** 罚掉减震而不拿掉撞速，
          冲击全进机身。官方塌陷量小不是腿更硬，是**根本没有硬着陆**。

    **撞速是前置条件，不是并列项。本轮只动撞速，腿的两项退回 R21 的低价（pw=2 / 600 / 150）。**

    ## 定价

    `approach_speed_penalty` 本身**已经是线性**（`tall_wall × clamp(vx − v_max, 0)`），
    没有 R21 那个平方形问题，纯粹是权重太低：

        现在 weight 6 → 每局 **−0.111**（R21 −0.117 / R22 −0.111，很稳）
        目标每局 ≈ −1.7（cc_hind_lift +2.53 的 67%；与 cc_yaw_ref −1.78 同量级）
        → weight **90**

    昨晚把它从 2 提到 6 —— **那只是从 1/50 提到 1/15，离能改变行为还差一个数量级。**

    ## 课程守卫（这条改法历史上塌过课程）

    09-17 加撞墙罚时课程从 4.75 掉到 0.32（升级线=降级线的棘轮，**任何让机器人变慢的改动都触发**）。
    `DOWN_FRAC=0.4` 之后棘轮已解（R21/R22 课程 6.5~6.6、升级 82~83%），所以这次值得重试，
    但仍设硬守卫：**课程末 < 2.0 即回滚**。

    ## 验收靶与守卫

        靶：**蹭近带实际速度 2.05 → <= 1.0**（官方 0.2~0.3，一步到位不现实，先看动不动）
        守卫：真正上台面 6/6、翻越最高机身 z >= 0.66、**课程末 >= 2.0**、climb_complete 不归零

    环境变量：S10_B3D_AW（默认 90）、S10_B3D_VMAX（默认沿用 0.6）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        aw = float(_os.environ.get("S10_B3D_AW", "90"))
        self.rewards.st_approach_speed.weight = -aw
        vm = _os.environ.get("S10_B3D_VMAX", "")
        if vm:
            self.rewards.st_approach_speed.params["v_max"] = float(vm)
        print(f"[s10-climb-B3d] 撞墙速度罚 w=-{aw:g}（昨晚 2→6 只是 1/50→1/15，差一个数量级）"
              f" | 每局 −0.111 → 目标 −1.7（cc_hind_lift +2.53 的 67%）| 腿的两项退回 R21 低价 | "
              f"靶 蹭近带速度 2.05→<=1.0；守卫 上台面6/6、机身>=0.66、**课程末>=2.0**", flush=True)


@configclass
class DeeproboticsS10ClimbB3eEnvCfg(DeeproboticsS10ClimbB3dEnvCfg):
    r"""**B3e = B3d 的撞墙速度罚再加价 90 → 300**（09-18 10:3x）。**这一轮是判别实验，不是调参。**

    ## 要判别什么

        「撞速降不下来」是 **价格不够**，还是 **策略根本做不到 / 从没采样过**？

    R23（w=90）读数：每局 −0.111 → **−0.823**，撞速 2.05 → **1.87（只降 9%）**，守卫全过。

        价格假说 → 再加 3.3 倍，撞速应继续明显下降（靶 <=1.4）
        能力/探索假说 → 撞速卡在 1.7~1.9 不动，罚金线性涨上去而行为不变

    **这两种结果导向完全不同的下一步，所以必须先分开，不能继续盲目加价。**

    ## 已经排除的（别再重复）

        · 速度指令：R21 模型在 V_CLIMB=0.40/0.25/0.15 三档下实际速度 2.23/2.25/2.17 —— **指令完全无效**
        · RSI 墙前出生：**早就开着**（60% 回合在离墙 0.31~0.56 m 静止出生），slow 门仍 0%
          —— 从静止到墙只有 0.3 m，轮子照样加到 2 m/s
        · 昨晚六种蹬腿杠杆：全卡在 slow 门（0~8%），且定价差两个数量级（R21 查明）

    ## 预注册的判读与下一步（先写死，免得事后找补）

        撞速 <= 1.4  → 价格假说成立：继续加价到位，然后把腿的两项价格提回 R22 水平
        撞速 1.4~1.7 → 部分有效：再加一轮到 w=600 定论
        撞速 >  1.7  → **能力/探索假说成立**：奖励侧已穷尽（价格、门、出生态、指令全试过），
                       剩下的是「高速撞墙直接终止回合」这类**不可支付**的约束，
                       或动作缩放/执行器这类**硬约束**——**后者必须作者拍板**（记忆：硬约束不能碰）

    守卫：上台面 6/6、翻越最高机身 z >= 0.66、课程末 >= 2.0、climb_complete 不归零。

    环境变量：S10_B3D_AW（本类默认 300）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        aw = float(_os.environ.get("S10_B3D_AW", "300"))
        self.rewards.st_approach_speed.weight = -aw
        print(f"[s10-climb-B3e] **判别实验**：撞墙速度罚 90 → {aw:g} | "
              f"R23 w=90 → 每局 −0.823、撞速 2.05→1.87（只降 9%）| "
              f"预注册判读：撞速<=1.4 价格假说成立；>1.7 则奖励侧已穷尽，转不可支付约束或请作者拍板硬约束", flush=True)


@configclass
class DeeproboticsS10ClimbB3fEnvCfg(DeeproboticsS10ClimbB3eEnvCfg):
    r"""**B3f = B3e（撞速已降到 0.81）+ 腿的两项提回 R22 价位**（09-18 10:5x）。

    ## 为什么现在才提腿价 —— 三轮建立的因果顺序

        R21  六种蹬腿杠杆失败的共同原因 = 定价差两个数量级
        R22  定价补上 → 被罚量首次真的降（后腿 0.169→0.139、前腿 0.126→0.089），
             **但守卫全破**（机身 0.559、侧倾 33°、恢复 nan）—— 当时撞速仍 2.09，
             **缩腿是那个速度下唯一的减震**，罚掉减震＝把冲击全塞进机身
        R23/R24 撞速罚 6→90→300：2.05 → 1.87 → **0.81**（冲击能量降到 15%）
        ⇒ **前置条件已满足**，同一套腿价现在应当不再换掉稳定性

    ## R24 的成果与代价

        向官方跳进的：领先前轮 0.048→**0.326**、抬头 13.3→**33.5°**、
                      最深帧离墙 0.289→**−0.050**（官方 −0.046~−0.012，**落进区间**）、
                      抬腿时后膝 −1.64→**−1.433**（官方 −1.55~−1.46，**落进区间**）、
                      后腿长变化 −0.047→**−0.008**、下陷 0.260→0.178
        代价：台上侧倾 17.9→**24.8**、前腿塌陷最差 **0.157**、课程 5.90→**2.47**（守卫 2.0）

    **侧倾和前腿塌陷正是作者要的「站稳」，也正是腿的两项要治的。**

    ## 本轮改动（单一）

        cc_front_total  pw=2 w=600  → **pw=1 w=6000**（R22 价位，每局约 −0.72）
        cc_hind_push    pw=2 w=150  → **pw=1 w=1200**（R22 价位，每局约 −0.60）

    撞速罚维持 300 不动 —— 课程已压到 2.47，不能再往下压。

    ## 验收靶与守卫

        靶：**台上侧倾 24.8 → <=18**、**前腿塌陷最差 0.157 → <=0.10**、后腿长变化 −0.008 → 转正
        守卫：上台面 6/6、翻越最高机身 z >= 0.66、**课程末 >= 2.0**、climb_complete 不归零、
              **R24 已拿到的形状不许丢**：领先前轮 >= 0.25、抬头 >= 25°、最深帧离墙 <= 0.15

    环境变量：S10_B3F_FW（6000）、S10_B3F_HW（1200）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        fw = float(_os.environ.get("S10_B3F_FW", "6000"))
        hw = float(_os.environ.get("S10_B3F_HW", "1200"))
        self.rewards.cc_front_total = RewTerm(func=climb.front_leg_support_total, weight=-fw,
                                              params={"lim": 0.06, "pw": 1.0})
        self.rewards.cc_hind_push = RewTerm(func=climb.hind_leg_push, weight=-hw,
                                            params={"lim": 0.06, "pw": 1.0})
        print(f"[s10-climb-B3f] 腿的两项提回 R22 价位：前腿合计 -{fw:g}、推起段后腿 -{hw:g}（线性）| "
              f"前置条件已满足：撞速 2.09→0.81，冲击能量降到 15%（R22 失败正是因为撞速还在）| "
              f"靶 台上侧倾<=18、前腿塌陷最差<=0.10、后腿长变化转正 | "
              f"守卫 上台面6/6、机身>=0.66、课程>=2.0、**R24 形状不许丢（领先前轮>=0.25、抬头>=25°）**", flush=True)


@configclass
class DeeproboticsS10ClimbB3gEnvCfg(DeeproboticsS10ClimbB3eEnvCfg):
    r"""**B3g = R24 配置 + 「上台后两前腿等长」+ 保留 R25 的蹬腿价**（09-18 11:4x）。

    作者 11:1x 收敛口径：「上墙你要蹬腿上去，左前轮要支撑，不要摔倒，没别的」，
    并补充「按你的想法来，综合分析，而不是只看这几个」。

    ## 三轮的账（为什么这样配）

        R24 = 撞速罚 300           → **全场最好**：6/6、机身 0.759、侧倾 24.8、形状最像官方
        R25 = R24 + 腿两项 ×10 价  → **① 首次达成 dleg +0.005**，但 3/6、机身 0.563、侧倾 33.9
        R25 里 **前腿项没动过自己的靶**（0.099 → 0.096），**后腿项动了**（−0.008 → +0.005）

    ⇒ 保留后腿高价（它管用），**撤掉前腿合计项的高价**（它在 R20/R22/R25 三轮都没动过自己的靶），
      换上从未被管过的「上台后两前腿等长」。

    ## 本轮配置（相对 R24）

        cc_hind_push    pw=2 w=150  → **pw=1 w=1200**（沿用 R25，每局约 −0.37，① 的杠杆）
        cc_front_total  维持 R24 的 pw=2 w=600（基本惰性，仅为连续性，不再指望它）
        cc_front_sym    **新增** lim=0.03 w=800（每局我方 −0.50、**官方恒为 0**）

    ## 预注册判读

        守卫恢复（6/6、机身>=0.66）→ **R25 的伤是前腿合计项造成的**，前腿项就此淘汰
        守卫仍破                   → 是后腿高价的代价，把 cc_hind_push 降到 600 再试
        侧倾降但 |fl−fr| 峰没降    → 对称项不是起效的原因，别归功给它

    ## 验收靶与守卫（作者三条为主）

        ① 蹬腿    dleg  +0.005 → 保住并趋向官方 +0.041
        ② 左前支撑 上台后 |fl−fr| 峰 0.096 → **<=0.04**（官方 0.028）
        ③ 不摔倒  台上侧倾 24.8/最差 47.2 → **<=18 / 最差 <=25**
        守卫：上台面 6/6、机身 z>=0.66、课程末>=2.0、climb_complete 不归零、
              **R24 形状不许丢**：领先前轮>=0.25、抬头 25~42°、最深帧离墙 −0.10~0.15

    ## 诚实标注

    「两前腿不等长 → 摔倒」**因果未证实**（轮次内去均值相关仅 +0.37 且符号不一致）。
    本项是**对齐官方一个从未被管过的维度**，不是已证实的病因。

    环境变量：S10_B3G_SYMW（800）、S10_B3G_SYMLIM（0.03）、S10_B3G_HW（1200）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        sw = float(_os.environ.get("S10_B3G_SYMW", "800"))
        sl = float(_os.environ.get("S10_B3G_SYMLIM", "0.03"))
        hw = float(_os.environ.get("S10_B3G_HW", "1200"))
        self.rewards.cc_front_sym = RewTerm(func=climb.front_leg_symmetry, weight=-sw, params={"lim": sl})
        self.rewards.cc_hind_push = RewTerm(func=climb.hind_leg_push, weight=-hw, params={"lim": 0.06, "pw": 1.0})
        print(f"[s10-climb-B3g] 新增 上台后两前腿等长 w=-{sw:g} lim={sl:g}（官方峰 0.028 / 我方 0.096；**官方每局恒为 0**）"
              f" + 保留 R25 蹬腿价 w=-{hw:g}（R25 里它把 dleg 打到 +0.005）| "
              f"撤掉前腿合计项高价（R20/R22/R25 三轮没动过自己的靶）| "
              f"靶 ①dleg 保正 ②|fl−fr|峰<=0.04 ③侧倾<=18/最差<=25；守卫 6/6、z>=0.66、R24 形状不许丢", flush=True)


@configclass
class DeeproboticsS10ClimbB3hEnvCfg(DeeproboticsS10ClimbB3eEnvCfg):
    r"""**B3h = R24 配置 + 「蓄力太短就罚」**（09-18 11:5x）。**单一改动。**

    ## 四轮教训：腿的动作治不动，因为它不是「选择」

        R20 逐腿塌陷   靶 0.0775 → 0.0735   没动
        R22 合计塌陷   靶 0.111  → 0.089    动了，但守卫全破
        R25 合计塌陷↑价 靶 0.089  → 0.096    没动，守卫破（3/6）
        R26 两腿等长   靶 0.096  → **0.150**  **更差**，守卫破（3/6）
        台上侧倾一路：24.8 → 33.9 → 37.6

    三个不同的量、三种形式、价格都按标定打到位，**全部治不动，落地一次比一次差**。
    ⇒ **落地时的腿部动作是被动力学逼出来的，不是策略在选**：
      机身以 38° 仰角冲到台沿再砸到前腿上，腿只能被动吸收。
      **要治「身子怎么到达的」，不是「腿到了之后做什么」。**

    ## 本轮的靶：判据表里最大的缺口，且从未被任何项管过

        蓄力→抬腿(s)   官方 **1.135~1.870**   R26 **0.312**   ← **4~6 倍**

    `st_hind_flex` 是 `StageRecord`：奖励**蹲得多深**（cap 0.9 = 膝角 −1.55），
    **不奖励蹲多久**。策略于是学成「蹲下去立刻弹起来」。

    ## 形式与标定

        cc_crouch_time = clamp(1.0 − 蓄力时长, 0)，在抬前轮后的 0.5 s 窗口内持续罚，w=20

    数值自检（状态机模拟，蹲 0.3 s vs 蹲 1.2 s vs 不在遭遇）：
    **我方每局 −0.700、官方恰好 0.000、场外 0.000。**
    官方蓄力 >=1.135 s，取 min_s=1.0 → **官方结构性为 0**。
    纯惩罚、不可刷分：门是「抬前轮」这个必须发生的事件，拖着不抬拿不到任何好处，
    还要照付撞墙罚与停滞罚。

    ## 起点与放弃项

    起点回到 **R24**（R25/R26 连破守卫）。**不带** R26 的对称项（它把自己的靶从 0.096 做到 0.150），
    **不带** R25/R26 的后腿高价（1200 是守卫破的原因：R25、R26 两轮都 3/6，唯一共同项就是它）。
    腿的三项全部回到 R24 的惰性水平。

    ## 预注册判读

        蓄力→抬腿 >=0.6 s 且守卫保住 → 方向对，继续加到官方区间
        蓄力→抬腿 没动               → 价格不够，w 20 → 60 再试一轮
        蓄力变长但侧倾没降           → 「到达方式」不是落地不稳的原因，回到落地端重找

    靶：蓄力→抬腿 0.312 → **>=0.60 s**；③ 台上侧倾 37.6 → <=25
    守卫：上台面 6/6、机身 z>=0.66、课程末>=2.0、climb_complete 不归零、
          R24 形状不许丢（领先前轮>=0.25、抬头 25~42°、① dleg 不得回到 −0.03 以下）

    环境变量：S10_B3H_W（20）、S10_B3H_MIN（1.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B3H_W", "20"))
        ms = float(_os.environ.get("S10_B3H_MIN", "1.0"))
        self.rewards.cc_crouch_time = RewTerm(func=climb.crouch_time_penalty, weight=-w, params={"min_s": ms})
        print(f"[s10-climb-B3h] **单一改动**：蓄力太短就罚 w=-{w:g} min_s={ms:g} | "
              f"靶是判据表最大缺口：蓄力→抬腿 官方 1.135~1.870s / 我方 0.312s（4~6倍），**从未被任何项管过** | "
              f"自检：我方每局 −0.700、**官方恰好 0.000** | "
              f"腿的三项全回 R24 惰性水平（四轮证明腿的动作治不动，是被动力学逼出来的）", flush=True)


@configclass
class DeeproboticsS10ClimbB3iEnvCfg(DeeproboticsS10ClimbB3hEnvCfg):
    r"""**B3i = B3h 蓄力罚 20 → 60**（09-18 12:2x）。按 B3h 预注册执行，**单一改动**。

    ## R27（w=20）读数

        cc_crouch_time 每局 **−0.5158** ⇒ 训练侧蓄力时长 ≈ **0.484 s**（罚 = clamp(1.0−dur,0)）
        评测侧 蓄力→抬腿 0.312 → **0.255 s**（R24 0.242）—— **没被治动**
        n=12 复核：上台面 **10/12**、机身 0.685 ✓、侧倾 **28.96**、① dleg −0.001、② 前腿塌陷 **0.085**

    ## 本轮的判读能力（为什么这轮能定论）

    `cc_crouch_time` 的每局值与蓄力时长是**解析关系**：每局 = clamp(1.0 − dur, 0)。
    所以从训练日志一眼反解出 dur，不必等评测：

        w=60 时  每局 −3.0 ⇒ dur=0   每局 −1.8 ⇒ dur=0.4   每局 −0.9 ⇒ dur=0.7   每局 0 ⇒ dur>=1.0

    **预注册判读**：

        dur >= 0.7 s → 价格问题，方向对，继续加到官方 1.135
        dur <  0.7 s → **探索问题**：策略压根没采样过「蹲住一秒再抬」，加价无效，**就此停手**，
                       转去查「为什么蹲不住」（是否有项在惩罚静止蓄力 / 姿态是否不可维持）
        dur 上去了但侧倾没降 → 「到达方式」不是落地不稳的原因，回落地端重找

    ## 已排除（别再重复）

        · `stall_penalty_encounter_free` **整个遭遇段已豁免**，不是它在罚静止蓄力
        · `cc_edge_clr`（全场最大项 +10~15）奖励的是**后轮越沿净空 0~12 cm 带**，
          与「立起来贴墙」无关 —— 我一度猜它driving过度仰角，**已查证撤回**
        · `st_hind_flex` 是 StageRecord，只奖励蹲得**深**（cap 0.9），不奖励蹲得**久**

    起点 R27 `2026-09-18_11-44-29/model_19575.pt`（n=12 复核 10/12、机身 0.685 守住）。

    靶：训练侧蓄力 dur 0.484 → **>=0.7 s**；③ 台上侧倾 28.96 → <=22
    守卫：上台面 >=10/12 等效（n=6 下 >=5/6）、机身 z>=0.66、课程末>=2.0、
          ① dleg 不得低于 −0.03、② 前腿塌陷 不得高于 0.10、领先前轮>=0.25

    环境变量：S10_B3H_W（本类默认 60）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B3H_W", "60"))
        self.rewards.cc_crouch_time.weight = -w
        print(f"[s10-climb-B3i] **单一改动**：蓄力罚 20 → {w:g} | "
              f"R27 反解蓄力时长 0.484 s（官方 1.135~1.870）| "
              f"本轮可从训练日志解析反解 dur：每局 −3.0⇒0s、−1.8⇒0.4s、−0.9⇒0.7s、0⇒>=1.0s | "
              f"预注册：dur>=0.7 继续加；dur<0.7 判定为**探索问题**、就此停手转查「为什么蹲不住」", flush=True)


@configclass
class DeeproboticsS10ClimbB3jEnvCfg(DeeproboticsS10ClimbB3hEnvCfg):
    r"""**B3j = B3h（蓄力罚回 w=20）+ 三轮支撑罚 −0.5 → −4.0**（09-18 13:0x）。**单一改动。**

    ## 蓄力这条路按预注册停手

        R28（w=60）训练侧蓄力 0.484 → **0.598 s**（价格有效，但没到我画的 0.7 线）
        **n=12 复核**：上台面 10/12 → **6/12**、侧倾 28.96 → **27.88（没降）**
        ⇒ 触发预注册第三条：**「到达方式」不是落地不稳的原因，回落地端重找。** 蓄力罚回 20。

    ## n=6 三次误导（本轮最该记的教训）

        R28 n=6：上台面 5/6、侧倾 **19.82** → n=12：**6/12、侧倾 27.88**
        侧倾在 n=6 下波动 ±8°、成功率 ±40%

    **这几轮据 n=6 读出的「小幅改善」大多不成立。** 只有 R24 那种大幅变化（撞速 2.05→0.81）是稳的。
    **今后：靶量的判定一律 n=12；n=6 只用来看守卫有没有塌。**

    ## 三个被固定效应检验打掉的假设（都别再用）

        两前腿长差 fdiff  跨轮次 +0.79 → **轮次内 +0.37**，各轮符号不一致
        最深帧抬头        跨轮次单调 → **轮次内 −0.26**，四轮全负
        腾空占比          官方 0% / 我方 7~13%，看着像 → **轮次内 +0.04**，不相关

    ## 唯一活下来的：支撑轮数 <=2 的时间占比

        轮次内相关 R13 **+0.34**、R27 **+0.81**、R28 **+0.98**
        **固定效应合并 n=36 = +0.78**

    对应的现成项就是 `support_deficit_climb`（触地轮 <3 时罚差额，翻越/遭遇段生效），
    当前 w=−0.5 → 每局仅 **−0.239**（`cc_edge_clr` 是 +8.23，差 34 倍）。

        w −0.5 → **−4.0**（8 倍）→ 每局约 **−1.9**，与 cc_yaw_ref −1.78 同量级

    ## 诚实标注（重要）

    **官方在同一窗口也有 54% 的时间 <=2 轮支撑**，所以这**不是「对齐官方」的靶**，
    而是**在我们自己这个策略族内部预测摔倒的量**。作者的 ③ 是「不摔倒」不是「动作一模一样」，
    故仍可用；但**不能宣称「官方就是这么做的」**。
    另：`support_deficit_climb` 当初被从 −1.5 降到 −0.5，注释写「三轮支撑只作软约束，
    不抹掉动态蹬起」—— 本轮加价可能与 ① 蹬腿冲突，**故守卫里盯死 ① 不得退化**。

    起点 R27 `2026-09-18_11-44-29/model_19575.pt`（n=12 10/12、机身 0.685，是近期最好的）。

    靶：③ 台上侧倾 **28.96 → <=22（n=12 判定）**；<=2 轮支撑时间占比 36% → <=25%
    守卫：上台面 >=10/12、机身 z>=0.66、课程末>=2.0、
          **① dleg 不得低于 −0.03**（加价可能压掉动态蹬起）、② 前腿塌陷<=0.10、领先前轮>=0.25

    环境变量：S10_B3J_W（默认 4.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        self.rewards.cc_crouch_time.weight = -20.0            # 蓄力罚回 R27 的 20（B3i 的 60 按预注册撤销）
        w = float(_os.environ.get("S10_B3J_W", "4.0"))
        self.rewards.support_deficit_climb.weight = -w
        print(f"[s10-climb-B3j] **单一改动**：三轮支撑罚 −0.5 → −{w:g}（每局 −0.239 → 约 −1.9）| "
              f"依据：<=2 轮支撑时间占比是**唯一通过固定效应检验**的量（轮次内 +0.34/+0.81/+0.98，合并 +0.78）| "
              f"fdiff / 抬头 / 腾空 三个假设均被同一把尺子打掉 | "
              f"蓄力罚回 20（B3i 的 60 在 n=12 下把上台面从 10/12 打到 6/12 且侧倾没降）| "
              f"靶 侧倾<=22（**n=12 判定**）；守卫 上台面>=10/12、z>=0.66、**① dleg>=−0.03**", flush=True)


@configclass
class DeeproboticsS10ClimbB3kEnvCfg(DeeproboticsS10ClimbB3jEnvCfg):
    r"""**B3k = B3j 的三轮支撑罚 −4 → −8**（09-18 13:2x）。**单一改动，并写死收尾条件。**

    ## R29（w=−4）n=12 读数：唯一验证过的杠杆，确实起作用

        ③ 台上侧倾中位  29.0 → **23.8**（R27/R28 都是 28~29）
        真正上台面      10/12 → **11/12**
        ① dleg          −0.001 → **+0.003**（保持正）
        support_deficit_climb 每局 −0.239 → **−0.843**

    ## 但代价明确

        翻越最高机身 z  0.685 → **0.605**（破守卫 0.66 —— 作者要的「翻越过去而不是贴着墙过去」在退化）
        恢复(s)         0.268 → **1.448**
        ③ 侧倾**最差**   49.1 → **60.9**（中位改善、尾部更糟：多了一种新失败模式）
        ② 前腿塌陷      0.082 → 0.111

    这正是该项当年从 −1.5 降到 −0.5 时注释写的顾虑：**「三轮支撑只作软约束，不抹掉动态蹬起」**。

    ## 本轮：再加一档，并**预先写死收尾条件**

        support_deficit_climb  −4.0 → **−8.0**

    **预注册判读（n=12 判定，n=6 只看守卫）**：

        侧倾中位 <=20 且 机身 z >=0.62  → 继续，把 ② 前腿塌陷 补回来
        侧倾中位 >20 或 机身 z <0.58    → **这条路到顶了，就此收尾**：
            · 交付版仍是 **R13**（12/12、侧倾 18.2、机身 0.769）
            · 同时交一份 **R29 作为「① 蹬腿达成」的备选**（dleg +0.003、11/12，但侧倾 23.8、机身 0.605）
            · 把「① 与 ③ 在当前奖励结构下互斥」作为**需作者拍板的结论**上报，
              附上剩余可动的底层选项（动作缩放 / 执行器 / 课程立面分档），**那些是硬约束**

    ## 为什么不再换新假设

    今晚提出的病因假设里，**三个被固定效应检验打掉**（两前腿长差 +0.37、抬头 −0.26、腾空 +0.04），
    **两个查证后撤回**（stall 已豁免、edge_clr 与仰角无关），**四个罚项没动过自己的靶**
    （逐腿塌陷 / 合计塌陷 ×2 / 两腿等长）。**只有「<=2 轮支撑时间占比」通过检验（+0.78）并真的见效。**
    在它见顶之前不引入新假设；见顶就收尾上报，不再空转。

    起点 R29 `2026-09-18_12-40-39/model_19774.pt`（n=12 11/12、① +0.003）。

    环境变量：S10_B3J_W（本类默认 8.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B3J_W", "8.0"))
        self.rewards.support_deficit_climb.weight = -w
        print(f"[s10-climb-B3k] **单一改动**：三轮支撑罚 −4 → −{w:g} | "
              f"R29 n=12：侧倾中位 29.0→**23.8**、上台面 **11/12**、① dleg **+0.003** —— 杠杆有效 | "
              f"代价：机身 z 0.685→**0.605**（破守卫）、恢复 0.268→1.448、侧倾最差 49→**60.9** | "
              f"**预注册收尾**：n=12 侧倾中位>20 或 机身z<0.58 → 这条路到顶，交付版回 R13 + R29 备选，"
              f"并把「①与③互斥」作为需作者拍板的结论上报", flush=True)


@configclass
class DeeproboticsS10ClimbB3mEnvCfg(DeeproboticsS10ClimbB3jEnvCfg):
    r"""**B3m = R29 配置 + 「上台后机身不许沉下去」**（09-18 14:0x）。作者追问引出的：

    > 「我觉得这样会不会站不稳，在现实中？会摔到台面上，腿还站得起来吗？」

    ## 先把作者的三问用数答了

        「站得稳吗」   仿真里 R13 站住 12/12、R29 11/12，末段侧倾回到 1.0°/1.7°、机身回 0.74
        「会摔到台面上吗」 **R29 机身最低只离台面 0.089 m**（R13 0.182），base 有厚度 → 离蹭到只剩几厘米
        「还站得起来吗」  每一次都起来了；真趴了 SDK 有样条 IK 起立，**但要切模式，1.4 自主模式下算失败**

        静态翻倒角（18.987 kg、质心高 0.316、半轮距 0.183）= **30.1°**
        R29 侧倾 23.8° → **只剩 21% 余量**；R13 中位 18.2° 但最差 35.3° **已越线**

    ## 本轮的靶：上台后机身净空 —— 官方与我方差 4 倍，且 B3 这一系从没管过

        上台后机身离台面最低   官方 **0.320**（三段 0.308/0.320/0.324）  R13 0.182  **R29 0.089**
        按各自站高归一         官方 **87%**                              R13 42%    **R29 21%**

    **官方上台之后机身基本不下沉；我方是一屁股坐下去再撑起来。**
    这也解释了为什么现实风险大：点云实测石笼顶起伏 ±10~25 cm（笼网/植被），
    而仿真台面是完全平的方盒子 —— **那 8.9 cm 余量现实里可能被一根翘起的笼网直接吃掉。**

    ## 与旁支 B2m 的关系（必须说清，别当成新发现）

    `top_low_posture` **早就存在**，09-17 在旁支 **B2m** 用过（w=30、阈值 0.387、平方形）。
    但：① 那一轮的对比在 09-14 记录里写着「**全部作废**」（n=1 / 课程塌陷那批）；
    ② **B3 这一系从未继承它**（R29 训练日志里 `cc_top_low` 读数根本不存在）；
    ③ B2m 阈值 0.387 = 0.90×0.43，而**官方自己最低 0.320 会被一起罚** —— 口径不干净。

    **本轮改用 阈值 0.25（ratio 0.5814×0.43）：官方三段每局值全部恰好 0.00000。**

    ## 标定（top_win 1.5 s 窗口，线性形）

        每局值   官方 **0.00000**（最差也 0）   R13 0.00148   **R29 0.01651**
        w=80 → R29 约 **−1.32**/局，R13 −0.12，**官方恒 0**

    改线性形的理由同今晚 cc_hind_push / cc_front_total：亏空量级 0.05~0.16，
    平方后剩 0.0025~0.026，价格被再砍一到两个数量级（`pw` 默认仍 2.0，B2m 行为不变）。

    起点 R29 `2026-09-18_12-40-39/model_19774.pt`（n=12：11/12、① dleg +0.003、③ 23.8）。
    **不从 R30 续**：R30 把支撑罚加到 −8 反而全面变差（侧倾 32.7、课程 1.09 破守卫），已判定 −4 是峰值。

    ## 验收靶与守卫（n=12 判定）

        靶：**机身离台面最低 0.089 → >=0.20**（官方 0.320）；③ 台上侧倾 23.8 → <=20
        守卫：上台面 >=10/12、翻越最高机身 z >=0.62、课程末>=2.0、
              ① dleg 不得低于 −0.01（R29 是 +0.003，别把蹬腿罚没了）

    环境变量：S10_B3M_W（80）、S10_B3M_THR（0.25）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B3M_W", "80"))
        thr = float(_os.environ.get("S10_B3M_THR", "0.25"))
        self.rewards.cc_top_low = RewTerm(func=climb.top_low_posture, weight=-w,
                                          params={"ratio": thr / 0.43, "nom": 0.43, "pw": 1.0})
        print(f"[s10-climb-B3m] 上台后机身不许沉下去: w=-{w:g} 阈值={thr:g}m 线性形 门=top_win | "
              f"先验：上台后机身离台面最低 官方 **0.320** / R13 0.182 / **R29 0.089**（按站高 87%/42%/21%）| "
              f"每局值 官方 **0.00000** / R29 0.01651 → w=80 我方约 −1.32、官方恒 0 | "
              f"静态翻倒角 30.1°，R29 侧倾 23.8° 只剩 21% 余量；石笼顶实测起伏 ±10~25cm | "
              f"靶 净空>=0.20、侧倾<=20；守卫 上台面>=10/12、z>=0.62、① dleg>=−0.01", flush=True)


@configclass
class DeeproboticsS10ClimbB3nEnvCfg(DeeproboticsS10ClimbB3mEnvCfg):
    r"""**B3n = B3m 的机身净空罚 80 → 200**（09-18 14:4x）。**单一改动。**

    ## R31（w=80）n=12：今晚第一个三条全线改善、且**全部不越翻倒线**的模型

        真正上台面 **12/12**   站住 **12/12**
        ① dleg          −0.043(R13) → **+0.009**（全场最高）
        ② 前腿塌陷      0.095(R13)  → **0.087**
        ③ 侧倾 中位 23.1（R13 18.2，退）/ **最差 26.8（R13 35.3）**
           **12/12 全部低于静态翻倒角 30.1°；R13 有 1 条 35.3° 越线、R29 有 1 条 60.9°**
        机身净空 0.079(R29) → **0.148**（被罚量 −1.32 → −0.34，降 74%）
        恢复 0.188（全场最好）  机身最高 z 0.749  撞速 1.05  课程末 3.04

    **比赛是一次性的：最差值比中位值更要命。按最差值口径，R31 已优于交付版 R13。**
    已存 `~/s10_logs/night/BEST2_R31/best2.onnx`（原 BEST=R13 未动）。

    ## 本轮：靶没到位，按同一杠杆再加一档

        预注册靶是 机身净空 >=0.20，实得 **0.148** —— 方向对、量没到
        官方 0.320，仍差一倍
        w 80 → **200**（每局 −0.34 → 约 −0.86；官方仍恒为 0）

    ## 预注册判读（n=12 判定）

        净空 >=0.20 且 侧倾最差 <30.1° 且 上台面 12/12 → 继续，**R31 被替换为新基线**
        净空没动                                      → 该杠杆见顶，停在 R31
        侧倾最差 >=30.1° 或 上台面 <11/12             → **回滚到 R31，定 R31 为交付候选**

    守卫：上台面 >=11/12、站住 >=11/12、翻越最高机身 z>=0.70、课程末>=2.0、
          **① dleg>=0**（R31 已是 +0.009，不许退回负）、② 前腿塌陷<=0.10

    环境变量：S10_B3M_W（本类默认 200）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B3M_W", "200"))
        self.rewards.cc_top_low.weight = -w
        print(f"[s10-climb-B3n] **单一改动**：机身净空罚 80 → {w:g} | "
              f"R31 n=12 **12/12 上台面 + 12/12 站住 + ① dleg +0.009 + 侧倾最差 26.8（全部低于翻倒线 30.1）** | "
              f"净空 0.079→0.148，靶 >=0.20 未到（官方 0.320）| "
              f"预注册：净空>=0.20 且 侧倾最差<30.1 且 12/12 → 继续；否则回滚 R31 定交付候选", flush=True)


@configclass
class DeeproboticsS10ClimbB3pEnvCfg(DeeproboticsS10ClimbB3mEnvCfg):
    r"""**B3p = R31 配置 + 落地 1.5 s 的侧倾罚**（09-18 15:4x）。**单一改动，补定价漏洞。**

    ## 为什么是这一项

    查到一个**定价漏洞**：`cc_roll_ph` w=−60 的门是 `phase`（翻越相位），
    **四轮一上台面、前方没墙，phase 立刻关掉，−60 随之消失**；那之后只剩全局 `cc_roll` w=−15，
    被 30 s episode 稀释到每局 **−0.299**。而**台上侧倾峰实测在上台后 0.4~0.6 s** ——
    **落地窗口是整套动作里侧倾定价最弱的一段。**

        cc_top_roll = clamp(|g_y| − sin10°, 0) × top_win，w=**400**
        每局值：官方 **0.00000**（三段最差也 0）/ R13 0.00099 / R31 0.00322 → R31 约 −1.29、官方恒 0

    ## 此前两轮的状态

        R31（净空罚 w=80）n=12：**12/12 上台面 + 12/12 站住 + ① dleg +0.009 + 侧倾最差 26.8（0/12 越线）**
        R32（净空罚 w=200）n=12：净空 **没再动**（0.158→0.151）→ 该杠杆见顶；
            侧倾中位更好（19.3）但**最差 30.2 压线**、① 掉回 +0.000 → 按预注册回滚，**R31 为交付候选**

    ## 粗糙台面旁证（09-18 15:2x，新建 `bench_top33_rough.xml`，93 个 3~8 cm 起伏块，种子 20260918）

        R13 6/6 上台面 6/6 站住 侧倾 16.8/21.5   R31 6/6 6/6 24.7/26.0   R32 6/6 6/6 25.5/29.3
        **三个模型都不会因为台面不平就翻**（0/6 越线）。
        限制：n=6；起伏块仅 3~8 cm；接近段仍是平地。

    起点 **R31** `2026-09-18_14-09-49/model_19973.pt`（不从 R32 续，净空罚 200 已判见顶）。

    ## 预注册判读（n=12）

        侧倾中位 <=18 且 最差 <30.1 且 上台面 12/12 且 ① dleg>=0 → **R31 被替换为新基线**
        侧倾没动                                                → 该漏洞不是主因，停手上报
        上台面 <11/12 或 ① dleg<0                               → 回滚 R31

    环境变量：S10_B3P_W（400）、S10_B3P_THR（10.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B3P_W", "400"))
        thr = float(_os.environ.get("S10_B3P_THR", "10.0"))
        self.rewards.cc_top_roll = RewTerm(func=climb.top_roll_penalty, weight=-w, params={"thr_deg": thr})
        print(f"[s10-climb-B3p] **补定价漏洞**：落地 1.5s 侧倾罚 w=-{w:g} 阈值={thr:g}° 门=top_win | "
              f"漏洞：cc_roll_ph(−60) 的门是 phase，**四轮上台后 phase 关掉、−60 消失**，"
              f"而侧倾峰在上台后 0.4~0.6s；那段只剩全局 −15（每局 −0.299）| "
              f"每局值 官方 **0.00000** / R13 0.00099 / R31 0.00322 → R31 约 −1.29、官方恒 0 | "
              f"靶 侧倾中位<=18、最差<30.1、12/12、① dleg>=0", flush=True)


@configclass
class DeeproboticsS10ClimbB3qEnvCfg(DeeproboticsS10ClimbB3mEnvCfg):
    r"""**B3q = R31 配置 + 台上后膝不许折到限位**（09-18 16:2x）。**单一改动。**

    ## 本轮的依据：推翻了我自己上一条「撑不住」的说法

    我先前写「机身沉下去是力矩撑不住」。**查关节角后证伪**：机身最低那一刻

        后膝角  官方 **2.185**（腿长 0.207 m）  /  R31 **2.693**（腿长 0.147 m）
        **关节限位 2.7227 —— 离限位只剩 0.029 rad，顶死了**

    **不是撑不住，是策略主动把后腿折到机械限位。** 官方的后腿比我们长 6 厘米。
    这与「净空 vs 侧倾」轮次内相关 **−0.83 / −0.82 / −0.88 / −0.97（连续四轮）** 接成完整因果：
    **后腿折死 → 机身低 → 侧倾大。**

    ## 与 B2g 的关系

    `on_top_knee_limit` 罚的就是这件事、lim 也是 2.30，但**门是「四轮绝对高度>0.18」，
    而 z 相对 env 出生点 → 多级地形上爬过任何东西后永久为真**，罚落到抬前腿上，登顶 6/6→2/6。
    **那是门的 bug，不是想法失败。** 本项用后来修好的 `top_win`（状态转变计时），**从没被有效检验过**。

    ## 标定

        每局值 官方 **0.00000**（三段最差也 0）/ R13 0.01678 / R31 0.01896 → w=80 我方约 −1.52、官方恒 0

    ## 此前的收敛状态（全部 n=12）

        R31 上台面 12/12、站住 12/12、① dleg +0.009、③ 侧倾最差 26.8（**0/12 越 30.1° 翻倒线**）、净空 0.158
        R32 净空罚加价 → 净空见顶（0.151）           R33 落地侧倾罚 → 侧倾不动（24.4）、① 转负
        R34 同配置续训 → 无改善（侧倾 25.2、净空 0.096）**⇒ R31 已收敛，是这条线的峰值**

    起点 **R31** `2026-09-18_14-09-49/model_19973.pt`。

    ## 预注册判读（n=12）

        后膝峰 <=2.45 且 净空 >=0.20 且 侧倾中位 <=20 且 12/12 且 ① dleg>=0 → **替换 R31 为新基线**
        后膝峰没动（>2.6）                                                  → 加价一轮到 w=240
        上台面 <11/12 或 ① dleg<0                                           → 回滚 R31

    环境变量：S10_B3Q_W（80）、S10_B3Q_LIM（2.30）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import climb_rewards as climb
        w = float(_os.environ.get("S10_B3Q_W", "80"))
        lim = float(_os.environ.get("S10_B3Q_LIM", "2.30"))
        self.rewards.cc_top_knee2 = RewTerm(func=climb.top_hind_knee_limit, weight=-w, params={"lim": lim})
        print(f"[s10-climb-B3q] **台上后膝不许折到限位** w=-{w:g} lim={lim:g} 门=top_win（修好的那个）| "
              f"实测机身最低那刻后膝：官方 **2.185**（腿 0.207）/ R31 **2.693**（腿 0.147），"
              f"**限位 2.7227，离限位只剩 0.029 rad —— 不是撑不住，是主动折死** | "
              f"每局值 官方 **0.00000** / R31 0.01896 → 我方约 −1.52 | "
              f"B2g 试过同一想法但门有 bug（绝对高度门在多级地形永久为真，6/6→2/6），**本项从没被有效检验过**", flush=True)


@configclass
class DeeproboticsS10ClimbB3rEnvCfg(DeeproboticsS10ClimbB3qEnvCfg):
    r"""**B3r = B3q 的台上后膝罚 80 → 240**（09-18 17:0x）。按 B3q 预注册执行，**单一改动**。

    ## R35（w=80）n=12：机制确认，且分布说清了 ③ 到底是什么问题

        侧倾  18.6 19.2 19.2 19.7 19.8 20.6 20.7 20.8 21.5 │ 27.9 28.8 50.0
        净空  .196 .229 .226 .236 .227 .247 .245 .247 .248 │ .090 .096 .103

    **完全二元，界线就是机身有没有撑住**：净空 >=0.17 → 侧倾 18.6~21.5°；净空 <=0.14 → 25.5~50°。
    R31 是 6/12 撑住，**R35 是 9/12**。

    ⇒ **③「不摔倒」不是「把侧倾压低」，是「让每一条都撑住」。**
      撑住那 9 条的最差只有 **21.5°**；12 条全进这一档，③ 就明显优于 R13（中位 18.2 / 最差 35.3）。

    ## R35 其它读数

        净空中位 0.158 → **0.228**（靶 >=0.20 **达成**）  ① dleg +0.009 → **+0.015**（全场最高）
        翻越最高机身 z 0.749 → **0.799**（全场最高）      侧倾中位 23.1 → 20.6
        上台面 11/12、站住 11/12
        **后膝峰仍 2.720（没动，靶 <=2.45）** —— 但每局罚从标定 1.52 掉到 **0.605**，
        说明**峰值仍触限位、但停留时间降了约 60%**

    ## 本轮

    按 B3q 预注册「后膝峰没动（>2.6）→ 加价到 w=240」。目标是把撑住的比例从 9/12 推到 12/12。

    **预注册判读（n=12）**：

        撑住（净空>=0.17）>=11/12 且 侧倾最差 <30.1 → **替换 R31 为交付基线**
        撑住比例没涨                                → 该杠杆见顶，**回 R31 定稿**
        上台面 <11/12 或 ① dleg<0                   → 回滚 R35

    起点 **R35** `2026-09-18_16-02-05/model_20172.pt`（11/12、① +0.015、净空 0.228）。

    环境变量：S10_B3Q_W（本类默认 240）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        w = float(_os.environ.get("S10_B3Q_W", "240"))
        self.rewards.cc_top_knee2.weight = -w
        print(f"[s10-climb-B3r] **单一改动**：台上后膝罚 80 → {w:g} | "
              f"R35 n=12：**撑住 9/12（R31 6/12）**、净空 0.158→**0.228**、① dleg→**+0.015**、机身 z→**0.799** | "
              f"分布二元：净空>=0.17 → 侧倾 18.6~21.5°；净空<=0.14 → 25.5~50° | "
              f"⇒ ③ 不是「压低侧倾」是「**让每条都撑住**」| "
              f"靶 撑住>=11/12 且 侧倾最差<30.1 → 替换 R31；没涨 → 回 R31 定稿", flush=True)


@configclass
class DeeproboticsS10TrotPhaseT10EnvCfg(DeeproboticsS10PerceptV1HTrotT10EnvCfg):
    r"""**Trot-Phase-T10（09-18 17:0x，作者拍板「直接上乙」）：观测 244 → 246，加步态相位钟。**

    ## 与已有 `DeeproboticsS10TrotPhaseEnvCfg` 的唯一区别：**起点换成 T10**

    那个类继承 **T11**，而 T11 在 09-15 已被判无效：
    「T11（tent 核）、T12（时间归一）、T13（权重 600）三轮打的都是症状」，
    且 **T11 把 15° 坡偏航打到 −20°**，文档写死「T11 无效，不做起点」。
    本类继承 **T10_15197**（`md5 ebef4dd2`，文档最终交付候选）。

    ## 为什么必须进观测（结构性结论，非调参）

    这条线**本来就有官方步频的节拍钟**：`st_trot_ref`（`TrotRefTrack`，权重 6.0，
    训练读数 1.41~1.64，量级足够），载入 `ref_step_official_v2.npz`，频率 **1.1 Hz** —— 就是官方步频。
    **钟按 1.1 Hz 走，策略跑 3.2 Hz，照样拿分。**

    原因：奖励是 `exp(−mean((q−q_ref)²)/σ²)`，而**观测里没有相位** →
    策略跑 3.2 Hz 时奖励**对相位取平均**，得到近似常数（实测 raw 0.24，每关节 RMS 误差 0.59 rad），
    **平均之后梯度就没了**。要拿满分需**同时**「降到 1.1 Hz」**且**「相位对齐」，
    而相位漂掉后没有反馈能纠回（RSI 只在 episode 开头对一次）——两个都必须同时猜对的稀疏目标。

    **加 sin/cos 两维后，既有的那一整套步频项（`step_rate_penalty` / `sg_swing_time` /
    `sg_diag_sync` / `knee_amp_penalty` / `lr_antiphase_reward`）立刻获得梯度。**

    ## 差距

        官方   步频 **恒 1.10 Hz**（0.5~1.9 m/s 全速段不变）  摆动 **恒 0.35 s**  靠**步幅**提速
        T10    步频 **3.23 / 3.39 / 3.77 Hz**（@0.5/1.0/1.8）  摆动 **0.13 s**  步幅约官方 40%

    ## 两处改动

    1. **观测**：末尾追加 `gait_phase`（2 维 sin/cos，1.10 Hz **开环**钟）→ **246**。
       **追加在 244 之后、不插中间** → runner 只需末尾补两维，前 244 维排布完全不动。
    2. **奖励**：`gp_swing`（`phase_swing_track`）把对角腿摆动绑到钟上 ——
       钟只是"看得见"，还得有东西把动作绑上去。摆动判定复用 `step_gait._swing`，与既有步态项同口径。
       该项是**双向惩罚**：不抬腿（步频 0）在"该摆"的半周期照样被罚，
       治的正是 [[project-s10-sim2sim-gait]] 记的那个退化终态（四腿锁死、纯轮子滚）。

    ## 同频同相（**这是整条路唯一的真机风险，写死在这里**）

        训练侧：φ += freq × env.step_dt，env.step_dt = decimation 4 × sim.dt 0.005 = **0.02 s（50 Hz）**
                回合重置（episode_length_buf <= 1）时 φ 归零
        runner：控制周期同为 **50 Hz**，必须以**同一个 freq** 推进，**策略接管那一刻 φ 归零**
        钟是**开环**的（不依赖接触反馈）—— 两边只要频率一致就同相，这是它能跨 sim2real 的前提

    ## 验收

        步频      3.23/3.39/3.77 → **<=1.5 Hz**（官方 1.10）
        摆动时长  0.13 → **>=0.28 s**（官方 0.35）
        `gp_swing` 读数**必须非零**（为零说明钟没被用上，先查门，别加权重）
        不得回退：膝 σ（Isaac >=0.06）、左右反相 <=−0.70（T9/T10 成果）、
                  10°坡偏航 +7.0 / 15°坡 −1.2 不得明显变差、平地 roll p95 <=6.8

    环境变量：S10_GAIT_FREQ（1.10）、S10_TROTP_W（`gp_swing` 权重，默认 1.0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import gait_clock as _gc
        f = float(_os.environ.get("S10_GAIT_FREQ", "1.10"))
        _gc.GAIT_FREQ_HZ[0] = f
        # **policy 组 244 -> 246**（末尾追加，前 244 维排布不动 —— 部署侧只需末尾补两维）
        self.observations.policy.gait_phase = ObsTerm(func=_gc.gait_phase_obs,
                                                     params={"freq": f}, scale=1.0)
        # critic 组 247 -> 249：价值函数同样是相位相关的，不给它看相位会白白增加方差。
        # critic **不参与部署**，加它没有接口风险。
        self.observations.critic.gait_phase = ObsTerm(func=_gc.gait_phase_obs,
                                                     params={"freq": f}, scale=1.0)
        w = float(_os.environ.get("S10_TROTP_W", "1.0"))
        self.rewards.gp_swing = RewTerm(func=_gc.phase_swing_track, weight=-w,
                                        params={"freq": f, "h_min": 0.04, "v_min": 0.3})
        # 航向保持：低步频下每秒能用来纠偏的落脚次数变少，原 −8（T10 上守住 ±5°）压不住了。
        # 09-18 实测：步频降到 2.09 后偏航从 18.8° 漂到 **124.8°**（机体系速度跟踪其实是好的，
        # 1.8 档机体系 +1.86 vs 基线 +1.92 —— 之前误判成"方向反了"，是把世界系当机体系读了）。
        yw = float(_os.environ.get("S10_TROTP_YAWW", "0"))
        if yw > 0:
            self.rewards.st_yaw_ref.weight = -yw
        # 左右轮**动作**偏置罚（09-18）。实测偏航漂的直接来源不是腿、也不是力矩，
        # 是一个**持续的左右轮速偏置 −2.86 rad/s** → wz +0.117 rad/s → 8 s 漂 54°。
        # 为什么现有项都管不住：
        #   · `track_ang_vel_z_exp` σ=√0.5 → 0.117 rad/s 只让它从 1.0 掉到 0.973，**罚 2.7%，等于免费**
        #   · `st_yaw_ref` 已加价到 −24（读数 −21.4，全场第二大）**仍压不住** → 不是它的价钱问题
        #   · `wheel_drive_balance` 罚的是**力矩**不均衡（每局 −0.27），不是速度偏置，口径不对
        # `wheel_action_lr_imbalance` 直接罚策略输出：|a_fl−a_fr| + |a_hl−a_hr|，有前进指令且不转向才算。
        # 它在 `.bak_h6` / `.bak_c17` 里用过（w=−0.2），当前配置已丢失。
        # 其 docstring 原话：「09-12 StairN-B 用力矩不均衡 −0.5 训 1000 步没改动左前轮独大，
        # 力矩受触地状态影响、信号弱；**动作差直接、可微**。」—— 与本次诊断完全一致。
        lrw = float(_os.environ.get("S10_TROTP_LRW", "0"))
        if lrw > 0:
            from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_balance as _wb
            self.rewards.wheel_action_lr = RewTerm(func=_wb.wheel_action_lr_imbalance, weight=-lrw)
        print(f"[s10-trot-PhaseT10] 航向罚 w={self.rewards.st_yaw_ref.weight:g}｜左右轮动作偏置罚 w=-{lrw:g} | "
              f"[s10-trot-PhaseT10] 观测 244 -> **246**（末尾追加 sin/cos 相位钟 {f:g} Hz）| "
              f"**起点 T10_15197**（原 TrotPhase 类继承的 T11 已被判无效、不做起点）| "
              f"新增 gp_swing w=-{w:g}（对角摆动绑钟，双向罚：不抬腿也罚）| "
              f"靶 步频 3.23/3.39/3.77 -> <=1.5 Hz、摆动 0.13 -> >=0.28 s；"
              f"守卫 膝σ>=0.06、反相<=−0.70、坡上偏航不劣、平地 roll p95<=6.8", flush=True)


@configclass
class DeeproboticsS10TrotPhaseT10SafeEnvCfg(DeeproboticsS10TrotPhaseT10EnvCfg):
    r"""**Trot-Phase-T10-Safe（09-19，作者：「T10 摔倒你要想办法解决……在 246 里解决，并完成 246 压步频」）**

    ## 真机事故与根因

    09-18 21:24 T10_15197 平地 roll 一歪，|动作| 0.9 → 5 → 44 → 1e14（1 s），控制台倾角急停。
    根因 = **训练/部署口径不一致**：观测里「上一动作」训练侧 clip ±100、runner 不钳位；动作本身两边都不钳位。
    正反馈环（动作↑ → 下帧"上一动作"↑ → 动作↑）在仿真被 ±100 截断，真机没有闸。

    离线自激复现（`s10_dev/selfexcite_gate.py`，把上一帧动作喂回观测迭代 15 步，打限位线 |a|=10.9）：

        T10_15197   roll 0/20/30/45 → 1.8 / **11.4** / **13.6** / **16.2**   ✗
        246-lr8     → 4.2 / **12.5** / **15.4** / **18.9**   ✗（继承并加重）
        V1H/R13/S16 → 2~6（健康参照；V1H 真机 09-12/13 走过）

    **246 有同样的病，且更重——按作者指令在 246 里解决。**

    ## 四道闸 + 一个惩罚（让策略在闸内学习、且不依赖闸）

        ① 观测 `actions` 项 clip ±100 → **±4**（policy 与 critic 组）—— 训练里正反馈环有闸
        ② 动作 cfg clip（处理后单位）：hipx ±0.5 rad、hipy/knee ±1.0 rad、轮 ±20 rad/s —— 仿真执行有闸
           （IsaacLab `JointAction.clip` 作用在 scale 之后；hipx 缩放 0.125、腿 0.25、轮 5 → 原始 ±4）
        ③ runner action clamp 腿 |a| ≤ 4.0 —— 部署闸，**待作者拍板**（AGX 改法已备）
        ④ `big_action_penalty`：原始 |a| 超过 3.0 的铰链平方 —— 正常运行恒为 0（真机常态 0.9、峰 1.7），
           越界越远罚越重，让策略自己不进大动作区
        ⑤ 倾斜进分布：reset roll/pitch ±0.3 → **±0.45 rad（±26°）**，推力加 roll 角速度 ±0.8 rad/s
           —— T10 在 20° 就爆，因为 20°+ 在训练里几乎没见过

    其余沿用 246 lr8 那版（步频 1.31/1.35/1.54、yaw 0.5 档 4°）：gp_swing 12、航向罚 24、轮偏置罚 8。
    起点 `2026-09-18_21-29-42/model_17192.pt`（246 维，直接续，不用再补列）。

    ## 验收（每轮）

        自激门 `selfexcite_gate.py --pass 7`：roll 0/20/30/45 不动点全部 <7（V1H 水平），**硬门，不过不导出交付**
        步频 <=1.5 Hz、跟踪 1.0→>=0.8 / 1.8→>=1.5、yaw 跨度 <=20°、roll p95 <=7、膝σ>=0.06

    环境变量：S10_SAFE_ACLIP（4.0）、S10_SAFE_BIGW（1.0）、S10_SAFE_BIGLIM（3.0）、S10_SAFE_TILT（0.45）、S10_SAFE_PUSHROLL（0.8）、
    S10_SAFE_ACTNOISE（0；safe2 用 1.5）、S10_SAFE_ACTNOISE_WHEEL（0；safe2 全维时=1.5，safe3 腿 1.5/轮 0）。
    """

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import gait_clock as _gc
        ac = float(_os.environ.get("S10_SAFE_ACLIP", "4.0"))
        # ① 观测里的上一动作：训练里也给环上闸
        self.observations.policy.actions.clip = (-ac, ac)
        self.observations.critic.actions.clip = (-ac, ac)
        # ①b（safe2，09-19 00:0x）：safe1 跑 100 点门读数纹丝不动（2.6/11.8/14.5/18.0 vs lr8 4.2/12.5/15.4/18.9）——
        # 闸只把环截断，不给"降增益"的梯度。直接打机制：观测里的上一动作加均匀噪声 ±an，策略无法精确依赖它，
        # ∂a/∂a_prev 自然变小（critic 组不加，保持真值）。噪声在 clip 之前作用（IsaacLab 顺序 func→noise→clip→scale）。
        an = float(_os.environ.get("S10_SAFE_ACTNOISE", "0.0"))
        # safe3（09-19 00:5x）：safe2 全 16 维 ±1.5 把门读数压到增益 2.2（过增益门）但平地 yaw 跨度 28/84/360°、步频 0.56/0.77——
        # 左右轮协调靠"上一动作"里的轮列，一起加噪就丢了航向。改成逐维：腿 12 维加噪、轮 4 维单独给（默认 0 = 干净）。
        anw = float(_os.environ.get("S10_SAFE_ACTNOISE_WHEEL", "0.0"))
        if an > 0 or anw > 0:
            import torch as _torch
            from isaaclab.utils.noise import UniformNoiseCfg as _Unoise   # IsaacLab 6.1 名字（operation 默认 add；n_min/n_max 可为张量，逐维）
            _n = _torch.tensor([an] * 12 + [anw] * 4, dtype=_torch.float32)   # 动作序：每腿(hipx,hipy,knee)×fl,fr,hl,hr + 轮 12..15
            self.observations.policy.actions.noise = _Unoise(n_min=-_n, n_max=_n)
        # ② 动作执行钳位（处理后单位 = 原始 ±ac × 各自缩放）
        self.actions.joint_pos.clip = {".*_hipx_joint": (-0.125 * ac, 0.125 * ac),
                                       ".*_hipy_joint": (-0.25 * ac, 0.25 * ac),
                                       ".*_knee_joint": (-0.25 * ac, 0.25 * ac)}
        self.actions.joint_vel.clip = {".*_wheel_joint": (-5.0 * ac, 5.0 * ac)}
        # ④ 大动作铰链罚
        bw = float(_os.environ.get("S10_SAFE_BIGW", "1.0")); bl = float(_os.environ.get("S10_SAFE_BIGLIM", "3.0"))
        self.rewards.big_action = RewTerm(func=_gc.big_action_penalty, weight=-bw, params={"lim": bl})
        # ⑥（safe5，03:3x）终止罚：safe4 崩法 = 每局总奖励大负数（航向罚 −20 等累计 −480/局）而 bad_orientation_2 终止不罚（bad_orientation_penalty
        # 只在更大角度触发，读数 −0.2）→ "早摔早解脱"成最优：翻倒率 0.20→0.79、局长 850→370、均奖反升。终止罚 −W 让死比活完整一局更贵。
        tw = float(_os.environ.get("S10_SAFE_TERMW", "0.0"))
        if tw > 0:
            from isaaclab.envs.mdp import is_terminated as _is_term     # 非超时终止=1（IsaacLab 通用）
            self.rewards.termination_penalty = RewTerm(func=_is_term, weight=-tw)
        # ⑤ 倾斜进分布
        tilt = float(_os.environ.get("S10_SAFE_TILT", "0.45")); pr = float(_os.environ.get("S10_SAFE_PUSHROLL", "0.8"))
        rb = self.events.randomize_reset_base.params["pose_range"]
        rb["roll"] = (-tilt, tilt); rb["pitch"] = (-tilt, tilt)
        self.events.randomize_push_robot.params["velocity_range"]["roll"] = (-pr, pr)
        print(f"[s10-trot-PhaseT10-Safe] 观测上一动作 clip ±{ac:g}｜动作 clip hipx ±{0.125*ac:g} 腿 ±{0.25*ac:g} 轮 ±{5*ac:g}｜"
              f"大动作罚 w={bw:g} lim={bl:g}（正常恒 0）｜上一动作观测噪声 腿 ±{an:g} 轮 ±{anw:g}｜reset 倾斜 ±{tilt:g} rad、推 roll ±{pr:g} rad/s | "
              f"终止罚 w={tw:g}｜治 T10 真机自激（roll20° 不动点 11.4 → 门 <7）", flush=True)


@configclass
class DeeproboticsS10StairExpertS16bEnvCfg(DeeproboticsS10StairExpertS16EnvCfg):
    r"""Stair-S16b（09-19 01:0x，从 **S16_15200** 续 600 步，作者「今晚自主完成 S16 完善」）。**入口鲁棒化，不动奖励。**

    真机 09-18 21:34 决赛楼梯脚下 S16 在场 11 s 没起步。离线定论：不是策略（S16≡S8 同输入）、高程图不是主因；
    仿真复现（T10 主策略、台沿前静止 5 s 起步、偏航 ±15°）S16 全 6/6、1.8~2.1 s 起步 → 仿真复现不出来。
    真机与训练分布的已知差异，一次全放进 DR（每项都对应一条真机事实）：
      ① 出生偏航 ±0.3 → **±0.5 rad**（真机右前轮先顶沿=偏航；训练从没见过 >17°）
      ② 关节复位缩放 (1.0,1.0) → **(0.75,1.25)**（T10 停车姿态左右不对称：hr 膝相对 −0.27、hipy −0.32；训练从没见过非默认站姿起步）
      ③ 高程图腐蚀：整图偏移 p 0.05→**0.25**（幅 0.15；真机 +0.09）、前方沟 p 0.02→**0.10**（−0.40）、
         **新增幻墙** p **0.15**（3 列 −0.55；真机单帧 46% 格子 0.3~0.6 m 假墙）
      ④ 静止→起步：rel_standing_envs 0.25 / 重采样 5 s 已有，不改。
    判据（训完一次性）：30 条验收全"上楼成功"；复现矩阵（T10 静止起步 yaw 0/±15/25、V1H 滚动）6/6；自激门过；站住漂移 ≤0.15 m。
    环境变量：S10_S16B_YAW（0.5）、S10_S16B_JSCALE（0.25）、S10_S16B_POFF（0.25）、S10_S16B_PTR（0.10）、S10_S16B_PWALL（0.15）。"""

    def __post_init__(self):
        super().__post_init__()
        import os as _os
        yw = float(_os.environ.get("S10_S16B_YAW", "0.5")); js = float(_os.environ.get("S10_S16B_JSCALE", "0.25"))
        po = float(_os.environ.get("S10_S16B_POFF", "0.25")); pt = float(_os.environ.get("S10_S16B_PTR", "0.10")); pw = float(_os.environ.get("S10_S16B_PWALL", "0.25"))
        pc = float(_os.environ.get("S10_S16B_PCHAOS", "0.08"))   # 01:2x 按 AGX 量化加猛：真机最差帧 86% 格子是墙、跳变 p95 0.52 m
        self.events.randomize_reset_base.params["pose_range"]["yaw"] = (-yw, yw)
        self.events.randomize_reset_joints.params["position_range"] = (1.0 - js, 1.0 + js)
        hs = self.observations.policy.height_scan
        hs.noise.p_offset = po; hs.noise.offset_max = 0.15; hs.noise.p_trench = pt; hs.noise.trench_val = -0.40
        hs.noise.p_wall = pw; hs.noise.wall_cols = 8; hs.noise.wall_val = -0.55          # 幻墙带宽 2~8 列随机
        hs.noise.p_chaos = pc; hs.noise.chaos_frac = 0.7; hs.noise.chaos_lo = -0.60; hs.noise.chaos_hi = 0.05   # 整帧混沌（70% 格子随机墙/坑）
        # S16c（02:1x）：S16b 门读数随 DR 训练单调变坏（15200 5.80 → 15400 6.43 → 15600 7.42 → 15799 8.30，roll45），
        # 用 246 线验证过的机制：腿 12 维上一动作加噪（轮干净）+ |a|>3 铰链罚。默认 0 = S16b 原样。
        an = float(_os.environ.get("S10_S16B_ACTNOISE", "0.0")); bw = float(_os.environ.get("S10_S16B_BIGW", "0.0"))
        if an > 0:
            import torch as _torch
            from isaaclab.utils.noise import UniformNoiseCfg as _Unoise
            _n = _torch.tensor([an] * 12 + [0.0] * 4, dtype=_torch.float32)
            self.observations.policy.actions.noise = _Unoise(n_min=-_n, n_max=_n)
        if bw > 0:
            from rl_training.tasks.manager_based.locomotion.velocity.mdp import gait_clock as _gc
            self.rewards.big_action = RewTerm(func=_gc.big_action_penalty, weight=-bw, params={"lim": 3.0})
        # S16d（04:1x）：S16c 门 roll45 在检查点间乱跳（33/28/41/10/7.5）——楼梯专家从没见过大侧倾，roll45 响应是碰运气。
        # 出生倾斜 DR（roll/pitch ±tl rad）让它成为训练过的区域（246 线 safe 系用 ±0.45 后门读数单调下降）。默认 0 = 不动。
        tl = float(_os.environ.get("S10_S16B_TILT", "0.0"))
        if tl > 0:
            rb = self.events.randomize_reset_base.params["pose_range"]; rb["roll"] = (-tl, tl); rb["pitch"] = (-tl, tl)
        print(f"[s10-stairN-S16b] 入口鲁棒化：出生偏航 ±{yw:g} rad｜关节复位缩放 ±{js:g}｜高程图 偏移 p{po:g}/0.15 沟 p{pt:g}/-0.40 幻墙 p{pw:g}/2~8列/-0.55 整帧混沌 p{pc:g}/70%｜"
              f"腿上一动作噪声 ±{an:g}（轮 0）｜大动作罚 w={bw:g}｜出生倾斜 ±{tl:g} rad｜奖励不动", flush=True)
