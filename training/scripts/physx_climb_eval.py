"""在训练仿真（IsaacLab/PhysX）里直接量「爬单级立面」成功率 —— 与 MuJoCo 部署栈对照，回答「训练侧到底学没学会」。

改自 scripts/reinforcement_learning/rsl_rl/play.py。地形只保留 climb_up（Inverted：出生坑底，四周立面 H），
指令固定前进 v（航向锁 0），跑 T 秒，成功 = 机身 z 相对出生高度升高 ≥ 0.6×H 且 x 前进过立面。
用法（在 /root/dl/src/rl_training 下、isaac venv）：
  python3 s10_dev/physx_climb_eval.py --task PerceptV12-Deeprobotics-M20-v0 --checkpoint <model.pt> --wall_h 0.33 --speed 0.5 --num_envs 64 --secs 15 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--wall_h", type=float, default=0.33)
parser.add_argument("--terrain", type=str, default="climb_up", choices=["climb_up", "approach_riser", "step_down", "flat"], help="climb_up=坑底四面墙；approach_riser=平地助跑 2 m 正对一道立面（V13 主地形）")
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--secs", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", action="store_true", help="每 50 步打印指令/速度/位移，用于验证评测装置本身")
parser.add_argument("--dump_obs", type=str, default="", help="把每步的观测(全部环境)与机身位姿存成 npz，用于与 MuJoCo 部署栈逐维对比")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--restitution", type=str, default="", help="覆盖机器人材质恢复系数范围，如 0,0.1（训练配方是 0~0.7，怀疑轮子撞沿反弹）")
parser.add_argument("--friction", type=str, default="", help="覆盖机器人材质摩擦范围，如 0.8,1.2")
parser.add_argument("--video", type=str, default="", help="录像输出 mp4 路径；跟拍 --video_env 号机器人的右侧视角（与作者现场视频同机位）")
parser.add_argument("--video_env", type=int, default=0)
parser.add_argument("--video_fps", type=int, default=25, help="录像帧率；策略 50 Hz，每 50/fps 步取一帧")
parser.add_argument("--video_eye", type=str, default="0.8,-2.4,0.8", help="相机相对机器人根的位置 (x,y,z)，-y 为机器人右侧")
parser.add_argument("--video_lookat", type=str, default="0.7,0.0,0.2")
parser.add_argument("--video_res", type=str, default="960x540", help="录像分辨率 WxH；训练并行时用 640x360 省渲染时间")
parser.add_argument("--pose_rand", type=str, default="", help="出生位姿随机 'dx,dy,dyaw'，如 0.3,0.5,0.35：来向偏差鲁棒性检查（默认全部固定为 0）")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True      # 无头模式下离屏渲染必须开
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import importlib.metadata as metadata  # noqa: E402
import rl_training.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # ---- 地形：只留 climb_up，立面钉在 H ----
    gen = env_cfg.scene.terrain.terrain_generator
    if args_cli.terrain == "flat":
        # 09-14：平地，只为「Isaac 观测 vs MuJoCo 部署观测」逐列对拍用（踏步策略在 Isaac 干净、导到 MuJoCo 塌成三轮滚，
        # 反作弊奖励项在 Isaac 里读数全是 0 —— 先确认两边喂给策略的 244 维是不是同一个东西）。不判翻越成功。
        import isaaclab.terrains as _tg
        gen.sub_terrains = {"flat": _tg.MeshPlaneTerrainCfg(proportion=1.0)}
        dx_need = 1e9
    elif args_cli.terrain == "step_down":
        # 下坎：倒金字塔台阶，出生在顶面平台上，四周向下 wall_h。踏面给足，只考察"敢不敢下"。
        import isaaclab.terrains as _tg
        dx_need = 1.8          # 出生在中央平台，走出平台边缘 1.8 m 且下降 ≥0.6H 判为下坎成功
        gen.sub_terrains = {"step_down": _tg.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(args_cli.wall_h, args_cli.wall_h),
            step_width=1.2, platform_width=3.0, border_width=1.0, holes=False)}
    elif args_cli.terrain == "approach_riser":
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.approach_riser import MeshApproachRiserTerrainCfg
        ar = MeshApproachRiserTerrainCfg(proportion=1.0, riser_height_range=(args_cli.wall_h, args_cli.wall_h), lane_width=4.0)
        gen.sub_terrains = {"approach_riser": ar}
        dx_need = 2.3          # 立面在 +2.0 m；机身中心过了立面 0.3 m 且升高 ≥ 0.25 = 上了台面
    else:
        up = gen.sub_terrains["climb_up"]
        up.step_height_range = (args_cli.wall_h, args_cli.wall_h)
        up.proportion = 1.0
        gen.sub_terrains = {"climb_up": up}
        dx_need = 1.5
    gen.curriculum = False
    gen.num_rows, gen.num_cols = 8, 8
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    if hasattr(env_cfg.curriculum, "command_levels"):
        env_cfg.curriculum.command_levels = None
    # ---- 指令：固定前进 v，航向锁 0（heading_command 会把偏航拉回 0）----
    c = env_cfg.commands.base_velocity
    c.ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
    c.ranges.lin_vel_y = (0.0, 0.0)
    c.ranges.ang_vel_z = (0.0, 0.0)
    c.ranges.heading = (0.0, 0.0)
    c.rel_standing_envs = 0.0
    # 09-08：UniformThresholdVelocityCommand 另有 rel_zero_vel/only_ang_z/only_lin_y/only_lin_x 四类采样（训练配方 0.2/0.2/0.02/0.02），
    # 评测里全部归零，让 64 台都有前进指令（此前只有约 58% 在动，统计分母被"站立台"稀释，录像也可能拍到不动的机器人）
    for attr in ("rel_zero_vel_envs", "rel_only_ang_z_envs", "rel_only_lin_y_envs", "rel_only_lin_x_envs"):
        if hasattr(c, attr):
            setattr(c, attr, 0.0)
    c.resampling_time_range = (1e6, 1e6)
    # ---- 关随机与推力，局长够跑完 ----
    env_cfg.observations.policy.enable_corruption = False
    mat = getattr(env_cfg.events, "randomize_rigid_body_material", None)
    if mat is not None and args_cli.restitution:
        lo, hi = (float(x) for x in args_cli.restitution.split(","))
        mat.params["restitution_range"] = (lo, hi); print("EVAL_SETUP 恢复系数范围覆盖为 (%.2f, %.2f)" % (lo, hi))
    if mat is not None and args_cli.friction:
        lo, hi = (float(x) for x in args_cli.friction.split(","))
        mat.params["static_friction_range"] = (lo, hi); mat.params["dynamic_friction_range"] = (lo, hi); print("EVAL_SETUP 摩擦范围覆盖为 (%.2f, %.2f)" % (lo, hi))
    for attr in ("randomize_apply_external_force_torque", "push_robot"):
        if hasattr(env_cfg.events, attr):
            setattr(env_cfg.events, attr, None)
    env_cfg.episode_length_s = args_cli.secs + 10.0
    if args_cli.num_envs <= 64:      # 09-10：与训练并行时显存紧，PhysX GPU 缓冲按 4096 台默认值会 OOM → 小规模评测缩小
        phys = getattr(env_cfg.sim, "physics", None) or getattr(env_cfg.sim, "physx", None)
        for k, v in dict(gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**17, gpu_found_lost_pairs_capacity=2**20,
                         gpu_found_lost_aggregate_pairs_capacity=2**20, gpu_total_aggregate_pairs_capacity=2**20).items():
            if phys is not None and hasattr(phys, k):
                setattr(phys, k, v)
    if getattr(env_cfg.events, "spawn_at_wall", None) is not None:
        env_cfg.events.spawn_at_wall = None          # 评测必须从 2 m 外正常走过来，不用训练里的墙前出生
    # 出生朝 +x、不偏移：遍历所有带 pose_range 的事件项（名字不一定叫 reset_base；之前 try/except 吞掉了失败 → 随机朝向 → dx 只有 0.28 m）
    fixed = []
    for name, term in vars(env_cfg.events).items():
        params = getattr(term, "params", None)
        if isinstance(params, dict) and "pose_range" in params:
            pr = dict(params["pose_range"]); pr["yaw"] = (0.0, 0.0); pr["x"] = (0.0, 0.0); pr["y"] = (0.0, 0.0)
            if args_cli.pose_rand:
                dx, dy, dyaw = (float(v) for v in args_cli.pose_rand.split(","))
                pr["x"] = (-dx, dx); pr["y"] = (-dy, dy); pr["yaw"] = (-dyaw, dyaw)
            params["pose_range"] = pr; fixed.append(name)
    if hasattr(c, "heading_command"):
        c.heading_command = True          # 有航向锁就让它把偏航拉回 0
    print("EVAL_SETUP 固定出生位姿的事件项: %s  heading_command=%s" % (fixed, getattr(c, "heading_command", None)))

    writer = None
    v_eye = v_lookat = None
    _viz = getattr(args_cli, "visualizer", None) or []
    gui = (("kit" in _viz) or not args_cli.headless) and not args_cli.video   # IsaacLab 6：默认无头，桌面窗口要 --viz kit
    if gui:
        # 桌面实时回放：GUI 视口跟拍 --video_env 号机器人（GUI 模式下 asset_root 跟随回调会正常触发）
        v = env_cfg.viewer
        v.origin_type = "asset_root"; v.asset_name = "robot"; v.env_index = args_cli.video_env
        v.eye = tuple(float(x) for x in args_cli.video_eye.split(","))
        v.lookat = tuple(float(x) for x in args_cli.video_lookat.split(","))
        print("GUI 实时回放：视口跟拍 env %d  eye=%s lookat=%s" % (v.env_index, v.eye, v.lookat))
    if args_cli.video:
        import cv2, os
        # 用一台独立的 Camera 传感器做跟拍（无头模式下 viewport 相机不跟随，实测两次都只拍到世界原点）：
        # 每步把相机放到 机器人根 + eye 偏移、看向 机器人根 + lookat 偏移，方向按世界系（−y = 机器人右侧，与作者现场视频同机位）
        # 必须每个 env 一台（全局单台相机在 scene.reset(env_ids) 时按 env 号索引会越界 → CUDA device-side assert）；
        # 用 TiledCamera 一次渲染全部，只有 --video_env 那台每步被摆到机器人旁边。
        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils
        env_cfg.scene.follow_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/FollowCam", update_period=1.0 / args_cli.video_fps, height=int(args_cli.video_res.split("x")[1]), width=int(args_cli.video_res.split("x")[0]), data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 80.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -2.6, 0.9), convention="world"))
        v_eye = [float(x) for x in args_cli.video_eye.split(",")]
        v_lookat = [float(x) for x in args_cli.video_lookat.split(",")]
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video)), exist_ok=True)
        print("VIDEO 跟拍 env %d  eye=%s lookat=%s → %s @%d fps" % (args_cli.video_env, v_eye, v_lookat, args_cli.video, args_cli.video_fps))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obs, _ = env.reset()
    p0 = robot.data.root_pos_w.clone()
    n = p0.shape[0]
    success = torch.zeros(n, dtype=torch.bool, device=p0.device)
    max_dz = torch.zeros(n, device=p0.device)
    max_dx = torch.zeros(n, device=p0.device)
    fell = torch.zeros(n, dtype=torch.bool, device=p0.device)
    steps = int(args_cli.secs / env.unwrapped.step_dt)
    # 约 1/3 环境被采成「站立指令」（rel_standing_envs 覆盖不生效），只在有前进指令的环境里统计
    cmd0 = env.unwrapped.command_manager.get_command("base_velocity")
    moving = cmd0[:, 0] > 0.1
    if args_cli.debug:
        cm = env.unwrapped.command_manager
        print("DEBUG step_dt=%.4f steps=%d  出生 z 中位 %.3f  地形原点 z 中位 %.3f" % (
            env.unwrapped.step_dt, steps, float(p0[:, 2].median()), float(env.unwrapped.scene.terrain.env_origins[:, 2].median())))
    dump_obs, dump_pos, dump_act = [], [], []
    for k in range(steps):
        if args_cli.dump_obs:
            o = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
            dump_obs.append(o.detach().cpu().numpy().copy())
            dump_pos.append(torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], 1).detach().cpu().numpy().copy())
        with torch.inference_mode():
            actions = policy(obs)
        if args_cli.dump_obs:
            dump_act.append(actions.detach().cpu().numpy().copy())
        # 09-08 修复：env.step 曾被缩进在 dump_obs 分支里，不带 --dump_obs 时环境根本不 step。
        obs, _, dones, _ = env.step(actions)
        p = robot.data.root_pos_w
        dz = p[:, 2] - p0[:, 2]
        dx = p[:, 0] - p0[:, 0]
        max_dz = torch.maximum(max_dz, dz)
        max_dx = torch.maximum(max_dx, dx)
        if args_cli.video:
            cam = env.unwrapped.scene["follow_cam"]
            rp = robot.data.root_pos_w
            rp = rp.torch if hasattr(rp, "torch") else rp
            root = rp[args_cli.video_env]
            eye = root + torch.tensor(v_eye, device=root.device, dtype=root.dtype)
            tgt = root + torch.tensor(v_lookat, device=root.device, dtype=root.dtype)
            cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0), env_ids=[args_cli.video_env])
            every = max(1, int(round(1.0 / (env.unwrapped.step_dt * args_cli.video_fps))))
            if k % every == 0 and k > 0:
                import numpy as np
                out = cam.data.output["rgb"]
                out = out.torch if hasattr(out, "torch") else out
                frame = out[args_cli.video_env].detach().cpu().numpy()
                fr = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])      # RGB→BGR
                if writer is None:
                    writer = cv2.VideoWriter(args_cli.video, cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (fr.shape[1], fr.shape[0]))
                t_s = (k + 1) * env.unwrapped.step_dt
                cv2.putText(fr, "%s  H=%.2f  v=%.2f  t=%.1fs  dz=%.2f dx=%.2f" % (args_cli.checkpoint.split("/")[-1], args_cli.wall_h, args_cli.speed, t_s,
                            float(dz[args_cli.video_env]), float(dx[args_cli.video_env])), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                writer.write(fr)
        if args_cli.terrain == "step_down":
            success |= (dz < -0.6 * args_cli.wall_h) & (dx.abs() > dx_need)   # 下坎：高度要降下去
        else:
            success |= (dz > 0.6 * args_cli.wall_h) & (dx > dx_need)      # 09-08：原来写死 dz>0.25，低墙评测全判失败
        if args_cli.debug and (k % 50 == 0 or k == steps - 1):
            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            v = robot.data.root_lin_vel_b
            qw = robot.data.root_quat_w
            qw = qw.torch if hasattr(qw, "torch") else qw
            x_, y_, z_, w_ = qw[:, 0], qw[:, 1], qw[:, 2], qw[:, 3]          # IsaacLab 6（warp）四元数是 xyzw
            yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ ** 2 + z_ ** 2))
            gb = robot.data.projected_gravity_b
            gb = gb.torch if hasattr(gb, "torch") else gb
            print("DEBUG k=%4d  cmd vx/vy/wz 均值 %.2f/%.2f/%.2f  实际 vx 均值 %.2f 中位 %.2f  dx 中位 %.2f  dz 中位 %.3f  偏航中位 %.0f°  抬头(-gx)中位 %.2f  终止累计 %d" % (
                k, float(cmd[:, 0].mean()), float(cmd[:, 1].mean()), float(cmd[:, 2].mean()),
                float(v[:, 0].mean()), float(v[:, 0].median()), float(dx.median()), float(dz.median()),
                float(torch.rad2deg(yaw).median()), float((-gb[:, 0]).median()), int(fell.sum())))
        if hasattr(dones, "bool"):
            fell |= dones.bool() & ~success       # 局内被终止（摔/超限）且未成功
    nm = int(moving.sum()); s = int((success & moving).sum()); f = int((fell & ~success & moving).sum())
    print("PHYSX_CLIMB task=%s ckpt=%s terrain=%s H=%.2f v=%.2f envs=%d(有前进指令 %d) secs=%.0f  成功 %d/%d (%.0f%%)  终止未成功 %d  max_dz中位 %.2f  max_dx中位 %.2f" % (
        args_cli.task, args_cli.checkpoint.split("/")[-1], args_cli.terrain, args_cli.wall_h, args_cli.speed, n, nm, args_cli.secs,
        s, nm, 100.0 * s / max(1, nm), f, float(max_dz[moving].median()), float(max_dx[moving].median())))
    if args_cli.dump_obs:
        import numpy as np
        np.savez_compressed(args_cli.dump_obs, obs=np.stack(dump_obs), pos=np.stack(dump_pos), act=np.stack(dump_act), p0=p0.detach().cpu().numpy(), step_dt=env.unwrapped.step_dt)
        print("OBS_DUMP 已存 %s  obs %s" % (args_cli.dump_obs, np.stack(dump_obs).shape))
    if writer is not None:
        writer.release()
        try:
            import imageio_ffmpeg, subprocess, shutil
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = args_cli.video + ".h264.mp4"
            subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", args_cli.video, "-c:v", "libx264", "-preset", "fast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp], check=True)
            shutil.move(tmp, args_cli.video)
            print("VIDEO 已写 %s（H.264）" % args_cli.video)
        except Exception as e:      # 转码失败就留 mp4v 原件
            print("VIDEO 已写 %s（mp4v，转 H.264 失败: %s）" % (args_cli.video, e))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
