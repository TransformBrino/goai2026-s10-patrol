# S10 感知策略任务族（2026-09-08 新机合并）：V16 配方原样，只把机器人换成与部署 runner 对齐的 S10 资产。
import gymnasium as gym

from . import agents

gym.register(
    id="PerceptS10-V0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptPPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-Climb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-FlatFast14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10FlatFast14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1Stair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1StairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1Down-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1DownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1H-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbH-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-FlatFast16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10FlatFast16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-FlatFast15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10FlatFast15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-FlatFast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10FlatFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-Trot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbG-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertGEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1G-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1GEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1DownB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1DownBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNB2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertB2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairND-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotV-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNE3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertE3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNE6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertE6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNE7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertE7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNE8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertE8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotH7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotH7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairNS14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC17-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC17EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC18EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC19-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC19EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC21-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC21EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC22-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC22EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairS15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC23-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC23EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairS16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairS16b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS16bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC24-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC24EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-V1HTrotT13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10PerceptV1HTrotT13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC25-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC25EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-StairS17-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10StairExpertS17EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC26-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC26EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC27EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC28-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC28EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC29EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC30-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC30EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC31-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC31EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC32-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC32EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC33-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC33EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC34-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC34EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC36-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC36EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC37-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC37EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC38-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC38EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbC39-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbExpertC39EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBranchA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBranchAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbApproach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbApproachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbMirror-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbMirrorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk3c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk3cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk14EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotPhase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk16EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk17-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk17EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk18EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk19-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk19EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbBlk21-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbBlk21EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2d-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2eEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2f-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2fEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2g-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2gEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2h-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2hEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2i-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2iEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2j-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2jEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2k-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2kEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2l-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2lEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2m-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2mEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2n-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2nEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2p-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2pEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2r-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2rEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2sEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2u-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2uEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2v-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2vEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2wEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2x-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2xEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2y-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2yEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB2z-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB2zEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3d-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3eEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3f-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3fEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3g-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3gEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3h-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3hEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3i-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3iEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3j-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3jEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3k-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3kEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3m-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3mEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3n-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3nEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3p-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3pEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3q-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3qEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-ClimbB3r-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10ClimbB3rEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotPhaseT10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotPhaseT10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)

gym.register(
    id="PerceptS10-TrotPhaseT10Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.s10_percept_env_cfg:DeeproboticsS10TrotPhaseT10SafeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsS10PerceptV1PPORunnerCfg",
    },
)
