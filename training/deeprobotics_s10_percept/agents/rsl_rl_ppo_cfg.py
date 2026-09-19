# S10 感知策略的 PPO runner 配置：沿用 M20 percept 的全部超参，只换 experiment_name（独立目录树，血脉不混）。
from isaaclab.utils.configclass import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20PerceptPPORunnerCfg,
)


@configclass
class DeeproboticsS10PerceptPPORunnerCfg(DeeproboticsM20PerceptPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_s10_percept"


@configclass
class DeeproboticsS10PerceptV1PPORunnerCfg(DeeproboticsS10PerceptPPORunnerCfg):
    """V1：从盲策略热启动时初始学习率 1e-3 → 3e-4。09-08 首跑前 50 迭代策略剧烈漂移塌成趴低爬行；
    新增的 187 列高程图输入从零起、价值函数从零起，优势噪声大，起步步子要小。adaptive 调度仍按 KL 自动升降。"""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
