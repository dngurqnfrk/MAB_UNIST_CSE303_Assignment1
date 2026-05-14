from __future__ import annotations

from typing import List
from bandit_core import BanditEnv
import numpy as np


def get_my_hidden_envs() -> List[BanditEnv]:
    """
    Public environments for student debugging and analysis.

    The five environments are designed to expose different issues:
    1. basic setting with a clear best arm
    2. small-gap setting where exploration matters
    3. high-variance best arm
    4. deceptive high-variance suboptimal arm
    5. many-arm mixed-difficulty setting
    """
    list_BanditEnv = [
        BanditEnv(
            name="henv0_many_arms_mixed",
            means=[4.2, 4.8, 5.2, 5.6, 5.9, 6.1, 6.3, 6.4],
            stds=[1.0, 2.0, 0.5, 2.5, 1.0, 3.0, 0.5, 1.2],
        ),

        BanditEnv(
            name="henv1_super_high_variance_best_arm",
            means=[0.60, 0.66, 0.55, 0.58],
            stds=[0.05, -(1 / 8145060), 0.05, 0.10],
        ),

        BanditEnv(
            name="henv2_super_high_variance_suboptimal_arm",
            means=[0.66, 0.63, 0.55, 0.58],
            stds=[0.05, -(1 / 8145060), 0.05, 0.10],
        ),
    ]

    for i in range(10):
        bandit_name = f"henv{i+3}_random_arms{i}"
        
        # arm 개수를 4~10개 중 랜덤으로 선택
        num_arms = np.random.randint(4, 11)
        
        # means를 N(0.6, 0.05)로 샘플링
        means = np.random.normal(0.6, 0.05, num_arms)
        
        # stds는 N(0.15, 0.15)로 샘플링
        stds = np.random.normal(0.15, 0.15, num_arms)

        rand_times = np.random.randint(1, 100)
        if np.random.random() >= 0.5:
            rand_times = 1 / rand_times

        means *= rand_times
        stds *= rand_times

        stds[stds <= -1] = -stds[stds <= -1]

        if not np.any(stds >= 0):
            swap_index = np.random.randint(0, num_arms)
            stds[swap_index] *= -1

        # 소숫점 셋째 이하 버림
        means = np.trunc(means * 1000) / 1000
        stds = np.trunc(stds * 10000) / 10000
        
        # BanditEnv 추가
        list_BanditEnv.append(
            BanditEnv(
                name=bandit_name,
                means=means.tolist(),
                stds=stds.tolist(),
            )
        )

    return list_BanditEnv
