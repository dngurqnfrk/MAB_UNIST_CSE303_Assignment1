from __future__ import annotations

from typing import Dict
import numpy as np
from bandit_core import BanditEnv


def student_algorithm1(env: BanditEnv, horizon: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    TODO:
    Implement your own exploration strategy.

    Suggested score form:
        score_i(t) = sample_mean_i(t) + bonus_i(t)
    """
    epsilon1 = 0.1
    
    K = env.K
    counts = np.zeros(K, dtype=int)
    reward_sums = np.zeros(K, dtype=float)
    reward_sq_sums = np.zeros(K, dtype=float)

    actions = []
    rewards = []

    total_counts = 0

    for t in range(horizon):
        scores = np.zeros(K, dtype=float)

        for i in range(K):
            if counts[i] == 0:
                scores[i] = np.inf
            else:
                mean_i = reward_sums[i] / counts[i]
                sd_i = np.sqrt(max(0, (reward_sq_sums[i] / counts[i] - (reward_sums[i] / counts[i]) ** 2)))

                # Placeholder: students should replace this.
                bonus_i = np.random.normal(0, sd_i / (counts[i]))

                scores[i] = mean_i + bonus_i

        total_counts += 1

        # SoftMax
        if np.any(counts == 0):
            arm = np.argmin(counts)
        else:
            temperature = 5 / total_counts

            means = scores

            logits = means / temperature
            logits = logits - np.max(logits)

            probs = np.exp(logits)
            probs = probs / np.sum(probs)

            arm = int(rng.choice(K, p=probs))
        
        r = env.pull(arm, rng)

        counts[arm] += 1
        reward_sums[arm] += r
        reward_sq_sums[arm] += r * r

        actions.append(arm)
        rewards.append(r)

    return {"actions": np.asarray(actions, dtype=int), "rewards": np.asarray(rewards, dtype=float)}

def student_algorithm2(env: BanditEnv, horizon: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    TODO:
    Implement your own exploration strategy.

    Suggested score form:
        score_i(t) = sample_mean_i(t) + bonus_i(t)
    """
    epsilon1 = 0.1
    
    K = env.K
    counts = np.zeros(K, dtype=int)
    reward_sums = np.zeros(K, dtype=float)
    reward_sq_sums = np.zeros(K, dtype=float)

    actions = []
    rewards = []

    total_counts = 0

    for t in range(horizon):
        scores = np.zeros(K, dtype=float)

        for i in range(K):
            if counts[i] == 0:
                scores[i] = np.inf
            else:
                mean_i = reward_sums[i] / counts[i]
                var_i = max(0, (reward_sq_sums[i] / counts[i] - (reward_sums[i] / counts[i]) ** 2))

                # 분산 및 표준편차 계산
                var_i = max(0, (reward_sq_sums[i] / counts[i] - mean_i ** 2))

                # 스케일에 무관하게 탐색하도록 표준편차(sd_i)를 곱해줌
                bonus_i = np.sqrt(2 * var_i * np.log(t + 1) / counts[i])
                correction_i  = 3 * np.log(t + 1) / counts[i]

                scores[i] = mean_i + bonus_i + correction_i

        total_counts += 1

        # SoftMax
        # if np.any(counts == 0):
        #     arm = np.argmin(counts)
        # else:
        #     temperature = 5 / total_counts

        #     means = reward_sums / counts

        #     logits = means / temperature
        #     logits = logits - np.max(logits)
        #     probs = np.exp(logits)
        #     probs = probs / np.sum(probs)

        #     arm = int(rng.choice(K, p=probs))
        if np.any(counts == 0):
            arm = np.argmin(counts)
        else:
            arm = int(np.argmax(scores))
        # if np.random.binomial(1, epsilon1):
        #     arm = int(rng.integers(0, K))
        
        r = env.pull(arm, rng)

        counts[arm] += 1
        reward_sums[arm] += r
        reward_sq_sums[arm] += r * r

        actions.append(arm)
        rewards.append(r)

    return {"actions": np.asarray(actions, dtype=int), "rewards": np.asarray(rewards, dtype=float)}
