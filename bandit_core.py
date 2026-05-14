from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np


@dataclass
class BanditEnv:
    name: str
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        self.stds = np.asarray(self.stds, dtype=float)

        if self.means.shape != self.stds.shape:
            raise ValueError("means and stds must have the same shape.")
        if np.any(self.stds < 0):
            print("[Warrning!] Stds have negative std")
            # raise ValueError("stds must be nonnegative.")

        self.K = int(self.means.shape[0])
        self.best_mean = float(np.max(self.means))
        self.best_arm = int(np.argmax(self.means))

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        # Binomial
        if self.stds[arm] < 0:
            # First get mean of lower -3σ
            mean3sig = 0
            i = 0
            # print(f"[pull] self.means{self.means}, self.stds{self.stds}")
            # print(f"[pull] Shape : means{self.means.shape} | stds{self.stds.shape}")
            for m, s in zip(self.means, self.stds):
                mean3sig += m - 3 * s if s >= 0 else 0
                i += 1 if s >= 0 else 0

            if i == 0:
                raise ValueError("At least one arm must have non-negative std")

            mean3sig /= i
            lotto_probability = -self.stds[arm]

            if not (0 <= lotto_probability <= 1):
                 raise ValueError(f"Invalid probability: {lotto_probability}. Must be in [0, 1]")
            
            lotto_prize = (self.means[arm] - (1 - lotto_probability) * mean3sig)
            lotto_prize /= lotto_probability
            
            return lotto_prize if rng.binomial(1, lotto_probability) else mean3sig

        # Normal Distribution
        return float(rng.normal(loc=self.means[arm], scale=self.stds[arm]))


def expected_cumulative_regret(actions: np.ndarray, means: np.ndarray, best_mean: float) -> np.ndarray:
    """
    Pseudo-regret:
        sum_t (mu_* - mu_{a_t})

    This is preferred to realized noisy regret for high-variance Gaussian rewards.
    """
    chosen_means = means[actions]
    return np.cumsum(best_mean - chosen_means)


def run_single_episode(
    env: BanditEnv,
    algo_fn: Callable[[BanditEnv, int, np.random.Generator], Dict[str, np.ndarray]],
    horizon: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = algo_fn(env, horizon, rng)

    if "rewards" not in result or "actions" not in result:
        raise ValueError("Algorithm must return a dict with 'rewards' and 'actions'.")

    rewards = np.asarray(result["rewards"], dtype=float)
    actions = np.asarray(result["actions"], dtype=int)

    if len(rewards) != horizon or len(actions) != horizon:
        raise ValueError("Length of rewards/actions must equal horizon.")
    if np.any(actions < 0) or np.any(actions >= env.K):
        raise ValueError("Algorithm returned invalid arm index.")

    regrets = expected_cumulative_regret(actions, env.means, env.best_mean)

    return {
        "rewards": rewards,
        "actions": actions,
        "regrets": regrets,
        "total_reward": np.array([np.sum(rewards)], dtype=float),
        "final_regret": np.array([regrets[-1]], dtype=float),
    }


def evaluate_algorithm(
    envs: List[BanditEnv],
    algo_fn: Callable[[BanditEnv, int, np.random.Generator], Dict[str, np.ndarray]],
    horizon: int,
    seeds: List[int],
) -> Dict[str, object]:
    per_env = []

    for env in envs:
        env_rewards = []
        env_final_regrets = []
        env_regret_curves = []

        for seed in seeds:
            result = run_single_episode(env, algo_fn, horizon, seed)
            env_rewards.append(float(result["total_reward"][0]))
            env_final_regrets.append(float(result["final_regret"][0]))
            env_regret_curves.append(result["regrets"])

        regret_curve_mean = np.mean(np.stack(env_regret_curves, axis=0), axis=0)

        per_env.append(
            {
                "env_name": env.name,
                "avg_total_reward": float(np.mean(env_rewards)),
                "std_total_reward": float(np.std(env_rewards, ddof=1)) if len(seeds) > 1 else 0.0,
                "avg_final_regret": float(np.mean(env_final_regrets)),
                "std_final_regret": float(np.std(env_final_regrets, ddof=1)) if len(seeds) > 1 else 0.0,
                "avg_regret_curve": regret_curve_mean,
            }
        )

    return {
        "per_env": per_env,
        "avg_reward_all": float(np.mean([x["avg_total_reward"] for x in per_env])),
        "avg_final_regret_all": float(np.mean([x["avg_final_regret"] for x in per_env])),
    }
