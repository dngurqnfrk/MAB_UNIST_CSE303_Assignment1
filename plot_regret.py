from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from algorithms_baseline import greedy, epsilon_greedy, softmax_policy
from algorithms_student import student_algorithm1, student_algorithm2
from bandit_core import run_single_episode
from public_envs import get_public_envs

from my_hidden_envs import get_my_hidden_envs


def compute_regret_curves(env, algo_fn, horizon, seeds):
    all_curves = []

    for seed in seeds:
        result = run_single_episode(env, algo_fn, horizon, seed)
        all_curves.append(result["regrets"])

    all_curves = np.stack(all_curves, axis=0)
    mean_curve = np.mean(all_curves, axis=0)
    stderr_curve = np.std(all_curves, axis=0, ddof=1) / np.sqrt(len(seeds))

    return mean_curve, stderr_curve


def main() -> None:
    envs = get_public_envs()
    envs = envs + get_my_hidden_envs()

    horizon = 20000
    seeds = list(range(200))

    algos = {
        "greedy": greedy,
        "epsilon_greedy": epsilon_greedy,
        "softmax_policy": softmax_policy,
        "student_algorithm1": student_algorithm1,
        "student_algorithm2": student_algorithm2,
    }

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    for env in envs:
        plt.figure(figsize=(8, 5))

        for algo_name, algo_fn in algos.items():
            mean_curve, stderr_curve = compute_regret_curves(env, algo_fn, horizon, seeds)
            x = np.arange(len(mean_curve))

            plt.plot(x, mean_curve, label=algo_name)
            plt.fill_between(
                x,
                mean_curve - 1.645 * stderr_curve,
                mean_curve + 1.645 * stderr_curve,
                alpha=0.2,
            )

        plt.xlabel("Round")
        plt.ylabel("Expected cumulative regret")
        means_str = ', '.join([f'{m:.2f}' for m in env.means])
        stds_str = ', '.join([f'{s:.2f}' for s in env.stds])
        plt.title(f"Public regret curves: {env.name}\nmeans=[{means_str}], stds=[{stds_str}]\nbest_mean={env.best_mean:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{env.name}_regret.png", dpi=150)
        plt.close()

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
