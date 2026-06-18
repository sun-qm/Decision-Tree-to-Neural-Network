"""Run the required performance-feasibility grid.

The default command can be expensive. Use ``examples.run_performance_experiment``
with fewer split seeds or alphas for a quick smoke run.
"""

from __future__ import annotations

from dt_to_nn.performance_baselines import main


if __name__ == "__main__":
    main(
        [
            "--datasets",
            "diabetes",
            "california_housing",
            "wine",
            "breast_cancer",
            "--settings",
            "single_tree",
            "ensemble_10",
            "--models",
            "source_tree",
            "coexplain_soft",
            "same_arch_random",
            "djinn",
            "path_expansion",
            "mlp",
            "--alphas",
            "0.5",
            "1",
            "2",
            "5",
            "10",
            "20",
            "50",
            "100",
            "--split-seeds",
            "0",
            "1",
            "2",
            "3",
            "4",
            "--stream-results",
            "--verbose",
        ]
    )
