"""Regression check for generation-level shared, rotating rollout seeds.

Runs without constructing a GPU device: a fake worker pool records the tasks
that ``evolve.run_generation`` schedules and returns deterministic scores.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import evolve


class RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def map(self, function, weights, seeds):
        return [function(weight, seed) for weight, seed in zip(weights, seeds)]


def check_shared_rotating_seed_batch() -> None:
    args = SimpleNamespace(
        seeds_per_candidate=3,
        population=3,
        elites=3,
        mutation_sigma=0.05,
        policy_architecture="stateless-128",
    )
    population = [np.array([candidate], dtype=np.float32) for candidate in range(3)]
    pool = RecordingPool()

    original_worker_rollout = evolve.worker_rollout
    try:
        def fake_rollout(weights: np.ndarray, seed: int) -> float:
            candidate = int(weights[0])
            pool.calls.append((candidate, seed))
            return float(candidate * 100 + seed % 97)

        evolve.worker_rollout = fake_rollout
        rng = np.random.default_rng(123)
        first = evolve.run_generation(population, args, rng, pool)
        first_calls = list(pool.calls)
        pool.calls.clear()
        second = evolve.run_generation(population, args, rng, pool)
    finally:
        evolve.worker_rollout = original_worker_rollout

    first_seeds = first[3]
    assert len(first_calls) == args.population * args.seeds_per_candidate
    for candidate in range(args.population):
        start = candidate * args.seeds_per_candidate
        assert [seed for _, seed in first_calls[start : start + args.seeds_per_candidate]] == first_seeds

    expected_first_fitness = float(np.mean([seed % 97 for seed in first_seeds]))
    assert first[1][0] == expected_first_fitness
    hardest_seed = first_seeds[int(np.argmax([seed % 97 for seed in first_seeds]))]
    assert first[2] == hardest_seed
    assert second[3] != first_seeds

    print(
        "[PASS] shared rotating seed batches "
        f"first={first_seeds} second={second[3]} replay_seed={first[2]}"
    )


if __name__ == "__main__":
    check_shared_rotating_seed_batch()
