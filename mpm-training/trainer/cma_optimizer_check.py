"""CMA convergence, reproducibility and evaluated-winner integration checks."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import evolve
from cma_optimizer import CmaOptimizer
from rollout_snapshot import RolloutSnapshot
from seed_schedule_check import RecordingPool


def check_search():
    for mode in ("diagonal", "full"):
        a = CmaOptimizer(np.ones(8), np.linspace(.2, 1., 8), .2, 16, 0, mode)
        b = CmaOptimizer(np.ones(8), np.linspace(.2, 1., 8), .2, 16, 0, mode)
        initial = None
        best = float("inf")
        for _ in range(100):
            xs = a.ask()
            np.testing.assert_array_equal(xs, b.ask())
            scores = [float(x @ x) for x in xs]
            initial = min(scores) if initial is None else initial
            best = min(best, min(scores))
            a.tell(scores)
            b.tell(scores)
        assert best < initial * 1e-5, (mode, initial, best)
        assert a.es.sigma != .2
        assert a.diagnostics() == b.diagnostics()
        if mode == "diagonal":
            # Production-size diagonal storage must never silently become dense.
            large = CmaOptimizer(np.zeros(6158), np.ones(6158), .05, 4, 0)
            large.ask(); large.tell([1., 2., 3., 4.])
            assert np.ndim(large.es.C) <= 1

    opt = CmaOptimizer(np.zeros(4), np.ones(4), .05, 4, 0)
    opt.ask(); opt.tell([np.nan, np.inf, -np.inf, np.inf])
    assert opt.restarts == 1 and opt.last_restart_reasons == ["all-invalid"]
    xs = opt.ask(); opt.tell([1., np.inf, np.nan, 2.])
    assert np.isfinite(opt.es.mean).all()
    opt.ask()
    try:
        opt.ask()
    except RuntimeError:
        pass
    else:
        raise AssertionError("double ask accepted")
    expected_winner = (opt.reference + opt.scales * opt.samples[2]).astype(np.float32)
    with patch.object(opt.es, "stop", return_value={"tolx": 1e-11}):
        opt.tell([4., 3., 1., 2.])
    np.testing.assert_array_equal(opt.reference, expected_winner)
    assert opt.last_restart_reasons == ["tolx"]
    assert opt.es.sigma == opt.initial_sigma


def check_generation():
    args = evolve.build_arg_parser().parse_args(["--population", "4", "--seeds-per-candidate", "2",
                                               "--particle-densities", ".5", "1"])
    evolve.finalize_policy_configuration(args)
    evolve.validate_fitness_configuration(args)
    assert args.optimizer == "cma-es"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "initial.npy"
        torch.manual_seed(0)
        initial = evolve.initial_weights(args)
        np.save(path, initial)
        args.initial_weights = path
        population, opt = evolve.initialize_search(args, np.random.default_rng(0))
        np.testing.assert_array_equal(opt.reference, initial)
        np.testing.assert_allclose(population[0], initial + opt.scales * opt.samples[0], rtol=1e-6, atol=1e-8)
        original = np.array(population)
        calls = []
        scores_in_order = [4., 1., 3., 2.]

        def worker(weights, seed, density, snapshot):
            index = next(i for i, w in enumerate(original) if np.array_equal(w, weights))
            score = scores_in_order[index] + density + (seed % 97) / 1000
            calls.append((index, seed, density))
            return RolloutSnapshot(score, None, None, {"candidate": index, "seed": seed, "density": density})

        with patch.object(evolve, "worker_rollout", worker), patch.object(opt, "tell", wraps=opt.tell) as tell:
            result = evolve.run_generation(population, args, np.random.default_rng(2), RecordingPool(),
                                           return_snapshot=True, optimizer=opt)
        actual_scores = tell.call_args.args[0]
        assert list(np.argsort(actual_scores)) == [1, 3, 2, 0]
        np.testing.assert_array_equal(result.winner_weights, original[1])
        assert not any(np.array_equal(result.winner_weights, x) for x in result.population)
        assert result.snapshot.diagnostics["candidate"] == 1
        assert result.snapshot.diagnostics["seed"] == result.winner_seed
        assert result.snapshot.diagnostics["density"] == result.winner_density == 1.
        assert len(calls) == 16
        for i in range(4):
            assert [seed for candidate, seed, q in calls if candidate == i and q == 1.] == result.evaluation_seeds
        assert result.fitnesses == sorted(actual_scores)
        assert opt.generation == 1

    for flags in (["--population", "1"], ["--mutation-sigma", "0"], ["--mutation-factors", "1", ".1"]):
        bad = evolve.build_arg_parser().parse_args(flags)
        try:
            evolve.validate_fitness_configuration(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"invalid CMA options accepted: {flags}")


if __name__ == "__main__":
    check_search()
    check_generation()
    print("[PASS] CMA convergence, seed-zero determinism, diagonal storage, invalid rollouts, warm start, shared cases and winner/preview identity")
