"""Exercise the real server loop with CPU rollouts and isolated output paths."""
import asyncio
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

import evolve
import train_server as server
from domain_fitness import target_mask
from raster import build_target_distance_field
from rollout_snapshot import RolloutSnapshot
from seed_schedule_check import RecordingPool
from targets import load_target
from update_rule import UpdateRule


class Pool(RecordingPool):
    def shutdown(self):
        pass


def main():
    args = server.parser.parse_args(["--generations", "2", "--population", "4",
                                    "--macro-steps", "4", "--particles", "64"])
    evolve.finalize_policy_configuration(args)
    evolve.finalize_density_configuration(args)
    evolve.validate_fitness_configuration(args)
    target = load_target(args.target)
    mask = target_mask(target, args.raster_resolution)
    evaluated = []

    def generation(*args, **kwargs):
        result = evolve.run_generation(*args, **kwargs)
        evaluated.append(result)
        return result

    def worker(weights, seed, density, return_snapshot=False):
        assert return_snapshot
        return RolloutSnapshot(float(weights @ weights), None, None, {})

    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        replacements = dict(args=args, target=target, target_raster=mask,
                            target_distance_field=build_target_distance_field(mask),
                            CHECKPOINTS_DIR=root, HISTORY_PATH=root / "history.jsonl",
                            SETTINGS_PATH=root / "settings.json", settings={},
                            latest_generation_message=None)
        for name, value in replacements.items():
            stack.enter_context(patch.object(server, name, value))
        stack.enter_context(patch.object(server, "_archive_previous_run"))
        stack.enter_context(patch.object(server, "build_pool", return_value=Pool()))
        stack.enter_context(patch.object(server, "_save_generation_images", return_value={}))
        stack.enter_context(patch.object(server, "broadcast", new=AsyncMock()))
        stack.enter_context(patch.object(server, "run_generation", side_effect=generation))
        stack.enter_context(patch.object(evolve, "worker_rollout", side_effect=worker))
        asyncio.run(server._training_loop_body())

        records = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()]
        model = UpdateRule(evolve.CHEM_CHANNELS, args.policy_architecture)
        for record, result in zip(records, evaluated):
            evolve.set_weights(model, result.winner_weights)
            assert record["weights"] == model.export_weights()
            assert record["best"] == result.fitnesses[0]
            assert record["optimizerState"]["generation"] == record["generation"] + 1
        winner = min(evaluated, key=lambda r: r.fitnesses[0])
        np.testing.assert_array_equal(np.load(root / "best.npy"), winner.winner_weights)
        metadata = json.loads((root / "best_meta.json").read_text())
        assert metadata["fitness"] == winner.fitnesses[0]
        assert metadata["optimizer"] == "cma-es"
        assert metadata["elites"] == 0
        assert metadata["reference_candidates"] == 1
        assert len(evaluated[0].fitnesses) == 4
        assert len(evaluated[1].fitnesses) == 5
        assert evaluated[1].fitnesses[0] <= evaluated[0].fitnesses[0]
        np.testing.assert_array_equal(evaluated[0].population[0], evaluated[0].winner_weights)
        assert metadata["optimizer_state"]["libraryVersion"] == "4.4.4"
        assert json.loads((root / "settings.json").read_text())["cmaCovariance"] == "diagonal"
    print("[PASS] CMA server history, previews and checkpoints use evaluated winners; diagnostics persisted")


if __name__ == "__main__":
    main()
