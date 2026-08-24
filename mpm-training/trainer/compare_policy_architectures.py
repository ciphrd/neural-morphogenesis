"""Run a paired stateless-128/stateful-64 evolution experiment.

Example:
  .venv/bin/python compare_policy_architectures.py --output comparisons/puddle \
    -- --target puddle --generations 50 --population 16 --workers 4

Every argument after ``--`` is forwarded identically to both evolve.py runs.
The script keeps checkpoints/logs separate and writes summary.json + report.html.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from policy_parameters import POLICY_ARCHITECTURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("evolve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.evolve_args[1:] if args.evolve_args[:1] == ["--"] else args.evolve_args
    forbidden = {"--_comparison-policy-architecture", "--checkpoint-dir"}
    if forbidden.intersection(forwarded):
        raise SystemExit("architecture and checkpoint directory are controlled by this comparison script")

    args.output.mkdir(parents=True, exist_ok=True)
    evolve = Path(__file__).with_name("evolve.py")
    rows: list[dict[str, object]] = []
    for architecture in POLICY_ARCHITECTURES:
        run_dir = args.output / architecture
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(evolve),
            "--_comparison-policy-architecture", architecture,
            "--checkpoint-dir", str(run_dir),
            *forwarded,
        ]
        print(f"[compare] starting {architecture}: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        with (run_dir / "training.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise SystemExit(f"{architecture} failed; inspect {run_dir / 'training.log'}")
        meta = json.loads((run_dir / "best_meta.json").read_text())
        parameter_count = int(np.load(run_dir / "best.npy", mmap_mode="r").size)
        rows.append(
            {
                "architecture": architecture,
                "hidden_dim": meta["hidden_dim"],
                "parameter_count": parameter_count,
                "best_fitness": meta["fitness"],
                "elapsed_seconds": elapsed,
                "checkpoint_dir": str(run_dir),
            }
        )
        print(f"[compare] finished {architecture}: fitness={meta['fitness']:.6g} time={elapsed:.1f}s", flush=True)

    summary = {"forwarded_arguments": forwarded, "runs": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['architecture']))}</td>"
        f"<td>{row['hidden_dim']}</td><td>{row['parameter_count']}</td>"
        f"<td>{float(row['best_fitness']):.6g}</td><td>{float(row['elapsed_seconds']):.1f}s</td>"
        "</tr>"
        for row in rows
    )
    report = f"""<!doctype html><meta charset="utf-8"><title>Policy architecture comparison</title>
<style>body{{font:15px system-ui;background:#111;color:#ddd;padding:2rem}}table{{border-collapse:collapse}}
th,td{{padding:.55rem .8rem;border:1px solid #444;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>Policy architecture comparison</h1>
<p>Paired runs use identical forwarded evolution settings and seed.</p>
<table><thead><tr><th>Architecture</th><th>Hidden</th><th>Parameters</th><th>Best fitness ↓</th><th>Wall time</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<pre>{html.escape(' '.join(forwarded))}</pre>"""
    (args.output / "report.html").write_text(report)
    print(f"[compare] report: {args.output / 'report.html'}")


if __name__ == "__main__":
    main()
