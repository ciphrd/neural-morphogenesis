"""Standalone visual playground for comparing distance.chamfer_distance
(direct nearest-neighbor point-cloud distance) against
raster.training_raster_distance (Gaussian-splat rasterization + a
distance-transform outside-shape penalty) — the two fitness metrics
evolve.py has swapped between over time (see that module's own module
docstring). NOT part of the training pipeline itself — a standalone
FastAPI app + single static HTML page (distance_playground.html, served
at GET /), built purely to let a human place particles by hand against a
real target shape and see, side by side, what each metric actually
returns for that exact placement — the kind of intuition-building a
training run itself never surfaces (it only ever reports one number per
rollout, never "what does this metric think of THIS specific
configuration").

Shows each metric in two forms, since the two live implementations
aren't directly comparable otherwise — raster.training_raster_distance
always searches over rotation about the target's own centroid (nothing
anchors a grown blob's pose to the target's, so a bare, no-search
comparison would penalize a correct SHAPE for having landed at the
"wrong" angle):
  - "Raw" — points exactly as placed, no rotation/centroid search:
    distance.chamfer_distance() and a direct rasterize_points_sum()
    (+ raster_distance()/outside_shape_penalty()) against the target, at
    whatever position/orientation the points were actually clicked.
  - "Best rotation" — alignment.best_alignment() and
    raster.training_raster_distance() — the SAME rotation-search-about-
    the-target's-centroid form evolve.py's own live fitness (raster) and
    render_rollout.py's own display alignment (chamfer) already use, so
    this half of the comparison is literally production code, not a
    reimplementation.

Usage:
    python distance_playground.py [--port 8010]
Then open http://localhost:8010/ in a browser.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from alignment import best_alignment
from distance import chamfer_distance
from raster import (
    build_target_distance_field,
    build_target_raster,
    outside_shape_penalty,
    raster_distance,
    rasterize_points_sum,
    training_raster_distance,
)
from targets import available_targets, load_target

app = FastAPI()

# MpmCore's own fixed [0,1]^2 domain — matches evolve.py's own RASTER_EXTENT.
EXTENT = (0.0, 1.0, 0.0, 1.0)


def _raster_to_data_uri(raster: np.ndarray) -> str:
    """Same pixel conversion as debug_images.save_raster_image() (row-
    flip for this project's y-up domain, clip to [0,1] since a sum-
    combined candidate raster can exceed 1 under piled-up particles —
    see rasterize_points_sum()'s own docstring), just to an in-memory
    PNG/base64 data URI instead of a file, for a JSON API response."""
    img = np.clip(raster[::-1], 0.0, 1.0)
    buf = BytesIO()
    Image.fromarray((img * 255.0).astype(np.uint8), mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _finite(v: float) -> float | None:
    """chamfer_distance()/best_alignment()/training_raster_distance() all
    return float('inf') for an empty point cloud (e.g. before any
    particle has been placed yet) — meaningful to Python, but
    Starlette's own JSONResponse renders with allow_nan=False (spec-
    compliant JSON has no Infinity token) and raises ValueError on it
    otherwise. None round-trips as JSON `null`; the frontend's own
    fmt() already treats anything that fails Number.isFinite() as "∞"."""
    return v if np.isfinite(v) else None


class ScoreRequest(BaseModel):
    target: str
    points: list[list[float]]
    raster_resolution: int = 128
    raster_sigma: float = 1.5
    outside_weight: float = 1.0


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "distance_playground.html").read_text()


@app.get("/targets")
def list_targets() -> dict:
    return {"targets": available_targets()}


@app.get("/target/{name}")
def get_target(name: str) -> dict:
    try:
        shape = load_target(name)
    except SystemExit as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"points": shape.points.tolist(), "texelSize": shape.texel_size()}


@app.post("/score")
def score(req: ScoreRequest) -> JSONResponse:
    try:
        target_shape = load_target(req.target)
    except SystemExit as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    points = np.array(req.points, dtype=np.float64).reshape(-1, 2)
    target_points = target_shape.points.astype(np.float64)

    target_raster = build_target_raster(
        target_points, req.raster_resolution, EXTENT, req.raster_sigma, half_size=target_shape.texel_size() / 2.0
    )
    target_distance_field = build_target_distance_field(target_raster)

    # --- Raw: points exactly as placed, no pose search. ---
    chamfer_raw = chamfer_distance(points, target_points)
    candidate_raw = rasterize_points_sum(points, req.raster_resolution, EXTENT, req.raster_sigma)
    coverage_raw = raster_distance(target_raster, candidate_raw)
    penalty_raw = outside_shape_penalty(points, target_distance_field, EXTENT)

    # --- Best rotation: literally the same rotation-search-about-the-
    # target's-centroid form production code already uses (see this
    # module's own module docstring).
    chamfer_aligned, aligned_points = best_alignment(points, target_points)
    raster_aligned, candidate_aligned, breakdown = training_raster_distance(
        points,
        target_points,
        target_raster,
        target_distance_field,
        req.raster_resolution,
        EXTENT,
        req.raster_sigma,
        outside_weight=req.outside_weight,
        return_breakdown=True,
    )

    return JSONResponse(
        {
            "chamfer": {"raw": _finite(chamfer_raw), "aligned": _finite(chamfer_aligned)},
            "raster": {
                "raw": {
                    "distance": _finite(coverage_raw + req.outside_weight * penalty_raw),
                    "coverage": _finite(coverage_raw),
                    "penalty": _finite(penalty_raw),
                },
                "aligned": {
                    "distance": _finite(raster_aligned),
                    "coverage": _finite(breakdown.coverage) if breakdown else None,
                    "spill": _finite(breakdown.spill) if breakdown else None,
                    "boundary": _finite(breakdown.boundary) if breakdown else None,
                    "crowding": _finite(breakdown.crowding) if breakdown else None,
                    "angle": _finite(breakdown.angle) if breakdown else None,
                },
            },
            "images": {
                "target": _raster_to_data_uri(target_raster),
                "candidateRaw": _raster_to_data_uri(candidate_raw),
                "candidateAligned": _raster_to_data_uri(candidate_aligned) if candidate_aligned is not None else None,
            },
            "alignedPoints": aligned_points.tolist(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8010)
    args = arg_parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
