import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from distance import chamfer_distance
from graph import Graph
from physics import relax
from target import TargetVolume

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGETS_DIR = Path(__file__).parent / "targets"
PREVIEW_POINTS = 250

graph = Graph.seed()
target: Optional[TargetVolume] = None


def _target_files() -> dict:
    if not TARGETS_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(TARGETS_DIR.glob("*.json"))}


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
    return points[idx]


def _load_named_volume(name: str) -> TargetVolume:
    files = _target_files()
    if name not in files:
        raise HTTPException(status_code=404, detail=f"unknown target '{name}'")
    data = json.loads(files[name].read_text())
    return TargetVolume.from_export(data)


@app.get("/")
def health():
    return {
        "status": "ok",
        "nodes": len(graph.positions),
        "triangles": len(graph.triangles),
        "growable": len(graph.triangles) - len(graph.grown),
        "target_loaded": target is not None,
    }


@app.post("/target/load")
def load_target(data: dict):
    """Load a voxel volume exported from the sculpture app
    (`{nx, ny, nz, voxels: [{x,y,z}, ...]}`) as the current growth target."""
    global target
    try:
        target = TargetVolume.from_export(data)
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"malformed voxel export: {exc}") from exc

    return {
        "status": "ok",
        "points": int(target.points.shape[0]),
        "resolution": list(target.resolution),
    }


@app.get("/target/distance")
def target_distance():
    if target is None:
        raise HTTPException(status_code=404, detail="no target loaded")
    return chamfer_distance(graph.positions_array(), target.points)


@app.get("/targets")
def list_targets():
    """Every *.json target file on disk, with a downsampled preview point
    cloud so the frontend can render a thumbnail without fetching the
    (potentially thousands of points) full volume for each one."""
    results = []
    for name, path in _target_files().items():
        try:
            data = json.loads(path.read_text())
            volume = TargetVolume.from_export(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        results.append(
            {
                "name": name,
                "points": int(volume.points.shape[0]),
                "resolution": list(volume.resolution),
                "preview": _downsample(volume.points, PREVIEW_POINTS).tolist(),
            }
        )
    return {"targets": results}


@app.post("/targets/{name}/load")
def load_named_target(name: str):
    global target
    target = _load_named_volume(name)
    return {
        "status": "ok",
        "points": int(target.points.shape[0]),
        "resolution": list(target.resolution),
    }


@app.get("/targets/{name}/distance")
def named_target_distance(name: str):
    """Distance from the current graph to a target file by name, without
    requiring it to be the currently-loaded/overlaid one — lets the
    frontend show a live distance reading for whichever target the user
    has selected independent of what's "active"."""
    volume = _load_named_volume(name)
    return chamfer_distance(graph.positions_array(), volume.points)


@app.post("/target/clear")
def clear_target():
    global target
    target = None
    return {"status": "ok"}


@app.get("/target/points")
def target_points():
    if target is None:
        raise HTTPException(status_code=404, detail="no target loaded")
    return {"points": target.points.tolist()}


def serialize_state() -> dict:
    return {
        "type": "state",
        "nodes": [
            {"id": i, "position": pos.tolist()} for i, pos in enumerate(graph.positions)
        ],
        "edges": [list(e) for e in graph.edges],
        "triangles": [
            {
                "id": tri_id,
                "vertices": list(vertices),
                "grown": tri_id in graph.grown,
                "apexPreview": (
                    None if tri_id in graph.grown else graph.apex_preview(tri_id).tolist()
                ),
            }
            for tri_id, vertices in graph.triangles.items()
        ],
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(serialize_state())

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("type") == "grow_triangle":
                tri_id = message.get("triangleId")
                if graph.grow_triangle(tri_id) is not None:
                    relaxed = relax(graph.positions_array(), list(graph.edges), graph.pinned)
                    graph.set_positions(relaxed)

                await websocket.send_json(serialize_state())

    except WebSocketDisconnect:
        pass
