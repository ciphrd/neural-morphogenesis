import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from alignment import best_fit_distance
from graph import Graph
from physics import RADIUS, relax
from substrate import field_and_gradient
from target import TargetShape
from update_rule import MAX_ENERGY, UpdateRule
from update_rule import step as apply_update_rule

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
target: Optional[TargetShape] = None
update_rule = UpdateRule()


def _target_files() -> dict:
    if not TARGETS_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(TARGETS_DIR.glob("*.json"))}


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
    return points[idx]


def _load_named_shape(name: str) -> TargetShape:
    files = _target_files()
    if name not in files:
        raise HTTPException(status_code=404, detail=f"unknown target '{name}'")
    data = json.loads(files[name].read_text())
    return TargetShape.from_export(data)


@app.get("/")
def health():
    return {
        "status": "ok",
        "nodes": len(graph.positions),
        "target_loaded": target is not None,
    }


@app.post("/target/load")
def load_target(data: dict):
    """Load a pixel shape exported from the sculpture app
    (`{nx, ny, pixels: [{x,y}, ...]}`) as the current growth target."""
    global target
    try:
        target = TargetShape.from_export(data)
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"malformed pixel export: {exc}") from exc

    return {
        "status": "ok",
        "points": int(target.points.shape[0]),
        "resolution": list(target.resolution),
    }


@app.get("/target/distance")
def target_distance():
    if target is None:
        raise HTTPException(status_code=404, detail="no target loaded")
    return best_fit_distance(graph.positions_array(), target.points)


@app.get("/targets")
def list_targets():
    """Every *.json target file on disk, with a downsampled preview point
    cloud so the frontend can render a thumbnail without fetching the
    (potentially thousands of points) full shape for each one."""
    results = []
    for name, path in _target_files().items():
        try:
            data = json.loads(path.read_text())
            shape = TargetShape.from_export(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        results.append(
            {
                "name": name,
                "points": int(shape.points.shape[0]),
                "resolution": list(shape.resolution),
                "preview": _downsample(shape.points, PREVIEW_POINTS).tolist(),
            }
        )
    return {"targets": results}


@app.post("/targets/{name}/load")
def load_named_target(name: str):
    global target
    target = _load_named_shape(name)
    return {
        "status": "ok",
        "points": int(target.points.shape[0]),
        "resolution": list(target.resolution),
    }


@app.get("/targets/{name}/distance")
def named_target_distance(name: str):
    """Best-fit distance from the current graph to a target file by name,
    without requiring it to be the currently-loaded/overlaid one — lets
    the frontend show a live distance reading for whichever target the
    user has selected independent of what's "active"."""
    shape = _load_named_shape(name)
    return best_fit_distance(graph.positions_array(), shape.points)


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


@app.get("/nodes/{node_id}")
def node_state(node_id: int):
    if node_id < 0 or node_id >= len(graph.positions):
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    return {
        "id": node_id,
        "position": graph.positions[node_id].tolist(),
        "idVector": graph.id_vectors[node_id].tolist(),
        "chemicals": graph.chemicals[node_id].tolist(),
        "energy": graph.energy[node_id],
        "spawnDirection": graph.spawn_directions[node_id].tolist(),
        "splitProb": graph.split_probs[node_id],
    }


@app.get("/substrate")
def substrate():
    """The N-layer substrate field, evaluated at each node's own
    position — the local "chemical" reading a growth rule would sense
    there. Nothing consumes this yet; it's exposed for inspection."""
    positions = graph.positions_array()
    values, gradients = field_and_gradient(positions, positions)
    return {
        "nodes": [
            {
                "id": i,
                "values": values[i].tolist(),
                "gradients": gradients[i].tolist(),
            }
            for i in range(positions.shape[0])
        ]
    }


def serialize_state() -> dict:
    return {
        "type": "state",
        "radius": RADIUS,
        "maxEnergy": MAX_ENERGY,
        "nodes": [
            {
                "id": i,
                "position": pos.tolist(),
                # spawnDirection, splitProb, and energy are broadcast for
                # every node (not chemicals/idVector, which stay
                # selected-node-only via GET /nodes/{id} — see
                # net/socket.ts): splitProb sets the "direction" color
                # mode's fill (red 0 -> green 1), spawnDirection draws its
                # tick, and energy draws the always-on ring every node
                # gets regardless of color mode.
                "spawnDirection": graph.spawn_directions[i].tolist(),
                "splitProb": graph.split_probs[i],
                "energy": graph.energy[i],
            }
            for i, pos in enumerate(graph.positions)
        ],
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(serialize_state())

    try:
        while True:
            message = await websocket.receive_json()

            if message.get("type") == "split_node":
                node_id = message.get("nodeId")
                if graph.split_node(node_id) is not None:
                    relaxed = relax(graph.positions_array(), graph.pinned, graph.id_array())
                    graph.set_positions(relaxed)

                await websocket.send_json(serialize_state())

            elif message.get("type") == "step":
                did_split = apply_update_rule(graph, update_rule)
                if did_split:
                    relaxed = relax(graph.positions_array(), graph.pinned, graph.id_array())
                    graph.set_positions(relaxed)

                await websocket.send_json(serialize_state())

    except WebSocketDisconnect:
        pass
