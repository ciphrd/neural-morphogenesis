"""Load shape targets from PNG alpha masks (and legacy pixel-export JSON).

PNG RGB supplies color supervision; alpha is continuous occupancy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix

TARGETS_DIR = Path(__file__).parent / "targets"
TARGET_SPAN_FRACTION = 1.0


@dataclass
class TargetShape:
    points: np.ndarray
    resolution: tuple[int, int]
    weights: np.ndarray | None = None
    occupancy: np.ndarray | None = None
    source_centroid: tuple[float, float] | None = None
    resolved_mask: np.ndarray | None = None
    target_center: tuple[float, float] | None = None

    rgb: np.ndarray | None = None
    resolved_rgb: np.ndarray | None = None
    _color_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        if self.weights is None:
            self.weights = np.ones(len(self.points), dtype=np.float32)
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if len(self.weights) != len(self.points):
            raise ValueError("target weights must match target points")

    @property
    def center(self) -> np.ndarray:
        if self.target_center is not None:
            return np.asarray(self.target_center, dtype=np.float64)
        if self.resolved_mask is not None:
            mask = np.asarray(self.resolved_mask, dtype=np.float64)
            ys, xs = np.indices(mask.shape, dtype=np.float64)
            mass = float(mask.sum())
            return np.array([
                float(((xs + 0.5) * mask).sum() / (mask.shape[1] * mass)),
                float(((ys + 0.5) * mask).sum() / (mask.shape[0] * mass)),
            ])
        if len(self.points) == 0:
            return np.array([0.5, 0.5])
        return np.average(self.points.astype(np.float64), axis=0, weights=self.weights)

    def filled_area(self) -> float:
        if self.resolved_mask is not None:
            return float(np.asarray(self.resolved_mask, dtype=np.float64).mean())
        if self.occupancy is not None:
            return float(np.sum(self.occupancy, dtype=np.float64)) * self.texel_size() ** 2
        return float(np.sum(self.weights, dtype=np.float64)) * self.texel_size() ** 2

    def texel_size(self) -> float:
        return TARGET_SPAN_FRACTION / max(self.resolution)

    def mask(self, resolution: int) -> np.ndarray:
        """Return cell-average occupancy in the simulation's unit domain."""
        if self.resolved_mask is not None:
            mask = np.asarray(self.resolved_mask, dtype=np.float64)
            if mask.shape != (resolution, resolution):
                raise ValueError(
                    f"embedded target mask is {mask.shape}, requested {(resolution, resolution)}"
                )
            return mask.copy()

        if self.occupancy is None:
            result = np.zeros((resolution, resolution), dtype=np.float64)
            half = self.texel_size() / 2
            for (x, y), weight in zip(self.points.astype(float), self.weights):
                lo = np.maximum(0, np.floor((np.array([x, y]) - half) * resolution).astype(int))
                hi = np.minimum(resolution, np.ceil((np.array([x, y]) + half) * resolution).astype(int))
                xs = np.arange(lo[0], hi[0]); ys = np.arange(lo[1], hi[1])
                wx = np.maximum(0, np.minimum(xs + 1, (x + half) * resolution)
                                - np.maximum(xs, (x - half) * resolution))
                wy = np.maximum(0, np.minimum(ys + 1, (y + half) * resolution)
                                - np.maximum(ys, (y - half) * resolution))
                result[np.ix_(ys, xs)] += float(weight) * wy[:, None] * wx
            return np.clip(result, 0, 1)

        height, width = self.occupancy.shape
        cx, cy = self.source_centroid
        scale = self.texel_size()

        def overlaps(count: int, centroid: float) -> np.ndarray:
            centers = (np.arange(count, dtype=np.float64) + 0.5 - centroid) * scale + 0.5
            lo, hi = centers - scale / 2, centers + scale / 2
            cell_lo = np.arange(resolution, dtype=np.float64)[:, None] / resolution
            cell_hi = (np.arange(resolution, dtype=np.float64)[:, None] + 1) / resolution
            return np.maximum(0.0, np.minimum(cell_hi, hi) - np.maximum(cell_lo, lo))

        # The overlap matrices are extremely sparse when a large source image
        # is reduced to the much smaller fitness grid. Sparse multiplication
        # keeps work roughly linear in source pixel count instead of cubic in
        # the source side length.
        wx = csr_matrix(overlaps(width, cx))
        wy = csr_matrix(overlaps(height, cy))
        rows_by_source_x = wy @ self.occupancy.astype(np.float64)
        result = (wx @ rows_by_source_x.T).T
        return np.clip(np.asarray(result) * resolution**2, 0, 1)

    @property
    def has_color(self) -> bool:
        return self.rgb is not None or self.resolved_rgb is not None

    def color_raster(self, resolution: int) -> np.ndarray | None:
        """Cell-average premultiplied RGB, with the same mapping as alpha."""
        if self.resolved_rgb is not None:
            rgb = np.asarray(self.resolved_rgb, dtype=float)
            if rgb.shape != (resolution, resolution, 3):
                raise ValueError("embedded target RGB resolution mismatch")
            return rgb.copy()
        if self.rgb is None:
            return None  # Legacy masks have no color supervision.
        if resolution not in self._color_cache:
            self._color_cache[resolution] = np.stack([
                replace(self, occupancy=self.occupancy*self.rgb[..., channel]).mask(resolution)
                for channel in range(3)
            ], axis=-1)
        return self._color_cache[resolution].copy()

    def wire(self, resolution: int) -> dict:
        mask = self.mask(resolution)
        data = {"mask": mask.ravel().tolist(), "resolution": resolution, "center": self.center.tolist()}
        rgb = self.color_raster(resolution)
        if rgb is not None:
            data["rgb"] = rgb.ravel().tolist()
        return data

    def overlay_points(self, resolution: int = 128) -> np.ndarray:
        """Return a bounded point approximation used only for visualization."""
        if self.occupancy is None and self.resolved_mask is None:
            return self.points
        mask = self.mask(resolution)
        ys, xs = np.nonzero(mask >= 0.5)
        if not len(xs):
            ys, xs = np.nonzero(mask > 0)
        return np.column_stack(((xs + 0.5) / resolution, (ys + 0.5) / resolution)).astype(np.float32)

    @classmethod
    def from_export(cls, data: dict) -> "TargetShape":
        nx, ny = int(data["nx"]), int(data["ny"])
        pixels = data["pixels"]
        if not pixels:
            return cls(np.zeros((0, 2), dtype=np.float32), (nx, ny))
        coords = np.unique(np.array([[p["x"], p["y"]] for p in pixels], dtype=np.float64), axis=0)
        centers = coords + 0.5
        centroid = centers.mean(axis=0)
        points = (centers - centroid) * (TARGET_SPAN_FRACTION / max(nx, ny)) + 0.5
        return cls(points.astype(np.float32), (nx, ny), target_center=(0.5, 0.5))

    @classmethod
    def from_png(cls, path: Path) -> "TargetShape":
        with Image.open(path) as image:
            if "A" not in image.getbands():
                raise ValueError(f"PNG target {path} must contain an alpha channel")
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        height, width = rgba.shape[:2]
        if width != height:
            raise ValueError(f"PNG target {path} must be square, got {width}x{height}")
        rgba = rgba[::-1].copy()  # image y-down -> simulation y-up
        occupancy = rgba[..., 3].astype(np.float32) / 255.0
        mass = float(occupancy.sum())
        if mass <= 0:
            raise ValueError(f"PNG target {path} has no occupied alpha pixels")
        ys, xs = np.nonzero(occupancy > 0)
        weights = occupancy[ys, xs]
        source_centroid = (
            float(np.sum((xs + 0.5) * weights) / mass),
            float(np.sum((ys + 0.5) * weights) / mass),
        )
        # Do not materialize one world-space point per source pixel: source
        # images may be arbitrarily large and production fitness uses the
        # occupancy raster directly. overlay_points() creates a bounded visual
        # approximation when an older point-oriented renderer needs one.
        return cls(np.zeros((0, 2), dtype=np.float32), (width, height),
                   occupancy=occupancy, source_centroid=source_centroid,
                   rgb=rgba[..., :3].astype(np.float32)/255.0,
                   target_center=(0.5, 0.5))

    @classmethod
    def from_wire(cls, data: dict) -> "TargetShape":
        resolution = int(data["resolution"])
        mask = np.asarray(data["mask"], dtype=np.float64).reshape(resolution, resolution)
        ys, xs = np.nonzero(mask > 0)
        points = np.column_stack(((xs + 0.5) / resolution, (ys + 0.5) / resolution))
        center = tuple(float(v) for v in data.get("center", (0.5, 0.5)))
        return cls(points, (resolution, resolution), mask[ys, xs], resolved_mask=mask,
                   target_center=center, resolved_rgb=(np.asarray(data["rgb"], dtype=float).reshape(resolution, resolution, 3)
                       if "rgb" in data else None))


def available_targets() -> list[str]:
    return sorted({p.stem for pattern in ("*.png", "*.json") for p in TARGETS_DIR.glob(pattern)})


def load_target(name: str) -> TargetShape:
    png_path = TARGETS_DIR / f"{name}.png"
    json_path = TARGETS_DIR / f"{name}.json"
    if png_path.is_file() and json_path.is_file():
        raise SystemExit(f"ambiguous target {name!r}: both {png_path.name} and {json_path.name} exist")
    try:
        if png_path.is_file():
            return TargetShape.from_png(png_path)
        if json_path.is_file():
            return TargetShape.from_export(json.loads(json_path.read_text()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid target {name!r}: {error}") from error
    raise SystemExit(f"unknown target {name!r} — choices: {available_targets()}")


def target_from_checkpoint(meta: dict) -> TargetShape:
    shape_target = meta.get("shape_settings", {}).get("shapeTarget")
    if shape_target and "mask" in shape_target:
        return TargetShape.from_wire(shape_target)
    return load_target(meta["target"])
