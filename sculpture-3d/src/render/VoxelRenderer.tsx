import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VoxelGrid } from "../voxel/VoxelGrid";
import { applyBrushStamp } from "../voxel/brush";
import { raycastVoxel, type BrushMode } from "./voxelRaycast";

export type Tool = "orbit" | "brush";
export type { BrushMode };

// World-space size of the bounding cube. Fixed regardless of resolution —
// callers derive cellSize = WORLD_SIZE / resolution so the cube never moves
// or resizes as the grid is subdivided more finely.
export const WORLD_SIZE = 20;

export type Axis = "x" | "y" | "z";
export type ClipKind = "min" | "max";

/** Normalized [0,1] fractions of the volume along each axis, defining the visible sub-region. */
export interface ClipBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

export const DEFAULT_CLIP: ClipBounds = { minX: 0, maxX: 1, minY: 0, maxY: 1, minZ: 0, maxZ: 1 };

const AXIS_DIRS: Record<Axis, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

const AXIS_COLORS: Record<Axis, number> = { x: 0xff5555, y: 0x5fd15f, z: 0x5599ff };

function clipKey(axis: Axis, kind: ClipKind): keyof ClipBounds {
  return `${kind}${axis.toUpperCase()}` as keyof ClipBounds;
}

// A slice plane is only shown once it's been dragged in from the volume's edge.
function isSliceActive(kind: ClipKind, value: number): boolean {
  const epsilon = 0.001;
  return kind === "min" ? value > epsilon : value < 1 - epsilon;
}

function planeCenter(axis: Axis, value: number): THREE.Vector3 {
  const half = WORLD_SIZE / 2;
  const v = value * WORLD_SIZE;
  if (axis === "x") return new THREE.Vector3(v, half, half);
  if (axis === "y") return new THREE.Vector3(half, v, half);
  return new THREE.Vector3(half, half, v);
}

// PlaneGeometry faces +Z by default; rotate it to face along each axis.
const PLANE_ROTATIONS: Record<Axis, [number, number, number]> = {
  x: [0, Math.PI / 2, 0],
  y: [-Math.PI / 2, 0, 0],
  z: [0, 0, 0],
};

// Each slider line is shifted diagonally away from the (0,0,0) corner — away
// from the cube along both of its other two axes, not hugging either face —
// so the three "min" handles never collapse onto the same point and the
// lines float visibly clear of the volume.
const DIAGONAL_OFFSETS: Record<Axis, THREE.Vector3> = {
  x: new THREE.Vector3(0, -1, -1).normalize(),
  y: new THREE.Vector3(-1, 0, -1).normalize(),
  z: new THREE.Vector3(-1, -1, 0).normalize(),
};
const LINE_OFFSET = WORLD_SIZE * 0.06;

function axisAnchor(axis: Axis): THREE.Vector3 {
  return DIAGONAL_OFFSETS[axis].clone().multiplyScalar(LINE_OFFSET);
}

function handlePosition(axis: Axis, valueFraction: number): THREE.Vector3 {
  return axisAnchor(axis).add(AXIS_DIRS[axis].clone().multiplyScalar(valueFraction * WORLD_SIZE));
}

function axisResolution(axis: Axis, dims: { nx: number; ny: number; nz: number }): number {
  return axis === "x" ? dims.nx : axis === "y" ? dims.ny : dims.nz;
}

interface VoxelRendererProps {
  grid: VoxelGrid;
  cellSize?: number;
  clip: ClipBounds;
  onClipChange: (clip: ClipBounds) => void;
  tool: Tool;
  brushMode: BrushMode;
  brushRadius: number;
  onPaint: (grid: VoxelGrid) => void;
}

export function VoxelRenderer({
  grid,
  cellSize = 1,
  clip,
  onClipChange,
  tool,
  brushMode,
  brushRadius,
  onPaint,
}: VoxelRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const instancedMeshRef = useRef<THREE.InstancedMesh | null>(null);
  const volumeGridRef = useRef<THREE.LineSegments | null>(null);
  const handleMeshesRef = useRef<Partial<Record<string, THREE.Mesh>>>({});
  const planeMeshesRef = useRef<Partial<Record<string, THREE.Mesh>>>({});

  const clipRef = useRef(clip);
  clipRef.current = clip;
  const onClipChangeRef = useRef(onClipChange);
  onClipChangeRef.current = onClipChange;
  const gridDimsRef = useRef({ nx: grid.nx, ny: grid.ny, nz: grid.nz });
  gridDimsRef.current = { nx: grid.nx, ny: grid.ny, nz: grid.nz };
  const gridRef = useRef(grid);
  gridRef.current = grid;
  const cellSizeRef = useRef(cellSize);
  cellSizeRef.current = cellSize;
  const toolRef = useRef(tool);
  toolRef.current = tool;
  const brushModeRef = useRef(brushMode);
  brushModeRef.current = brushMode;
  const brushRadiusRef = useRef(brushRadius);
  brushRadiusRef.current = brushRadius;
  const onPaintRef = useRef(onPaint);
  onPaintRef.current = onPaint;

  // Set up the scene, camera, and the draggable clip handles once.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(WORLD_SIZE, WORLD_SIZE, WORLD_SIZE * 1.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(WORLD_SIZE / 2, WORLD_SIZE / 2, WORLD_SIZE / 2);
    controls.enableDamping = true;
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(WORLD_SIZE, WORLD_SIZE * 2, WORLD_SIZE);
    scene.add(dirLight);

    // Slider tracks: a 1px colored line per axis, showing the full range the
    // handles on that axis can slide across.
    const tracks: THREE.Line[] = [];
    (["x", "y", "z"] as Axis[]).forEach((axis) => {
      const anchor = axisAnchor(axis);
      const end = anchor.clone().add(AXIS_DIRS[axis].clone().multiplyScalar(WORLD_SIZE));
      const trackGeometry = new THREE.BufferGeometry().setFromPoints([anchor, end]);
      const trackMaterial = new THREE.LineBasicMaterial({ color: AXIS_COLORS[axis] });
      const track = new THREE.Line(trackGeometry, trackMaterial);
      scene.add(track);
      tracks.push(track);
    });

    // Clip handles: one pair (min/max) per axis, sliding along the three
    // slider tracks.
    const handleGeometry = new THREE.SphereGeometry(WORLD_SIZE * 0.015, 16, 12);
    const handles: THREE.Mesh[] = [];
    (["x", "y", "z"] as Axis[]).forEach((axis) => {
      (["min", "max"] as ClipKind[]).forEach((kind) => {
        const material = new THREE.MeshBasicMaterial({ color: AXIS_COLORS[axis], depthTest: false });
        const mesh = new THREE.Mesh(handleGeometry, material);
        mesh.renderOrder = 10;
        mesh.userData = { axis, kind };
        const value = clipRef.current[clipKey(axis, kind)];
        mesh.position.copy(handlePosition(axis, value));
        scene.add(mesh);
        handles.push(mesh);
        handleMeshesRef.current[`${axis}-${kind}`] = mesh;
      });
    });

    // Semi-transparent planes visualizing each active slice cut.
    const planeGeometry = new THREE.PlaneGeometry(WORLD_SIZE, WORLD_SIZE);
    const planeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const planes: THREE.Mesh[] = [];
    (["x", "y", "z"] as Axis[]).forEach((axis) => {
      (["min", "max"] as ClipKind[]).forEach((kind) => {
        const mesh = new THREE.Mesh(planeGeometry, planeMaterial);
        mesh.rotation.set(...PLANE_ROTATIONS[axis]);
        const value = clipRef.current[clipKey(axis, kind)];
        mesh.position.copy(planeCenter(axis, value));
        mesh.visible = isSliceActive(kind, value);
        scene.add(mesh);
        planes.push(mesh);
        planeMeshesRef.current[`${axis}-${kind}`] = mesh;
      });
    });

    // Brush hover preview: a wireframe sphere showing where the brush would
    // paint, colored by add/erase mode.
    const brushPreviewGeometry = new THREE.SphereGeometry(1, 16, 12);
    const brushPreviewMaterial = new THREE.MeshBasicMaterial({
      color: 0x5fd15f,
      wireframe: true,
      transparent: true,
      opacity: 0.7,
      depthTest: false,
    });
    const brushPreview = new THREE.Mesh(brushPreviewGeometry, brushPreviewMaterial);
    brushPreview.renderOrder = 9;
    brushPreview.visible = false;
    scene.add(brushPreview);

    const updateBrushPreview = (target: [number, number, number]) => {
      const cs = cellSizeRef.current;
      const radius = Math.max(0.5, brushRadiusRef.current) * cs;
      brushPreview.position.set(
        (target[0] + 0.5) * cs,
        (target[1] + 0.5) * cs,
        (target[2] + 0.5) * cs
      );
      brushPreview.scale.setScalar(radius);
      brushPreviewMaterial.color.set(brushModeRef.current === "add" ? 0x5fd15f : 0xff5555);
    };

    // Drag interaction: raycast against a plane that contains the handle's
    // axis and faces the camera as directly as possible, so the intersection
    // point tracks the mouse smoothly along that axis.
    const raycaster = new THREE.Raycaster();
    let dragging: { axis: Axis; kind: ClipKind } | null = null;
    let rafId: number | null = null;
    let pendingClip: ClipBounds | null = null;

    // Brush stroke state: painting mutates a single working clone in place
    // for the whole stroke (so later dabs correctly see earlier ones as
    // surface/empty), and commits throttled, freshly-cloned snapshots to
    // React so state updates don't get dropped via reference equality.
    let painting = false;
    let strokeGrid: VoxelGrid | null = null;
    let paintRafId: number | null = null;

    const schedulePaintUpdate = () => {
      if (paintRafId !== null) return;
      paintRafId = requestAnimationFrame(() => {
        paintRafId = null;
        if (strokeGrid) onPaintRef.current(strokeGrid.clone());
      });
    };

    const getNDC = (event: PointerEvent): THREE.Vector2 => {
      const rect = container.getBoundingClientRect();
      return new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );
    };

    const axisFraction = (axis: Axis, ndc: THREE.Vector2): number | null => {
      raycaster.setFromCamera(ndc, camera);
      const axisDir = AXIS_DIRS[axis];
      const camDir = new THREE.Vector3();
      camera.getWorldDirection(camDir);
      const planeNormal = camDir.clone().sub(axisDir.clone().multiplyScalar(axisDir.dot(camDir)));
      if (planeNormal.lengthSq() < 1e-8) return null;
      planeNormal.normalize();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(planeNormal, axisAnchor(axis));
      const hit = new THREE.Vector3();
      if (!raycaster.ray.intersectPlane(plane, hit)) return null;
      const t = hit.dot(axisDir);
      return Math.min(1, Math.max(0, t / WORLD_SIZE));
    };

    const scheduleClipUpdate = (next: ClipBounds) => {
      pendingClip = next;
      if (rafId !== null) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        if (pendingClip) onClipChangeRef.current(pendingClip);
      });
    };

    const handlePointerDown = (event: PointerEvent) => {
      const ndc = getNDC(event);
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(handles, false);
      if (hits.length > 0) {
        const { axis, kind } = hits[0].object.userData as { axis: Axis; kind: ClipKind };
        dragging = { axis, kind };
        controls.enabled = false;
        container.style.cursor = "grabbing";
        event.preventDefault();
        return;
      }

      if (toolRef.current === "brush") {
        const target = raycastVoxel(
          raycaster.ray.origin,
          raycaster.ray.direction,
          gridRef.current,
          cellSizeRef.current,
          brushModeRef.current
        );
        if (!target) return;
        painting = true;
        strokeGrid = gridRef.current.clone();
        applyBrushStamp(strokeGrid, target, brushRadiusRef.current, brushModeRef.current === "add");
        updateBrushPreview(target);
        controls.enabled = false;
        event.preventDefault();
        schedulePaintUpdate();
      }
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (dragging) {
        const { axis, kind } = dragging;
        const frac = axisFraction(axis, getNDC(event));
        if (frac === null) return;

        // Snap to the nearest voxel boundary for the current resolution — no
        // in-between (sub-cell) slice positions.
        const resolution = axisResolution(axis, gridDimsRef.current);
        const step = 1 / resolution;
        const snapped = Math.min(1, Math.max(0, Math.round(frac / step) * step));

        const current = clipRef.current;
        const minK = clipKey(axis, "min");
        const maxK = clipKey(axis, "max");
        const next: ClipBounds = { ...current };
        if (kind === "min") {
          next[minK] = Math.min(snapped, current[maxK] - step);
        } else {
          next[maxK] = Math.max(snapped, current[minK] + step);
        }
        clipRef.current = next;

        const value = next[clipKey(axis, kind)];
        const mesh = handleMeshesRef.current[`${axis}-${kind}`];
        if (mesh) mesh.position.copy(handlePosition(axis, value));
        const plane = planeMeshesRef.current[`${axis}-${kind}`];
        if (plane) {
          plane.position.copy(planeCenter(axis, value));
          plane.visible = isSliceActive(kind, value);
        }

        scheduleClipUpdate(next);
        return;
      }

      if (painting && strokeGrid) {
        raycaster.setFromCamera(getNDC(event), camera);
        const target = raycastVoxel(
          raycaster.ray.origin,
          raycaster.ray.direction,
          strokeGrid,
          cellSizeRef.current,
          brushModeRef.current
        );
        if (target) {
          applyBrushStamp(strokeGrid, target, brushRadiusRef.current, brushModeRef.current === "add");
          updateBrushPreview(target);
          schedulePaintUpdate();
        }
        return;
      }

      // Hover, not dragging or painting: show handle grab cursor, or the
      // brush preview when the brush tool is active.
      const ndc = getNDC(event);
      raycaster.setFromCamera(ndc, camera);
      const hoveringHandle = raycaster.intersectObjects(handles, false).length > 0;
      if (hoveringHandle) {
        container.style.cursor = "grab";
        brushPreview.visible = false;
        return;
      }
      if (toolRef.current === "brush") {
        const target = raycastVoxel(
          raycaster.ray.origin,
          raycaster.ray.direction,
          gridRef.current,
          cellSizeRef.current,
          brushModeRef.current
        );
        if (target) {
          updateBrushPreview(target);
          brushPreview.visible = true;
          container.style.cursor = "crosshair";
        } else {
          brushPreview.visible = false;
          container.style.cursor = "default";
        }
      } else {
        brushPreview.visible = false;
        container.style.cursor = "default";
      }
    };

    const handlePointerUp = () => {
      if (dragging) {
        dragging = null;
        controls.enabled = true;
        container.style.cursor = "default";
      }
      if (painting) {
        painting = false;
        controls.enabled = true;
        if (strokeGrid) onPaintRef.current(strokeGrid.clone());
        strokeGrid = null;
      }
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    let frameId: number;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (paintRafId !== null) cancelAnimationFrame(paintRafId);
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      tracks.forEach((t) => {
        t.geometry.dispose();
        (t.material as THREE.Material).dispose();
      });
      handles.forEach((h) => (h.material as THREE.Material).dispose());
      handleGeometry.dispose();
      handleMeshesRef.current = {};
      planeMaterial.dispose();
      planeGeometry.dispose();
      planeMeshesRef.current = {};
      brushPreviewGeometry.dispose();
      brushPreviewMaterial.dispose();
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep handle and slice-plane positions in sync when clip changes externally (e.g. a reset button).
  useEffect(() => {
    (["x", "y", "z"] as Axis[]).forEach((axis) => {
      (["min", "max"] as ClipKind[]).forEach((kind) => {
        const value = clip[clipKey(axis, kind)];
        const handle = handleMeshesRef.current[`${axis}-${kind}`];
        if (handle) handle.position.copy(handlePosition(axis, value));
        const plane = planeMeshesRef.current[`${axis}-${kind}`];
        if (plane) {
          plane.position.copy(planeCenter(axis, value));
          plane.visible = isSliceActive(kind, value);
        }
      });
    });
  }, [clip]);

  // Rebuild the instanced mesh whenever the grid, cell size, or clip range changes.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (instancedMeshRef.current) {
      scene.remove(instancedMeshRef.current);
      instancedMeshRef.current.geometry.dispose();
      (instancedMeshRef.current.material as THREE.Material).dispose();
      instancedMeshRef.current = null;
    }
    if (volumeGridRef.current) {
      scene.remove(volumeGridRef.current);
      volumeGridRef.current.geometry.dispose();
      (volumeGridRef.current.material as THREE.Material).dispose();
      volumeGridRef.current = null;
    }

    // A static 3D lattice spanning the whole nx*ny*nz volume — one line per
    // unit step along each axis, independent of which voxels are filled or
    // clipped. Serves as the fixed reference frame the clip handles slide on.
    const { nx, ny, nz } = grid;
    const linesAlongX = (ny + 1) * (nz + 1);
    const linesAlongY = (nx + 1) * (nz + 1);
    const linesAlongZ = (nx + 1) * (ny + 1);
    const gridPositions = new Float32Array((linesAlongX + linesAlongY + linesAlongZ) * 2 * 3);
    let o = 0;
    for (let y = 0; y <= ny; y++) {
      for (let z = 0; z <= nz; z++) {
        gridPositions[o++] = 0;
        gridPositions[o++] = y * cellSize;
        gridPositions[o++] = z * cellSize;
        gridPositions[o++] = nx * cellSize;
        gridPositions[o++] = y * cellSize;
        gridPositions[o++] = z * cellSize;
      }
    }
    for (let x = 0; x <= nx; x++) {
      for (let z = 0; z <= nz; z++) {
        gridPositions[o++] = x * cellSize;
        gridPositions[o++] = 0;
        gridPositions[o++] = z * cellSize;
        gridPositions[o++] = x * cellSize;
        gridPositions[o++] = ny * cellSize;
        gridPositions[o++] = z * cellSize;
      }
    }
    for (let x = 0; x <= nx; x++) {
      for (let y = 0; y <= ny; y++) {
        gridPositions[o++] = x * cellSize;
        gridPositions[o++] = y * cellSize;
        gridPositions[o++] = 0;
        gridPositions[o++] = x * cellSize;
        gridPositions[o++] = y * cellSize;
        gridPositions[o++] = nz * cellSize;
      }
    }
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute("position", new THREE.BufferAttribute(gridPositions, 3));
    const gridMaterial = new THREE.LineBasicMaterial({ color: 0x4a5568, transparent: true, opacity: 0.2 });
    const volumeGrid = new THREE.LineSegments(gridGeometry, gridMaterial);
    scene.add(volumeGrid);
    volumeGridRef.current = volumeGrid;

    const { minX, maxX, minY, maxY, minZ, maxZ } = clip;
    const filledVoxels: [number, number, number][] = [];
    for (const [x, y, z] of grid.filled()) {
      if (x + 0.5 < minX * nx || x + 0.5 > maxX * nx) continue;
      if (y + 0.5 < minY * ny || y + 0.5 > maxY * ny) continue;
      if (z + 0.5 < minZ * nz || z + 0.5 > maxZ * nz) continue;
      filledVoxels.push([x, y, z]);
    }
    if (filledVoxels.length === 0) return;

    const geometry = new THREE.BoxGeometry(cellSize * 0.95, cellSize * 0.95, cellSize * 0.95);
    const material = new THREE.MeshStandardMaterial({ color: 0x4f8cff, roughness: 0.5, metalness: 0.1 });
    const mesh = new THREE.InstancedMesh(geometry, material, filledVoxels.length);

    const dummy = new THREE.Object3D();
    let i = 0;
    for (const [x, y, z] of filledVoxels) {
      dummy.position.set((x + 0.5) * cellSize, (y + 0.5) * cellSize, (z + 0.5) * cellSize);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      i++;
    }
    mesh.instanceMatrix.needsUpdate = true;

    scene.add(mesh);
    instancedMeshRef.current = mesh;
  }, [grid, cellSize, clip]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
