import { useMemo, useState } from "react";
import { VoxelGrid } from "./voxel/VoxelGrid";
import { populateSphere } from "./voxel/populate";
import { resampleVoxelGrid } from "./voxel/resample";
import { downloadVoxelGridJSON } from "./voxel/exportJson";
import {
  VoxelRenderer,
  WORLD_SIZE,
  DEFAULT_CLIP,
  type ClipBounds,
  type Tool,
  type BrushMode,
} from "./render/VoxelRenderer";
import { Controls } from "./ui/Controls";

function buildSphereGrid(resolution: number, radius: number): VoxelGrid {
  const grid = new VoxelGrid(resolution, resolution, resolution);
  const center = resolution / 2;
  populateSphere(grid, { centerX: center, centerY: center, centerZ: center, radius });
  return grid;
}

export default function App() {
  const [resolution, setResolution] = useState(24);
  const [radius, setRadius] = useState(8);
  const [populated, setPopulated] = useState(true);
  const [grid, setGrid] = useState<VoxelGrid>(() => buildSphereGrid(24, 8));
  const [clip, setClip] = useState<ClipBounds>(DEFAULT_CLIP);
  const [tool, setTool] = useState<Tool>("orbit");
  const [brushMode, setBrushMode] = useState<BrushMode>("add");
  const [brushRadius, setBrushRadius] = useState(1);

  const voxelCount = useMemo(() => grid.count(), [grid]);
  // Derived from the grid actually being rendered, not the pending slider
  // value, so the bounding cube's world-space size never shifts underneath it.
  const cellSize = WORLD_SIZE / grid.nx;

  const handlePopulate = () => {
    setPopulated(true);
    setGrid(buildSphereGrid(resolution, radius));
  };

  const handleClear = () => {
    setPopulated(false);
    setGrid(new VoxelGrid(resolution, resolution, resolution));
  };

  const handleRadiusChange = (value: number) => {
    setRadius(value);
    // Radius is an explicit sphere-technique parameter, so re-run the
    // technique rather than resampling — resampling is only for keeping the
    // existing volume in place across a resolution change.
    if (populated) setGrid(buildSphereGrid(resolution, value));
  };

  const handleResolutionChange = (value: number) => {
    setResolution(value);
    setRadius((r) => Math.min(r, value / 2));
    setGrid((prev) => resampleVoxelGrid(prev, value, value, value));
  };

  const handleExport = () => {
    downloadVoxelGridJSON(grid);
  };

  const handleResetClip = () => {
    setClip(DEFAULT_CLIP);
  };

  const handlePaint = (newGrid: VoxelGrid) => {
    // Brush edits take the grid out of "pure sphere technique" mode, same as
    // Clear — so the radius slider won't clobber manual edits, while the
    // resolution slider still resamples them in place.
    setPopulated(false);
    setGrid(newGrid);
  };

  return (
    <div className="app">
      <Controls
        resolution={resolution}
        onResolutionChange={handleResolutionChange}
        radius={radius}
        onRadiusChange={handleRadiusChange}
        voxelCount={voxelCount}
        onPopulate={handlePopulate}
        onClear={handleClear}
        onExport={handleExport}
        onResetClip={handleResetClip}
        tool={tool}
        onToolChange={setTool}
        brushMode={brushMode}
        onBrushModeChange={setBrushMode}
        brushRadius={brushRadius}
        onBrushRadiusChange={setBrushRadius}
      />
      <div className="viewport">
        <VoxelRenderer
          grid={grid}
          cellSize={cellSize}
          clip={clip}
          onClipChange={setClip}
          tool={tool}
          brushMode={brushMode}
          brushRadius={brushRadius}
          onPaint={handlePaint}
        />
      </div>
    </div>
  );
}
