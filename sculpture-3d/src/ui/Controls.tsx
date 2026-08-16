import type { Tool, BrushMode } from "../render/VoxelRenderer";

interface ControlsProps {
  resolution: number;
  onResolutionChange: (value: number) => void;
  radius: number;
  onRadiusChange: (value: number) => void;
  voxelCount: number;
  onPopulate: () => void;
  onClear: () => void;
  onExport: () => void;
  onResetClip: () => void;
  tool: Tool;
  onToolChange: (tool: Tool) => void;
  brushMode: BrushMode;
  onBrushModeChange: (mode: BrushMode) => void;
  brushRadius: number;
  onBrushRadiusChange: (value: number) => void;
}

export function Controls({
  resolution,
  onResolutionChange,
  radius,
  onRadiusChange,
  voxelCount,
  onPopulate,
  onClear,
  onExport,
  onResetClip,
  tool,
  onToolChange,
  brushMode,
  onBrushModeChange,
  brushRadius,
  onBrushRadiusChange,
}: ControlsProps) {
  return (
    <div className="controls">
      <h1>Sculpture</h1>
      <p className="subtitle">Populate a voxel grid, then export it.</p>

      <section>
        <h2>Grid</h2>
        <label>
          Resolution ({resolution}³)
          <input
            type="range"
            min={4}
            max={64}
            value={resolution}
            onChange={(e) => onResolutionChange(Number(e.target.value))}
          />
        </label>
      </section>

      <section>
        <h2>Tool</h2>
        <div className="tabs">
          <button className={tool === "orbit" ? "active" : ""} onClick={() => onToolChange("orbit")}>
            Orbit
          </button>
          <button className={tool === "brush" ? "active" : ""} onClick={() => onToolChange("brush")}>
            Brush
          </button>
        </div>

        {tool === "brush" && (
          <>
            <p className="hint">
              Click or drag on the volume (or empty space inside the grid) to paint. Camera orbit is
              disabled while this tool is active.
            </p>
            <div className="tabs">
              <button
                className={brushMode === "add" ? "active" : ""}
                onClick={() => onBrushModeChange("add")}
              >
                Add
              </button>
              <button
                className={brushMode === "erase" ? "active" : ""}
                onClick={() => onBrushModeChange("erase")}
              >
                Erase
              </button>
            </div>
            <label>
              Brush radius ({brushRadius.toFixed(1)})
              <input
                type="range"
                min={0}
                max={5}
                step={0.5}
                value={brushRadius}
                onChange={(e) => onBrushRadiusChange(Number(e.target.value))}
              />
            </label>
          </>
        )}
      </section>

      <section>
        <h2>Technique: Sphere</h2>
        <label>
          Radius ({radius.toFixed(1)})
          <input
            type="range"
            min={1}
            max={resolution / 2}
            step={0.5}
            value={radius}
            onChange={(e) => onRadiusChange(Number(e.target.value))}
          />
        </label>
        <button onClick={onPopulate}>Populate sphere</button>
        <button onClick={onClear} className="secondary">
          Clear
        </button>
      </section>

      <section>
        <h2>Slice</h2>
        <p className="hint">
          Drag the colored handles on the corner of the bounding box in the viewport (red = X, green = Y,
          blue = Z) to slice into the volume.
        </p>
        <button onClick={onResetClip} className="secondary">
          Reset slicing
        </button>
      </section>

      <section>
        <h2>Export</h2>
        <button onClick={onExport} disabled={voxelCount === 0}>
          Export as JSON
        </button>
      </section>

      <section>
        <h2>Stats</h2>
        <p>{voxelCount} voxels filled</p>
      </section>
    </div>
  );
}
