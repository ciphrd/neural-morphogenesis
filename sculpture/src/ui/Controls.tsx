interface ControlsProps {
  resolution: number;
  onResolutionChange: (value: number) => void;
  mode: "add" | "erase";
  onModeChange: (mode: "add" | "erase") => void;
  pixelCount: number;
  onClear: () => void;
  onExport: () => void;
}

export function Controls({
  resolution,
  onResolutionChange,
  mode,
  onModeChange,
  pixelCount,
  onClear,
  onExport,
}: ControlsProps) {
  return (
    <div className="controls">
      <h1>Sculpture 2D</h1>
      <p className="subtitle">Paint a pixel grid, then export it.</p>

      <section>
        <h2>Grid</h2>
        <label>
          Resolution ({resolution}²)
          <input
            type="range"
            min={8}
            max={64}
            value={resolution}
            onChange={(e) => onResolutionChange(Number(e.target.value))}
          />
        </label>
      </section>

      <section>
        <h2>Brush</h2>
        <p className="hint">Click or drag on the canvas to paint.</p>
        <div className="tabs">
          <button className={mode === "add" ? "active" : ""} onClick={() => onModeChange("add")}>
            Add
          </button>
          <button
            className={mode === "erase" ? "active" : ""}
            onClick={() => onModeChange("erase")}
          >
            Erase
          </button>
        </div>
        <button onClick={onClear} className="secondary">
          Clear
        </button>
      </section>

      <section>
        <h2>Export</h2>
        <button onClick={onExport} disabled={pixelCount === 0}>
          Export as JSON
        </button>
      </section>

      <section>
        <h2>Stats</h2>
        <p>{pixelCount} pixels filled</p>
      </section>
    </div>
  );
}
