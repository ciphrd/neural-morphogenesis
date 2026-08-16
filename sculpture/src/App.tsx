import { useState } from "react";
import { PixelGrid } from "./pixel/PixelGrid";
import { downloadPixelGridJSON } from "./pixel/exportJson";
import { PixelCanvas } from "./render/PixelCanvas";
import { Controls } from "./ui/Controls";

function buildGrid(resolution: number): PixelGrid {
  return new PixelGrid(resolution, resolution);
}

export default function App() {
  const [resolution, setResolution] = useState(32);
  const [grid, setGrid] = useState<PixelGrid>(() => buildGrid(32));
  const [mode, setMode] = useState<"add" | "erase">("add");
  const [, forceUpdate] = useState(0);

  const handleResolutionChange = (value: number) => {
    setResolution(value);
    setGrid(buildGrid(value));
  };

  const handleClear = () => {
    grid.clear();
    forceUpdate((n) => n + 1);
  };

  const handleExport = () => {
    downloadPixelGridJSON(grid);
  };

  return (
    <div className="app">
      <Controls
        resolution={resolution}
        onResolutionChange={handleResolutionChange}
        mode={mode}
        onModeChange={setMode}
        pixelCount={grid.count()}
        onClear={handleClear}
        onExport={handleExport}
      />
      <div className="viewport">
        <PixelCanvas grid={grid} mode={mode} onChange={() => forceUpdate((n) => n + 1)} />
      </div>
    </div>
  );
}
