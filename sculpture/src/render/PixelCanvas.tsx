import { useEffect, useRef, type PointerEvent } from "react";
import { brushCells, PixelGrid } from "../pixel/PixelGrid";

export const CANVAS_SIZE = 512;

interface PixelCanvasProps {
  grid: PixelGrid;
  mode: "add" | "erase";
  brushSize: number;
  /** Called once a paint stroke ends, so the parent can re-read grid.count()
   * etc. Painting itself mutates `grid` in place and redraws imperatively —
   * going through React state on every cell would be unnecessary churn
   * during a drag. */
  onChange: () => void;
}

export function PixelCanvas({ grid, mode, brushSize, onChange }: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paintingRef = useRef(false);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const brushSizeRef = useRef(brushSize);
  brushSizeRef.current = brushSize;
  // Grid cell currently under the pointer (null when the pointer isn't
  // over the canvas at all) — drives the hover preview in draw() below.
  const hoverRef = useRef<[number, number] | null>(null);

  const cellSize = CANVAS_SIZE / grid.nx;

  const draw = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#111318";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctx.fillStyle = "#4f8cff";
    for (const [x, y] of grid.filled()) {
      ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= grid.nx; i++) {
      const p = i * cellSize;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, CANVAS_SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(CANVAS_SIZE, p);
      ctx.stroke();
    }

    // Hover preview: exactly the cells brushCells() would paint, clipped
    // to the grid — same call painting itself makes (see paintAt), so
    // this can't drift out of sync with what a click actually does.
    if (hoverRef.current) {
      const [hx, hy] = hoverRef.current;
      const cells = brushCells(hx, hy, brushSizeRef.current).filter(([x, y]) => grid.inBounds(x, y));
      if (cells.length > 0) {
        const accent = modeRef.current === "add" ? "#4f8cff" : "#ff5a5a";
        ctx.fillStyle = modeRef.current === "add" ? "rgba(79,140,255,0.35)" : "rgba(255,90,90,0.35)";
        for (const [x, y] of cells) {
          ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const [x, y] of cells) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(
          minX * cellSize + 0.75,
          minY * cellSize + 0.75,
          (maxX - minX + 1) * cellSize - 1.5,
          (maxY - minY + 1) * cellSize - 1.5
        );
      }
    }
  };

  useEffect(draw, [grid, cellSize, brushSize, mode]);

  const cellFromEvent = (clientX: number, clientY: number): [number, number] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * grid.nx);
    const y = Math.floor(((clientY - rect.top) / rect.height) * grid.ny);
    return [x, y];
  };

  const paintAt = (cx: number, cy: number) => {
    for (const [x, y] of brushCells(cx, cy, brushSizeRef.current)) {
      grid.set(x, y, modeRef.current === "add");
    }
    draw();
  };

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const cell = cellFromEvent(e.clientX, e.clientY);
    paintingRef.current = true;
    hoverRef.current = cell;
    paintAt(cell[0], cell[1]);
  };
  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const cell = cellFromEvent(e.clientX, e.clientY);
    hoverRef.current = cell;
    if (paintingRef.current) {
      paintAt(cell[0], cell[1]);
    } else {
      draw();
    }
  };
  const stopPainting = () => {
    if (!paintingRef.current) return;
    paintingRef.current = false;
    onChange();
  };
  const handlePointerLeave = () => {
    hoverRef.current = null;
    stopPainting();
    draw();
  };

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      className="pixel-canvas"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={stopPainting}
      onPointerLeave={handlePointerLeave}
    />
  );
}
