import { useEffect, useRef, type PointerEvent } from "react";
import { PixelGrid } from "../pixel/PixelGrid";

export const CANVAS_SIZE = 512;

interface PixelCanvasProps {
  grid: PixelGrid;
  mode: "add" | "erase";
  /** Called once a paint stroke ends, so the parent can re-read grid.count()
   * etc. Painting itself mutates `grid` in place and redraws imperatively —
   * going through React state on every cell would be unnecessary churn
   * during a drag. */
  onChange: () => void;
}

export function PixelCanvas({ grid, mode, onChange }: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paintingRef = useRef(false);
  const modeRef = useRef(mode);
  modeRef.current = mode;

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
  };

  useEffect(draw, [grid, cellSize]);

  const paintAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * grid.nx);
    const y = Math.floor(((clientY - rect.top) / rect.height) * grid.ny);
    grid.set(x, y, modeRef.current === "add");
    draw();
  };

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    paintingRef.current = true;
    paintAt(e.clientX, e.clientY);
  };
  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!paintingRef.current) return;
    paintAt(e.clientX, e.clientY);
  };
  const stopPainting = () => {
    if (!paintingRef.current) return;
    paintingRef.current = false;
    onChange();
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
      onPointerLeave={stopPainting}
    />
  );
}
