import { useEffect, useRef } from "react";
import type { GraphNode } from "../net/socket";

export type Tool = "add" | "select" | "move";

/**
 * "solid" is the original look (blue, green-for-latest) and is what
 * every caller gets by default — only the Training tab's client-side
 * sim actually has chemicals/idVector data to color by, so this is opt
 * in, not a behavior change for anything that doesn't pass it.
 * "direction" fills red(0)->green(1) by splitProb (the network's own
 * raw split probability, before the energy gate scales it down) and
 * draws a white tick pointing spawnDirection — unlike channels/id both
 * are populated on both InteractiveView and TrainingView (see
 * net/socket.ts's GraphNode), since they're small enough to always
 * broadcast.
 */
export type ColorMode = "solid" | "channels" | "id" | "direction";

/**
 * Background layer behind the nodes themselves:
 * - "none" — just the grid, no field overlay.
 * - "field" — the chemical field's raw value (first 3 channels,
 *   Gaussian-summed from every node's `chemicals`). Channel 0/1/2 drive
 *   R/G/B independently (no blending into one shared scalar); a neutral
 *   gray is the zero-signal floor each channel rests at, and a negative
 *   value darkens its own channel from there down toward pure black —
 *   see FIELD_BASE_GRAY.
 * - "gradientX" / "gradientY" — that same field's *signed* ∂/∂x or ∂/∂y
 *   component per channel, instead of its value — the actual derivative
 *   a node's own sensing step differentiates
 *   (substrate.weighted_field_and_gradient) — "field" shows what's
 *   *there*, these show what a node standing at that point would feel
 *   pulling it one way or another along that axis. Deliberately a
 *   signed *component*, not the vector's magnitude: magnitude
 *   (`hypot(∂x, ∂y)`) is a square-root-of-squares and can structurally
 *   never be negative, which is the wrong shape for a black=negative /
 *   white=positive reading — a single axis's raw value is what's
 *   actually signed. Same R/G/B-per-channel, negative-toward-black
 *   treatment as "field" (see FIELD_BASE_GRAY) — just picking `dx` or
 *   `dy` out of the same computation instead of `hypot(dx, dy)`.
 */
export type FieldMode = "none" | "field" | "gradientX" | "gradientY";

interface GraphRendererProps {
  nodes: GraphNode[];
  radius: number;
  targetPoints: [number, number][] | null;
  tool: Tool;
  selectedNodeId: number | null;
  onSplitNode: (nodeId: number) => void;
  onSelectNode: (nodeId: number | null) => void;
  colorMode?: ColorMode;
  /** Only meaningful with tool="move" — omit for callers (e.g. InteractiveView)
   * that don't offer it; the tool simply won't do anything without them. */
  onDragNode?: (nodeId: number, position: [number, number]) => void;
  onDragEnd?: (nodeId: number) => void;
  /** See FieldMode. Needs per-node chemicals (Training tab only — same
   * reason ColorMode's "channels"/"id" are Training-only). */
  fieldMode?: FieldMode;
  /** Kernel bandwidth for the field preview — should match whatever
   * sensingSigma the replay actually used (ReplayConfig.sensingSigma),
   * not be a fixed guess, so the preview matches what nodes really sense. */
  fieldSigma?: number;
  /** Denominator for the always-on energy ring (see drawEnergyRing) —
   * should match the real per-run MAX_ENERGY (ReplayConfig.maxEnergy /
   * GraphState.maxEnergy), not be a fixed guess. */
  maxEnergy?: number;
}

// Mirrors update_rule.py's SENSING_SIGMA — only used as a fallback for
// callers that don't have a real per-run sensingSigma to pass (there
// currently are none; InteractiveView doesn't offer a background field
// at all, same reason it doesn't offer "channels"/"id" color modes).
const DEFAULT_FIELD_SIGMA = 1.15;

// Mirrors update_rule.py's MAX_ENERGY — only used as a fallback for a
// caller that hasn't received a real per-run value yet (e.g. before the
// first websocket/generation message arrives).
const DEFAULT_MAX_ENERGY = 100;

// Gap between the node's own edge and the energy ring drawn around it,
// in screen pixels — matches the "off by 2px from the circle
// boundaries" spec exactly, not scaled with zoom, so the ring stays a
// crisp, consistent 2px gap regardless of how far zoomed in/out the
// viewport is.
const ENERGY_RING_GAP_PX = 2;
const ENERGY_RING_COLOR = "#3b82f6";
const ENERGY_RING_WIDTH = 2.5;

// Screen-space cell size for the field preview. draw() can fire on
// every pointermove (panning, hover) not just on real state changes, so
// this trades a bit of visible blockiness for staying cheap enough to
// not jank the interaction it's layered under.
const FIELD_CELL_PX = 12;

// Same scale as ColorMode "channels" (CHEMICAL_CLIP) — deliberately not
// rescaled for the fact that this is a *sum* over every contributing
// node rather than one node's own value: several nearby saturated nodes
// driving the field further into saturation than any single node could
// is real information (their influence is genuinely stacking), not a
// display bug worth hiding.
const FIELD_COLOR_SCALE = 10;

// Shared "no signal" floor for every background mode below: a spot with
// zero chemical value / zero gradient component renders as this neutral
// gray in every channel, not the arbitrary mid-value a symmetric
// [-1,1]->[0,255] lerp happens to land on. Each signed quantity (raw
// value, or a gradient's ∂x/∂y component) descends from here toward
// pure black as it goes negative, and climbs toward white as it goes
// positive. Each of R/G/B is still driven independently by its own
// chemical channel (0->R, 1->G, 2->B, no blending into one shared
// scalar) — this only changes the byte curve each channel is put
// through, not which channel drives which color axis.
const FIELD_BASE_GRAY = 50;

function signedToByte(v: number, scale: number): number {
  const t = Math.max(-1, Math.min(1, v / scale));
  return t < 0
    ? Math.round(FIELD_BASE_GRAY * (1 + t)) // [-1, 0) -> [0, FIELD_BASE_GRAY)
    : Math.round(FIELD_BASE_GRAY + (255 - FIELD_BASE_GRAY) * t); // [0, 1] -> [FIELD_BASE_GRAY, 255]
}

function signedFieldColor(values: [number, number, number], scale: number): string {
  return `rgb(${signedToByte(values[0], scale)}, ${signedToByte(values[1], scale)}, ${signedToByte(values[2], scale)})`;
}

function drawField(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: GraphNode[],
  v: { scale: number; panX: number; panY: number },
  sigma: number
) {
  const sources = nodes.filter((n): n is GraphNode & { chemicals: number[] } => (n.chemicals?.length ?? 0) >= 3);
  if (sources.length === 0) return;

  const sigma2 = sigma * sigma;
  for (let sy = 0; sy < canvas.height; sy += FIELD_CELL_PX) {
    const wy = -(sy + FIELD_CELL_PX / 2 - canvas.height / 2 - v.panY) / v.scale;
    for (let sx = 0; sx < canvas.width; sx += FIELD_CELL_PX) {
      const wx = (sx + FIELD_CELL_PX / 2 - canvas.width / 2 - v.panX) / v.scale;

      let r = 0;
      let g = 0;
      let b = 0;
      for (const n of sources) {
        const dx = n.position[0] - wx;
        const dy = n.position[1] - wy;
        const kernel = Math.exp(-(dx * dx + dy * dy) / (2 * sigma2));
        r += kernel * n.chemicals[0];
        g += kernel * n.chemicals[1];
        b += kernel * n.chemicals[2];
      }

      ctx.fillStyle = signedFieldColor([r, g, b], FIELD_COLOR_SCALE);
      ctx.fillRect(sx, sy, FIELD_CELL_PX, FIELD_CELL_PX);
    }
  }
}

// Derived, not guessed: a single node holding a fully-saturated
// chemical value (CHEMICAL_CLIP), measured along the axis it's directly
// offset on (so the *other* axis's offset is zero and doesn't shrink
// this one via the shared kernel), contributes a gradient component of
// `|v|*kernel(d)*d/sigma²` as a function of distance `d` from it, which
// peaks at `d = sigma` with value `|v|*exp(-1/2)/sigma` — "the biggest
// single-axis contribution one saturated node can produce, right where
// it's strongest." Built from FIELD_COLOR_SCALE (== CHEMICAL_CLIP) and
// DEFAULT_FIELD_SIGMA (== SENSING_SIGMA) rather than redefining them, so
// this stays derived instead of a second number to keep in sync with
// the first by hand. Several overlapping saturated nodes (routine once
// packed at CONTACT_DISTANCE spacing, well inside one sigma of each
// other) still push well past this ceiling — their contributions
// genuinely stack, same reasoning as FIELD_COLOR_SCALE's own comment —
// which is real signal, not a rendering bug to hide by inflating the
// scale further.
const GRADIENT_COMPONENT_SCALE = (FIELD_COLOR_SCALE * Math.exp(-0.5)) / DEFAULT_FIELD_SIGMA;

// Same Gaussian-sum machinery as drawField, but accumulating each
// channel's signed analytic gradient *component* along one axis
// (∂/∂x or ∂/∂y) instead of its value — mirroring
// substrate.weighted_field_and_gradient / substrate.ts's math (see
// their comments for the derivation): for a source at `p` contributing
// value `v` at bandwidth `sigma`, `∇(v·kernel) = -(query-p)/sigma² ·
// v·kernel`, summed over sources. Deliberately a signed component, not
// the vector's magnitude (`hypot(∂x, ∂y)`) — a square root of squares
// can never be negative, so it can't render as "negative -> black" the
// way FieldMode's doc comment on "gradientX"/"gradientY" promises; a
// single raw axis is what's actually signed.
function drawFieldGradientComponent(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  nodes: GraphNode[],
  v: { scale: number; panX: number; panY: number },
  sigma: number,
  axis: "x" | "y"
) {
  const sources = nodes.filter((n): n is GraphNode & { chemicals: number[] } => (n.chemicals?.length ?? 0) >= 3);
  if (sources.length === 0) return;

  const sigma2 = sigma * sigma;
  for (let sy = 0; sy < canvas.height; sy += FIELD_CELL_PX) {
    const wy = -(sy + FIELD_CELL_PX / 2 - canvas.height / 2 - v.panY) / v.scale;
    for (let sx = 0; sx < canvas.width; sx += FIELD_CELL_PX) {
      const wx = (sx + FIELD_CELL_PX / 2 - canvas.width / 2 - v.panX) / v.scale;

      let g0 = 0;
      let g1 = 0;
      let g2 = 0;
      for (const n of sources) {
        // Same dx/dy convention as drawField (source minus query); the
        // gradient of a Gaussian bump points toward its source, so the
        // per-source contribution is +weight*dx (or *dy) in this
        // convention (see the function comment for the sign
        // derivation).
        const dx = n.position[0] - wx;
        const dy = n.position[1] - wy;
        const kernel = Math.exp(-(dx * dx + dy * dy) / (2 * sigma2));
        const d = axis === "x" ? dx : dy;
        g0 += ((n.chemicals[0] * kernel) / sigma2) * d;
        g1 += ((n.chemicals[1] * kernel) / sigma2) * d;
        g2 += ((n.chemicals[2] * kernel) / sigma2) * d;
      }

      ctx.fillStyle = signedFieldColor([g0, g1, g2], GRADIENT_COMPONENT_SCALE);
      ctx.fillRect(sx, sy, FIELD_CELL_PX, FIELD_CELL_PX);
    }
  }
}

// A little more generous than the true radius, so clicking near a node's
// edge still registers.
const HOVER_PADDING = 1.4;

const FALLBACK_COLOR = "#4f8cff";

// Chemical channels start roughly in [-1, 1] (uniform-random init) but
// nothing keeps them there — an untrained network's per-step deltas are
// additive with no decay, so they routinely run out past that range
// within a handful of steps, all the way out to runner.ts's own
// CHEMICAL_CLIP of ±10 (a numerical safety bound, not a meaningful
// display range). Map that full span linearly via `scale` rather than
// squashing it, so a pinned-white or pinned-black node is showing an
// honest "this channel is sitting at its hard clip," not an artifact of
// the color mapping. idVector is a different distribution — runner.ts
// renormalizes it to unit length every step, so its components already
// live genuinely in [-1, 1] and want scale=1; using chemicals' scale=10
// here would crush every id color down toward gray, which is exactly the
// "dim" bug this comment is now guarding against.
function channelColor(values: number[] | undefined, scale: number): string | null {
  if (!values || values.length < 3) return null;
  const toByte = (v: number) => Math.round(((Math.max(-1, Math.min(1, v / scale)) + 1) / 2) * 255);
  return `rgb(${toByte(values[0])}, ${toByte(values[1])}, ${toByte(values[2])})`;
}

// splitProb is the network's own raw split probability (sigmoid output,
// before update_rule.py's energy gate scales it down) — a single scalar
// in [0, 1], so a literal red-to-green lerp reads directly as "how much
// this node wants to split" without needing a legend. Endpoints match
// colors already used elsewhere in this file (targetPoints' red,
// "solid" mode's latest-node green) rather than arbitrary new ones.
const SPLIT_PROB_LOW = [239, 68, 68]; // matches targetPoints' red
const SPLIT_PROB_HIGH = [74, 222, 128]; // matches "solid" mode's latest green
function splitProbColor(splitProb: number | undefined): string | null {
  if (splitProb === undefined) return null;
  const t = Math.max(0, Math.min(1, splitProb));
  const [r, g, b] = SPLIT_PROB_LOW.map((lo, i) => Math.round(lo + (SPLIT_PROB_HIGH[i] - lo) * t));
  return `rgb(${r}, ${g}, ${b})`;
}

// Drawn around *every* node regardless of colorMode — energy is
// orthogonal to whatever a node's fill is currently encoding, and
// nodes that haven't gotten a broadcast energy yet (n.energy
// undefined — shouldn't happen once main.py/runner.ts are both current,
// but a stale message during a reconnect is plausible) just skip the
// ring rather than drawing a misleading empty/full one.
function drawEnergyRing(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  screenRadius: number,
  energy: number | undefined,
  maxEnergy: number
) {
  if (energy === undefined) return;
  const fraction = Math.max(0, Math.min(1, energy / maxEnergy));
  if (fraction <= 0) return;
  const ringRadius = screenRadius + ENERGY_RING_GAP_PX;
  const startAngle = -Math.PI / 2; // 12 o'clock, so a full ring reads clockwise from the top
  ctx.strokeStyle = ENERGY_RING_COLOR;
  ctx.lineWidth = ENERGY_RING_WIDTH;
  ctx.beginPath();
  ctx.arc(sx, sy, ringRadius, startAngle, startAngle + fraction * Math.PI * 2);
  ctx.stroke();
}

export function GraphRenderer({
  nodes,
  radius,
  targetPoints,
  tool,
  selectedNodeId,
  onSplitNode,
  onSelectNode,
  colorMode = "solid",
  onDragNode,
  onDragEnd,
  fieldMode = "none",
  fieldSigma = DEFAULT_FIELD_SIGMA,
  maxEnergy = DEFAULT_MAX_ENERGY,
}: GraphRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // scale = screen pixels per world unit; pan = screen-pixel offset of the
  // world origin from canvas center. Kept as refs (not React state) since
  // they update on every mousemove/wheel tick — redraw is imperative, no
  // need to push that through React's render cycle.
  const viewRef = useRef({ scale: 90, panX: 0, panY: 0 });
  const hoveredIdRef = useRef<number | null>(null);
  const draggingRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  // Separate from draggingRef (pan): which node, if any, tool="move" is
  // currently repositioning.
  const nodeDragRef = useRef<{ nodeId: number } | null>(null);
  const dataRef = useRef({
    nodes,
    radius,
    targetPoints,
    tool,
    selectedNodeId,
    colorMode,
    fieldMode,
    fieldSigma,
    maxEnergy,
  });
  dataRef.current = {
    nodes,
    radius,
    targetPoints,
    tool,
    selectedNodeId,
    colorMode,
    fieldMode,
    fieldSigma,
    maxEnergy,
  };
  const onSplitRef = useRef(onSplitNode);
  onSplitRef.current = onSplitNode;
  const onSelectRef = useRef(onSelectNode);
  onSelectRef.current = onSelectNode;
  const onDragNodeRef = useRef(onDragNode);
  onDragNodeRef.current = onDragNode;
  const onDragEndRef = useRef(onDragEnd);
  onDragEndRef.current = onDragEnd;

  const draw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const { nodes, radius, targetPoints, tool, selectedNodeId, colorMode, fieldMode, fieldSigma, maxEnergy } =
      dataRef.current;
    const v = viewRef.current;
    const w2s = (x: number, y: number) => ({
      x: canvas.width / 2 + v.panX + x * v.scale,
      y: canvas.height / 2 + v.panY - y * v.scale,
    });

    ctx.fillStyle = "#111318";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (fieldMode === "field") drawField(ctx, canvas, nodes, v, fieldSigma);
    else if (fieldMode === "gradientX") drawFieldGradientComponent(ctx, canvas, nodes, v, fieldSigma, "x");
    else if (fieldMode === "gradientY") drawFieldGradientComponent(ctx, canvas, nodes, v, fieldSigma, "y");

    // faint 1-world-unit grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    const origin = w2s(0, 0);
    const startX = ((origin.x % v.scale) + v.scale) % v.scale;
    const startY = ((origin.y % v.scale) + v.scale) % v.scale;
    for (let x = startX; x < canvas.width; x += v.scale) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = startY; y < canvas.height; y += v.scale) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    const hoveredId = hoveredIdRef.current;
    const hoverColor = tool === "select" ? "#7dd3fc" : tool === "move" ? "#c084fc" : "#ffb84f";
    // node ids are assigned sequentially and never reused, so the last
    // element is always the most recently added node
    const latestId = nodes.length > 0 ? nodes[nodes.length - 1].id : null;
    const screenRadius = radius * v.scale;
    for (const n of nodes) {
      const s = w2s(n.position[0], n.position[1]);
      const isHovered = n.id === hoveredId;
      const isLatest = n.id === latestId;
      const isSelected = n.id === selectedNodeId;

      let fillColor: string;
      if (colorMode === "solid") {
        // Original look: latest-added node is a solid fill color, same
        // as hover/selected already worked before colorMode existed.
        fillColor = isHovered ? hoverColor : isLatest ? "#4ade80" : FALLBACK_COLOR;
      } else {
        const dataColor =
          colorMode === "channels"
            ? channelColor(n.chemicals, 10)
            : colorMode === "id"
              ? channelColor(n.idVector, 1)
              : splitProbColor(n.splitProb);
        fillColor = isHovered ? hoverColor : dataColor ?? FALLBACK_COLOR;
      }

      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.arc(s.x, s.y, screenRadius, 0, Math.PI * 2);
      ctx.fill();

      drawEnergyRing(ctx, s.x, s.y, screenRadius, n.energy, maxEnergy);

      // A literal tick pointing the vector's actual way makes "direction"
      // mode's spawn direction readable at a glance, on top of the fill
      // now carrying splitProb instead. Skipped for the [0, 0] "no
      // direction yet" case — nothing meaningful to point at.
      if (colorMode === "direction" && n.spawnDirection) {
        const [dx, dy] = n.spawnDirection;
        const mag = Math.hypot(dx, dy);
        if (mag >= 1e-6) {
          const ux = dx / mag;
          const uy = -dy / mag; // world y-up -> screen y-down
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(s.x, s.y);
          ctx.lineTo(s.x + ux * screenRadius * 1.8, s.y + uy * screenRadius * 1.8);
          ctx.stroke();
        }
      }

      // In a data-driven color mode the fill is taken, so "latest" gets
      // a ring instead of overriding the fill like it does in "solid".
      if (isLatest && colorMode !== "solid") {
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = 0.6;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(s.x, s.y, screenRadius * 1.6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      if (isHovered) {
        ctx.strokeStyle = hoverColor;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(s.x, s.y, screenRadius * 2.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      if (isSelected) {
        ctx.strokeStyle = "#7dd3fc";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(s.x, s.y, screenRadius * 2.8, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    if (targetPoints) {
      ctx.fillStyle = "rgba(239,68,68,0.7)";
      for (const [x, y] of targetPoints) {
        const s = w2s(x, y);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  // Resize the canvas to fill its container, and redraw whenever the
  // container size changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resize = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      draw();
    };
    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Redraw whenever any relevant prop actually changes.
  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, radius, targetPoints, tool, selectedNodeId, colorMode, fieldMode, fieldSigma, maxEnergy]);

  // Hover/click/pan/zoom handling — set up once; live data is read from
  // refs so this effect never needs to depend on props.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const screenToWorld = (sx: number, sy: number) => {
      const v = viewRef.current;
      return {
        x: (sx - canvas.width / 2 - v.panX) / v.scale,
        y: -(sy - canvas.height / 2 - v.panY) / v.scale,
      };
    };

    const findHoveredNode = (worldX: number, worldY: number): number | null => {
      const { nodes, radius } = dataRef.current;
      const threshold = radius * HOVER_PADDING;
      let best: number | null = null;
      let bestDist = threshold;
      for (const n of nodes) {
        const d = Math.hypot(n.position[0] - worldX, n.position[1] - worldY);
        if (d < bestDist) {
          bestDist = d;
          best = n.id;
        }
      }
      return best;
    };

    const handlePointerMove = (ev: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const { x, y } = screenToWorld(sx, sy);

      if (nodeDragRef.current) {
        onDragNodeRef.current?.(nodeDragRef.current.nodeId, [x, y]);
        draw();
        return;
      }

      if (draggingRef.current) {
        const d = draggingRef.current;
        viewRef.current.panX = d.panX + (ev.clientX - d.x);
        viewRef.current.panY = d.panY + (ev.clientY - d.y);
        draw();
        return;
      }

      const hit = findHoveredNode(x, y);
      if (hit !== hoveredIdRef.current) {
        hoveredIdRef.current = hit;
        draw();
      }
    };

    const handlePointerDown = (ev: PointerEvent) => {
      // right or middle button: pan
      if (ev.button === 2 || ev.button === 1) {
        draggingRef.current = {
          x: ev.clientX,
          y: ev.clientY,
          panX: viewRef.current.panX,
          panY: viewRef.current.panY,
        };
        ev.preventDefault();
        return;
      }
      if (ev.button !== 0) return;

      const { tool, selectedNodeId } = dataRef.current;
      const hit = hoveredIdRef.current;

      if (tool === "move") {
        if (hit === null) {
          console.debug("[trainer] move: click missed every node");
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const { x, y } = screenToWorld(ev.clientX - rect.left, ev.clientY - rect.top);
        console.debug("[trainer] dragging node", hit);
        nodeDragRef.current = { nodeId: hit };
        onDragNodeRef.current?.(hit, [x, y]);
        return;
      }

      if (tool === "add") {
        if (hit !== null) {
          console.debug("[trainer] splitting node", hit);
          onSplitRef.current(hit);
        } else {
          console.debug("[trainer] click missed every node");
        }
        return;
      }

      // select tool: clicking the already-selected node deselects it;
      // clicking empty space also clears the selection
      const next = hit !== null && hit !== selectedNodeId ? hit : null;
      console.debug("[trainer] selecting node", next);
      onSelectRef.current(next);
    };

    const handlePointerUp = () => {
      draggingRef.current = null;
      if (nodeDragRef.current) {
        onDragEndRef.current?.(nodeDragRef.current.nodeId);
        nodeDragRef.current = null;
      }
    };

    const handleWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const before = screenToWorld(sx, sy);

      const v = viewRef.current;
      const oldScale = v.scale;
      const factor = Math.exp(-ev.deltaY * 0.001);
      const newScale = Math.min(400, Math.max(15, oldScale * factor));

      v.panX += before.x * (oldScale - newScale);
      v.panY -= before.y * (oldScale - newScale);
      v.scale = newScale;

      draw();
    };

    const preventContextMenu = (ev: MouseEvent) => ev.preventDefault();

    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointerup", handlePointerUp);
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    canvas.addEventListener("contextmenu", preventContextMenu);

    return () => {
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("wheel", handleWheel);
      canvas.removeEventListener("contextmenu", preventContextMenu);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} className="graph-canvas" />
    </div>
  );
}
