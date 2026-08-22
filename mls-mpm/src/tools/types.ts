/** Selectable interaction tools — see gridUpdate.wgsl's own Mouse.mode
 * for what each one actually does to the grid. main.ts owns turning
 * mouse events + the active tool into that uniform's fields each frame;
 * this file is just the registry the Tool dropdown is built from. */
export type ToolId = "move" | "force" | "add" | "attractPoint";

export interface ToolDef {
  id: ToolId;
  label: string;
  hint: string;
}

export const TOOLS: readonly ToolDef[] = [
  {
    id: "move",
    label: "Move",
    hint: "Drag (either button) to carry particles along with the cursor.",
  },
  {
    id: "force",
    label: "Attract / Repel",
    hint: "Left-drag to attract, right-drag to repel.",
  },
  {
    id: "add",
    label: "Add Particles",
    hint: "Click and hold to spawn new particles at the cursor.",
  },
  {
    id: "attractPoint",
    label: "Attract to Point",
    // 2-click gesture (main.ts owns the awaitingTarget state machine) —
    // see gpu/attract.wgsl's own header for the pick/highlight/commit/
    // apply mechanism this drives.
    hint: "Click a particle to pick it, then click a spot — it'll drift there.",
  },
];

export const DEFAULT_TOOL: ToolId = "move";
