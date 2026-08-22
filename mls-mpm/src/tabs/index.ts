import { blocksWorld, discWorld, growthWorld, organismWorld } from "../worlds";
import type { Tab } from "./types";

export type { Tab } from "./types";

/** Every tab, in tab-bar order. First tab ("Sandbox") is the original,
 * unscoped app — both its worlds, all controls — kept exactly as it
 * was; add a new tab by giving it its own `worlds` subset (existing
 * worlds, or a brand new one next to worlds/organism.ts) rather than
 * folding a new experiment into Sandbox's own list. */
export const TABS: readonly Tab[] = [
  { id: "sandbox", label: "Sandbox", worlds: [blocksWorld, discWorld] },
  { id: "organism", label: "Organism", worlds: [organismWorld] },
  { id: "growth", label: "Growth", worlds: [growthWorld] },
];

export const DEFAULT_TAB: Tab = TABS[0];

export function tabById(id: string): Tab | undefined {
  return TABS.find((t) => t.id === id);
}
