import type { World } from "../worlds/types";

/** One top-level example/experiment — main.ts's tab bar switches
 * between these. Each tab scopes its own World dropdown to `worlds`
 * (a subset of worlds/index.ts's full WORLDS list, in dropdown order);
 * switching tabs auto-loads `worlds[0]`. Nothing about the underlying
 * simulation is tab-specific (same MpmSimulation/MpmRenderer instance
 * throughout, same Tool/Field controls available in every tab) — a tab
 * is purely "which worlds does the World dropdown offer right now,"
 * kept as its own concept instead of overloading World itself so a
 * focused single-world tab (Organism) doesn't need a dropdown with one
 * option to feel first-class. */
export interface Tab {
  id: string;
  label: string;
  worlds: readonly World[];
}
