import { blocksWorld } from "./blocks";
import { discWorld } from "./disc";
import { growthWorld } from "./growth";
import { organismWorld } from "./organism";
import type { World } from "./types";

export type { SceneData, World, WorldDefaults } from "./types";
export { blocksWorld, discWorld, growthWorld, organismWorld };

/** Every world that exists, regardless of which tab(s) surface it —
 * add a new file next to blocks.ts/disc.ts/organism.ts (own SceneData/
 * World export, same shape) and list it here so worldById() can find it
 * (main.ts's generic World-select `change` handler always looks up by
 * id through this list, whichever tab is active). Which worlds actually
 * appear in the World dropdown for a given tab is tabs/index.ts's own
 * concern, not this file's — see that module. See gpu/mpm.ts's
 * MAX_PARTICLES for the one constraint a new world must respect: its
 * SceneData.count can't exceed that cap. */
export const WORLDS: readonly World[] = [blocksWorld, discWorld, organismWorld, growthWorld];

export const DEFAULT_WORLD: World = WORLDS[0];

export function worldById(id: string): World | undefined {
  return WORLDS.find((w) => w.id === id);
}
