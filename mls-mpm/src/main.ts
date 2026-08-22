import { acquireGpuDevice, watchDeviceLoss } from "./gpu/device";
import {
  MpmSimulation,
  DEFAULT_GRAVITY,
  DEFAULT_SUBSTEPS,
  DEFAULT_E,
  DEFAULT_NU,
  DEFAULT_HARDENING,
  DEFAULT_ELASTICITY,
  DEFAULT_DAMPING,
  DEFAULT_FIELD_RESOLUTION,
  DEFAULT_SPLAT_RADIUS,
  DEFAULT_REPULSION_STRENGTH,
  DEFAULT_ATTRACT_STRENGTH,
  FIELD_RESOLUTIONS,
  DT,
  MAX_PARTICLES,
  MOUSE_FORCE_STRENGTH,
  MOUSE_FORCE_RADIUS,
  MOUSE_MOVE_RADIUS,
} from "./gpu/mpm";
import type { FieldResolution } from "./gpu/mpm";
import type { FieldMode } from "./gpu/render";
import { DEFAULT_POINT_RADIUS_PX, MpmRenderer } from "./gpu/render";
import { DEFAULT_TAB, TABS } from "./tabs";
import type { Tab } from "./tabs";
import { TOOLS, DEFAULT_TOOL } from "./tools/types";
import type { ToolId } from "./tools/types";
import { allocateScene, hexToRgb, setColor, setRestState } from "./worlds/util";
import { worldById } from "./worlds";
import type { World } from "./worlds";

// "Add Particles" tool config — see spawnDotAt()/spawnAlongDrag() below.
// A drag places one dot every ADD_STRING_SPACING domain units traveled
// (not one per rendered frame), so the result is an evenly spaced
// "string" of particles tracing the cursor's path regardless of frame
// rate or drag speed; ADD_MAX_STEPS_PER_FRAME caps how many can land in
// one frame so a cursor jump (e.g. after switching tools mid-drag,
// or a stalled frame) can't dump an enormous run in one go. Colors cycle
// once per mousedown gesture (not per spawned particle) so a single
// click-drag reads as one string, visually distinct from the next.
const ADD_STRING_SPACING = 0.012;
const ADD_MAX_STEPS_PER_FRAME = 64;
const ADD_COLORS = [0xed553b, 0xf2b134, 0x068587, 0x7dd3fc, 0xa78bfa];

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const banner = document.getElementById("banner") as HTMLDivElement;
const tabBar = document.getElementById("tabBar") as HTMLDivElement;
const worldSelect = document.getElementById("worldSelect") as HTMLSelectElement;
const toolSelect = document.getElementById("toolSelect") as HTMLSelectElement;
const toolHint = document.getElementById("toolHint") as HTMLParagraphElement;
const fieldModeInput = document.getElementById("fieldMode") as HTMLSelectElement;
const particleSizeInput = document.getElementById("particleSize") as HTMLInputElement;
const particleSizeValue = document.getElementById("particleSizeValue") as HTMLSpanElement;
const substepsInput = document.getElementById("substeps") as HTMLInputElement;
const substepsValue = document.getElementById("substepsValue") as HTMLSpanElement;
const gravityInput = document.getElementById("gravity") as HTMLInputElement;
const gravityValue = document.getElementById("gravityValue") as HTMLSpanElement;
const stiffnessInput = document.getElementById("stiffness") as HTMLInputElement;
const stiffnessValue = document.getElementById("stiffnessValue") as HTMLSpanElement;
const poissonInput = document.getElementById("poisson") as HTMLInputElement;
const poissonValue = document.getElementById("poissonValue") as HTMLSpanElement;
const hardeningInput = document.getElementById("hardening") as HTMLInputElement;
const hardeningValue = document.getElementById("hardeningValue") as HTMLSpanElement;
const elasticityInput = document.getElementById("elasticity") as HTMLInputElement;
const elasticityValue = document.getElementById("elasticityValue") as HTMLSpanElement;
const dampingInput = document.getElementById("damping") as HTMLInputElement;
const dampingValue = document.getElementById("dampingValue") as HTMLSpanElement;
const repulsionResolutionInput = document.getElementById("repulsionResolution") as HTMLSelectElement;
const repulsionStrengthInput = document.getElementById("repulsionStrength") as HTMLInputElement;
const repulsionStrengthValue = document.getElementById("repulsionStrengthValue") as HTMLSpanElement;
const splatRadiusInput = document.getElementById("splatRadius") as HTMLInputElement;
const splatRadiusValue = document.getElementById("splatRadiusValue") as HTMLSpanElement;
const attractStrengthInput = document.getElementById("attractStrength") as HTMLInputElement;
const attractStrengthValue = document.getElementById("attractStrengthValue") as HTMLSpanElement;
const pauseBtn = document.getElementById("pauseBtn") as HTMLButtonElement;
const resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;
const particleCountEl = document.getElementById("particleCount") as HTMLParagraphElement;

function showBanner(message: string): void {
  banner.textContent = message;
  banner.classList.remove("hidden");
}

/** Keeps the canvas a square, sized to fill whichever of
 * window width/height is smaller — the simulation's domain is always
 * [0,1]^2, so a non-square canvas would visibly stretch it. */
function resizeCanvas(): number {
  const dpr = window.devicePixelRatio || 1;
  const cssSize = Math.min(window.innerWidth, window.innerHeight);
  const pixelSize = Math.max(1, Math.round(cssSize * dpr));
  canvas.style.width = `${cssSize}px`;
  canvas.style.height = `${cssSize}px`;
  if (canvas.width !== pixelSize) canvas.width = pixelSize;
  if (canvas.height !== pixelSize) canvas.height = pixelSize;
  return pixelSize;
}

function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

function populateSelect(select: HTMLSelectElement, options: readonly { id: string; label: string }[]): void {
  select.innerHTML = "";
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.id;
    el.textContent = opt.label;
    select.appendChild(el);
  }
}

async function main(): Promise<void> {
  const result = await acquireGpuDevice();
  if (!result.ok) {
    showBanner(`WebGPU isn't available: ${result.reason}`);
    return;
  }
  const device = result.device;
  watchDeviceLoss(device, (message) => showBanner(message));

  const context = canvas.getContext("webgpu");
  if (!context) {
    showBanner('canvas.getContext("webgpu") returned null.');
    return;
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const sim = new MpmSimulation(device, DEFAULT_TAB.worlds[0].buildScene());
  const renderer = new MpmRenderer(device, format, sim);

  // TEMP DEBUG HOOK — for manually cross-checking mpm-training/trainer's
  // headless Python extraction against this actual browser sandbox (see
  // mpm-training/trainer/README.md's own "what this does NOT prove yet"
  // section). Not meant to be permanent — revert this block once that
  // comparison is done. readPositions() mirrors the same copy-to-staging
  // + mapAsync readback mpm-training/trainer/mpm_core.py's own
  // read_positions() does via wgpu's queue.read_buffer() convenience.
  (window as unknown as { __mpmDebug: unknown }).__mpmDebug = {
    sim,
    device,
    async readPositions(): Promise<number[][]> {
      const count = sim.activeCount;
      const byteSize = count * 2 * 4;
      const staging = device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(sim.positions, 0, staging, 0, byteSize);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const arr = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      const positions: number[][] = [];
      for (let i = 0; i < count; i++) positions.push([arr[i * 2], arr[i * 2 + 1]]);
      return positions;
    },
  };

  // --- Tab / World / Tool selects ---
  populateSelect(toolSelect, TOOLS);
  toolSelect.value = DEFAULT_TOOL;

  // Field Resolution options come from gpu/mpm.ts's own FIELD_RESOLUTIONS
  // — single source of truth, same pattern as worldSelect/toolSelect
  // above — rather than hardcoding the 5 <option> values in index.html.
  populateSelect(
    repulsionResolutionInput,
    FIELD_RESOLUTIONS.map((r) => ({ id: String(r), label: String(r) }))
  );
  repulsionResolutionInput.value = String(DEFAULT_FIELD_RESOLUTION);

  let activeTab: Tab = DEFAULT_TAB;
  let activeWorld: World = DEFAULT_TAB.worlds[0];
  let activeTool: ToolId = DEFAULT_TOOL;

  fieldModeInput.addEventListener("change", () => {
    renderer.setFieldMode(fieldModeInput.value as FieldMode);
  });

  let paused = false;
  let substeps = DEFAULT_SUBSTEPS;
  substepsInput.value = String(DEFAULT_SUBSTEPS);
  substepsValue.textContent = String(DEFAULT_SUBSTEPS);
  gravityInput.value = String(DEFAULT_GRAVITY);
  gravityValue.textContent = String(DEFAULT_GRAVITY);
  stiffnessInput.value = String(DEFAULT_E);
  stiffnessValue.textContent = String(DEFAULT_E);
  poissonInput.value = String(DEFAULT_NU);
  poissonValue.textContent = DEFAULT_NU.toFixed(2);
  hardeningInput.value = String(DEFAULT_HARDENING);
  hardeningValue.textContent = String(DEFAULT_HARDENING);
  elasticityInput.value = String(DEFAULT_ELASTICITY);
  elasticityValue.textContent = DEFAULT_ELASTICITY.toFixed(2);
  dampingInput.value = String(DEFAULT_DAMPING);
  dampingValue.textContent = formatPct(DEFAULT_DAMPING);
  particleSizeInput.value = String(DEFAULT_POINT_RADIUS_PX);
  particleSizeValue.textContent = `${DEFAULT_POINT_RADIUS_PX}px`;
  repulsionStrengthInput.value = String(DEFAULT_REPULSION_STRENGTH);
  repulsionStrengthValue.textContent = DEFAULT_REPULSION_STRENGTH.toFixed(3);
  splatRadiusInput.value = String(DEFAULT_SPLAT_RADIUS);
  splatRadiusValue.textContent = DEFAULT_SPLAT_RADIUS.toFixed(3);
  attractStrengthInput.value = String(DEFAULT_ATTRACT_STRENGTH);
  attractStrengthValue.textContent = String(DEFAULT_ATTRACT_STRENGTH);

  // The 4 material sliders all feed into one setMaterial(E, nu,
  // hardening, elasticity) call (they share a single uniform buffer —
  // see gpu/mpm.ts's own note on why) — track the current value of
  // whichever ones a given slider *isn't* controlling, rather than
  // reading all 4 input elements back on every change.
  let gravity = DEFAULT_GRAVITY;
  let stiffness = DEFAULT_E;
  let poisson = DEFAULT_NU;
  let hardening = DEFAULT_HARDENING;
  let elasticity = DEFAULT_ELASTICITY;
  let damping = DEFAULT_DAMPING;
  let particleSize = DEFAULT_POINT_RADIUS_PX;
  const applyMaterial = () => sim.setMaterial(stiffness, poisson, hardening, elasticity);
  // Separate from applyMaterial() (its own uniform, gridUpdate.wgsl's
  // `damping`, not p2g/g2p.wgsl's Material) but needs `substeps` too —
  // see gpu/mpm.ts's perSubstepDamping() for why a per-*frame*
  // percentage's actual GPU-side value depends on the current substep
  // count, so this has to be called again whenever *either* changes.
  const applyDamping = () => sim.setDamping(damping, substeps);

  /** Applies `world`'s own slider defaults (worlds/types.ts's
   * WorldDefaults — every field optional) to both the live sim and the
   * slider UI. Unlike an earlier version of this function, every field
   * is always resolved (world's own value ?? gpu/mpm.ts's bare
   * DEFAULT_* fallback) rather than only touching fields the world
   * bothers to specify — worlds/organism.ts is the reason: it sets
   * every material field to something far from the defaults (see its
   * own file), and a world that *doesn't* specify, say, elasticity
   * (worlds/blocks.ts) needs to reliably land back on 0, not silently
   * inherit whatever Organism left behind. Called on every world
   * switch, not just once at startup. */
  function applyWorldDefaults(world: World): void {
    const d = world.defaults ?? {};
    gravity = d.gravity ?? DEFAULT_GRAVITY;
    gravityInput.value = String(gravity);
    gravityValue.textContent = String(gravity);
    sim.setGravity(gravity);

    stiffness = d.stiffness ?? DEFAULT_E;
    stiffnessInput.value = String(stiffness);
    stiffnessValue.textContent = String(stiffness);

    poisson = d.poisson ?? DEFAULT_NU;
    poissonInput.value = String(poisson);
    poissonValue.textContent = poisson.toFixed(2);

    hardening = d.hardening ?? DEFAULT_HARDENING;
    hardeningInput.value = String(hardening);
    hardeningValue.textContent = String(hardening);

    elasticity = d.elasticity ?? DEFAULT_ELASTICITY;
    elasticityInput.value = String(elasticity);
    elasticityValue.textContent = elasticity.toFixed(2);

    applyMaterial();

    damping = d.damping ?? DEFAULT_DAMPING;
    dampingInput.value = String(damping);
    dampingValue.textContent = formatPct(damping);
    applyDamping();

    particleSize = d.particleSize ?? DEFAULT_POINT_RADIUS_PX;
    particleSizeInput.value = String(particleSize);
    particleSizeValue.textContent = `${particleSize}px`;
    renderer.setPointRadiusPx(particleSize);

    const repulsionStrength = d.repulsionStrength ?? DEFAULT_REPULSION_STRENGTH;
    repulsionStrengthInput.value = String(repulsionStrength);
    repulsionStrengthValue.textContent = repulsionStrength.toFixed(3);
    sim.setRepulsionStrength(repulsionStrength);
  }

  worldSelect.addEventListener("change", () => {
    const world = worldById(worldSelect.value);
    if (!world) return;
    activeWorld = world;
    sim.loadScene(world.buildScene());
    applyWorldDefaults(world);
    updateParticleCount();
  });

  /** Switches the active tab: repopulates the World dropdown from
   * `tab.worlds` (see tabs/types.ts — a tab is just "which worlds does
   * the World dropdown offer"), auto-loads its first world, and updates
   * the tab bar's own active-button styling. Shared by both the initial
   * load and every later tab-button click, so there's exactly one path
   * that can leave worldSelect/activeWorld/the sim out of sync. */
  function selectTab(tab: Tab): void {
    activeTab = tab;
    populateSelect(worldSelect, tab.worlds);
    const world = tab.worlds[0];
    worldSelect.value = world.id;
    activeWorld = world;
    sim.loadScene(world.buildScene());
    applyWorldDefaults(world);
    updateParticleCount();
    for (const btn of tabBar.querySelectorAll<HTMLButtonElement>(".tab-button")) {
      btn.classList.toggle("is-active", btn.dataset.tabId === tab.id);
    }
  }

  for (const tab of TABS) {
    const btn = document.createElement("button");
    btn.className = "tab-button";
    btn.textContent = tab.label;
    btn.dataset.tabId = tab.id;
    btn.addEventListener("click", () => {
      if (tab.id === activeTab.id) return;
      selectTab(tab);
    });
    tabBar.appendChild(btn);
  }
  selectTab(DEFAULT_TAB);

  function updateToolHint(): void {
    toolHint.textContent = TOOLS.find((t) => t.id === activeTool)?.hint ?? "";
  }
  updateToolHint();

  toolSelect.addEventListener("change", () => {
    activeTool = toolSelect.value as ToolId;
    updateToolHint();
    // A drag in progress under the old tool's semantics (e.g. a "move"
    // grab) means nothing to the new one — drop it rather than having
    // the next applyMouse() frame reinterpret an old button-held state
    // under a different mode.
    mouseActive = false;
    // Same reasoning for "Attract to Point"'s own 2-click gesture — a
    // pick made under this tool shouldn't silently resolve into a target
    // click after switching away and back.
    attractAwaitingTarget = false;
  });

  substepsInput.addEventListener("input", () => {
    substeps = Number(substepsInput.value);
    substepsValue.textContent = String(substeps);
    // Damping's own GPU-side value depends on substeps (see
    // perSubstepDamping()) — re-derive it so "X% lost per frame" stays
    // true at the new substep count instead of silently drifting.
    applyDamping();
  });
  gravityInput.addEventListener("input", () => {
    gravity = Number(gravityInput.value);
    gravityValue.textContent = String(gravity);
    sim.setGravity(gravity);
  });
  stiffnessInput.addEventListener("input", () => {
    stiffness = Number(stiffnessInput.value);
    stiffnessValue.textContent = String(stiffness);
    applyMaterial();
  });
  poissonInput.addEventListener("input", () => {
    poisson = Number(poissonInput.value);
    poissonValue.textContent = poisson.toFixed(2);
    applyMaterial();
  });
  hardeningInput.addEventListener("input", () => {
    hardening = Number(hardeningInput.value);
    hardeningValue.textContent = String(hardening);
    applyMaterial();
  });
  elasticityInput.addEventListener("input", () => {
    elasticity = Number(elasticityInput.value);
    elasticityValue.textContent = elasticity.toFixed(2);
    applyMaterial();
  });
  dampingInput.addEventListener("input", () => {
    damping = Number(dampingInput.value);
    dampingValue.textContent = formatPct(damping);
    applyDamping();
  });
  particleSizeInput.addEventListener("input", () => {
    particleSize = Number(particleSizeInput.value);
    particleSizeValue.textContent = `${particleSize}px`;
    renderer.setPointRadiusPx(particleSize);
  });
  repulsionResolutionInput.addEventListener("change", () => {
    // Unlike every other control here, this rebuilds GPU resources (see
    // MpmSimulation.setFieldResolution()'s own docstring) rather than
    // just writing a uniform — the renderer's own display bind group has
    // to be rebuilt right after, in this exact order, since it points at
    // sim.densityTexture's own (now-stale) identity otherwise.
    sim.setFieldResolution(Number(repulsionResolutionInput.value) as FieldResolution);
    renderer.rebuildRepulsionDisplay();
  });
  repulsionStrengthInput.addEventListener("input", () => {
    const strength = Number(repulsionStrengthInput.value);
    repulsionStrengthValue.textContent = strength.toFixed(3);
    sim.setRepulsionStrength(strength);
  });
  splatRadiusInput.addEventListener("input", () => {
    const radius = Number(splatRadiusInput.value);
    splatRadiusValue.textContent = radius.toFixed(3);
    sim.setSplatRadius(radius);
  });
  attractStrengthInput.addEventListener("input", () => {
    const strength = Number(attractStrengthInput.value);
    attractStrengthValue.textContent = String(strength);
    sim.setAttractStrength(strength);
  });
  pauseBtn.addEventListener("click", () => {
    paused = !paused;
    pauseBtn.textContent = paused ? "▶ Play" : "⏸ Pause";
  });
  resetBtn.addEventListener("click", () => {
    sim.loadScene(activeWorld.buildScene());
    updateParticleCount();
  });

  window.addEventListener("resize", () => {
    renderer.setCanvasSizePx(resizeCanvas());
  });
  renderer.setCanvasSizePx(resizeCanvas());

  // --- Mouse interaction ---
  // Two fundamentally different tools share this same event plumbing —
  // see gridUpdate.wgsl's own MODE_FORCE/MODE_MOVE comment for the
  // actual per-tool physics. `mouseActive` gates everything (a hover
  // with no button held affects nothing); `mouseButton` only matters to
  // the force tool (attract vs repel); prevMouseX/Y is the move tool's
  // own state (see applyMouse() below) — updated once per *rendered
  // frame*, not per mousemove event, so its delta always corresponds to
  // "how far the cursor moved this frame," matching the cadence
  // sim.step(substeps) itself already runs at.
  let mouseActive = false;
  let mouseButton: 0 | 2 = 0;
  let mouseX = 0.5;
  let mouseY = 0.5;
  let prevMouseX = 0.5;
  let prevMouseY = 0.5;
  // Bumped once per mousedown gesture (not per spawned particle) by the
  // "add" tool's own mousedown handling below — see spawnDotAt()/
  // spawnAlongDrag(). addLastSpawnX/Y is that tool's own separate cursor-
  // tracking state (distinct from prevMouseX/Y, which the "move" tool
  // owns) — the domain point the *last* spawned dot landed at, so
  // spawnAlongDrag() knows how far the cursor has traveled since then
  // regardless of how many rendered frames that took.
  let addColorIdx = 0;
  let addLastSpawnX = 0.5;
  let addLastSpawnY = 0.5;
  // "Attract to Point" tool's own 2-click gesture state (tools/types.ts,
  // gpu/attract.wgsl) — false = the next click picks a particle, true =
  // the next click sets that particle's own target. Lives in main.ts, not
  // gpu/mpm.ts: MpmSimulation itself has no notion of "which click this
  // is," it just exposes the two underlying GPU operations
  // (pickParticleAt/commitAttractorAt) for this state machine to call.
  let attractAwaitingTarget = false;

  // Canvas pixel space is Y-down (CSS/DOM convention); the sim's [0,1]^2
  // domain is Y-up (see render.wgsl's own note) — flip here, the one
  // place mouse coordinates cross from one convention to the other.
  function updateMouseFromEvent(e: MouseEvent): void {
    const rect = canvas.getBoundingClientRect();
    mouseX = (e.clientX - rect.left) / rect.width;
    mouseY = 1 - (e.clientY - rect.top) / rect.height;
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("mousedown", (e) => {
    updateMouseFromEvent(e);
    mouseActive = true;
    mouseButton = e.button === 2 ? 2 : 0;
    // Zero the move tool's first-frame delta — otherwise grabbing far
    // from wherever the cursor last happened to be (anywhere, since it
    // isn't tracked while no button is held) would read as one huge
    // instantaneous jump, exactly the "too abrupt" behavior this tool
    // exists to avoid.
    prevMouseX = mouseX;
    prevMouseY = mouseY;
    if (activeTool === "add") {
      addColorIdx++;
      addLastSpawnX = mouseX;
      addLastSpawnY = mouseY;
      // A plain click (mousedown with no subsequent drag) still needs to
      // place a dot — spawnAlongDrag() below only fires once the cursor
      // has moved at least ADD_STRING_SPACING past addLastSpawnX/Y, which
      // a click that never moves would never satisfy.
      spawnDotAt(mouseX, mouseY);
    } else if (activeTool === "attractPoint") {
      // 2-click gesture — see attractAwaitingTarget's own comment. Each
      // click here is a single instantaneous GPU operation (not tied to
      // mouseActive/applyMouse()'s own per-*frame* cadence the way move/
      // force are), so there's nothing more to do once it's dispatched.
      if (!attractAwaitingTarget) {
        sim.pickParticleAt(mouseX, mouseY);
        attractAwaitingTarget = true;
      } else {
        sim.commitAttractorAt(mouseX, mouseY);
        attractAwaitingTarget = false;
      }
    }
  });
  // Listened on window (not canvas) for move/up so a drag that leaves
  // the canvas mid-gesture still tracks correctly and reliably releases.
  window.addEventListener("mousemove", (e) => {
    if (mouseActive) updateMouseFromEvent(e);
  });
  window.addEventListener("mouseup", () => {
    mouseActive = false;
  });

  /** Appends exactly one particle at domain coordinates (x, y), straight
   * into the live sim via MpmSimulation.addParticles() — the "arbitrary
   * spot" primitive both a plain click (mousedown handler above) and
   * every step of a drag (spawnAlongDrag() below) funnel through. Reuses
   * worlds/util.ts's own scene-building helpers (allocateScene/
   * setRestState/setColor) so a spawned particle starts from the exact
   * same at-rest state (F=identity, Jp=1) every worlds/*.ts file's own
   * initial particles do — there is no separate "just-spawned" state in
   * the physics. */
  function spawnDotAt(x: number, y: number): void {
    const { positions, velocities, F, C, Jp, colors } = allocateScene(1);
    const [r, g, b] = hexToRgb(ADD_COLORS[addColorIdx % ADD_COLORS.length]);
    positions[0] = x;
    positions[1] = y;
    setRestState(F, Jp, 0);
    setColor(colors, 0, r, g, b);
    sim.addParticles({ count: 1, positions, velocities, F, C, Jp, colors });
  }

  /** Fills the gap between addLastSpawnX/Y (wherever the last dot landed)
   * and the cursor's current position with dots spaced ADD_STRING_SPACING
   * domain units apart, advancing addLastSpawnX/Y to match — called once
   * per rendered frame the "add" tool is held down. A no-op below that
   * spacing threshold (holding the cursor still spawns nothing further,
   * rather than stacking dots on top of each other every frame), and the
   * *rate* the cursor travels doesn't change the result: a slow drag and
   * a fast drag across the same path both end up with dots at the same
   * fixed spacing, just accumulated over a different number of frames —
   * this is what makes the result read as one continuous "string" rather
   * than a speed-dependent trail. */
  function spawnAlongDrag(): void {
    const dx = mouseX - addLastSpawnX;
    const dy = mouseY - addLastSpawnY;
    const dist = Math.hypot(dx, dy);
    if (dist < ADD_STRING_SPACING) return;

    const steps = Math.min(Math.floor(dist / ADD_STRING_SPACING), ADD_MAX_STEPS_PER_FRAME);
    const ux = dx / dist;
    const uy = dy / dist;
    const { positions, velocities, F, C, Jp, colors } = allocateScene(steps);
    const [r, g, b] = hexToRgb(ADD_COLORS[addColorIdx % ADD_COLORS.length]);
    for (let i = 0; i < steps; i++) {
      addLastSpawnX += ux * ADD_STRING_SPACING;
      addLastSpawnY += uy * ADD_STRING_SPACING;
      positions[i * 2] = addLastSpawnX;
      positions[i * 2 + 1] = addLastSpawnY;
      setRestState(F, Jp, i);
      setColor(colors, i, r, g, b);
    }
    sim.addParticles({ count: steps, positions, velocities, F, C, Jp, colors });
  }

  /** Pushes the current tool/mouse state into gpu/mpm.ts's Mouse uniform
   * — called once per rendered frame (not per mousemove), right before
   * sim.step(substeps), so its effect applies to every substep in that
   * step() call uniformly. */
  function applyMouse(): void {
    if (!mouseActive) {
      sim.setMouse({ x: 0, y: 0, velX: 0, velY: 0, strength: 0, radius: 0, mode: 0 });
      prevMouseX = mouseX;
      prevMouseY = mouseY;
      return;
    }

    if (activeTool === "add") {
      // Spawning particles is the whole effect this tool has — it
      // doesn't also drive gridUpdate.wgsl's Mouse uniform the way
      // move/force do, so leave that at mode 0 (off).
      spawnAlongDrag();
      sim.setMouse({ x: 0, y: 0, velX: 0, velY: 0, strength: 0, radius: 0, mode: 0 });
      prevMouseX = mouseX;
      prevMouseY = mouseY;
      return;
    }

    if (activeTool === "attractPoint") {
      // Both clicks of this tool's own gesture already fired (and did
      // everything they need to) from the mousedown handler above — this
      // tool has no continuous, held-down behavior, so just make sure
      // holding the button doesn't fall through to the Force tool's own
      // branch below (the only one with no explicit tool check).
      sim.setMouse({ x: 0, y: 0, velX: 0, velY: 0, strength: 0, radius: 0, mode: 0 });
      prevMouseX = mouseX;
      prevMouseY = mouseY;
      return;
    }

    if (activeTool === "move") {
      // Convert this frame's on-screen cursor delta into a domain-
      // velocity such that applying it uniformly across `substeps` DT
      // integrations reproduces that same delta by the end of the frame
      // (pos += DT*v each substep => substeps*DT*v ≈ this frame's actual
      // mouse movement) — grabbed material tracks the cursor's real
      // on-screen speed, nothing more, nothing less. Zero when the mouse
      // is held still (prevMouseX/Y == mouseX/Y), which correctly reads
      // as "hold in place," not "let go."
      const denom = Math.max(substeps * DT, 1e-9);
      const velX = (mouseX - prevMouseX) / denom;
      const velY = (mouseY - prevMouseY) / denom;
      sim.setMouse({ x: mouseX, y: mouseY, velX, velY, strength: 0, radius: MOUSE_MOVE_RADIUS, mode: 2 });
    } else {
      const strength = mouseButton === 2 ? -MOUSE_FORCE_STRENGTH : MOUSE_FORCE_STRENGTH;
      sim.setMouse({ x: mouseX, y: mouseY, velX: 0, velY: 0, strength, radius: MOUSE_FORCE_RADIUS, mode: 1 });
    }

    prevMouseX = mouseX;
    prevMouseY = mouseY;
  }

  function updateParticleCount(): void {
    particleCountEl.textContent = `Particles: ${sim.activeCount.toLocaleString()} / ${MAX_PARTICLES.toLocaleString()}`;
  }
  updateParticleCount();

  const frame = () => {
    if (!paused) {
      applyMouse();
      sim.step(substeps);
      updateParticleCount();
    }
    renderer.render(context);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);

  window.addEventListener("beforeunload", () => {
    sim.destroy();
    renderer.destroy();
  });
}

main().catch((err) => {
  console.error("[mls-mpm]", err);
  showBanner(`Failed to start: ${String(err)}`);
});
