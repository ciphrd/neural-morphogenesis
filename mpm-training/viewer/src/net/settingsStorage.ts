import type { RunSettings } from "../gpu/types";

const STORAGE_KEY = "mpm-training:last-run-settings:v1";

// First-ever offline visits have no browser cache to restore. These mirror the
// trainer's ordinary CLI/simulation defaults closely enough to build a valid
// randomized rollout; once any backend settings response arrives it replaces
// them in localStorage and becomes the next startup default.
export const DEFAULT_RUN_SETTINGS: RunSettings = {
  target: "circle",
  particles: 400,
  initialParticleCount: 5,
  macroSteps: 160,
  growthSteps: null,
  substepsPerMacro: 16,
  gravity: 200,
  spawnX: 0.5,
  spawnY: 0.5,
  spawnHalfWidth: 0.08,
  channels: 8,
  fieldN: 256,
  morphologyBlurSigma: 0.01,
  morphologyDensityReference: 1,
  neuralUpdatesPerMacro: 4,
  communicationSpeed: 1,
  elasticStrainScale: 0.15,
  elasticStrainInputsEnabled: true,
  hiddenDim: 128,
  decay: 0,
  depositRate: 1,
  maxAccel: 0,
  maxStrafe: 0,
  maxEnvWrite: 1,
  maxAngularAccel: 1.4,
  angularDamping: 0.8,
  maxAngularVelocity: 0.1,
  depositDistance: 0,
  depositSigma: 0.4,
  splitDisplacement: 0.0027,
  divisionCooldown: 1,
  friction: 0.9,
  massRampMacroSteps: 1,
  growthDuration: 48,
  growthMax: 2,
  growthThreshold: 0.85,
  mpmEnabled: true,
  chirality: true,
  damping: 0.039306956424563166,
  materialE: 10_000,
  materialNu: 0.2,
  materialHardening: 3,
  materialElasticity: 0.2,
  splatRadius: 0.004,
  repulsionStrength: 0,
  repulsionMaxDelta: 40,
  population: 16,
  elites: 3,
  mutationSigma: 0.05,
  runSeed: 0,
  totalGenerations: 50,
  checkpointEvery: 5,
};

function looksLikeSettings(value: unknown): value is Partial<RunSettings> {
  if (!value || typeof value !== "object") return false;
  const settings = value as Record<string, unknown>;
  return (
    typeof settings.target === "string" &&
    typeof settings.particles === "number" && settings.particles > 0 &&
    typeof settings.channels === "number" && settings.channels > 0 &&
    typeof settings.hiddenDim === "number" && settings.hiddenDim > 0 &&
    typeof settings.fieldN === "number" && settings.fieldN > 0
  );
}

/** Returns the last backend-provided settings, or a valid first-run fallback.
 * Merging over the fallback lets caches from older frontend versions acquire
 * newly-added settings fields without making an offline startup invalid. */
export function loadDefaultRunSettings(): RunSettings {
  try {
    const serialized = window.localStorage.getItem(STORAGE_KEY);
    if (!serialized) return DEFAULT_RUN_SETTINGS;
    const cached: unknown = JSON.parse(serialized);
    return looksLikeSettings(cached)
      ? { ...DEFAULT_RUN_SETTINGS, ...cached }
      : DEFAULT_RUN_SETTINGS;
  } catch {
    // localStorage may be unavailable (privacy/security policy) or corrupt.
    return DEFAULT_RUN_SETTINGS;
  }
}

/** Best-effort by design: browser storage policy must never stop a live
 * backend response from loading into the viewer. */
export function storeRunSettings(settings: RunSettings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Continue with in-memory settings when storage is unavailable/full.
  }
}
