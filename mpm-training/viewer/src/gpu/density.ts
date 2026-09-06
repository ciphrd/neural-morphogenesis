import densityModelConfig from "../../../core/config.json";
const densityModel = densityModelConfig.density;

export const DENSITY_MODEL_VERSION = densityModel.MODEL_VERSION;
export const MIN_SUPPORTED_DENSITY = densityModel.MIN_SUPPORTED_MULTIPLIER;
export const MAX_SUPPORTED_DENSITY = densityModel.MAX_SUPPORTED_MULTIPLIER;

export interface DensityReference {
  particleCap: number;
  initialParticles: number;
  chemicalFieldN: number;
  particleMass: number;
  particleVolume: number;
  chemicalGradientInputScale: number;
  repulsionStrength: number;
  repulsionMaxDelta: number;
}

export interface ResolvedDensity {
  modelVersion: number;
  multiplier: number;
  spacingScale: number;
  spacing: number;
  initialSpacing: number;
  initialParticles: number;
  particleCap: number;
  particleMass: number;
  particleVolume: number;
  splatRadius: number;
  chemicalGradientInputScale: number;
  repulsionStrength: number;
  repulsionMaxDelta: number;
}

export function validateDensityMultiplier(multiplier: number, allowUnsafe = false): number {
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(`particle density multiplier must be finite and positive, got ${multiplier}`);
  }
  if (!allowUnsafe && (multiplier < MIN_SUPPORTED_DENSITY || multiplier > MAX_SUPPORTED_DENSITY)) {
    throw new Error(
      `particle density multiplier ${multiplier} is outside the supported range ` +
      `[${MIN_SUPPORTED_DENSITY}, ${MAX_SUPPORTED_DENSITY}]`,
    );
  }
  return multiplier;
}

export function resolveDensity(
  reference: DensityReference,
  multiplier: number,
  allowUnsafe = false,
): ResolvedDensity {
  const q = validateDensityMultiplier(multiplier, allowUnsafe);
  if (reference.particleCap < 1 || reference.initialParticles < 1) {
    throw new Error("reference particle counts must be positive");
  }
  if (reference.initialParticles > reference.particleCap) {
    throw new Error("reference initial particle count cannot exceed its cap");
  }
  if (reference.chemicalFieldN < 1) throw new Error("chemical field resolution must be positive");

  const spacingScale = 1 / Math.sqrt(q);
  const spacing = densityModelConfig.run.sampleSpacing * spacingScale;
  return {
    modelVersion: DENSITY_MODEL_VERSION,
    multiplier: q,
    spacingScale,
    spacing,
    // Keep seed triangles near the normal post-split scale at every density.
    initialSpacing: densityModel.INITIAL_SPACING_IN_SAMPLE_SPACINGS * spacing,
    initialParticles: Math.max(1, Math.floor(reference.initialParticles * q + 0.5)),
    particleCap: Math.max(1, Math.floor(reference.particleCap * q + 0.5)),
    particleMass: reference.particleMass / q,
    particleVolume: reference.particleVolume / q,
    splatRadius: densityModel.REPULSION_RADIUS_IN_CELLS * spacing,
    chemicalGradientInputScale: reference.chemicalGradientInputScale,
    repulsionStrength: reference.repulsionStrength * spacingScale * spacingScale,
    repulsionMaxDelta: reference.repulsionMaxDelta * spacingScale,
  };
}

/** Resolve a reference q=1 run configuration into an actual playback config. */
export function configAtDensity<T extends {
  particles: number;
  initialParticleCount: number;
  sampleSpacing: number;
  splatRadius: number;
  baseResolution: number;
  particleMass: number;
  particleVolume: number;
  chemicalGradientInputScale: number;
  repulsionStrength: number;
  repulsionMaxDelta: number;
}>(config: T, multiplier: number): T & {
  particleDensityMultiplier: number;
  particleMass: number;
  particleVolume: number;
  chemicalGradientInputScale: number;
} {
  const resolved = resolveDensity({
    particleCap: config.particles,
    initialParticles: config.initialParticleCount,
    chemicalFieldN: config.baseResolution,
    particleMass: config.particleMass,
    particleVolume: config.particleVolume,
    chemicalGradientInputScale: config.chemicalGradientInputScale,
    repulsionStrength: config.repulsionStrength,
    repulsionMaxDelta: config.repulsionMaxDelta,
  }, multiplier, true);
  return {
    ...config,
    particles: resolved.particleCap,
    initialParticleCount: resolved.initialParticles,
    particleDensityMultiplier: resolved.multiplier,
    particleMass: resolved.particleMass,
    particleVolume: resolved.particleVolume,
    chemicalGradientInputScale: resolved.chemicalGradientInputScale,
    sampleSpacing: config.sampleSpacing * resolved.spacingScale,
    splatRadius: config.splatRadius * resolved.spacingScale,
    repulsionStrength: resolved.repulsionStrength,
    repulsionMaxDelta: resolved.repulsionMaxDelta,
  };
}
