import channelConfig from "../../../core/chemical_channels.json";

export interface ChemicalChannelProfile {
  scale: string;
  resolutionScale: number;
  relaxationTime: number;
  fieldResponseTime: number;
  decayExponent: number;
  diffusionMultiplier: number;
  depositSigmaMultiplier: number;
  role?: string;
}

type PartialProfile = Partial<Omit<ChemicalChannelProfile, "scale">> & { scale?: string };

function positive(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved <= 0) throw new Error(`chemical channel ${name} must be positive`);
  return resolved;
}

function nonnegative(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < 0) throw new Error(`chemical channel ${name} must be nonnegative`);
  return resolved;
}

function normalize(raw: PartialProfile, defaults: PartialProfile = {}): ChemicalChannelProfile {
  return {
    scale: raw.scale ?? defaults.scale ?? "local",
    resolutionScale: positive(raw.resolutionScale, defaults.resolutionScale ?? 1, "resolutionScale"),
    relaxationTime: positive(raw.relaxationTime, defaults.relaxationTime ?? 1, "relaxationTime"),
    fieldResponseTime: positive(raw.fieldResponseTime, defaults.fieldResponseTime ?? 1, "fieldResponseTime"),
    decayExponent: positive(raw.decayExponent, defaults.decayExponent ?? 1, "decayExponent"),
    diffusionMultiplier: nonnegative(raw.diffusionMultiplier, defaults.diffusionMultiplier ?? 1, "diffusionMultiplier"),
    depositSigmaMultiplier: positive(raw.depositSigmaMultiplier, defaults.depositSigmaMultiplier ?? 1, "depositSigmaMultiplier"),
    ...(raw.role ? { role: raw.role } : {}),
  };
}

export function defaultChemicalChannelProfiles(channels: number): ChemicalChannelProfile[] {
  if (channelConfig.channels.length !== channels) {
    return Array.from({ length: channels }, () => normalize({ scale: "local" }));
  }
  return channelConfig.channels.map((channel) => {
    const defaults = channelConfig.profiles[channel.scale as keyof typeof channelConfig.profiles];
    return normalize(channel, { scale: channel.scale, ...defaults });
  });
}

export function homogeneousChemicalChannelProfiles(channels: number): ChemicalChannelProfile[] {
  return Array.from({ length: channels }, () => normalize({ scale: "local" }));
}

export function resolveChemicalChannelProfiles(
  channels: number,
  profiles?: readonly PartialProfile[],
): ChemicalChannelProfile[] {
  const resolved = profiles ? profiles.map((profile) => normalize(profile)) : homogeneousChemicalChannelProfiles(channels);
  if (resolved.length !== channels) throw new Error(`expected ${channels} chemical channel profiles, got ${resolved.length}`);
  return resolved;
}

export interface PackedChemicalLayout {
  profiles: ChemicalChannelProfile[];
  widths: number[];
  heights: number[];
  offsets: number[];
  total: number;
  maxWidth: number;
  maxHeight: number;
  shaderConstants: Record<string, string | number>;
}

function wgslArray(kind: "u32" | "f32", values: readonly number[]): string {
  const suffix = kind === "u32" ? "u" : "";
  return `array<${kind}, ${values.length}>(${values.map((value) => `${value}${suffix}`).join(", ")})`;
}

export function packChemicalChannelLayout(
  width: number,
  height: number,
  rawProfiles: readonly PartialProfile[],
): PackedChemicalLayout {
  const profiles = resolveChemicalChannelProfiles(rawProfiles.length, rawProfiles);
  const widths = profiles.map((profile) => Math.max(1, Math.round(width * profile.resolutionScale)));
  const heights = profiles.map((profile) => Math.max(1, Math.round(height * profile.resolutionScale)));
  const offsets: number[] = [];
  let total = 0;
  for (let i = 0; i < profiles.length; i++) {
    offsets.push(total);
    total += widths[i] * heights[i];
  }
  return {
    profiles,
    widths,
    heights,
    offsets,
    total,
    maxWidth: Math.max(...widths),
    maxHeight: Math.max(...heights),
    shaderConstants: {
      FIELD_WIDTHS: wgslArray("u32", widths),
      FIELD_HEIGHTS: wgslArray("u32", heights),
      FIELD_OFFSETS: wgslArray("u32", offsets),
      FIELD_RELAXATION_TIMES: wgslArray("f32", profiles.map((profile) => profile.relaxationTime)),
      FIELD_RESPONSE_TIMES: wgslArray("f32", profiles.map((profile) => profile.fieldResponseTime)),
      FIELD_DECAY_EXPONENTS: wgslArray("f32", profiles.map((profile) => profile.decayExponent)),
      FIELD_DIFFUSION_MULTIPLIERS: wgslArray("f32", profiles.map((profile) => profile.diffusionMultiplier)),
      FIELD_DEPOSIT_SIGMA_MULTIPLIERS: wgslArray("f32", profiles.map((profile) => profile.depositSigmaMultiplier)),
      FIELD_TOTAL: total,
      FIELD_MAX_WIDTH: Math.max(...widths),
      FIELD_MAX_HEIGHT: Math.max(...heights),
    },
  };
}
