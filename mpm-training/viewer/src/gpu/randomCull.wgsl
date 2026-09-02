// Viewer-only random population cull. The host samples victims uniformly from
// the complete live population, then pairs each victim below the post-cull
// boundary with a surviving tail agent. Each invocation performs one such
// move, copying every physical and neural state field before the host shrinks
// the compact live prefix. Victims already in the tail need no GPU copy.

struct ParticleMeta {
  rng: u32,
  cooldown: f32,
  heading: f32,
  angularVelocity: f32,
  color: vec4<f32>,
  divisionHazard: f32,
  divisionThreshold: f32,
  privateState: array<f32, 8>,
  chemicalState: array<f32, __CHANNELS__>,
}

struct AgentState {
  growthCount: atomic<u32>,
  _padding: array<u32, 63>,
  particleMeta: array<ParticleMeta>,
}

struct ParticleRest {
  growthF: vec4<f32>,
  jp: f32,
  cycleActive: f32,
  growthAngle: f32,
  growthAnisotropy: f32,
  divisionBias: f32,
  growthFrameHeading: f32,
  appearanceScale: f32,
  _padding: f32,
}

struct CullParams {
  replacementCount: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> particleF: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> particleC: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> particleRest: array<ParticleRest>;
@group(0) @binding(5) var<storage, read_write> agentState: AgentState;
@group(0) @binding(6) var<storage, read> replacementPairs: array<vec2<u32>>;
@group(0) @binding(7) var<uniform> params: CullParams;

@compute @workgroup_size(64)
fn compactRandomCull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let replacementIndex = gid.x;
  if (replacementIndex >= params.replacementCount) { return; }

  let destination = replacementPairs[replacementIndex].x;
  let source = replacementPairs[replacementIndex].y;
  positions[destination] = positions[source];
  velocities[destination] = velocities[source];
  particleF[destination] = particleF[source];
  particleC[destination] = particleC[source];
  particleRest[destination] = particleRest[source];
  agentState.particleMeta[destination] = agentState.particleMeta[source];
}
