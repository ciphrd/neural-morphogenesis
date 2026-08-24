// Grid-native, NN-controlled contact viscosity.
//
// P2G deposits each particle's weldExpression as an additional mass-weighted
// accumulator channel. This pass runs after gridUpdate has resolved momentum
// into velocity and before G2P. It reads only grid fields and writes a separate
// velocity buffer, so there are no particle pairs and no in-place neighbor
// races. Once two fronts share MPM nodes, expressed regions resist relative
// motion through bounded velocity diffusion.

const GRID_N: u32 = __GRID_N__u;
const STRIDE: u32 = GRID_N + 1u;
const NODE_COUNT: u32 = STRIDE * STRIDE;
const DX: f32 = __DX__;
const DT: f32 = __DT__;
const SCALE: f32 = 4096.0;

const CH_MASS: u32 = 2u;
const CH_WELD_MASS: u32 = 3u;
const CHANNELS: u32 = 4u;

struct WeldingParams {
  // Diffusion rate in inverse simulation-time units. Zero is an exact copy.
  strength: f32,
  // Hard numerical cap on one pass's neighbor-average blend.
  maxBlend: f32,
  // Grid mass at which the local material is considered dense enough.
  densityReference: f32,
  _padding: f32,
}

@group(0) @binding(0) var<storage, read_write> gridAccum: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> sourceVelocity: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> weldedVelocity: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: WeldingParams;

fn nodeIndex(x: u32, y: u32) -> u32 {
  return x * STRIDE + y;
}

fn nodeMass(index: u32) -> f32 {
  return f32(atomicLoad(&gridAccum[index * CHANNELS + CH_MASS])) / SCALE;
}

fn nodeActivation(index: u32, mass: f32) -> f32 {
  if (mass <= 0.0) { return 0.0; }
  let weldMass = f32(atomicLoad(
    &gridAccum[index * CHANNELS + CH_WELD_MASS]
  )) / SCALE;
  let expression = clamp(weldMass / max(mass, 1e-6), 0.0, 1.0);
  let expressionGate = smoothstep(0.05, 0.8, expression);
  let densityGate = smoothstep(
    0.25 * max(params.densityReference, 1e-6),
    max(params.densityReference, 1e-6),
    mass,
  );
  return expressionGate * densityGate;
}

@compute @workgroup_size(64)
fn applyGridWelding(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= NODE_COUNT) { return; }
  let x = index / STRIDE;
  let y = index % STRIDE;
  // P2G deliberately uses only [0, GRID_N) in the toroidal domain; preserve
  // the allocated compatibility strip as zero rather than coupling through it.
  if (x >= GRID_N || y >= GRID_N) {
    weldedVelocity[index] = vec2<f32>(0.0);
    return;
  }

  let centerVelocity = sourceVelocity[index];
  let mass = nodeMass(index);
  if (params.strength <= 0.0 || mass <= 0.0) {
    weldedVelocity[index] = centerVelocity;
    return;
  }

  let activation = nodeActivation(index, mass);

  let left = nodeIndex((x + GRID_N - 1u) % GRID_N, y);
  let right = nodeIndex((x + 1u) % GRID_N, y);
  let down = nodeIndex(x, (y + GRID_N - 1u) % GRID_N);
  let up = nodeIndex(x, (y + 1u) % GRID_N);
  let neighbors = array<u32, 4>(left, right, down, up);

  // Symmetric edge flux: both endpoints independently evaluate the same
  // min(activation) and harmonic mass. Their momentum changes are therefore
  // equal and opposite even when node masses differ.
  var momentumFlux = vec2<f32>(0.0);
  for (var k = 0u; k < 4u; k = k + 1u) {
    let neighbor = neighbors[k];
    let m = nodeMass(neighbor);
    if (m > 0.0) {
      let edgeActivation = min(activation, nodeActivation(neighbor, m));
      let harmonicMass = 2.0 * mass * m / max(mass + m, 1e-6);
      momentumFlux = momentumFlux
        + edgeActivation * harmonicMass * (sourceVelocity[neighbor] - centerVelocity);
    }
  }

  // Exponential rate conversion makes the effect stable across different
  // physics-substep counts; maxBlend protects the explicit four-edge flux.
  let blend = min(
    1.0 - exp(-max(params.strength, 0.0) * DT),
    clamp(params.maxBlend, 0.0, 0.1),
  );
  weldedVelocity[index] = centerVelocity + blend * momentumFlux / max(mass, 1e-6);
}
