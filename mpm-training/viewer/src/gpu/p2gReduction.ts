/** Accumulate nearby particles in a bounded workgroup hash before touching
 * the global grid. Both reductions retain float32 values, including tiny mass
 * and momentum. Hash collisions fall back to the original global writer.
 */
export function p2gReductionShader(source: string): string {
  return workgroupReductionShader(source, "addGridFloat", ["p2g"]);
}

export function workgroupReductionShader(source: string, writer: string, entries: string[], slots = 256): string {
  if (!source.includes(`fn ${writer}(`)) throw new Error(`Missing ${writer} writer`);
  let specialized = source.split(`${writer}(`).join("addLocalFloat(").replace("fn addLocalFloat(", "fn addGlobalFloat(");
  for (const name of entries) {
    const entry = `@compute @workgroup_size(64)\nfn ${name}(@builtin(global_invocation_id) gid: vec3<u32>)`;
    if (!specialized.includes(entry)) throw new Error(`Unexpected ${name} shader layout`);
    specialized = specialized.replace(entry, `fn workgroup_${name}(gid: vec3<u32>)`);
  }
  return specialized + `
const LOCAL_SLOTS: u32 = ${slots}u;
const EMPTY_KEY: u32 = 0xffffffffu;
var<workgroup> localKeys: array<atomic<u32>, ${slots}>;
var<workgroup> localValues: array<atomic<i32>, ${slots}>;
fn addLocalFloat(index: u32, value: f32) {
  if (value == 0.0) { return; }
  var hash = index;
  hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
  hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
  var slot = (hash ^ (hash >> 16u)) & (LOCAL_SLOTS - 1u);
  for (var probe = 0u; probe < 16u; probe++) {
    let found = atomicCompareExchangeWeak(&localKeys[slot], EMPTY_KEY, index);
    // A spurious weak-CAS failure on an empty slot can safely continue probing;
    // duplicate buckets are combined by the final floating-point global sum.
    if (found.exchanged || found.old_value == index) {
      var previous = atomicLoad(&localValues[slot]);
      loop {
        let next = bitcast<i32>(bitcast<f32>(previous) + value);
        let result = atomicCompareExchangeWeak(&localValues[slot], previous, next);
        if (result.exchanged) { return; }
        previous = result.old_value;
      }
    }
    slot = (slot + 1u) & (LOCAL_SLOTS - 1u);
  }
  addGlobalFloat(index, value);
}
${entries.map(name => `@compute @workgroup_size(64)
fn ${name}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  for (var slot = lane; slot < LOCAL_SLOTS; slot += 64u) {
    atomicStore(&localKeys[slot], EMPTY_KEY);
    atomicStore(&localValues[slot], 0);
  }
  workgroupBarrier();
  workgroup_${name}(gid);
  workgroupBarrier();
  for (var slot = lane; slot < LOCAL_SLOTS; slot += 64u) {
    let key = atomicLoad(&localKeys[slot]);
    if (key != EMPTY_KEY) { addGlobalFloat(key, bitcast<f32>(atomicLoad(&localValues[slot]))); }
  }
}`).join("\n")}
`;
}

/** Preserve the density kernel's existing per-contribution rounding, but avoid
 * overflowing its global 32-bit fixed-point sum in dense populations. */
export function densityReductionShader(source: string): string {
  const write = "atomicAdd(&densityAccum[idx], i32(round(weight * SCALE)));";
  const read = "f32(atomicLoad(&densityAccum[idx])) / SCALE";
  if (!source.includes(write) || !source.includes(read)) throw new Error("Unexpected density shader layout");
  const specialized = source.replace(write, "addDensityFloat(idx, round(weight * SCALE) / SCALE);")
    .replace(read, "bitcast<f32>(atomicLoad(&densityAccum[idx]))") + `
fn addDensityFloat(index: u32, value: f32) {
  if (value == 0.0) { return; }
  var previous = atomicLoad(&densityAccum[index]);
  loop {
    let next = bitcast<i32>(bitcast<f32>(previous) + value);
    let result = atomicCompareExchangeWeak(&densityAccum[index], previous, next);
    if (result.exchanged) { return; }
    previous = result.old_value;
  }
}
`;
  return workgroupReductionShader(specialized, "addDensityFloat", ["splatDensity"], 512);
}
