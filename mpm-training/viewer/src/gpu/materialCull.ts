/** Disjoint tail-to-hole copies preserve every surviving sample, including its neural state. */
export function materialCullPairs(mask: Uint8Array): { survivors: number; pairs: Uint32Array<ArrayBuffer> } {
  let killed=0
  for(const victim of mask) if(victim) killed++
  const survivors=mask.length-killed
  const pairs:number[]=[]
  let source=survivors
  for(let destination=0;destination<survivors;destination++) {
    if(!mask[destination]) continue
    while(source<mask.length && mask[source]) source++
    pairs.push(destination,source++)
  }
  return {survivors,pairs:new Uint32Array(pairs)}
}

/** Keep sample centers in a central disk; diameter is a fraction of world width. */
export function outsideCenterCircleMask(positions: Float32Array, diameter = 0.4): Uint8Array {
  const radiusSquared = (diameter / 2) ** 2
  const mask = new Uint8Array(positions.length / 2)
  for (let i = 0; i < mask.length; i++) {
    const x = positions[i * 2] - 0.5, y = positions[i * 2 + 1] - 0.5
    // Small tolerance retains boundary points represented as float32.
    mask[i] = Number.isFinite(x) && Number.isFinite(y) && x * x + y * y <= radiusSquared + 1e-8 ? 0 : 1
  }
  return mask
}
