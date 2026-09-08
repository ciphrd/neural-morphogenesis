import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import ts from "typescript"
const source = readFileSync(new URL("../src/gpu/weightTransition.ts", import.meta.url), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { WeightTransition, BRAIN_TRANSITION_SECONDS } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)

test("brain weights and biases blend smoothly and finish at the exact target", () => {
 const blend = new WeightTransition()
 const current = blend.load(new Float32Array([0, -2, 0.5]))
 const target = new Float32Array([2, 4, -0.5])
 blend.start(target)
 assert.deepEqual(Array.from(current), [0, -2, 0.5])
 assert.equal(blend.advance(0), null)
 assert.deepEqual(Array.from(blend.advance(BRAIN_TRANSITION_SECONDS / 2)), [1, 1, 0])
 assert.deepEqual(blend.advance(BRAIN_TRANSITION_SECONDS / 2), target)
 assert.equal(blend.advance(1), null)
})
test("repeated randomization starts at the current values and loading cancels it", () => {
 const blend = new WeightTransition()
 blend.load(new Float32Array([0]))
 blend.start(new Float32Array([4]))
 const current = blend.advance(BRAIN_TRANSITION_SECONDS / 2)
 blend.start(new Float32Array([10]))
 assert.equal(current[0], 2)
 assert.equal(blend.advance(BRAIN_TRANSITION_SECONDS / 2)[0], 6)
 blend.load(new Float32Array([3]))
 assert.equal(blend.advance(10), null)
 assert.throws(() => blend.start(new Float32Array([1, 2])))
})
test("different frame cadences give the same result and invalid time is ignored", () => {
 const a = new WeightTransition(), b = new WeightTransition()
 for(const blend of [a,b]) { blend.load(new Float32Array([0])); blend.start(new Float32Array([10])) }
 const expected = a.advance(BRAIN_TRANSITION_SECONDS / 2)[0]
 let result
 for(let i=0;i<25;i++) result = b.advance(BRAIN_TRANSITION_SECONDS / 50)
 assert.ok(Math.abs(result[0]-expected)<1e-5)
 assert.equal(b.advance(NaN),null)
 assert.equal(b.advance(-1),null)
})
