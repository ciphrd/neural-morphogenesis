import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import ts from "typescript"
const source = readFileSync(new URL("../src/gpu/materialCull.ts", import.meta.url), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { materialCullPairs, outsideCenterCircleMask } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)
test("compaction preserves exactly the surviving records for all masks up to 8 samples", () => {
  for(let n=0;n<=8;n++) for(let bits=0;bits<2**n;bits++) {
    const mask=Uint8Array.from({length:n},(_,i)=>(bits>>i)&1)
    const {survivors,pairs}=materialCullPairs(mask)
    const records=Array.from({length:n},(_,i)=>({id:i,physics:i*7,memory:i*11}))
    for(let i=0;i<pairs.length;i+=2) {
      assert.ok(pairs[i]<survivors && pairs[i+1]>=survivors)
      records[pairs[i]]=records[pairs[i+1]]
    }
    assert.deepEqual(records.slice(0,survivors).map(r=>r.id).sort((a,b)=>a-b),Array.from(mask.keys()).filter(i=>!mask[i]))
  }
})

test("circle cull keeps the central 40%-diameter disk, including its boundary", () => {
 const positions=new Float32Array([.5,.5, .7,.5, .3,.5, .5,.7, .5,.3, .701,.5, .65,.65, 0,0, NaN,.5])
 assert.deepEqual(Array.from(outsideCenterCircleMask(positions)),[0,0,0,0,0,1,1,1,1])
 assert.equal(materialCullPairs(outsideCenterCircleMask(positions)).survivors,5)
 assert.equal(outsideCenterCircleMask(new Float32Array()).length,0)
})

test("90% auto-prune disk preserves its boundary and removes corners", () => {
 const points=new Float32Array([.5,.5, .05,.5, .95,.5, .5,.05, .5,.95, .049,.5, .951,.5, .85,.85])
 assert.deepEqual(Array.from(outsideCenterCircleMask(points,.9)),[0,0,0,0,0,1,1,1])
 assert.equal(outsideCenterCircleMask(new Float32Array([.8,.5]))[0],1)
 assert.equal(outsideCenterCircleMask(new Float32Array([.8,.5]),.9)[0],0)
})
