import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import ts from "typescript"
const source = readFileSync(new URL("../src/gpu/cutPush.ts", import.meta.url), "utf8")
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const { makeCutPush, isCutPush } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)
const a={x:.2,y:.3},b={x:.4,y:.3}
test("blade width and geometry are independent of pointer speed",()=>{
 const slow=makeCutPush(a,b,200,1,320),fast=makeCutPush(a,b,50,1,320)
 assert.ok(fast.speed>slow.speed)
 assert.equal(slow.radius,fast.radius)
 assert.deepEqual(slow.from,fast.from)
 assert.deepEqual(slow.to,fast.to)
 assert.equal(slow.radius,2/320)
 assert.equal(makeCutPush(a,b,0,1,320),null)
 assert.equal(makeCutPush(a,a,10,1,320),null)
})
test("zoom preserves the on-screen blade width",()=>{
 const normal=makeCutPush(a,b,100,1,320),zoomed=makeCutPush(a,b,100,2,320)
 assert.equal(zoomed.speed,normal.speed)
 assert.equal(zoomed.radius,normal.radius/2)
 assert.equal(zoomed.from.x,.35)
 assert.equal(zoomed.from.y,.6)
})
test("speed is consistent across event frequencies for the same motion",()=>{
 const whole=makeCutPush(a,b,100,1,320)
 const half=makeCutPush(a,{x:.3,y:.3},50,1,320)
 assert.ok(Math.abs(whole.speed-half.speed)<1e-12)
 assert.equal(makeCutPush(a,b,10,NaN,320),null)
 assert.equal(isCutPush({...whole,radius:-1}),false)
})

test("zoomed-out margins do not clamp strokes onto the material boundary",()=>{
 assert.equal(makeCutPush({x:.1,y:.1},{x:.2,y:.2},100,.5,320),null)
 const clipped=makeCutPush({x:0,y:.5},{x:1,y:.5},100,.5,320)
 assert.deepEqual(clipped.from,{x:0,y:.5})
 assert.deepEqual(clipped.to,{x:1,y:.5})
})
