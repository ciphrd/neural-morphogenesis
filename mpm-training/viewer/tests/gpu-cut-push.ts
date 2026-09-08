import { MaterialCutter } from "../src/gpu/materialCutter"
import { GpuSimulation } from "../src/gpu/simulation"
import { createPerformanceConfig } from "../src/performance/config"
import type { CutPush } from "../src/gpu/cutPush"
const output=document.querySelector("pre")!
const log=(text:string)=>output.textContent+=text+"\n"
const check=(ok:boolean,text:string)=>{if(!ok)throw Error(text);log("PASS: "+text)}
async function run() {
 output.textContent=""
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU")
 const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}})
 device.pushErrorScope("validation")
 const triangles=[
  [.45,.45,.55,.45,.5,.55], // crossed
  [.3,.499,.3,.65,.4,.65], // only tip meets blade; centroid misses
  [.3,.7,.4,.7,.35,.8], // distant
  [.97,.48,.03,.48,.99,.55], // periodic seam
  [.1,.1,.4,.1,.25,.4], // blade wholly inside
 ]
 const count=triangles.length
 const positions=new Float32Array(count*2),rest=new Float32Array(count*16)
 triangles.forEach((v,i)=>{positions.set([(v[0]+v[2]+v[4])/3,(v[1]+v[3]+v[5])/3],i*2);rest.set([1,0,0,1,1,0,0,0,...v,.01,1],i*16)})
 const buffer=(data:Float32Array)=>{const b=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,data);return b}
 const core={positions:buffer(positions),rest:buffer(rest),activeCount:count}
 const cutter=new MaterialCutter(device,core)
 const stroke:CutPush={from:{x:.2,y:.5},to:{x:.8,y:.5},radius:.003,speed:1}
 const whole=await cutter.classify([stroke])
 check(Array.from(whole).join()==="1,1,0,0,0","cut hits triangle edges even when their particle centers miss")
 const split=Array.from({length:60},(_,i)=>({...stroke,from:{x:.2+i*.01,y:.5},to:{x:.2+(i+1)*.01,y:.5},speed:8}))
 check((await cutter.classify(split)).every((v,i)=>v===whole[i]),"subdividing a fast stroke produces the same cut")
 check((await cutter.classify(Array.from({length:260},()=>stroke))).every((v,i)=>v===whole[i]),"repeated strokes and batches are idempotent")
 check((await cutter.classify([{...stroke,from:{x:.245,y:.2},to:{x:.255,y:.2}}]))[4]===1,"blade entirely inside a triangle cuts it")
 const seam=await cutter.classify([{...stroke,from:{x:0,y:.5},to:{x:.015,y:.5}}])
 check(seam[3]===1,"periodic domain fragment at the seam is cut")
 const invalid=await cutter.classify([{...stroke,radius:NaN}])
 check(invalid.every(v=>v===0),"invalid blade is ignored")
 cutter.destroy();core.positions.destroy();core.rest.destroy()
 const sim=new GpuSimulation(device,"rgba8unorm")
 const config=createPerformanceConfig(42)
 config.fastAccumulation=false
 config.initialParticleCount=100
 sim.loadGeneration(config)
 const before=await sim.readPositions()
 const line={...stroke,from:{x:0,y:before[1]},to:{x:1,y:before[1]},radius:.002}
 const killed=await sim.cutMaterial([line])
 const after=await sim.readPositions()
 check(killed>0 && after.length>0 && after.length===before.length-killed*2,"cut removes intersected material while preserving other particles")
 const original=new Set(Array.from({length:before.length/2},(_,i)=>`${before[2*i]},${before[2*i+1]}`))
 check(Array.from({length:after.length/2},(_,i)=>original.has(`${after[2*i]},${after[2*i+1]}`)).every(Boolean),"survivor positions are copied exactly, with no artificial displacement")
 check(await sim.cutMaterial([line])===0,"cutting the same path twice does not remove extra material")
 for(let i=0;i<8;i++)await sim.step()
 check((await sim.readPositions()).every(Number.isFinite),"physics remains finite after cutting")
 const pending=sim.cutMaterial([line]);sim.restartRollout()
 const ignored=await pending
 check(ignored===0 && sim.particleCount===before.length/2,`restart invalidates an in-flight cut (removed=${ignored}, count=${sim.particleCount}, expected=${before.length/2})`)
 const cover=Array.from({length:11},(_,i)=>({...stroke,from:{x:0,y:i/10},to:{x:1,y:i/10},radius:.1}))
 await sim.cutMaterial(cover)
 check(sim.particleCount===0,"cut can remove the final piece without leaving a ghost particle")
 await sim.step()
 check(sim.particleCount===0,"empty simulation stays valid after cutting")
 sim.destroy()
 const error=await device.popErrorScope();if(error)throw error
 device.destroy();log("ALL CHECKS PASSED")
}
run().catch(error=>log("FAIL: "+error.message))
