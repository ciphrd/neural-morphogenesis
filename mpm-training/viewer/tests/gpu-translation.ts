import config from "../../core/config.json";
const DT=config.simulation.DT;
import { MpmCore } from "../src/gpu/mpmCore";
const out=document.querySelector("pre")!;
const log=(s:string)=>out.textContent+=s+"\n";
async function run() {
 out.textContent="";
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU");
 const device=await adapter.requestDevice();device.pushErrorScope("validation");
 for(const reduced of [false,true]) for(const count of [1,65]) for(const clustered of [false,true]) for(const q of [1,1e-6,1e-8,1e-12]) {
  const core=new MpmCore(device,reduced);
  const positions=new Float32Array(count*2),velocities=new Float32Array(count*2),F=new Float32Array(count*4),domain=new Float32Array(count*6);
  for(let i=0;i<count;i++) {
   const x=clustered?.502:.102+(i%8)*.08,y=clustered?.502:.102+Math.floor(i/8)*.08;
   positions.set([x,y],i*2);velocities.set([1,-.5],i*2);F.set([1,0,0,1],i*4);
   domain.set([x-.0002,y-.0002,x+.0004,y-.0002,x-.0002,y+.0004],i*6);
  }
  core.loadScene({count,positions,velocities,F,C:new Float32Array(count*4),Jp:new Float32Array(count).fill(1),
   quadratureWeights:new Float32Array(count).fill(q),domainGeometry:"triangle-vertices",domain});
  core.setMaterial(0,.2,0,1,0,0,1);core.setGravity(0);core.setDamping(0,1);core.setRepulsionStrength(0,0);
  const encoder=device.createCommandEncoder();core.encodeSteps(encoder,100);device.queue.submit([encoder.finish()]);
  const p=await core.readPositions();
  for(let i=0;i<count;i++) {
   const errorX=Math.abs(p[2*i]-positions[2*i]-100*DT),errorY=Math.abs(p[2*i+1]-positions[2*i+1]+50*DT);
   if(!Number.isFinite(errorX+errorY)||errorX>7e-6 || errorY>7e-6) throw Error(`Lost translation reduced=${reduced} count=${count} clustered=${clustered} q=${q}: ${errorX}, ${errorY}`);
  }
  log(`PASS reduced=${reduced}, count=${count}, clustered=${clustered}, weight=${q}: translation preserved`);
  core.destroy();
 }
 const error=await device.popErrorScope();if(error)throw error;device.destroy();log("ALL CHECKS PASSED");
}
run().catch(e=>log("FAIL: "+e.message));
