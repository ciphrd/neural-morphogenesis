import { GpuSimulation } from '../src/gpu/simulation';
import { createPerformanceConfig } from '../src/performance/config';
import type { Environment } from '../src/gpu/environment';
const out=document.querySelector('pre')!;
const log=(s:string)=>out.textContent+=s+'\n';
async function run(){
 out.textContent='';
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No GPU');
 const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('float32-blendable')?['float32-blendable']:[],requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
 device.pushErrorScope('validation');
 const pipeline=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:`
 @group(0) @binding(0) var<storage,read> field:array<f32>;
 @group(0) @binding(1) var<storage,read_write> result:array<f32>;
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<arrayLength(&field)){result[id.x]=field[id.x];}}`}),entryPoint:'main'}});
 async function read(sim:GpuSimulation){
  const env=(sim as unknown as {environment:Environment}).environment;
  const size=env.layout.total*4;
  const result=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const staging=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:env.buffers[env.parity]}},{binding:1,resource:{buffer:result}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(env.layout.total/64));pass.end();encoder.copyBufferToBuffer(result,0,staging,0,size);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
  const data=new Float32Array(staging.getMappedRange().slice(0));staging.destroy();result.destroy();return data;
 }
 // Isolate the appended audio input: all other weights and biases are zero.
 for (const legacy of [false, true]) {
  const config=createPerformanceConfig(42,true);
  config.weights.fc1w.forEach(row=>row.fill(0));config.weights.fc1b.fill(0);
  config.weights.fc2w.forEach(row=>row.fill(0));config.weights.fc2b.fill(0);
  config.weights.fc1w[0][config.weights.fc1w[0].length-1]=1;
  config.weights.fc2w[0][0]=1;
  if(legacy)config.weights.fc1w=config.weights.fc1w.map(row=>row.slice(0,-1));
  const sim=new GpuSimulation(device,'rgba8unorm');
  sim.loadGeneration({...config,particles:64,initialParticleCount:64,mpmEnabled:false,growthSteps:0});
  sim.setAudioEnergy(0);await sim.step();
  if((await read(sim)).some(v=>v!==0))throw Error('Silence injected chemistry');
  sim.setAudioEnergy(1);await sim.step();
  const peak=Math.max(...await read(sim));
  if(legacy ? peak!==0 : !(peak>0))throw Error('GPU audio/legacy compatibility failed');
  sim.destroy();log(`PASS GPU audio input: legacy=${legacy}, peak=${peak}`);
 }
 for(const count of [64,4200]){
  const sims=[false,true].map(fast=>{const sim=new GpuSimulation(device,'rgba8unorm');sim.loadGeneration({...createPerformanceConfig(42,fast),particles:count,initialParticleCount:count,mpmEnabled:false,growthSteps:0,sampleSpacing:.001});return sim;});
  for(let step=1;step<=60;step++){
   for(const sim of sims)await sim.step();
   if(step===1||step===60){
    const [a,b]=await Promise.all(sims.map(read));let error=0,scale=0;
    for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('Nonfinite substrate');error=Math.max(error,Math.abs(a[i]-b[i]));scale=Math.max(scale,Math.abs(a[i]));}
    const env=(sims[0] as unknown as {environment:Environment}).environment;
    const saturated=env.layout.offsets.map((offset,c)=>{const end=offset+env.layout.widths[c]*env.layout.heights[c];let n=0,peak=0;for(let i=offset;i<end;i++){if(Math.abs(a[i])>=2)n++;peak=Math.max(peak,Math.abs(a[i]));}return {channel:c,peak,saturated:n};});
    log(`count=${count}, step=${step}, max=${scale}, fast/reference error=${error}, channels=${JSON.stringify(saturated)}`);
    if(error>1e-5+scale*.002)throw Error('Optimized substrate diverged from reference');
    // This fixed-seed, stationary fixture reached 3.33 with the post-merge
    // chemistry defaults; training-compatible defaults peak at 1.73.
    if(count===4200 && step===60 && scale>=2)throw Error('Training-default fixture saturated the substrate display');
   }
  }
  sims.forEach(sim=>sim.destroy());
 }
 const error=await device.popErrorScope();if(error)throw error;device.destroy();log('ALL CHECKS PASSED');
}
run().catch(e=>log('FAIL: '+(e.stack??e.message)));
