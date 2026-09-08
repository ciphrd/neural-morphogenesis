import { GpuSimulation } from "../src/gpu/simulation";
import { createPerformanceConfig } from "../src/performance/config";
const out=document.querySelector("pre")!;
const log=(s:string)=>out.textContent+=s+"\n";
async function run(){
 out.textContent="";
 const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU");
 if(!adapter.features.has("timestamp-query"))throw Error("GPU timestamps unavailable");
 const features:GPUFeatureName[]=["timestamp-query"];
 if(adapter.features.has("float32-blendable"))features.push("float32-blendable");
 const device=await adapter.requestDevice({requiredFeatures:features,requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
 device.addEventListener("uncapturederror",e=>log("GPU ERROR: "+e.error.message));
 const query=device.createQuerySet({type:"timestamp",count:4096});
 const resolve=device.createBuffer({size:4096*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const staging=device.createBuffer({size:4096*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const createCompute=device.createComputePipeline.bind(device);
 device.createComputePipeline=(d)=>createCompute({...d,label:d.compute.entryPoint});
 const createRender=device.createRenderPipeline.bind(device);
 device.createRenderPipeline=(d)=>createRender({...d,label:d.label??d.vertex.entryPoint??"render"});
 let recording=false,names:string[]=[];
 const createEncoder=device.createCommandEncoder.bind(device);
 device.createCommandEncoder=(d)=>{
  const encoder=createEncoder(d);
  for(const kind of ["beginComputePass","beginRenderPass"] as const){
   const original=encoder[kind].bind(encoder) as any;
   (encoder as any)[kind]=(desc:any={})=>{
    const index=names.length;
    if(recording) names.push(kind);
    const pass=original(recording?{...desc,timestampWrites:{querySet:query,beginningOfPassWriteIndex:index*2,endOfPassWriteIndex:index*2+1}}:desc);
    const set=pass.setPipeline.bind(pass);
    pass.setPipeline=(pipeline:any)=>{if(recording)names[index]=pipeline.label;set(pipeline);};
    return pass;
   };
  }
  return encoder;
 };
 const count=Number(new URLSearchParams(location.search).get("count")??20000);
 const sim=new GpuSimulation(device,"rgba8unorm");
 const config={...createPerformanceConfig(42,true),particles:count,initialParticleCount:Math.floor(count/2),sampleSpacing:0.001*Math.sqrt(20000/count),initialCondition:"none" as const};
 sim.loadGeneration(config);
 log(`Agents: ${sim.particleCount}, hidden width: ${config.hiddenDim}, physics substeps: ${config.substepsPerMacro}`);
 for(let i=0;i<3;i++)await sim.step();await device.queue.onSubmittedWorkDone();
 const all=new Map<string,number[]>();
 for(let trial=0;trial<5;trial++){
  names=[];recording=true;const start=performance.now();await sim.step();recording=false;
  const commands=createEncoder();commands.resolveQuerySet(query,0,names.length*2,resolve,0);commands.copyBufferToBuffer(resolve,0,staging,0,names.length*16);
  device.queue.submit([commands.finish()]);await staging.mapAsync(GPUMapMode.READ);
  const times=new BigUint64Array(staging.getMappedRange());const sums=new Map<string,number>();
  for(let i=0;i<names.length;i++)sums.set(names[i],(sums.get(names[i])??0)+Number(times[2*i+1]-times[2*i])/1e6);
  staging.unmap();log(`Step ${trial}: ${(performance.now()-start).toFixed(2)} ms wall, ${[...sums.values()].reduce((a,b)=>a+b,0).toFixed(2)} ms GPU`);
  for(const [name,ms]of sums)all.set(name,[...(all.get(name)??[]),ms]);
 }
 log("Mean GPU time per macro step:");
 for(const [name,values] of [...all].sort((a,b)=>b[1].reduce((s,v)=>s+v,0)-a[1].reduce((s,v)=>s+v,0)))log(`${name}: ${(values.reduce((s,v)=>s+v,0)/values.length).toFixed(3)} ms`);
 sim.destroy();device.destroy();log("DONE");
}
run().catch(e=>log("FAIL: "+(e.stack??e)));
