import {MpmCore,REPULSION_FIELD_N} from "../src/gpu/mpmCore";
const out=document.querySelector("pre")!;const log=(s:string)=>out.textContent+=s+"\n";
async function run(){
 out.textContent="";const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU");
 const device=await adapter.requestDevice({requiredFeatures:["float32-blendable"]});device.pushErrorScope("validation");
 const n=REPULSION_FIELD_N,bytes=n*n*4;
 const module=device.createShaderModule({code:`
@group(0) @binding(0) var field:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(8,8) fn read(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<${n}u&&id.y<${n}u){result[id.y*${n}u+id.x]=textureLoad(field,vec2<i32>(id.xy),0).x;}}
`});
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
 const pipeline=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"read"}});
 for(const clustered of [false,true]){
  const results:Float32Array[]=[];
  for(const reduced of [false,true]){
   const count=4097,positions=new Float32Array(count*2),F=new Float32Array(count*4);
   for(let i=0;i<count;i++){positions.set(clustered?[.999,.001]:[(i*17%997)/997,(i*31%991)/991],i*2);F.set([1,0,0,1],i*4);}
   const core=new MpmCore(device,reduced);core.loadScene({count,positions,velocities:new Float32Array(count*2),F,C:new Float32Array(count*4),Jp:new Float32Array(count).fill(1)});
   core.setSplatRadius(.016);core.setMorphology(.01,1);
   const output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:core.densityTexture.createView()},{binding:1,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder();core.encodeMorphology(encoder);const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(n/8),Math.ceil(n/8));pass.end();
   encoder.copyBufferToBuffer(output,0,staging,0,bytes);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);results.push(new Float32Array(staging.getMappedRange().slice(0)));staging.unmap();staging.destroy();output.destroy();core.destroy();
  }
  let max=0;
  for(let i=0;i<n*n;i++){const error=Math.abs(results[0][i]-results[1][i]);max=Math.max(max,error/Math.max(1,results[0][i]));if(!Number.isFinite(error)||error>1e-4+Math.abs(results[0][i])*.0003)throw Error(`Density mismatch ${clustered} at ${i}: ${results[0][i]} vs ${results[1][i]}`);}
  log(`PASS Gaussian splats vs float workgroup density, clustered=${clustered}, max relative error=${max.toExponential(2)}`);
 }
 const error=await device.popErrorScope();if(error)throw error;device.destroy();log("ALL CHECKS PASSED");
}
run().catch(e=>log("FAIL: "+e.message));
