import {workgroupReductionShader} from "../src/gpu/p2gReduction";
const out=document.querySelector("pre")!;const log=(s:string)=>out.textContent+=s+"\n";
async function run(){
 out.textContent="";const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU");
 const device=await adapter.requestDevice();device.pushErrorScope("validation");
 for(const spread of [false,true])for(const count of [1,65,4097]){
  const keys=spread?2048:9,bytes=keys*4;
  const expected=new Float64Array(keys),magnitudes=new Float64Array(keys);
  for(let p=0;p<count;p++)for(let j=0;j<27;j++){
   const key=(p*27+j)%keys,value=Math.fround((p%5-2)*Math.fround(1e-11));expected[key]+=value;magnitudes[key]+=Math.abs(value);
  }
  const source=workgroupReductionShader(`
@group(0) @binding(0) var<storage,read_write> sums: array<atomic<i32>>;
fn addFloat(index:u32,value:f32){
 var previous=atomicLoad(&sums[index]);loop{
  let next=bitcast<i32>(bitcast<f32>(previous)+value);let result=atomicCompareExchangeWeak(&sums[index],previous,next);
  if(result.exchanged){return;}previous=result.old_value;
 }
}
@compute @workgroup_size(64)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>){
 if(gid.x>=${count}u){return;}
 for(var j=0u;j<27u;j++){addFloat((gid.x*27u+j)%${keys}u,f32(i32(gid.x%5u)-2)*1e-11);}
}`,"addFloat",["deposit"]);
  const module=device.createShaderModule({code:source});const pipeline=device.createComputePipeline({layout:"auto",compute:{module,entryPoint:"deposit"}});
  const sums=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const staging=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:sums}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
  encoder.copyBufferToBuffer(sums,0,staging,0,bytes);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
  const actual=new Float32Array(staging.getMappedRange());
  for(let i=0;i<keys;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>1e-18+magnitudes[i]*2e-6)throw Error(`Reduction mismatch spread=${spread} count=${count} key=${i}: ${actual[i]} vs ${expected[i]}`);
  staging.unmap();staging.destroy();sums.destroy();log(`PASS count=${count}, ${spread?"hash saturation / fallback":"dense collision / cancellation"}, signed 1e-11 contributions`);
 }
 const error=await device.popErrorScope();if(error)throw error;device.destroy();log("ALL CHECKS PASSED");
}
run().catch(e=>log("FAIL: "+e.message));
