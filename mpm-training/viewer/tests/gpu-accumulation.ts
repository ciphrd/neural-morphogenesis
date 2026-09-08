import { GpuSimulation } from "../src/gpu/simulation";
import { createPerformanceConfig } from "../src/performance/config";
import { ChemicalSplats } from "../src/gpu/chemicalSplats";
import { packChemicalChannelLayout, homogeneousChemicalChannelProfiles } from "../src/gpu/chemicalChannels";

const output = document.querySelector("pre")!;
const log = (s: string) => output.textContent += s + "\n";
async function run() {
  output.textContent = "";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error("No WebGPU adapter");
  const hasSplats = adapter.features.has("float32-blendable");
  const device = await adapter.requestDevice({requiredFeatures: hasSplats ? ["float32-blendable"] : [], requiredLimits: {maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage}});
  device.addEventListener("uncapturederror", e => log("GPU ERROR: " + e.error.message));
  const read = async (source: GPUBuffer, bytes: number) => {
    const staging = device.createBuffer({size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy(); return result;
  };
  if(!hasSplats) {log("SKIP splats: float32-blendable unavailable"); return;}
  for (const clustered of [false, true]) {
    const count = 4096, channels = 2;
    const layout = packChemicalChannelLayout(32, 24, homogeneousChemicalChannelProfiles(channels).map((p,c)=>({...p,resolutionScale: c ? .5 : 1})));
    const data = new Float32Array(count*(channels+3));
    const expected = new Float64Array(layout.total*2);
    for(let p=0; p<count; p++) {
      const base = p*(channels+3);
      data[base] = clustered ? .001 : ((p*13)%997)/997;
      data[base+1] = clustered ? .999 : ((p*31)%991)/991;
      data[base+2] = 1e-8 * (1 + p%5);
      data[base+3] = p%3 ? 2.5 : -4;
      data[base+4] = -70000; // Signed and outside float16 range.
      for(let c=0;c<channels;c++) {
        const w=layout.widths[c],h=layout.heights[c];
        const px=data[base]*w-.5, py=data[base+1]*h-.5;
        const bx=Math.floor(px-.5),by=Math.floor(py-.5),fx=px-bx,fy=py-by;
        const weights=(f:number)=>[.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2];
        const wx=weights(fx),wy=weights(fy);
        for(let x=0;x<3;x++) for(let y=0;y<3;y++) {
          const i=layout.offsets[c]+((by+y+h)%h)*w+(bx+x+w)%w;
          const area=data[base+2]*wx[x]*wy[y];
          expected[i]+=area*data[base+3+c]; expected[layout.total+i]+=area;
        }
      }
    }
    device.pushErrorScope("validation");
    const scratch = device.createBuffer({size: Math.max(data.length,layout.total*2)*4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    const splats = new ChemicalSplats(device,layout,scratch);
    device.queue.writeBuffer(scratch,0,data);
    const encoder=device.createCommandEncoder(); splats.encode(encoder,count); device.queue.submit([encoder.finish()]);
    const error=await device.popErrorScope(); if(error) throw error;
    const actual=await read(scratch,layout.total*2*4);
    let relative=0;
    for(let i=0;i<actual.length;i++) {
      const err=Math.abs(actual[i]-expected[i]);
      const tolerance=1e-9+Math.abs(expected[i])*0.002;
      if(!Number.isFinite(actual[i]) || err>tolerance) throw Error(`Splat mismatch ${clustered} at ${i}: ${actual[i]} vs ${expected[i]}`);
      relative=Math.max(relative,err/Math.max(1e-9,Math.abs(expected[i])));
    }
    log(`PASS ${clustered ? "clustered / wrapped" : "spread"} splats: 4096 particles, mixed resolutions, signed values; max relative error ${relative.toExponential(2)}`);
    const referenceModule = device.createShaderModule({code: `
@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<atomic<i32>>;
fn add(index: u32, value: f32) {
  if(value == 0.0) { return; }
  var previous = atomicLoad(&sums[index]);
  loop {
    let next = bitcast<i32>(bitcast<f32>(previous) + value);
    let result = atomicCompareExchangeWeak(&sums[index], previous, next);
    if(result.exchanged) { return; }
    previous = result.old_value;
  }
}
@compute @workgroup_size(64) fn deposit(@builtin(global_invocation_id) id: vec3<u32>) {
  if(id.x >= ${count}u) {return;}
  let base = id.x * ${channels+3}u;
  let widths = array<u32,2>(${layout.widths.map(v=>v+"u").join(",")});
  let heights = array<u32,2>(${layout.heights.map(v=>v+"u").join(",")});
  let offsets = array<u32,2>(${layout.offsets.map(v=>v+"u").join(",")});
  for(var c=0u;c<2u;c++) {
    let size = vec2<f32>(f32(widths[c]),f32(heights[c]));
    let p = vec2<f32>(samples[base],samples[base+1u])*size-vec2<f32>(0.5);
    let b = vec2<i32>(floor(p-vec2<f32>(0.5))); let f=p-vec2<f32>(b);
    let w = array<vec2<f32>,3>(0.5*(vec2<f32>(1.5)-f)*(vec2<f32>(1.5)-f),
      vec2<f32>(0.75)-(f-vec2<f32>(1.0))*(f-vec2<f32>(1.0)),
      0.5*(f-vec2<f32>(0.5))*(f-vec2<f32>(0.5)));
    for(var x=0u;x<3u;x++) {for(var y=0u;y<3u;y++) {
      let ix = u32((b.x+i32(x)+i32(widths[c]))%i32(widths[c]));
      let iy = u32((b.y+i32(y)+i32(heights[c]))%i32(heights[c]));
      let i=offsets[c]+iy*widths[c]+ix;let area=samples[base+2u]*w[x].x*w[y].y;
      add(i,area*samples[base+3u+c]);add(${layout.total}u+i,area);
    }}
  }
}`});
    const referencePipeline=device.createComputePipeline({layout:"auto",compute:{module:referenceModule,entryPoint:"deposit"}});
    const samples=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(samples,0,data);
    const referenceGroup=device.createBindGroup({layout:referencePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:samples}},{binding:1,resource:{buffer:scratch}},
    ]});
    const benchmark=async (raster:boolean) => {
      // Warm-up plus several separately timed batches, including submission overhead.
      const timings:number[]=[];
      for(let repeat=0;repeat<4;repeat++) {
        await device.queue.onSubmittedWorkDone();const start=performance.now();
        for(let iteration=0;iteration<5;iteration++) {
          const commands=device.createCommandEncoder();
          if(raster) { device.queue.writeBuffer(scratch,0,data);splats.encode(commands,count); }
          else {
            commands.clearBuffer(scratch);const pass=commands.beginComputePass();
            pass.setPipeline(referencePipeline);pass.setBindGroup(0,referenceGroup);
            pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
          }
          device.queue.submit([commands.finish()]);
        }
        await device.queue.onSubmittedWorkDone();if(repeat)timings.push((performance.now()-start)/5);
      }
      return timings.sort((a,b)=>a-b)[1];
    };
    const casMs=await benchmark(false),splatMs=await benchmark(true);
    log(`TIMING ${clustered?"clustered":"spread"}: CAS ${casMs.toFixed(2)} ms, splats ${splatMs.toFixed(2)} ms per deposition (wall time incl. submissions)`);
    samples.destroy();splats.destroy();scratch.destroy();
  }
  for (const fast of [false, true]) for (const architecture of ["persistent-environment", "cell-owned-projection"] as const) {
    device.pushErrorScope("validation");
    const sim = new GpuSimulation(device, "rgba8unorm");
    sim.loadGeneration({...createPerformanceConfig(42, fast), particles: 256,
      chemicalCommunicationArchitecture: architecture});
    for(let i=0;i<5;i++) await sim.step();
    await device.queue.onSubmittedWorkDone();
    const error = await device.popErrorScope(); if(error) throw error;
    const positions = await sim.readPositions();
    if(!positions.length || positions.some(v=>!Number.isFinite(v))) throw Error("Invalid simulation positions");
    log(`PASS full simulation: fast=${fast}, ${architecture}, ${sim.particleCount} particles`);
    sim.destroy();
  }
  for (const architecture of ["persistent-environment", "cell-owned-projection"] as const) {
    device.pushErrorScope("validation");
    const sim = new GpuSimulation(device, "rgba8unorm");
    sim.loadGeneration({...createPerformanceConfig(42, true), particles: 5000, initialParticleCount: 2500,
      sampleSpacing: .001, chemicalCommunicationArchitecture: architecture});
    for(let i=0;i<3;i++) await sim.step();
    sim.killFraction(.8);
    for(let i=0;i<3;i++) await sim.step();
    const positions = await sim.readPositions();
    if(!positions.length || positions.some(v=>!Number.isFinite(v))) throw Error("Invalid positions across accumulation mode switch");
    const error = await device.popErrorScope();if(error)throw error;
    log(`PASS high-count reduction → splat switch after culling: ${architecture}, ${sim.particleCount} particles`);
    sim.destroy();
  }
  device.destroy();
  const fallbackAdapter = await navigator.gpu.requestAdapter();
  if (!fallbackAdapter) throw Error("No fallback adapter");
  const fallbackDevice = await fallbackAdapter.requestDevice({requiredLimits: {maxStorageBuffersPerShaderStage: fallbackAdapter.limits.maxStorageBuffersPerShaderStage}});
  fallbackDevice.pushErrorScope("validation");
  for (const architecture of ["persistent-environment", "cell-owned-projection"] as const) {
    const sim = new GpuSimulation(fallbackDevice, "rgba8unorm");
    sim.loadGeneration({...createPerformanceConfig(42, true), particles: 256, chemicalCommunicationArchitecture: architecture});
    for(let i=0;i<5;i++) await sim.step();
    const positions=await sim.readPositions();
    if(!positions.length || positions.some(v=>!Number.isFinite(v))) throw Error("Invalid fallback positions");
    log(`PASS floating-point chemistry fallback without float32 blending: ${architecture}`);
    sim.destroy();
  }
  await fallbackDevice.queue.onSubmittedWorkDone();
  const fallbackError=await fallbackDevice.popErrorScope(); if(fallbackError) throw fallbackError;
  fallbackDevice.destroy();
  log("ALL CHECKS PASSED");
}
run().catch(e=>log("FAIL: "+(e.stack ?? e.message ?? String(e))));
