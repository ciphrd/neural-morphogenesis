import { Renderer, type ParticleShape, type ParticleColorMode } from "../src/gpu/render";
import { MpmCore } from "../src/gpu/mpmCore";
import { Environment } from "../src/gpu/environment";
import { VIEWER_DEFAULTS } from "../src/viewerConfig";
const out = document.querySelector("pre")!;
const log = (text: string) => out.textContent += text + "\n";
const check = (condition: boolean, message: string) => { if (!condition) throw Error(message); log("PASS " + message); };
async function run() {
  out.textContent = "";
  const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw Error("No GPU");
  const device = await adapter.requestDevice({requiredLimits: {maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage}});
  device.pushErrorScope("validation");
  const core = new MpmCore(device);
  core.loadScene({count:1, positions:new Float32Array([.5,.5]), velocities:new Float32Array(2),
    F:new Float32Array([1,0,0,1]),C:new Float32Array(4),Jp:new Float32Array([1]),
    domainGeometry:"triangle-vertices",domain:new Float32Array([.3,.3,.7,.3,.5,.7])});
  const environment = new Environment(device, {channels:3,width:16,height:16,decay:0,depositRate:0}, core.gridVel);
  const data = new Float32Array(environment.layout.total);
  for (let c=0;c<3;c++) {
    const w=environment.layout.widths[c], h=environment.layout.heights[c];
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) data[environment.layout.offsets[c]+y*w+x] = c===0 ? -2+4*(x+.5)/w : c===1 ? -2+4*(y+.5)/h : -2;
  }
  for(const buffer of environment.buffers) device.queue.writeBuffer(buffer,0,data);
  const meta=device.createBuffer({size:256+80,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const renderer=new Renderer(device,"rgba8unorm",core,environment,meta,core.growthField);
  renderer.setCanvasSizePx(256,256);renderer.setZoom(1);renderer.setParticleAlpha(1);
  renderer.setFieldMode("none");renderer.setDomainVisible(false);renderer.setTargetVisible(false);
  renderer.setBloom({...VIEWER_DEFAULTS.rendering.bloom,enabled:false});
  const canvas=document.querySelector("canvas")!; const context=canvas.getContext("webgpu")!;
  context.configure({device,format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const capture=async(shape:ParticleShape,color:ParticleColorMode)=>{
    renderer.setParticleShape(shape);renderer.setParticleColorMode(color);renderer.render(context,1);
    const staging=device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:context.getCurrentTexture()},{buffer:staging,bytesPerRow:1024},{width:256,height:256});
    device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
    const pixels=new Uint8Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return pixels;
  };
  const pixel=(image:Uint8Array,x:number,y:number)=>Array.from(image.slice((y*256+x)*4,(y*256+x)*4+3));
  // Real GPU validation catches WGSL/bind-group errors that TypeScript cannot.
  const creationError=await device.popErrorScope();if(creationError)throw creationError;
  device.pushErrorScope("validation");
  for(const shape of ["dot","dot-center","triangle","domain","domain-wireframe"] as ParticleShape[])
    for(const color of ["white","neural-color","growth-magnitude","neural-memory","chemical-memory","boundary-value","neurons","substrate"] as ParticleColorMode[]) await capture(shape,color);
  const error=await device.popErrorScope();if(error)throw error;
  log("PASS all 40 shape/color combinations render without GPU validation errors");
  const filled=await capture("domain","substrate"), wire=await capture("domain-wireframe","substrate");
  const interior=pixel(wire,128,140), background=pixel(wire,10,10);
  check(interior.every((v,i)=>Math.abs(v-background[i])<3),"wireframe interior stays empty");
  check(pixel(filled,128,140)[0]>50,"filled domain retains its interior");
  const left=pixel(wire,100,178),right=pixel(wire,156,178);
  check(right[0]>left[0]+25 && left[1]>30,"substrate color interpolates along a domain edge");
  renderer.setPointRadiusPx(32);renderer.setCenterDotSize(.18);
  const plain=await capture("dot","neural-color"),centered=await capture("dot-center","neural-color");
  check(pixel(centered,128,128)[0]>pixel(plain,128,128)[0]+50,"white center appears over the selected color");
  check(pixel(centered,140,128)[0]<30,"small center leaves surrounding dot unchanged");
  renderer.setCenterDotSize(.5);const larger=await capture("dot-center","neural-color");
  check(pixel(larger,140,128)[0]>50,"center size changes its visible radius");
  renderer.setParticleAlpha(0);const transparent=await capture("dot-center","neural-color");
  check(pixel(transparent,128,128)[0]>50,"semi-transparent center remains visible at zero base opacity");
  await capture("domain-wireframe","substrate");
  log("ALL CHECKS PASSED");
  renderer.destroy();environment.destroy();core.destroy();meta.destroy();device.destroy();
}
run().catch(error=>log("FAIL: "+error.message));
