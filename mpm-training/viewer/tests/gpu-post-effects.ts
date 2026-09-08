import { BloomPostProcess } from '../src/gpu/bloom';
import { POST_EFFECT_DEFAULTS } from '../src/gpu/postEffects';
const out=document.querySelector('pre')!;const log=(s:string)=>out.textContent+=s+'\n';
async function run(){
 out.textContent='';const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No GPU');
 const device=await adapter.requestDevice();device.pushErrorScope('validation');
 const effect=new BloomPostProcess(device,'rgba8unorm');effect.resize(128,128);
 const base={...POST_EFFECT_DEFAULTS,enabled:false,intensity:1,threshold:.2,radiusPx:2,scatter:.5,levels:4};
 const shader=device.createShaderModule({code:`
 @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{let p=array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3));return vec4<f32>(p[i],0,1);}
 @fragment fn fragment(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32>{let checker=f32((u32(p.x)/2u+u32(p.y)/2u)%2u);return vec4<f32>(vec3<f32>(checker),1);}`});
 const pipeline=device.createRenderPipeline({layout:'auto',vertex:{module:shader,entryPoint:'vertex'},fragment:{module:shader,entryPoint:'fragment',targets:[{format:'rgba16float'}]}});
 async function render(settings:typeof base,label?:string){
  effect.setSettings(settings);
  if(settings.strobeEnabled) (effect as unknown as {strobePhases:number[]}).strobePhases=[.25,.5,.75];
  const target=device.createTexture({size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
  const staging=device.createBuffer({size:128*128*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const encoder=device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:effect.sceneView,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});pass.setPipeline(pipeline);pass.draw(3);pass.end();effect.encode(encoder,target.createView());
  encoder.copyTextureToBuffer({texture:target},{buffer:staging,bytesPerRow:512},[128,128]);device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
  const data=new Uint8ClampedArray(staging.getMappedRange().slice(0));staging.destroy();target.destroy();
  if(label){const canvas=document.createElement('canvas');canvas.width=128;canvas.height=128;canvas.title=label;canvas.getContext('2d')!.putImageData(new ImageData(data,128,128),0,0);document.querySelector('#images')!.append(canvas);}
  return data;
 }
 const original=await render(base,'Original');
 const dof=await render({...base,dofEnabled:true},'Depth of field');
 let outerDifference=0;
 for(let y=0;y<128;y++)for(let x=0;x<128;x++){
  const i=(y*128+x)*4;
  if(y>=56&&y<72&&dof[i]!==original[i])throw Error('Focus band was blurred');
  if(y<16||y>=112)outerDifference+=Math.abs(dof[i]-original[i]);
 }
 if(outerDifference<10000)throw Error('Outer regions were not blurred');
 const rgb=await render({...base,rgbEnabled:true},'RGB slicer');
 if(!rgb.some((v,i)=>i%4!==3&&v!==original[i]))throw Error('RGB slicer had no effect');
 if(!rgb.some((v,i)=>i%4===0&&(v!==rgb[i+1]||v!==rgb[i+2])))throw Error('RGB channels were synchronized');
 const neutral=await render({...base,rgbEnabled:true,rgbMix:0,dofEnabled:true,dofTop:0,dofFront:0});
 if(neutral.some((v,i)=>v!==original[i]))throw Error('Neutral effects changed pixels');
 await render({...base,enabled:true,dofEnabled:true,rgbEnabled:true},'Combined with bloom');
 const strobe=await render({...base,strobeEnabled:true,strobeRedSpeed:0,strobeGreenSpeed:0,strobeBlueSpeed:0},'Strobe: independent frozen channels');
 for(let i=0;i<strobe.length;i+=4){
  if(strobe[i]!==original[i]||strobe[i+1]!==0||strobe[i+2]!==0)throw Error('Strobe is not a uniform per-channel multiplication');
 }
 const strobeNeutral=await render({...base,strobeEnabled:true,strobeMix:0,strobeRedSpeed:0,strobeGreenSpeed:0,strobeBlueSpeed:0});
 if(strobeNeutral.some((v,i)=>v!==original[i]))throw Error('Zero strobe mix changed pixels');
 const error=await device.popErrorScope();if(error)throw error;
 effect.destroy();device.destroy();log('PASS: sharp focus band, blurred top/front, independent RGB modulation, neutral identity, combined bloom.');log('PASS: spatially uniform independent strobe channels and zero-mix identity.');log('ALL CHECKS PASSED');
}
run().catch(e=>log('FAIL: '+(e.stack??e.message)));
