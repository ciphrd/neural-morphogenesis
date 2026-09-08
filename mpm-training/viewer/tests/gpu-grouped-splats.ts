import {ChemicalSplats} from "../src/gpu/chemicalSplats";
import {packChemicalChannelLayout,homogeneousChemicalChannelProfiles} from "../src/gpu/chemicalChannels";
const out=document.querySelector("pre")!;const log=(s:string)=>out.textContent+=s+"\n";
async function run(){
 out.textContent="";const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error("No GPU");
 const device=await adapter.requestDevice({requiredFeatures:["float32-blendable"]});device.pushErrorScope("validation");
 for(const resolution of [1,2,32]){
  const channels=7,count=65;
  const layout=packChemicalChannelLayout(resolution,resolution,homogeneousChemicalChannelProfiles(channels).map((p,c)=>({...p,resolutionScale:c===3||c===6?.5:1})));
  const data=new Float32Array(count*(channels+3)),expected=new Float64Array(layout.total*2);
  for(let p=0;p<count;p++){
   const b=p*(channels+3);data[b]=p%2?.999:.001;data[b+1]=p%3?.001:.999;data[b+2]=1e-8;
   for(let c=0;c<channels;c++){
    data[b+3+c]=(c%2?-1:1)*(c+1)*1000;
    const w=layout.widths[c],h=layout.heights[c],x=data[b]*w-.5,y=data[b+1]*h-.5,bx=Math.floor(x-.5),by=Math.floor(y-.5);
    const weights=(f:number)=>[.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2];const wx=weights(x-bx),wy=weights(y-by);
    for(let i=0;i<3;i++)for(let j=0;j<3;j++){
     const index=layout.offsets[c]+((by+j+h)%h)*w+(bx+i+w)%w,area=data[b+2]*wx[i]*wy[j];
     expected[index]+=area*data[b+3+c];expected[layout.total+index]+=area;
    }
   }
  }
  const bytes=Math.max(data.length,layout.total*2)*4;
  const scratch=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:layout.total*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  device.queue.writeBuffer(scratch,0,data);const splats=new ChemicalSplats(device,layout,scratch);
  const encoder=device.createCommandEncoder();splats.encode(encoder,count);encoder.copyBufferToBuffer(scratch,0,read,0,layout.total*8);device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
  for(let i=0;i<actual.length;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>1e-10+Math.abs(expected[i])*0.0003)throw Error(`Mismatch resolution=${resolution} index=${i}: ${actual[i]} vs ${expected[i]}`);
  read.unmap();read.destroy();scratch.destroy();splats.destroy();log(`PASS 7 packed channels, wrapped boundaries, resolution ${resolution}`);
 }
 const error=await device.popErrorScope();if(error)throw error;device.destroy();log("ALL CHECKS PASSED");
}
run().catch(e=>log("FAIL: "+e.message));
