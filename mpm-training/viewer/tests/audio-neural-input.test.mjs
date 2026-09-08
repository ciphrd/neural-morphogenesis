import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
test('audio input changes policy output and legacy weights remain compatible',async()=>{
 const server=await createServer({server:{middlewareMode:true,hmr:false}});
 try {
  const {randomWeights}=await server.ssrLoadModule('/src/gpu/agents.ts');
  const {evalPolicy,policyWeightsShapeError}=await server.ssrLoadModule('/src/gpu/policyEval.ts');
  for(const architecture of ['stateless-128','stateful-64']){
   const weights=randomWeights(9,8,architecture,42);
   const base=33+(architecture==='stateful-64'?8:0);
   assert.equal(weights.fc1w[0].length,base+1);
   weights.fc1w.forEach(row=>row.fill(0));weights.fc1b.fill(0);
   weights.fc2w.forEach(row=>row.fill(0));weights.fc2b.fill(0);
   weights.fc1w[0][base]=1;weights.fc2w[0][0]=1;
   const input=new Float32Array(base+1);
   assert.equal(evalPolicy(input,weights,9,8,1,architecture).envWrite[0],0);
   input[base]=1;
   assert.ok(evalPolicy(input,weights,9,8,1,architecture).envWrite[0]>.7);
   const legacy={...weights,fc1w:weights.fc1w.map(row=>row.slice(0,-1))};
   assert.equal(policyWeightsShapeError(legacy,9,8,architecture),null);
   assert.equal(evalPolicy(input,legacy,9,8,1,architecture).envWrite[0],0);
  }
 }finally{await server.close();}
});
