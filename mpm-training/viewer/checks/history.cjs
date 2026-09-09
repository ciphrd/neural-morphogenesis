const fs = require('fs');
const Module = require('module');
const ts = require(process.cwd()+'/viewer/node_modules/typescript');
const assert = require('assert/strict');
const originalLoad = Module._load;
Module._load = function(request, parent, isMain) {
  if(request === '../gpu/agents') return {randomWeights:()=>({})};
  return originalLoad.call(this,request,parent,isMain);
};
require.extensions['.ts'] = (module, filename) => {
  const output=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,esModuleInterop:true,resolveJsonModule:true}}).outputText;
  module._compile(output,filename);
};
const {applyHistory,applyGeneration,deriveState,EMPTY_ACCUMULATOR}=require(process.cwd()+'/viewer/src/net/trainingSocket.ts');
const summaries=Array.from({length:1100},(_,i)=>({generation:i,best:1,mean:2,worst:3,allTimeBest:1,seed:i,optimizerState:{sigma:0.1}}));
const latest={...summaries.at(-1),weights:{fc1w:[42]}};
const start=performance.now();
let acc=applyHistory({...EMPTY_ACCUMULATOR,settings:{channels:12}}, {generations:summaries,latestGeneration:latest});
let state=deriveState(acc);
assert.equal(state.history.length,1100);
assert.equal(state.configByGeneration.size,1);
assert.equal(state.latest.weights.fc1w[0],42);
for(let i=1100;i<1120;i++) acc=applyGeneration(acc,{...latest,generation:i});
// Late HTTP backfill must not erase newer websocket records.
acc=applyHistory(acc,{generations:summaries,latestGeneration:latest});
state=deriveState(acc);
assert.equal(state.history.length,1120);
assert.equal(state.configByGeneration.size,8);
assert.equal(state.latest.generation,1119);
assert(state.history.every(r=>r.optimizerState.sigma===.1));
// Legacy full history is also accepted, but large policies are bounded.
const legacy=deriveState(applyHistory({...EMPTY_ACCUMULATOR,settings:{}},{generations:summaries.map(r=>({...r,weights:{fc1w:[1]}}))}));
assert.equal(legacy.history.length,1100);assert.equal(legacy.configByGeneration.size,8);
console.log('[PASS] compact/legacy backfill, all chart generations, eight-policy cache, websocket/backfill race; '+(performance.now()-start).toFixed(1)+' ms');
