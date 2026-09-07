const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const ts = require('typescript');
function load(relative) {
  const source = fs.readFileSync(path.join(__dirname, '..', relative), 'utf8');
  const code = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
  const module = {exports:{}};
  new Function('require','module','exports',code)(() => ({}),module,module.exports);
  return module.exports;
}
const {averageTimings,timingReport} = load('src/ui/timingAverages.ts');
const {applyGeneration,applyHistory,deriveState,EMPTY_ACCUMULATOR} = load('src/net/trainingSocket.ts');
const timing = (seconds, count) => ({seconds,poolSeconds:seconds-1,selectionSeconds:0,previewSeconds:1,checkpointSeconds:0,otherSeconds:0,
  rollouts:{seconds,count,meanSeconds:seconds/count,maxSeconds:seconds,stages:{physics:{seconds,count,maxSeconds:seconds/count}}}});
assert.equal(averageTimings([]),undefined);
const a=timing(10,1), b=timing(30,9);
b.rollouts.stages.setup={seconds:2,count:1,maxSeconds:2};
a.winner={seconds:4,stages:{physics:{seconds:4,count:2,maxSeconds:3}}};
const avg=averageTimings([{generation:0,timing:a},{generation:5,timing:b}]);
assert.equal(avg.seconds,20);
assert.equal(avg.rollouts.stages.physics.seconds,20);
assert.equal(avg.rollouts.stages.physics.count,5);
assert.equal(avg.rollouts.stages.setup.seconds,1); // Missing stage contributes zero for that generation.
assert.equal(avg.rollouts.meanSeconds,4); // Weighted across calls/rollouts, not mean of means.
assert.equal(avg.rollouts.stages.physics.maxSeconds,10);
assert.equal(avg.winner.seconds,4); // Missing winner is excluded, not averaged as zero.
let acc=EMPTY_ACCUMULATOR;
for(let generation=0;generation<600;generation++) acc=applyGeneration(acc,{generation,timing:a});
assert.equal(acc.records.size,500);
assert.equal(deriveState(acc).timingHistory.length,600);
acc=applyGeneration(acc,{generation:599,timing:b});
assert.equal(deriveState(acc).timingHistory.length,600);
let restored=applyHistory(EMPTY_ACCUMULATOR,{generations:[],timings:deriveState(acc).timingHistory});
assert.equal(deriveState(restored).timingHistory.length,600);
assert.equal(deriveState(restored).timingHistory[599].timing.seconds,30);
assert.equal(deriveState(applyHistory(EMPTY_ACCUMULATOR,{generations:[{generation:1}]})).timingHistory.length,0);
console.log('[PASS] Run averages, weighted means, missing data, maxima, deduplication and timing history beyond 500 generations');

a.rollouts.gpu={supported:true,samples:1,seconds:.01,stages:{gpuPhysics:{seconds:.01,count:1,maxSeconds:.01}}};
b.rollouts.gpu={supported:true,samples:3,seconds:.09,stages:{gpuPhysics:{seconds:.09,count:3,maxSeconds:.04}}};
const gpuAverage=averageTimings([{generation:0,timing:a},{generation:1,timing:b}]);
const gpu=timingReport(gpuAverage,'gpu-all');
assert.equal(gpu.gpu.samples,4);
assert.ok(Math.abs(gpu.stages.gpuPhysics.seconds-.025)<1e-12);
assert.equal(gpu.stages.gpuPhysics.maxSeconds,.04);
assert.equal(timingReport(gpuAverage,'gpu-winner'),undefined);
console.log('[PASS] GPU sample-weighted averages, per-step normalization and unavailable GPU history');
