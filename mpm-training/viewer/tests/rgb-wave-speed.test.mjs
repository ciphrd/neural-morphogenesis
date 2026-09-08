import test from 'node:test'
import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import ts from 'typescript'
const source=readFileSync(new URL('../src/gpu/postEffects.ts',import.meta.url),'utf8')
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText
const {normalizePostEffects,advanceRgbPhases,advanceStrobePhases}=await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const close=(a,b)=>a.forEach((value,i)=>assert.ok(Math.abs(value-b[i])<1e-10))
test('independent speeds pause, advance, and reverse each channel',()=>{
 const settings=normalizePostEffects({redPeriod:4,greenPeriod:4,bluePeriod:4,redSpeed:0,greenSpeed:2,blueSpeed:-1})
 close(advanceRgbPhases([.1,.1,.1],1,settings),[.1,.6,.85])
})
test('phase stays continuous through speed changes and frame subdivisions',()=>{
 const settings=normalizePostEffects({})
 const first=advanceRgbPhases([0,.3,.6],.1,settings)
 close(advanceRgbPhases(first,.1,settings),advanceRgbPhases([0,.3,.6],.2,settings))
 const paused=normalizePostEffects({redSpeed:0,greenSpeed:0,blueSpeed:0})
 close(advanceRgbPhases(first,100,paused),first)
 close(advanceRgbPhases(first,0,normalizePostEffects({redSpeed:-4})),first)
})
test('older scenes default to normal speed and invalid values stay safe',()=>{
 const defaults=normalizePostEffects({redSpeed:NaN,greenSpeed:99999,blueSpeed:-99999})
 assert.equal(defaults.redSpeed,1);assert.equal(defaults.greenSpeed,1000);assert.equal(defaults.blueSpeed,-1000)
})

test('strobe defaults give synchronized 8 Hz sharp pulses and independent rates',()=>{
 const settings=normalizePostEffects({})
 assert.equal(settings.strobeEnabled,false)
 close(advanceStrobePhases([0,0,0],1/32,settings),[.25,.25,.25])
 assert.equal(settings.strobeRedExponent,16)
 close(advanceStrobePhases([0,0,0],1/32,{...settings,strobeRedSpeed:0,strobeGreenSpeed:2,strobeBlueSpeed:-1}),[0,.5,.75])
})
