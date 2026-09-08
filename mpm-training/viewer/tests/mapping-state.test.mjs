import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
const source=readFileSync(new URL('../src/audio/mappingState.ts',import.meta.url),'utf8')
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText
const load=async suffix=>import(`data:text/javascript;base64,${Buffer.from(compiled+'\n// '+suffix).toString('base64')}`)
const {appendAudioMapping,uniqueMappingIds}=await load('initial')
const targets=[{key:'physics.communicationSpeed',min:1,max:10},{key:'physics.gravity',min:0,max:20}]
test('adding after module reload preserves the first mapping and uses a different ID',async()=>{
 const first=appendAudioMapping([],targets)
 const reloaded=await load('hot reload')
 const next=reloaded.appendAudioMapping(first,targets)
 assert.deepEqual(next[0],first[0])
 assert.notEqual(next[0].id,next[1].id)
 const edited=next.map(row=>row.id===next[1].id?{...row,min:5}:row)
 assert.deepEqual(edited[0],first[0])
 assert.equal(edited[1].min,5)
})
test('replayed and batched additions use current state for IDs and targets',()=>{
 const first=appendAudioMapping([],targets)
 assert.deepEqual(appendAudioMapping(first,targets),appendAudioMapping(first,targets))
 const second=appendAudioMapping(first,targets)
 assert.notEqual(second[0].target,second[1].target)
 const next=appendAudioMapping(second.slice(1),targets)
 assert.notEqual(next[0].id,next[1].id)
 assert.equal(next[1].target,targets[0].key)
})
test('existing duplicate IDs are repaired without changing row settings',()=>{
 const rows=[{id:1,target:targets[0].key,min:1,max:10},{id:1,target:targets[1].key,min:2,max:8},{id:2,target:targets[0].key,min:3,max:7}]
 const result=uniqueMappingIds(rows)
 assert.equal(new Set(result.map(row=>row.id)).size,3)
 assert.deepEqual(result.map(({id,...settings})=>settings),rows.map(({id,...settings})=>settings))
 assert.equal(uniqueMappingIds(result),result)
})
