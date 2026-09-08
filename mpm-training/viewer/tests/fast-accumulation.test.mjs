import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

test("splat specialization writes private records without changing policy chemistry", async () => {
  const server = await createServer({configFile: false, server: {middlewareMode: true}, appType: "custom"});
  try {
    const {chemicalSplatShader, chemicalSplatGroups} = await server.ssrLoadModule("/src/gpu/chemicalSplats.ts");
    const { packChemicalChannelLayout, homogeneousChemicalChannelProfiles } = await server.ssrLoadModule("/src/gpu/chemicalChannels.ts");
    const packed = packChemicalChannelLayout(32, 24, homogeneousChemicalChannelProfiles(7).map((p,i)=>({...p,resolutionScale:i<4?1:.5})));
    assert.deepEqual(chemicalSplatGroups(packed), [[0,1,2],[3],[4,5,6]]);
    const {p2gReductionShader} = await server.ssrLoadModule("/src/gpu/p2gReduction.ts");
    const p2g = await readFile("../core/p2g.wgsl", "utf8");
    const reduced = p2gReductionShader(p2g);
    assert.ok(reduced.includes("fn addGlobalFloat("));
    assert.ok(reduced.includes("workgroup_p2g(gid);\n  workgroupBarrier();"));
    assert.ok(!reduced.includes("round("));
    const agents = await readFile("../core/agents.wgsl", "utf8");
    const splats = chemicalSplatShader(agents);
    assert.ok(splats.includes("depositMaterialSample(pi, levels,"));
    assert.ok(splats.includes("depositMaterialSample(pi, result.envWrite,"));
    assert.ok(!splats.includes("atomicCompareExchangeWeak(&depositScratch"));
    assert.ok(splats.includes("depositScratch: array<f32>"));
    const policy = agents.slice(agents.indexOf("struct PolicyOutput"));
    assert.equal(splats.slice(splats.indexOf("struct PolicyOutput")), policy.replace("depositMaterialSample(result.envWrite,", "depositMaterialSample(pi, result.envWrite,"));
    const core = await readFile("src/gpu/mpmCore.ts", "utf8");
    assert.ok(!core.includes("fastAccumulationShader"));
  } finally { await server.close(); }
});
