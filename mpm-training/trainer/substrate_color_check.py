"""GPU checks for spatial RGB sampling and the optional neural head."""
import numpy as np

from config import CONFIG
from agents_gpu import AgentsGPU, weight_layout, PARTICLE_META_BUFFER_OFFSET
from device import pick_device
from environment_gpu import EnvironmentGPU
from mpm_core import MpmCore
from training_sim import TrainingRollout
from policy_parameters import policy_heads


def check_browser_layouts():
    import json
    import subprocess
    from pathlib import Path
    script = r"""
const fs = require('fs'), path = require('path'), Module = require('module');
const ts = require('./viewer/node_modules/typescript');
const oldLoad = Module._load;
Module._load = function(request, parent, ...args) {
  if (request.endsWith('?raw')) return fs.readFileSync(path.resolve(path.dirname(parent.filename), request.slice(0, -4)), 'utf8');
  return oldLoad.call(this, request, parent, ...args);
};
require.extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
    esModuleInterop: true, resolveJsonModule: true,
  }}).outputText, filename);
};
const config = require('./core/config.json');
const {randomWeights} = require('./viewer/src/gpu/agents.ts');
const {evalPolicy, policyWeightsShapeError} = require('./viewer/src/gpu/policyEval.ts');
const result = [];
const channels = config.chemistry.channels.length;
for (const source of ['substrate', 'neural']) {
  config.coloring.source = source;
  for (const [architecture, hidden] of [['stateless-128',128], ['stateful-64',64], ['stateful-128',128]]) {
    const weights = randomWeights(channels, hidden, architecture, 17);
    if (policyWeightsShapeError(weights,channels,hidden,architecture)) throw Error('layout mismatch');
    const output = evalPolicy(new Float32Array(weights.fc1w[0].length),weights,channels,hidden,1,architecture);
    if ((output.color === null) !== (source === 'substrate')) throw Error('incorrect color source');
    result.push([source,architecture,hidden,weights.fc2b.length]);
  }
}
process.stdout.write(JSON.stringify(result));
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['node', '-e', script], cwd=root, capture_output=True, text=True, check=True)
    original = CONFIG['coloring']['source']
    try:
        for source, architecture, hidden, outputs in json.loads(result.stdout):
            CONFIG['coloring']['source'] = source
            assert outputs == weight_layout(len(CONFIG['chemistry']['channels']), hidden, architecture)['out_dim']
    finally:
        CONFIG['coloring']['source'] = original
    print('[PASS] Browser random weights and CPU evaluator match Python layouts in both color modes')


def main():
    check_browser_layouts()
    device = pick_device()
    channels = len(CONFIG['chemistry']['channels'])
    original = CONFIG['coloring']['source']
    try:
        for source in ('substrate', 'neural'):
            CONFIG['coloring']['source'] = source
            for architecture, hidden in (('stateless-128', 128), ('stateful-64', 64), ('stateful-128', 128)):
                for communication in ('persistent-environment', 'cell-owned-projection'):
                    core = MpmCore(device)
                    env = EnvironmentGPU(device, channels, 16, 16, 1., 1., chemical_communication_architecture=communication)
                    agents = AgentsGPU(device, core, env, channels, hidden, 1., 16, .01, 1., 1., .5, .5,
                                       policy_architecture=architecture, chemical_communication_architecture=communication)
                    TrainingRollout(core, agents, env, (.5, .5), .01, 0., 17, initial_particle_count=1)
                    layout = weight_layout(len(CONFIG['chemistry']['channels']), hidden, architecture)
                    assert ('color' in [h.name for h in policy_heads(channels, architecture)]) == (source == 'neural')
                    weights = np.zeros(layout['total_floats'], np.float32)
                    # Large chemical writes must not affect this tick's color.
                    weights[layout['fc2b_offset']:layout['fc2b_offset'] + channels] = 10.
                    neural_logits = np.array([-2., 0., 2.], np.float32)
                    if source == 'neural':
                        weights[-3:] = neural_logits
                    agents.load_weights(weights)
                    for x in (.25, .375, .75):
                        device.queue.write_buffer(core.positions, 0, np.array([x, .5], np.float32))
                        field = np.zeros(env.total_values, np.float32)
                        raw_rgb = []
                        for rgb, channel in enumerate(CONFIG['coloring']['channels']):
                            width, height = env.channel_widths[channel], env.channel_heights[channel]
                            offset = env.channel_offsets[channel]
                            # Cell-centered ramp: interpolation at x yields (4*x - 2) + rgb.
                            plane = np.tile(4 * (np.arange(width) + .5) / width - 2 + rgb, (height, 1))
                            field[offset:offset + width * height] = plane.ravel()
                            raw_rgb.append(4 * x - 2 + rgb)
                        heading_channel = CONFIG['simulation']['HEADING_CHANNEL_INDEX']
                        assert heading_channel not in CONFIG['coloring']['channels']
                        width, height = env.channel_widths[heading_channel], env.channel_heights[heading_channel]
                        offset = env.channel_offsets[heading_channel]
                        # RGB varies horizontally; the relocated heading varies vertically.
                        field[offset:offset + width * height] = np.repeat(
                            4 * (np.arange(height) + .5) / height - 2, width)
                        device.queue.write_buffer(env.buffers[env.parity], 0, field)
                        encoder = device.create_command_encoder()
                        if communication == 'persistent-environment':
                            env.encode_sense(encoder)
                        agents.encode_step(encoder, env.parity, commit_growth=False)
                        device.queue.submit([encoder.finish()])
                        expected = (1 / (1 + np.exp(-np.clip(neural_logits, -20, 20)))
                                    if source == 'neural' else np.array(raw_rgb))
                        np.testing.assert_allclose(agents.read_colors(1)[0], expected, atol=2e-6)
                        if communication == 'persistent-environment':
                            raw = device.queue.read_buffer(agents._agent_state_buffer,
                                PARTICLE_META_BUFFER_OFFSET, agents._particle_meta_dtype.itemsize)
                            alignment = np.frombuffer(raw, agents._particle_meta_dtype)['alignment'][0]
                            assert abs(alignment[0]) < 1e-6 and alignment[1] > 0, alignment
            print(f'[PASS] {source}: RGB values, independent heading, movement, pre-update sampling, both chemistry modes and all policy architectures')
    finally:
        CONFIG['coloring']['source'] = original


if __name__ == '__main__':
    main()
