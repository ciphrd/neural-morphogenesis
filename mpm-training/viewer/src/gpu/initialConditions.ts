import defaultsConfig from '../../../core/config.json';
const defaults = defaultsConfig.initialConditions;
import { spawnUniform01 } from './rng';
import type { SceneData } from './types';
import type { Environment } from './environment';

export const initialConditionPresets = defaults.presets;
export type InitialConditionPreset = 'none' | 'chemical-pole' | 'chemical-gradient' | 'chemical-noise' | 'geometric-bias' | 'internal-state' | 'mechanical-bias' | 'handed-chemistry';

/** One-time, seed-relative cues. Mirrors trainer/initial_conditions.py. */
export class InitialCondition {
  private c: number;
  private s: number;
  private phases: number[];
  constructor(readonly preset: InitialConditionPreset, readonly strength: number,
    readonly channel: number, seed: number, private center: readonly number[], private radius: number) {
    if (!initialConditionPresets.some(p => p.id === preset)) throw new Error(`Unknown initial condition: ${preset}`);
    if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error('Initial-condition strength must be between 0 and 1');
    const angle = 2*Math.PI*spawnUniform01(seed, 100);
    this.c = Math.cos(angle); this.s = Math.sin(angle);
    this.phases = Array.from({length: 4}, (_, i) => 2*Math.PI*spawnUniform01(seed, 101+i));
    this.radius = Math.max(radius, 1e-8);
  }
  signal(x: number, y: number, secondary = false): number {
    const dx = delta(x-this.center[0]), dy = delta(y-this.center[1]);
    const u = (dx*this.c+dy*this.s)/this.radius, v = (-dx*this.s+dy*this.c)/this.radius;
    let bump = Math.exp(-((u-(secondary ? 0 : .55))**2+(v-(secondary ? .55 : 0))**2)/(2*.45**2));
    if (this.preset === 'chemical-gradient') bump = .5*(1+Math.tanh(1.5*u))*Math.exp(-.5*(u*u+v*v)**2);
    else if (this.preset === 'chemical-noise') {
      const p = this.phases;
      bump = .25*(Math.sin(2*u+p[0])+Math.sin(2*v+p[1])+Math.sin(3*u+2*v+p[2])+Math.sin(u-3*v+p[3]))*Math.exp(-.5*(u*u+v*v)**2);
    }
    return this.strength*bump;
  }
  deform(scene: SceneData): void {
    const c = this.c, s = this.s;
    if (this.preset === 'geometric-bias') {
      const a = Math.exp(.5*this.strength), xx = c*c*a+s*s/a, xy = c*s*(a-1/a), yy = s*s*a+c*c/a;
      for (const values of [scene.positions, scene.domain]) {
        if (!values) continue;
        for (let i = 0; i < values.length; i += 2) {
          const x = delta(values[i]-this.center[0]), y = delta(values[i+1]-this.center[1]);
          values[i] = wrap(xx*x+xy*y+this.center[0]); values[i+1] = wrap(xy*x+yy*y+this.center[1]);
        }
      }
    } else if (this.preset === 'mechanical-bias') {
      for (let i = 0; i < scene.count; i++) {
        const e = .15*this.signal(scene.positions[2*i], scene.positions[2*i+1]);
        const a = Math.exp(e), b = Math.exp(-e);
        scene.F.set([c*c*a+s*s*b, c*s*(a-b), c*s*(a-b), s*s*a+c*c*b], i*4);
      }
    }
  }
  states(positions: Float32Array, channels: number): { chemistry: Float32Array; privateState: Float32Array } {
    if (!Number.isInteger(this.channel) || this.channel < 0 || this.channel >= channels) throw new Error('Initial chemical channel must exist in this policy');
    if (this.preset === 'handed-chemistry' && channels < 2) throw new Error('Handed chemistry requires two channels');
    const n = positions.length/2, chemistry = new Float32Array(n*channels), privateState = new Float32Array(n*8);
    for (let i = 0; i < n; i++) {
      const value = this.signal(positions[2*i], positions[2*i+1]);
      if (this.isChemical) {
        chemistry[i*channels+this.channel] = value;
        if (this.preset === 'handed-chemistry') chemistry[i*channels+(this.channel+1)%channels] = this.signal(positions[2*i], positions[2*i+1], true);
      } else if (this.preset === 'internal-state') privateState[i*8] = value;
    }
    return { chemistry, privateState };
  }
  private get isChemical(): boolean { return this.preset.startsWith('chemical-') || this.preset === 'handed-chemistry'; }
  seedEnvironment(device: GPUDevice, environment: Environment): void {
    if (!this.isChemical) return;
    const field = new Float32Array(environment.layout.total);
    const indices = [this.channel];
    if (this.preset === 'handed-chemistry') indices.push((this.channel+1)%environment.channels);
    indices.forEach((channel, k) => {
      const w = environment.layout.widths[channel], h = environment.layout.heights[channel], offset = environment.layout.offsets[channel];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) field[offset+y*w+x] = this.signal((x+.5)/w, (y+.5)/h, k === 1);
    });
    for (const buffer of environment.buffers) device.queue.writeBuffer(buffer, 0, field);
  }
}
const delta = (v: number) => v-Math.floor(v+.5);
const wrap = (v: number) => v-Math.floor(v);
