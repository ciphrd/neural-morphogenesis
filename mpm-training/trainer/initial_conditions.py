"""Seeded, physical-space perturbations, mirrored in gpu/initialConditions.ts."""
import json
from pathlib import Path
import numpy as np
from agents_gpu import _spawn_uniform01

DEFAULTS = json.loads((Path(__file__).resolve().parent.parent / 'core/initial_conditions.json').read_text())
PRESETS = tuple(p['id'] for p in DEFAULTS['presets'])


def validate_initial_condition(preset, strength, channel, channels, recurrent):
    if preset not in PRESETS:
        raise ValueError(f'Unknown initial condition: {preset}')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Initial-condition strength must be between 0 and 1')
    if not 0 <= channel < channels:
        raise ValueError('Initial chemical channel must exist in this policy')
    if preset == 'internal-state' and not recurrent:
        raise ValueError('Internal-state initial condition requires recurrent cell memory')
    if preset == 'handed-chemistry' and channels < 2:
        raise ValueError('Handed chemistry requires at least two chemical channels')


class InitialCondition:
    def __init__(self, preset, strength, channel, seed, center, radius):
        self.preset, self.strength, self.channel = preset, strength, channel
        self.center, self.radius = np.asarray(center), max(radius, 1e-8)
        angle = 2*np.pi*_spawn_uniform01(seed, 100)
        self.c, self.s = np.cos(angle), np.sin(angle)
        self.phases = [2*np.pi*_spawn_uniform01(seed, 101+i) for i in range(4)]

    def coordinates(self, positions):
        d = (np.asarray(positions, dtype=float)-self.center+.5)%1-.5
        return (d[..., 0]*self.c+d[..., 1]*self.s)/self.radius, (-d[..., 0]*self.s+d[..., 1]*self.c)/self.radius

    def signal(self, positions, secondary=False):
        u, v = self.coordinates(positions)
        bump = np.exp(-((u-(0 if secondary else .55))**2+(v-(.55 if secondary else 0))**2)/(2*.45**2))
        if self.preset == 'chemical-gradient':
            bump = .5*(1+np.tanh(1.5*u))*np.exp(-.5*(u*u+v*v)**2)
        elif self.preset == 'chemical-noise':
            p = self.phases
            bump = .25*(np.sin(2*u+p[0])+np.sin(2*v+p[1])+np.sin(3*u+2*v+p[2])+np.sin(u-3*v+p[3]))*np.exp(-.5*(u*u+v*v)**2)
        return self.strength*bump

    def deform(self, scene):
        if self.preset == 'geometric-bias':
            a = np.exp(.5*self.strength)
            matrix = np.array([[self.c*self.c*a+self.s*self.s/a, self.c*self.s*(a-1/a)],
                               [self.c*self.s*(a-1/a), self.s*self.s*a+self.c*self.c/a]])
            for values in (scene[0], scene[5].reshape(-1, 2)):
                d = (values.astype(float)-self.center+.5)%1-.5
                values[:] = (d@matrix.T+self.center)%1
        elif self.preset == 'mechanical-bias':
            e = .15*self.signal(scene[0])
            a, b = np.exp(e), np.exp(-e)
            scene[2][:, 0] = self.c*self.c*a+self.s*self.s*b
            scene[2][:, 1] = scene[2][:, 2] = self.c*self.s*(a-b)
            scene[2][:, 3] = self.s*self.s*a+self.c*self.c*b

    def states(self, positions, channels):
        chemistry = np.zeros((len(positions), channels), np.float32)
        private = np.zeros((len(positions), 8), np.float32)
        if self.preset.startswith('chemical-') or self.preset == 'handed-chemistry':
            chemistry[:, self.channel] = self.signal(positions)
            if self.preset == 'handed-chemistry':
                chemistry[:, (self.channel+1)%channels] = self.signal(positions, True)
        elif self.preset == 'internal-state':
            private[:, 0] = self.signal(positions)
        return chemistry, private

    def seed_environment(self, environment):
        if not (self.preset.startswith('chemical-') or self.preset == 'handed-chemistry'):
            return
        field = np.zeros(environment.total_values, np.float32)
        indices = [self.channel]
        if self.preset == 'handed-chemistry':
            indices.append((self.channel+1)%environment.channels)
        for k, channel in enumerate(indices):
            w, h = environment.channel_widths[channel], environment.channel_heights[channel]
            y, x = np.mgrid[:h, :w]
            values = self.signal(np.stack(((x+.5)/w, (y+.5)/h), axis=-1), k == 1).ravel()
            offset = environment.channel_offsets[channel]
            field[offset:offset+w*h] = values
        for buffer in environment.buffers:
            environment.device.queue.write_buffer(buffer, 0, field)
