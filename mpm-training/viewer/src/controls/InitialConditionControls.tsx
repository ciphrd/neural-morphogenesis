import defaults from '../../../core/initial_conditions.json'
import { initialConditionPresets, type InitialConditionPreset } from '../gpu/initialConditions'
import type { RunSettings } from '../gpu/types'

export type InitialConditionSettings = Pick<RunSettings, 'initialCondition' | 'initialConditionStrength' | 'initialConditionChannel'>
export function InitialConditionControls({ config, recurrent, onChange }: {
  config: RunSettings | null; recurrent: boolean;
  onChange: (settings: InitialConditionSettings) => void;
}) {
  const preset = config?.initialCondition ?? 'none'
  const strength = config?.initialConditionStrength ?? defaults.defaultStrength
  const channel = config?.initialConditionChannel ?? defaults.defaultChannel
  const change = (patch: InitialConditionSettings) => onChange({ initialCondition: preset,
    initialConditionStrength: strength, initialConditionChannel: channel, ...patch })
  const chemical = preset.startsWith('chemical-') || preset === 'handed-chemistry'
  return <>
    <div className="stat-row">
      <span>Initial condition</span>
      <select className="select" aria-label="Initial condition" value={preset} disabled={!config}
        title="One-time asymmetry. Changing this restarts playback; training uses its recorded preset."
        onChange={e => change({ initialCondition: e.target.value as InitialConditionPreset })}>
        {initialConditionPresets.map(p => <option key={p.id} value={p.id}
          disabled={(p.id === 'internal-state' && !recurrent) || (p.id === 'handed-chemistry' && (config?.channels ?? 0) < 2)}>
          {p.label}{p.id === 'internal-state' && !recurrent ? ' (requires recurrent memory)' : ''}
        </option>)}
      </select>
    </div>
    {preset !== 'none' && <div className="stat-row">
      <span>Asymmetry strength</span>
      <input aria-label="Asymmetry strength" type="range" min="0" max="1" step="0.05"
        value={strength} onChange={e => change({ initialConditionStrength: Number(e.target.value) })}/>
      <span>{strength.toFixed(2)}</span>
    </div>}
    {chemical && <div className="stat-row">
      <span>Initial chemical channel</span>
      <select className="select" aria-label="Initial chemical channel" value={channel}
        onChange={e => change({ initialConditionChannel: Number(e.target.value) })}>
        {Array.from({ length: config?.channels ?? 0 }, (_, i) => <option key={i} value={i}>
          {i}{i === 3 ? ' (orientation)' : ''}{preset === 'handed-chemistry' ? ` + ${(i+1)%(config?.channels ?? 1)}` : ''}
        </option>)}
      </select>
    </div>}
  </>
}
