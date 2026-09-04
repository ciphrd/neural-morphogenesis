import type { DevelopmentalSettings } from "../gpu/developmentalFields";
import { Slider } from "./Slider";

interface Props {
  value: DevelopmentalSettings;
  onChange: (value: DevelopmentalSettings) => void;
  onReseed: () => void;
}

export function DevelopmentalFieldsPanel({ value, onChange, onReseed }: Props) {
  const set = <K extends keyof DevelopmentalSettings>(key: K, next: DevelopmentalSettings[K]) => {
    onChange({ ...value, [key]: next });
  };
  const slider = (
    key: Exclude<keyof DevelopmentalSettings, "enabled">,
    label: string, min: number, max: number, step: number, digits = 3,
  ) => (
    <label className="slider-row">
      <span>{label}</span>
      <Slider min={min} max={max} step={step} value={value[key]} onChange={(next) => set(key, next)} />
      <span className="slider-value">{value[key].toFixed(digits)}</span>
    </label>
  );

  return (
    <section>
      <h2>Developmental fields</h2>
      <label className="checkbox-row">
        <input type="checkbox" checked={value.enabled} onChange={(event) => set("enabled", event.target.checked)} />
        Enable experimental AP organizers
      </label>
      <button className="button secondary" type="button" onClick={onReseed} disabled={!value.enabled}>
        Reseed poles
      </button>
      <p className="hint">Separate from the nine policy chemicals. The founder heading seeds anterior and posterior fields; the NN cannot read or write them.</p>
      {slider("timeScale", "Development speed", 0, 4, 0.05, 2)}
      {slider("seedOffset", "Pole offset", 0.002, 0.08, 0.001)}
      {slider("seedSigma", "Seed radius", 0.002, 0.04, 0.001)}
      {slider("activatorDiffusion", "Activator diffusion", 0, 0.0002, 0.000002, 6)}
      {slider("inhibitorDiffusion", "Inhibitor diffusion", 0, 0.001, 0.00001, 6)}
      <details className="settings-category">
        <summary>Reaction parameters</summary>
        {slider("sourceProduction", "Organizer production", 0, 6, 0.05, 2)}
        {slider("activatorDecay", "Morphogen decay", 0.005, 0.5, 0.005, 3)}
        {slider("inhibitorProduction", "Inhibitor production", 0, 4, 0.05, 2)}
        {slider("inhibitorDecay", "Inhibitor decay", 0, 1, 0.01, 2)}
        {slider("inhibitorSuppression", "Inhibitor suppression", 0, 8, 0.1, 2)}
        {slider("occupancyHalfSaturation", "Occupancy half-saturation", 0.01, 1, 0.01, 2)}
        {slider("occupancyHillExponent", "Occupancy exponent", 1, 8, 0.25, 2)}
      </details>
    </section>
  );
}
