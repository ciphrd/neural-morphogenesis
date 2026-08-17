import type { SpawnDistribution } from "../gpu/rng";

interface Option {
  value: SpawnDistribution;
  label: string;
  hint: string;
  icon: JSX.Element;
}

// Three small hand-drawn SVG icons (20x20, currentColor strokes/fills so
// they inherit the button's own text color / active-state tint) rather
// than pulling in an icon library for three glyphs — each one is meant
// to read as a miniature of what the distribution actually looks like:
// a lone dot for the tight default jitter, scattered dots across the
// whole square for "square", scattered dots confined to a smaller
// centered square for "quarterSquare".
const OPTIONS: Option[] = [
  {
    value: "default",
    label: "Default",
    hint: "Tight jitter at the grid center — matches what training itself uses.",
    icon: (
      <svg viewBox="0 0 20 20" width="18" height="18" fill="none">
        <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.3" opacity="0.4" />
        <circle cx="10" cy="10" r="1.8" fill="currentColor" />
      </svg>
    ),
  },
  {
    value: "square",
    label: "Square",
    hint: "Uniformly scattered across the entire grid.",
    icon: (
      <svg viewBox="0 0 20 20" width="18" height="18" fill="none">
        <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.3" opacity="0.4" />
        <circle cx="5.5" cy="6" r="1.1" fill="currentColor" />
        <circle cx="14" cy="4.5" r="1.1" fill="currentColor" />
        <circle cx="9.5" cy="10" r="1.1" fill="currentColor" />
        <circle cx="15.5" cy="14.5" r="1.1" fill="currentColor" />
        <circle cx="4.5" cy="15" r="1.1" fill="currentColor" />
        <circle cx="12" cy="16" r="1.1" fill="currentColor" />
        <circle cx="7" cy="13" r="1.1" fill="currentColor" />
      </svg>
    ),
  },
  {
    value: "quarterSquare",
    label: "Smaller square",
    hint: "Uniformly scattered inside a centered square a quarter of the grid's area.",
    icon: (
      <svg viewBox="0 0 20 20" width="18" height="18" fill="none">
        <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.3" opacity="0.25" />
        <rect x="6" y="6" width="8" height="8" rx="1" stroke="currentColor" strokeWidth="1.3" opacity="0.7" />
        <circle cx="8" cy="8.3" r="1" fill="currentColor" />
        <circle cx="12.3" cy="7.3" r="1" fill="currentColor" />
        <circle cx="9" cy="12" r="1" fill="currentColor" />
        <circle cx="12.7" cy="11.3" r="1" fill="currentColor" />
      </svg>
    ),
  },
];

interface SpawnDistributionToggleProps {
  value: SpawnDistribution;
  onChange: (value: SpawnDistribution) => void;
}

/** Icon-button segmented control for GridCanvas's viewer-only initial
 * agent spread (see gpu/rng.ts's SpawnDistribution docstring) — doesn't
 * affect training, just how a replayed rollout scatters agents at step
 * 0, for visually probing how a trained update rule behaves starting
 * from a spread it was never actually trained on. */
export function SpawnDistributionToggle({ value, onChange }: SpawnDistributionToggleProps) {
  return (
    <div className="icon-toggle-group" role="radiogroup" aria-label="Initial spawn distribution">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          title={`${option.label} — ${option.hint}`}
          className={`icon-toggle-button${value === option.value ? " is-active" : ""}`}
          onClick={() => onChange(option.value)}
        >
          {option.icon}
        </button>
      ))}
    </div>
  );
}
