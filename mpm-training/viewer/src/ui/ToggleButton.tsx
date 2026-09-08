interface ToggleButtonProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  hideLabel?: boolean
  className?: string
  title?: string
}

export function ToggleButton({
  label, checked, onChange, disabled = false, hideLabel = false, className = "", title,
}: ToggleButtonProps) {
  return (
    <label className={`toggle-control ${className}`} title={title}>
      {!hideLabel && <span>{label}</span>}
      <button
        type="button"
        className="toggle-button"
        aria-label={label}
        aria-pressed={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
      >
        {checked ? "ON" : "OFF"}
      </button>
    </label>
  )
}
