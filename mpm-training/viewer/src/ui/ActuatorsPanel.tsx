import { useEffect, useState } from "react"
import {
  ATTRACTOR_CONTROLS,
  AttractorMotion,
  type AttractorPosition,
  type AttractorSettings,
  normalizeAttractor,
} from "../performance/actuators"
import {
  PERFORMANCE_CHANNEL_NAME,
  type ProjectionToControllerMessage,
} from "../performance/types"
import { Slider } from "./Slider"
import { ToggleButton } from "./ToggleButton"

export function ActuatorsPanel({
  value,
  effective,
  onChange,
}: {
  value: AttractorSettings
  effective: AttractorSettings
  onChange: (value: AttractorSettings) => void
}) {
  const [open, setOpen] = useState(true)
  const [live, setLive] = useState<{
    position: AttractorPosition
    at: number
  } | null>(null)
  useEffect(() => {
    const channel = new BroadcastChannel(PERFORMANCE_CHANNEL_NAME)
    channel.onmessage = (
      event: MessageEvent<ProjectionToControllerMessage>
    ) => {
      if (event.data.type === "attractor")
        setLive({ position: event.data.position, at: Date.now() })
    }
    const timer = window.setInterval(
      () =>
        setLive((current) =>
          current && Date.now() - current.at > 2000 ? null : current
        ),
      1000
    )
    return () => {
      channel.close()
      window.clearInterval(timer)
    }
  }, [])
  const settings = normalizeAttractor(effective)
  const position = live?.position ?? new AttractorMotion().advance(settings, 0)
  const color = settings.strength < 0 ? "#fda4af" : "#7dd3fc"
  return (
    <section>
      <div className="physics-panel-header">
        <button
          className="physics-panel-toggle"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>
            ▸
          </span>
          <h2>Actuators</h2>
        </button>
        {open && (
          <div className="attractor-preview" title={`${live ? "Live output" : "Open output for live position"} · ${settings.enabled ? settings.strength < 0 ? "Repel" : "Attract" : "Off"}`}>
            <svg
              viewBox="0 0 200 200"
              role="img"
              aria-label={`Attractor position ${position.x.toFixed(2)}, ${position.y.toFixed(2)}; range ${settings.range.toFixed(2)}`}
            >
              <path d="M100 0V200M0 100H200" stroke="#252525" />
              <g opacity={settings.enabled ? 1 : 0.35}>
                {[-1, 0, 1].flatMap((dx) =>
                  [-1, 0, 1].map((dy) => (
                    <circle
                      key={`${dx},${dy}`}
                      cx={(position.x + dx) * 200}
                      cy={(1 - position.y + dy) * 200}
                      r={settings.range * 200}
                      fill={color}
                      fillOpacity="0.08"
                      stroke={color}
                      strokeOpacity="0.5"
                    />
                  ))
                )}
                <circle
                  cx={position.x * 200}
                  cy={(1 - position.y) * 200}
                  r="3"
                  fill={color}
                />
              </g>
            </svg>
          </div>
        )}
      </div>
      {open && (
        <div className="physics-panel-body">
          <ToggleButton
            label="Attractor"
            checked={value.enabled}
            onChange={(enabled) => onChange({ ...value, enabled })}
          />
          {ATTRACTOR_CONTROLS.map(({ key, label, min, max, step }) => (
            <label data-audio-target={`attractor.${key}`} className="slider-row" key={key}>
              <span>{label}</span>
              <Slider
                value={value[key]}
                min={min}
                max={max}
                step={step}
                onChange={(number) => onChange({ ...value, [key]: number })}
              />
              <span className="slider-value">
                {settings[key].toFixed(key === "phase" ? 0 : 3)}
              </span>
            </label>
          ))}
        </div>
      )}
    </section>
  )
}
