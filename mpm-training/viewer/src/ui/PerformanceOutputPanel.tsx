import type { Dispatch, SetStateAction } from "react"
import type { PerformanceControls } from "../performance/workspacePresets"
import { OutputPreview } from "./OutputPreview"
import type { ProjectionTelemetry } from "../performance/types"

export interface PerformanceOutputStatus {
  connected: boolean
  telemetry: ProjectionTelemetry | null
}

export function PerformanceOutputPanel({ connected, telemetry, controls, onControlsChange }: PerformanceOutputStatus & { controls: PerformanceControls; onControlsChange: Dispatch<SetStateAction<PerformanceControls>> }) {
  const live = connected && telemetry !== null
  const openOutput = () => {
    const url = new URL(window.location.href)
    url.searchParams.set("output", "1")
    url.hash = ""
    window.open(url, "mpm-projection", "popup,width=1280,height=720")?.focus()
  }

  return (
    <section className="performance-dashboard-card">
      <div className="physics-panel-header">
        <h2>Output</h2>
        <div className="performance-output-actions">
          <span role="status" className={"performance-connection" + (connected ? " is-connected" : "")}>
            {connected ? "Live" : "Offline"}
          </span>
          <button type="button" className="select performance-open-output" onClick={openOutput} aria-label="Open output">Open</button>
        </div>
      </div>
      <dl className="performance-current-state">
        <div><dt>FPS</dt><dd>{live ? telemetry.fps.toFixed(0) : "—"}</dd></div>
        <div><dt>Step</dt><dd>{live ? telemetry.step.toLocaleString() : "—"}</dd></div>
        <div><dt>Cells</dt><dd>{live ? telemetry.particleCount.toLocaleString() : "—"}</dd></div>
      </dl>
      <OutputPreview connected={connected} enabled={controls.previewEnabled ?? true}
        setEnabled={previewEnabled => onControlsChange(current => ({ ...current, previewEnabled }))}
        cutEnabled={controls.cutEnabled ?? false}
        setCutEnabled={cutEnabled => onControlsChange(current => ({ ...current, cutEnabled }))} />
    </section>
  )
}
