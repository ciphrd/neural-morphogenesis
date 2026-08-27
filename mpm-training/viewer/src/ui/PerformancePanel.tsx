import { useEffect, useRef, useState } from "react"
import type { SimulationConfig } from "../gpu/types"
import { interpolateSnapshot } from "../performance/interpolate"
import {
  PERFORMANCE_CHANNEL_NAME,
  type ControllerToProjectionMessage,
  type PerformanceScene,
  type PerformanceSnapshot,
  type ProjectionTelemetry,
  type ProjectionToControllerMessage,
} from "../performance/types"
import { Slider } from "./Slider"

const SCENES_STORAGE_KEY = "mpm-training-performance-scenes-v1"

interface PerformancePanelProps {
  config: SimulationConfig | null
  snapshot: PerformanceSnapshot
  onApplySnapshot: (snapshot: PerformanceSnapshot) => void
  onRestart: () => void
}

function loadScenes(): PerformanceScene[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(SCENES_STORAGE_KEY) ?? "[]")
    return Array.isArray(parsed) ? parsed as PerformanceScene[] : []
  } catch {
    return []
  }
}

export function PerformancePanel({
  config,
  snapshot,
  onApplySnapshot,
  onRestart,
}: PerformancePanelProps) {
  const [open, setOpen] = useState(true)
  const [scenes, setScenes] = useState<PerformanceScene[]>(loadScenes)
  const [sceneName, setSceneName] = useState("")
  const [transitionMs, setTransitionMs] = useState(1200)
  const [outputConnected, setOutputConnected] = useState(false)
  const [telemetry, setTelemetry] = useState<ProjectionTelemetry | null>(null)
  const channelRef = useRef<BroadcastChannel | null>(null)
  const configRef = useRef(config)
  const snapshotRef = useRef(snapshot)
  const transitionFrameRef = useRef(0)
  configRef.current = config
  snapshotRef.current = snapshot

  useEffect(() => {
    const channel = new BroadcastChannel(PERFORMANCE_CHANNEL_NAME)
    channelRef.current = channel
    channel.onmessage = (event: MessageEvent<ProjectionToControllerMessage>) => {
      if (event.data.type === "hello") {
        setOutputConnected(true)
        channel.postMessage({ type: "config", config: configRef.current } satisfies ControllerToProjectionMessage)
        channel.postMessage({ type: "snapshot", snapshot: snapshotRef.current } satisfies ControllerToProjectionMessage)
      } else if (event.data.type === "telemetry") {
        setOutputConnected(true)
        setTelemetry(event.data.telemetry)
      }
    }
    return () => {
      cancelAnimationFrame(transitionFrameRef.current)
      channel.close()
      channelRef.current = null
    }
  }, [])

  useEffect(() => {
    channelRef.current?.postMessage({ type: "config", config } satisfies ControllerToProjectionMessage)
  }, [config])

  useEffect(() => {
    channelRef.current?.postMessage({ type: "snapshot", snapshot } satisfies ControllerToProjectionMessage)
  }, [snapshot])

  useEffect(() => {
    localStorage.setItem(SCENES_STORAGE_KEY, JSON.stringify(scenes))
  }, [scenes])

  useEffect(() => {
    if (!outputConnected) return
    const timer = window.setInterval(() => {
      if (!telemetry || Date.now() - telemetry.updatedAt > 2000) {
        setOutputConnected(false)
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [outputConnected, telemetry])

  const openOutput = () => {
    const url = new URL(window.location.href)
    url.searchParams.set("output", "1")
    url.hash = ""
    window.open(url, "mpm-projection", "popup,width=1280,height=720")?.focus()
  }

  const restart = () => {
    onRestart()
    channelRef.current?.postMessage({ type: "command", command: "restart" } satisfies ControllerToProjectionMessage)
  }

  const saveScene = () => {
    const trimmed = sceneName.trim()
    const name = trimmed || `Scene ${scenes.length + 1}`
    const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
    setScenes((current) => [...current, { id, name, snapshot }])
    setSceneName("")
  }

  const recallScene = (scene: PerformanceScene) => {
    cancelAnimationFrame(transitionFrameRef.current)
    if (transitionMs === 0) {
      onApplySnapshot(scene.snapshot)
      return
    }
    const from = snapshotRef.current
    const startedAt = performance.now()
    const animate = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / transitionMs)
      const eased = progress * progress * (3 - 2 * progress)
      onApplySnapshot(interpolateSnapshot(from, scene.snapshot, eased))
      if (progress < 1) transitionFrameRef.current = requestAnimationFrame(animate)
    }
    transitionFrameRef.current = requestAnimationFrame(animate)
  }

  return (
    <section>
      <div className="physics-panel-header">
        <button className="physics-panel-toggle" onClick={() => setOpen((value) => !value)}>
          <span className={"physics-panel-chevron" + (open ? " is-open" : "")}>▸</span>
          <h2>Performance</h2>
        </button>
        <span className={"performance-connection" + (outputConnected ? " is-connected" : "")}>
          {outputConnected ? "Output live" : "No output"}
        </span>
      </div>
      {open && (
        <div className="physics-panel-body performance-panel-body">
          <div className="performance-actions">
            <button className="select" onClick={openOutput}>Open output</button>
            <button className="select" onClick={restart}>Restart</button>
            <button
              className={"select performance-blackout" + (snapshot.blackout ? " is-active" : "")}
              onClick={() => onApplySnapshot({ ...snapshot, blackout: !snapshot.blackout })}
            >
              {snapshot.blackout ? "Restore" : "Blackout"}
            </button>
          </div>
          {telemetry && outputConnected && (
            <div className="performance-telemetry">
              <span>{telemetry.fps.toFixed(0)} FPS</span>
              <span>step {telemetry.step}</span>
              <span>{telemetry.particleCount} cells</span>
            </div>
          )}
          <label className="slider-row">
            <span>Transition</span>
            <Slider min={0} max={5000} step={50} value={transitionMs} onChange={setTransitionMs} />
            <span className="slider-value">{(transitionMs / 1000).toFixed(2)}s</span>
          </label>
          <div className="performance-save-row">
            <input
              className="number-input"
              value={sceneName}
              placeholder="Scene name"
              onChange={(event) => setSceneName(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") saveScene() }}
            />
            <button className="icon-button" onClick={saveScene} aria-label="Save current scene" title="Save current scene">+</button>
          </div>
          {scenes.length === 0 && <p className="hint">Save the current look to begin a scene set.</p>}
          <div className="performance-scenes">
            {scenes.map((scene, index) => (
              <div className="performance-scene" key={scene.id}>
                <button onClick={() => recallScene(scene)}><span>{index + 1}</span>{scene.name}</button>
                <button
                  className="audio-remove"
                  onClick={() => setScenes((current) => current.filter(({ id }) => id !== scene.id))}
                  aria-label={`Delete ${scene.name}`}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
