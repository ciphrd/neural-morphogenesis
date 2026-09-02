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
import { PerformanceRenderingPanel } from "./PerformanceRenderingPanel"
import { Slider } from "./Slider"

const SCENES_STORAGE_KEY = "mpm-training-performance-scenes-v1"
const AUTO_PRUNE_STORAGE_KEY = "mpm-training-performance-auto-prune-99-5-v1"
const LEGACY_AUTO_PRUNE_STORAGE_KEYS = [
  "mpm-training-performance-auto-prune-98-v1",
  "mpm-training-performance-auto-prune-95-v1",
  "mpm-training-performance-auto-prune-80-v1",
]
const AUTO_PRUNE_DELAY_STORAGE_KEY = "mpm-training-performance-auto-prune-delay-v1"
const AUTO_RANDOMIZE_STORAGE_KEY = "mpm-training-performance-auto-randomize-v1"
const AUTO_RANDOMIZE_DURATION_STORAGE_KEY = "mpm-training-performance-auto-randomize-duration-v1"
const AUTO_RESET_STORAGE_KEY = "mpm-training-performance-auto-reset-v1"
const AUTO_RESET_DURATION_STORAGE_KEY = "mpm-training-performance-auto-reset-duration-v1"

function loadAutoPrune(): boolean {
  const stored = localStorage.getItem(AUTO_PRUNE_STORAGE_KEY)
    ?? LEGACY_AUTO_PRUNE_STORAGE_KEYS
      .map((key) => localStorage.getItem(key))
      .find((value) => value !== null)
  return stored === "true"
}

function loadAutoRandomize(): boolean {
  return localStorage.getItem(AUTO_RANDOMIZE_STORAGE_KEY) === "true"
}

function loadAutoReset(): boolean {
  return localStorage.getItem(AUTO_RESET_STORAGE_KEY) === "true"
}

function loadAutomationDurationSeconds(storageKey: string): number {
  const raw = localStorage.getItem(storageKey)
  if (raw === null) return 20
  const stored = Number(raw)
  return Number.isFinite(stored) ? Math.max(1, Math.min(120, stored)) : 20
}

function loadAutoPruneDelaySeconds(): number {
  const raw = localStorage.getItem(AUTO_PRUNE_DELAY_STORAGE_KEY)
  if (raw === null) return 30
  const stored = Number(raw)
  return Number.isFinite(stored) ? Math.max(0, Math.min(120, stored)) : 30
}

interface PerformancePanelProps {
  config: SimulationConfig | null
  snapshot: PerformanceSnapshot
  onApplySnapshot: (snapshot: PerformanceSnapshot) => void
  onRestart: () => void
  onRandomize: (restart: boolean) => void
  onKillFraction: (fraction: number) => void
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
  onRandomize,
  onKillFraction,
}: PerformancePanelProps) {
  const [open, setOpen] = useState(true)
  const [scenes, setScenes] = useState<PerformanceScene[]>(loadScenes)
  const [sceneName, setSceneName] = useState("")
  const [transitionMs, setTransitionMs] = useState(1200)
  const [autoPruneEnabled, setAutoPruneEnabled] = useState(loadAutoPrune)
  const [autoPruneDelaySeconds, setAutoPruneDelaySeconds] = useState(loadAutoPruneDelaySeconds)
  const [autoRandomize, setAutoRandomize] = useState(loadAutoRandomize)
  const [autoRandomizeDurationSeconds, setAutoRandomizeDurationSeconds] = useState(
    () => loadAutomationDurationSeconds(AUTO_RANDOMIZE_DURATION_STORAGE_KEY),
  )
  const [autoReset, setAutoReset] = useState(loadAutoReset)
  const [autoResetDurationSeconds, setAutoResetDurationSeconds] = useState(
    () => loadAutomationDurationSeconds(AUTO_RESET_DURATION_STORAGE_KEY),
  )
  const [outputConnected, setOutputConnected] = useState(false)
  const [telemetry, setTelemetry] = useState<ProjectionTelemetry | null>(null)
  const channelRef = useRef<BroadcastChannel | null>(null)
  const configRef = useRef(config)
  const snapshotRef = useRef(snapshot)
  const autoPruneRef = useRef(autoPruneEnabled)
  const autoPruneDelayRef = useRef(autoPruneDelaySeconds)
  const autoRandomizeRef = useRef(autoRandomize)
  const autoRandomizeDurationRef = useRef(autoRandomizeDurationSeconds)
  const autoResetRef = useRef(autoReset)
  const autoResetDurationRef = useRef(autoResetDurationSeconds)
  const transitionFrameRef = useRef(0)
  configRef.current = config
  snapshotRef.current = snapshot
  autoPruneRef.current = autoPruneEnabled
  autoPruneDelayRef.current = autoPruneDelaySeconds
  autoRandomizeRef.current = autoRandomize
  autoRandomizeDurationRef.current = autoRandomizeDurationSeconds
  autoResetRef.current = autoReset
  autoResetDurationRef.current = autoResetDurationSeconds

  useEffect(() => {
    const channel = new BroadcastChannel(PERFORMANCE_CHANNEL_NAME)
    channelRef.current = channel
    channel.onmessage = (event: MessageEvent<ProjectionToControllerMessage>) => {
      if (event.data.type === "hello") {
        setOutputConnected(true)
        channel.postMessage({ type: "config", config: configRef.current } satisfies ControllerToProjectionMessage)
        channel.postMessage({ type: "snapshot", snapshot: snapshotRef.current } satisfies ControllerToProjectionMessage)
        channel.postMessage({
          type: "auto-prune",
          fraction: autoPruneRef.current ? 0.995 : null,
          delayMs: autoPruneDelayRef.current * 1000,
        } satisfies ControllerToProjectionMessage)
        channel.postMessage({
          type: "auto-randomize",
          intervalMs: autoRandomizeRef.current ? autoRandomizeDurationRef.current * 1000 : null,
        } satisfies ControllerToProjectionMessage)
        channel.postMessage({
          type: "auto-reset",
          intervalMs: autoResetRef.current ? autoResetDurationRef.current * 1000 : null,
        } satisfies ControllerToProjectionMessage)
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
    localStorage.setItem(AUTO_PRUNE_STORAGE_KEY, String(autoPruneEnabled))
    localStorage.setItem(AUTO_PRUNE_DELAY_STORAGE_KEY, String(autoPruneDelaySeconds))
    channelRef.current?.postMessage({
      type: "auto-prune",
      fraction: autoPruneEnabled ? 0.995 : null,
      delayMs: autoPruneDelaySeconds * 1000,
    } satisfies ControllerToProjectionMessage)
  }, [autoPruneDelaySeconds, autoPruneEnabled])

  useEffect(() => {
    localStorage.setItem(AUTO_RANDOMIZE_STORAGE_KEY, String(autoRandomize))
    localStorage.setItem(AUTO_RANDOMIZE_DURATION_STORAGE_KEY, String(autoRandomizeDurationSeconds))
    channelRef.current?.postMessage({
      type: "auto-randomize",
      intervalMs: autoRandomize ? autoRandomizeDurationSeconds * 1000 : null,
    } satisfies ControllerToProjectionMessage)
  }, [autoRandomize, autoRandomizeDurationSeconds])

  useEffect(() => {
    localStorage.setItem(AUTO_RESET_STORAGE_KEY, String(autoReset))
    localStorage.setItem(AUTO_RESET_DURATION_STORAGE_KEY, String(autoResetDurationSeconds))
    channelRef.current?.postMessage({
      type: "auto-reset",
      intervalMs: autoReset ? autoResetDurationSeconds * 1000 : null,
    } satisfies ControllerToProjectionMessage)
  }, [autoReset, autoResetDurationSeconds])

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

  const randomize = (restartAfter: boolean) => {
    onRandomize(restartAfter)
    channelRef.current?.postMessage({
      type: "command",
      command: restartAfter ? "randomize-and-restart" : "randomize",
    } satisfies ControllerToProjectionMessage)
  }

  const killTwentyPercent = () => {
    onKillFraction(0.2)
    channelRef.current?.postMessage({
      type: "command",
      command: "kill-20-percent",
    } satisfies ControllerToProjectionMessage)
  }

  const killEightyPercent = () => {
    onKillFraction(0.8)
    channelRef.current?.postMessage({
      type: "command",
      command: "kill-80-percent",
    } satisfies ControllerToProjectionMessage)
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
            <button className="select" onClick={() => randomize(false)}>Randomize</button>
            <button className="select" onClick={() => randomize(true)}>Rand + res</button>
            <button
              className={"select performance-auto-randomize" + (autoRandomize ? " is-active" : "")}
              aria-pressed={autoRandomize}
              title={`Randomize the live neural weights every ${autoRandomizeDurationSeconds} seconds without restarting`}
              onClick={() => setAutoRandomize((enabled) => !enabled)}
            >
              Auto randomize · {autoRandomizeDurationSeconds}s
            </button>
            <button
              className={"select performance-auto-reset" + (autoReset ? " is-active" : "")}
              aria-pressed={autoReset}
              title={`Restart the live rollout every ${autoResetDurationSeconds} seconds without changing its neural weights`}
              onClick={() => setAutoReset((enabled) => !enabled)}
            >
              Auto reset · {autoResetDurationSeconds}s
            </button>
            <button className="select performance-kill" onClick={killTwentyPercent}>Kill 20%</button>
            <button className="select performance-kill" onClick={killEightyPercent}>Kill 80%</button>
            <button
              className={"select performance-auto-prune" + (autoPruneEnabled ? " is-active" : "")}
              aria-pressed={autoPruneEnabled}
              title="Randomly cull 99.5% of the live population after it remains at the particle cap for the configured delay"
              onClick={() => setAutoPruneEnabled((enabled) => !enabled)}
            >
              Auto prune 99.5%
            </button>
            <button
              className={"select performance-blackout" + (snapshot.blackout ? " is-active" : "")}
              onClick={() => onApplySnapshot({ ...snapshot, blackout: !snapshot.blackout })}
            >
              {snapshot.blackout ? "Restore" : "Blackout"}
            </button>
          </div>
          {autoRandomize && (
            <label className="slider-row" title="Time between automatic neural-weight randomizations">
              <span>Randomize every</span>
              <Slider
                min={1}
                max={120}
                step={1}
                value={autoRandomizeDurationSeconds}
                onChange={setAutoRandomizeDurationSeconds}
              />
              <span className="slider-value">{autoRandomizeDurationSeconds.toFixed(0)}s</span>
            </label>
          )}
          {autoReset && (
            <label className="slider-row" title="Time between automatic rollout resets">
              <span>Reset every</span>
              <Slider
                min={1}
                max={120}
                step={1}
                value={autoResetDurationSeconds}
                onChange={setAutoResetDurationSeconds}
              />
              <span className="slider-value">{autoResetDurationSeconds.toFixed(0)}s</span>
            </label>
          )}
          {autoPruneEnabled && (
            <label className="slider-row" title="Time spent at the population cap before automatic pruning">
              <span>Prune delay</span>
              <Slider
                min={0}
                max={120}
                step={1}
                value={autoPruneDelaySeconds}
                onChange={setAutoPruneDelaySeconds}
              />
              <span className="slider-value">{autoPruneDelaySeconds.toFixed(0)}s</span>
            </label>
          )}
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
      <PerformanceRenderingPanel
        config={config}
        value={snapshot.render}
        onChange={(render) => onApplySnapshot({ ...snapshot, render })}
      />
    </section>
  )
}
