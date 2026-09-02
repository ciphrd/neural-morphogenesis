import { useEffect, useRef, useState } from "react"
import type { SimulationConfig } from "./gpu/types"
import {
  PERFORMANCE_CHANNEL_NAME,
  type ControllerToProjectionMessage,
  type PerformanceSnapshot,
  type ProjectionToControllerMessage,
} from "./performance/types"
import type { GridCanvasHandle } from "./render/GridCanvas"
import { GridCanvas } from "./render/GridCanvas"

const DEFAULT_DEFORM_SETTINGS = {
  direction: "outward" as const,
  strength: 1,
  radius: 0.08,
  mode: "velocity" as const,
}

function visualSignature(snapshot: PerformanceSnapshot): string {
  return JSON.stringify({ ...snapshot, physics: null })
}

export function ProjectionView() {
  const canvasRef = useRef<GridCanvasHandle>(null)
  const channelRef = useRef<BroadcastChannel | null>(null)
  const visualSignatureRef = useRef("")
  const telemetryRef = useRef({ lastSentAt: 0, lastFrameAt: performance.now(), fps: 0 })
  const particleCapRef = useRef(0)
  const autoPruneFractionRef = useRef<number | null>(null)
  const autoPruneDelayMsRef = useRef(30_000)
  const autoPruneArmedRef = useRef(true)
  const autoPruneReachedCapAtRef = useRef<number | null>(null)
  const autoRandomizeIntervalMsRef = useRef<number | null>(null)
  const autoRandomizeNextAtRef = useRef<number | null>(null)
  const autoResetIntervalMsRef = useRef<number | null>(null)
  const autoResetNextAtRef = useRef<number | null>(null)
  const [config, setConfig] = useState<SimulationConfig | null>(null)
  const [snapshot, setSnapshot] = useState<PerformanceSnapshot | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const channel = new BroadcastChannel(PERFORMANCE_CHANNEL_NAME)
    channelRef.current = channel
    channel.onmessage = (event: MessageEvent<ControllerToProjectionMessage>) => {
      const message = event.data
      setConnected(true)
      if (message.type === "config") {
        setConfig(message.config)
      } else if (message.type === "snapshot") {
        particleCapRef.current = message.snapshot.particleCap
        if (message.snapshot.physics) canvasRef.current?.setPhysics(message.snapshot.physics)
        const signature = visualSignature(message.snapshot)
        if (signature !== visualSignatureRef.current) {
          visualSignatureRef.current = signature
          setSnapshot(message.snapshot)
        }
      } else if (message.type === "auto-prune") {
        autoPruneFractionRef.current = message.fraction
        autoPruneDelayMsRef.current = Math.max(0, message.delayMs)
        autoPruneArmedRef.current = true
        autoPruneReachedCapAtRef.current = null
      } else if (message.type === "auto-randomize") {
        autoRandomizeIntervalMsRef.current = message.intervalMs
        autoRandomizeNextAtRef.current = message.intervalMs === null
          ? null
          : performance.now() + Math.max(0, message.intervalMs)
      } else if (message.type === "auto-reset") {
        autoResetIntervalMsRef.current = message.intervalMs
        autoResetNextAtRef.current = message.intervalMs === null
          ? null
          : performance.now() + Math.max(0, message.intervalMs)
      } else if (message.type === "command") {
        if (message.command === "restart") canvasRef.current?.restart()
        else if (message.command === "randomize") canvasRef.current?.randomizeWeights(false)
        else if (message.command === "randomize-and-restart") canvasRef.current?.randomizeWeights(true)
        else if (message.command === "kill-20-percent") canvasRef.current?.killFraction(0.2)
        else if (message.command === "kill-80-percent") canvasRef.current?.killFraction(0.8)
      }
    }
    channel.postMessage({ type: "hello" } satisfies ProjectionToControllerMessage)
    return () => {
      channel.close()
      channelRef.current = null
    }
  }, [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "f" || event.repeat) return
      event.preventDefault()
      if (document.fullscreenElement) void document.exitFullscreen()
      else void document.documentElement.requestFullscreen()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  const onStep = (step: number, particleCount: number) => {
    const now = performance.now()
    const autoPruneFraction = autoPruneFractionRef.current
    const particleCap = particleCapRef.current
    if (autoPruneFraction !== null && particleCap > 0) {
      if (particleCount < particleCap) {
        autoPruneArmedRef.current = true
        autoPruneReachedCapAtRef.current = null
      } else if (autoPruneArmedRef.current) {
        if (autoPruneReachedCapAtRef.current === null) {
          autoPruneReachedCapAtRef.current = now
        }
        if (now - autoPruneReachedCapAtRef.current >= autoPruneDelayMsRef.current) {
          autoPruneArmedRef.current = false
          autoPruneReachedCapAtRef.current = null
          canvasRef.current?.killFraction(autoPruneFraction)
        }
      }
    } else {
      autoPruneReachedCapAtRef.current = null
    }
    const autoRandomizeIntervalMs = autoRandomizeIntervalMsRef.current
    const autoRandomizeNextAt = autoRandomizeNextAtRef.current
    if (
      autoRandomizeIntervalMs !== null &&
      autoRandomizeNextAt !== null &&
      now >= autoRandomizeNextAt
    ) {
      canvasRef.current?.randomizeWeights(false)
      autoRandomizeNextAtRef.current = now + autoRandomizeIntervalMs
    }
    const autoResetIntervalMs = autoResetIntervalMsRef.current
    const autoResetNextAt = autoResetNextAtRef.current
    if (
      autoResetIntervalMs !== null &&
      autoResetNextAt !== null &&
      now >= autoResetNextAt
    ) {
      canvasRef.current?.restart()
      autoResetNextAtRef.current = now + autoResetIntervalMs
    }
    const elapsed = now - telemetryRef.current.lastFrameAt
    telemetryRef.current.lastFrameAt = now
    const instantaneousFps = elapsed > 0 ? 1000 / elapsed : 0
    telemetryRef.current.fps += (instantaneousFps - telemetryRef.current.fps) * 0.08
    if (now - telemetryRef.current.lastSentAt < 250) return
    telemetryRef.current.lastSentAt = now
    channelRef.current?.postMessage({
      type: "telemetry",
      telemetry: {
        step,
        particleCount,
        fps: telemetryRef.current.fps,
        updatedAt: Date.now(),
      },
    } satisfies ProjectionToControllerMessage)
  }

  return (
    <main className={"projection-view" + (snapshot?.blackout ? " is-blackout" : "")}>
      {snapshot && (
        <GridCanvas
          ref={canvasRef}
          config={config}
          targetPoints={null}
          targetVisible={false}
          physics={snapshot.physics}
          particleCap={snapshot.particleCap}
          initialParticleCount={snapshot.initialParticleCount}
          noiseDisplacementStrength={snapshot.noiseDisplacementStrength ?? 0}
          fieldMode={snapshot.render.fieldMode}
          substrateChannelStart={snapshot.render.substrateChannelStart}
          accent={snapshot.render.accent}
          morphologyGradientVisible={snapshot.render.morphologyGradientVisible}
          morphologyDensityVisible={snapshot.render.morphologyDensityVisible}
          blur={snapshot.render.blur}
          gradientExponent={snapshot.render.gradientExponent}
          particleRenderMode={snapshot.render.particleRenderMode}
          zoom={snapshot.render.zoom}
          autoZoom={snapshot.render.autoZoom}
          bloom={snapshot.render.bloom}
          particleRadiusPx={snapshot.render.particleRadiusPx}
          whiteDotsAlpha={snapshot.render.whiteDotsAlpha}
          activationAlpha={snapshot.render.activationAlpha}
          neuralColorAlpha={snapshot.render.neuralColorAlpha}
          internalStateAlpha={snapshot.render.internalStateAlpha}
          boundaryGradientScale={snapshot.render.boundaryGradientScale}
          internalStateChannelStart={snapshot.render.internalStateChannelStart}
          chemicalMemoryOpponentSubtraction={snapshot.render.chemicalMemoryOpponentSubtraction}
          growthAxisLengthPx={snapshot.render.growthAxisLengthPx}
          deformSettings={DEFAULT_DEFORM_SETTINGS}
          onStep={onStep}
          loopAtTrainedSteps={snapshot.loopAtTrainedSteps}
          paused={snapshot.paused}
        />
      )}
      {!snapshot && (
        <div className="projection-waiting">
          <strong>{connected ? "Waiting for simulation state…" : "Waiting for controller…"}</strong>
          <span>Open the Performance panel in the main viewer.</span>
        </div>
      )}
      <div className="projection-blackout" />
    </main>
  )
}
