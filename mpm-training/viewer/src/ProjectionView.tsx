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
        if (message.snapshot.physics) canvasRef.current?.setPhysics(message.snapshot.physics)
        const signature = visualSignature(message.snapshot)
        if (signature !== visualSignatureRef.current) {
          visualSignatureRef.current = signature
          setSnapshot(message.snapshot)
        }
      } else if (message.type === "command" && message.command === "restart") {
        canvasRef.current?.restart()
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
          particleRadiusPx={snapshot.render.particleRadiusPx}
          whiteDotsAlpha={snapshot.render.whiteDotsAlpha}
          activationAlpha={snapshot.render.activationAlpha}
          neuralColorAlpha={snapshot.render.neuralColorAlpha}
          internalStateAlpha={snapshot.render.internalStateAlpha}
          boundaryGradientScale={snapshot.render.boundaryGradientScale}
          internalStateChannelStart={snapshot.render.internalStateChannelStart}
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
