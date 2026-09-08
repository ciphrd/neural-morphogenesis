import { PostProcessingPanel } from "./PostProcessingPanel"
import { PerformanceToolbar } from "./PerformanceToolbar"
import type { PerformanceOutputStatus } from "./PerformanceOutputPanel"
import { useEffect, useRef, useState } from "react"
import { SequenceScheduler, type PerformanceSequence, type SequenceAction } from "../performance/sequences"
import type { SimulationConfig } from "../gpu/types"
import {
  PERFORMANCE_CHANNEL_NAME,
  type ControllerToProjectionMessage,
  type PerformanceSnapshot,
  type ProjectionTelemetry,
  type ProjectionToControllerMessage,
} from "../performance/types"
import { PerformanceRenderingPanel } from "./PerformanceRenderingPanel"

interface PerformanceControllerProps {
  onPolicyWeights: (weights: SimulationConfig["weights"]) => void
  sequences: PerformanceSequence[]
  onOutputStatusChange: (status: PerformanceOutputStatus) => void
  config: SimulationConfig | null
  snapshot: PerformanceSnapshot
  onApplySnapshot: (snapshot: PerformanceSnapshot) => void
  onRestart: () => void
  onRandomize: (restart: boolean) => void
  onKillFraction: (fraction: number) => void
  onKillOutsideCircle: () => void
}

export function PerformanceController({
  onPolicyWeights,
  sequences,
  onOutputStatusChange,
  config,
  snapshot,
  onApplySnapshot,
  onRestart,
  onRandomize,
  onKillFraction,
  onKillOutsideCircle,
}: PerformanceControllerProps) {
  const policyCallback = useRef(onPolicyWeights)
  policyCallback.current = onPolicyWeights
  const [outputConnected, setOutputConnected] = useState(false)
  const [telemetry, setTelemetry] = useState<ProjectionTelemetry | null>(null)
  const channelRef = useRef<BroadcastChannel | null>(null)
  const configRef = useRef(config)
  const snapshotRef = useRef(snapshot)
  configRef.current = config
  snapshotRef.current = snapshot

  useEffect(() => {
    const channel = new BroadcastChannel(PERFORMANCE_CHANNEL_NAME)
    channelRef.current = channel
    channel.onmessage = (event: MessageEvent<ProjectionToControllerMessage>) => {
      if (event.data.type === "policy-weights") {
        policyCallback.current(event.data.weights)
      } else if (event.data.type === "hello") {
        setOutputConnected(true)
        channel.postMessage({ type: "config", config: configRef.current } satisfies ControllerToProjectionMessage)
        channel.postMessage({ type: "snapshot", snapshot: snapshotRef.current } satisfies ControllerToProjectionMessage)

      } else if (event.data.type === "telemetry") {
        setOutputConnected(true)
        setTelemetry(event.data.telemetry)
      }
    }
    return () => {
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
    if (!outputConnected) return
    const timer = window.setInterval(() => {
      if (!telemetry || Date.now() - telemetry.updatedAt > 2000) {
        setOutputConnected(false)
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [outputConnected, telemetry])

  useEffect(() => {
    onOutputStatusChange({ connected: outputConnected, telemetry })
  }, [onOutputStatusChange, outputConnected, telemetry])

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

  const keepCenterCircle = () => {
    onKillOutsideCircle()
    channelRef.current?.postMessage({ type: "command", command: "keep-center-circle" } satisfies ControllerToProjectionMessage)
  }

  const executeAction = (action: SequenceAction) => {
    if (action === "restart") restart()
    else if (action === "randomize") randomize(false)
    else if (action === "randomize-and-restart") randomize(true)
    else if (action === "kill-20-percent") killTwentyPercent()
    else if (action === "kill-80-percent") killEightyPercent()
    else if (action === "prune") {
      onKillFraction(0.995)
      channelRef.current?.postMessage({ type: "command", command: "prune" } satisfies ControllerToProjectionMessage)
    }
  }

  const actionsRef = useRef(executeAction)
  actionsRef.current = executeAction
  const sequencesRef = useRef(sequences)
  sequencesRef.current = sequences
  const schedulerRef = useRef(new SequenceScheduler())
  useEffect(() => {
    // Reconcile edits immediately, including a quick off/on toggle.
    for (const action of schedulerRef.current.tick(sequences, performance.now())) actionsRef.current(action)
  }, [sequences])
  useEffect(() => {
    const timer = window.setInterval(() => {
      for (const action of schedulerRef.current.tick(sequencesRef.current, performance.now())) {
        actionsRef.current(action)
      }
    }, 50)
    return () => window.clearInterval(timer)
  }, [])

  return (
    <>
    <PerformanceToolbar onRestart={restart} onRandomize={randomize} onKillTwenty={killTwentyPercent} onKillEighty={killEightyPercent} onKeepCircle={keepCenterCircle} autoPruneCircle={snapshot.autoPruneCircle ?? false} onAutoPruneCircleChange={autoPruneCircle => onApplySnapshot({ ...snapshot, autoPruneCircle })} />
    <div className="performance-dashboard-card">
      <PerformanceRenderingPanel
        config={config}
        value={snapshot.render}
        onChange={(render) => onApplySnapshot({ ...snapshot, render })}
      />
    </div>
    <div className="performance-dashboard-card">
    <PostProcessingPanel
      value={snapshot.render}
      onChange={render => onApplySnapshot({ ...snapshot, render })}
    />
    </div>
    </>
  )
}
