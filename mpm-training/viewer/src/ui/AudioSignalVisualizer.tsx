import { useEffect, useRef, type RefObject } from "react"
import { ENERGY_HISTORY_DURATION_MS, type AudioAnalysisFrame, type AudioEnergySample } from "../audio/useAudioInput"

interface AudioSignalVisualizerProps {
  analysis: RefObject<AudioAnalysisFrame | null>
  energyHistory: RefObject<AudioEnergySample[]>
  active: boolean
  gain: number
  threshold: number
}

function processAmplitude(value: number, gain: number, threshold: number): number {
  const magnitude = Math.max(0, Math.abs(value) - threshold) * gain
  return Math.sign(value) * Math.min(1, magnitude)
}

function prepareCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr))
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  const context = canvas.getContext("2d")
  context?.setTransform(dpr, 0, 0, dpr, 0, 0)
  return context
}

function drawGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.fillStyle = "#000000"
  context.fillRect(0, 0, width, height)
  context.strokeStyle = "rgba(91, 166, 183, 0.12)"
  context.lineWidth = 1
  for (let x = 0; x <= width; x += width / 8) {
    context.beginPath()
    context.moveTo(Math.round(x) + 0.5, 0)
    context.lineTo(Math.round(x) + 0.5, height)
    context.stroke()
  }
  for (let y = 0; y <= height; y += height / 4) {
    context.beginPath()
    context.moveTo(0, Math.round(y) + 0.5)
    context.lineTo(width, Math.round(y) + 0.5)
    context.stroke()
  }
}

function signalGradient(context: CanvasRenderingContext2D, width: number, opacity: number) {
  const gradient = context.createLinearGradient(0, 0, width, 0)
  gradient.addColorStop(0, `rgba(244, 86, 86, ${opacity})`)
  gradient.addColorStop(0.5, `rgba(244, 211, 94, ${opacity})`)
  gradient.addColorStop(1, `rgba(97, 211, 137, ${opacity})`)
  return gradient
}

function drawWaveform(
  canvas: HTMLCanvasElement,
  frame: AudioAnalysisFrame | null,
  gain: number,
  threshold: number,
) {
  const context = prepareCanvas(canvas)
  if (!context) return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  const plotHeight = height - 18
  context.strokeStyle = "rgba(160, 170, 165, 0.15)"
  context.beginPath()
  context.moveTo(0, plotHeight / 2 + 0.5)
  context.lineTo(width, plotHeight / 2 + 0.5)
  context.stroke()
  if (!frame) return
  context.strokeStyle = "rgba(255, 255, 255, 0.7)"
  context.lineWidth = 1.4
  context.beginPath()
  const stride = Math.max(1, Math.floor(frame.waveform.length / width))
  for (let x = 0; x < width; x++) {
    const rawValue = frame.waveform[Math.min(frame.waveform.length - 1, x * stride)]
    const value = processAmplitude(rawValue, gain, threshold)
    const y = plotHeight / 2 - value * plotHeight * 0.43
    if (x === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  }
  context.stroke()
}

function drawSpectrum(
  canvas: HTMLCanvasElement,
  frame: AudioAnalysisFrame | null,
  gain: number,
  threshold: number,
) {
  const context = prepareCanvas(canvas)
  if (!context) return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  drawGrid(context, width, height)
  const plotHeight = height - 18
  if (frame) {
    const minFrequency = 20
    const maxFrequency = Math.min(20_000, frame.sampleRate / 2)
    context.strokeStyle = signalGradient(context, width, 0.3)
    context.fillStyle = signalGradient(context, width, 0.14)
    context.lineWidth = 1.5
    context.beginPath()
    for (let x = 0; x < width; x++) {
      const frequency = minFrequency * Math.pow(maxFrequency / minFrequency, x / Math.max(1, width - 1))
      const bin = Math.min(
        frame.spectrum.length - 1,
        Math.round((frequency / (frame.sampleRate / 2)) * (frame.spectrum.length - 1)),
      )
      const rawAmplitude = Math.pow(10, frame.spectrum[bin] / 20)
      const processedAmplitude = Math.max(0, rawAmplitude - threshold) * gain
      const processedDb = processedAmplitude > 0 ? 20 * Math.log10(processedAmplitude) : -100
      const level = Math.max(0, Math.min(1, (processedDb + 100) / 80))
      const y = plotHeight - level * (plotHeight - 4)
      if (x === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    }
    context.lineTo(width, plotHeight)
    context.lineTo(0, plotHeight)
    context.closePath()
    context.fill()
    context.stroke()
  }
  context.fillStyle = "#60747b"
  context.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace"
  const labels = [20, 60, 250, 1000, 4000, 16000]
  const maxFrequency = frame ? Math.min(20_000, frame.sampleRate / 2) : 20_000
  for (const frequency of labels) {
    const x = Math.log(frequency / 20) / Math.log(maxFrequency / 20) * width
    if (x < 0 || x > width) continue
    const label = frequency >= 1000 ? `${frequency / 1000}k` : `${frequency}`
    context.fillText(label, Math.min(width - 18, Math.max(2, x + 2)), height - 5)
  }
}

function drawEnergyHistory(canvas: HTMLCanvasElement, history: AudioEnergySample[], now: number) {
  const context = prepareCanvas(canvas)
  if (!context) return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  drawGrid(context, width, height)
  const left = 30
  const right = Math.max(left + 1, width - 8)
  const top = 8
  const bottom = height - 20
  const start = now - ENERGY_HISTORY_DURATION_MS
  context.strokeStyle = "#8fc1cc"
  context.lineWidth = 1.4
  context.beginPath()
  let started = false
  for (const sample of history) {
    if (sample.time < start || sample.time > now) continue
    const x = left + ((sample.time - start) / ENERGY_HISTORY_DURATION_MS) * (right - left)
    const y = bottom - Math.max(0, Math.min(1, sample.value)) * (bottom - top)
    if (!started) context.moveTo(x, y)
    else context.lineTo(x, y)
    started = true
  }
  context.stroke()
  context.fillStyle = "#819397"
  context.font = "9px sans-serif"
  context.fillText("100%", 2, top + 4)
  context.fillText("0%", 2, bottom)
  context.fillText(`−${ENERGY_HISTORY_DURATION_MS / 1000}s`, left, height - 5)
  context.textAlign = "center"
  context.fillText(`−${ENERGY_HISTORY_DURATION_MS / 2000}s`, (left + right) / 2, height - 5)
  context.textAlign = "right"
  context.fillText("Now", right, height - 5)
  context.textAlign = "left"
}

export function AudioSignalVisualizer({
  analysis,
  energyHistory,
  active,
  gain,
  threshold,
}: AudioSignalVisualizerProps) {
  const signalRef = useRef<HTMLCanvasElement>(null)
  const energyRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let animationFrame = 0
    const draw = () => {
      if (signalRef.current) {
        // Draw the filled spectrum first, then overlay the time-domain signal.
        drawSpectrum(signalRef.current, analysis.current, gain, threshold)
        drawWaveform(signalRef.current, analysis.current, gain, threshold)
      }
      if (energyRef.current) {
        drawEnergyHistory(energyRef.current, energyHistory.current ?? [], performance.now())
      }
      animationFrame = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(animationFrame)
  }, [analysis, energyHistory, gain, threshold])

  return (
    <div className={"audio-analyzers" + (active ? " is-active" : "")}>
      <figure>
        <figcaption><span>Signal + spectrum</span><small>POST · TIME / LOG Hz</small></figcaption>
        <canvas ref={signalRef} aria-label="Audio waveform layered over frequency spectrum" />
      </figure>
      <figure>
        <figcaption><span>Energy</span><small>POST · LAST {ENERGY_HISTORY_DURATION_MS / 1000}s</small></figcaption>
        <canvas ref={energyRef} aria-label={`Audio energy over the last ${ENERGY_HISTORY_DURATION_MS / 1000} seconds, from 0 to 100 percent`} />
      </figure>
    </div>
  )
}
