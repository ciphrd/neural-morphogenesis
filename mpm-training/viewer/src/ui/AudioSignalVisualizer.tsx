import { useEffect, useRef, type RefObject } from "react"
import type { AudioAnalysisFrame } from "../audio/useAudioInput"

interface AudioSignalVisualizerProps {
  analysis: RefObject<AudioAnalysisFrame | null>
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
  context.fillStyle = "#080d11"
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
  drawGrid(context, width, height)
  context.strokeStyle = "rgba(129, 163, 171, 0.25)"
  context.beginPath()
  context.moveTo(0, height / 2 + 0.5)
  context.lineTo(width, height / 2 + 0.5)
  context.stroke()
  if (!frame) return
  context.strokeStyle = "#8fc1cc"
  context.lineWidth = 1.4
  context.beginPath()
  const stride = Math.max(1, Math.floor(frame.waveform.length / width))
  for (let x = 0; x < width; x++) {
    const rawValue = frame.waveform[Math.min(frame.waveform.length - 1, x * stride)]
    const value = processAmplitude(rawValue, gain, threshold)
    const y = height / 2 - value * height * 0.43
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
    context.strokeStyle = "#8fc1cc"
    context.fillStyle = "rgba(106, 155, 166, 0.14)"
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

export function AudioSignalVisualizer({
  analysis,
  active,
  gain,
  threshold,
}: AudioSignalVisualizerProps) {
  const waveformRef = useRef<HTMLCanvasElement>(null)
  const spectrumRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let animationFrame = 0
    const draw = () => {
      if (waveformRef.current) {
        drawWaveform(waveformRef.current, analysis.current, gain, threshold)
      }
      if (spectrumRef.current) {
        drawSpectrum(spectrumRef.current, analysis.current, gain, threshold)
      }
      animationFrame = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(animationFrame)
  }, [analysis, gain, threshold])

  return (
    <div className={"audio-analyzers" + (active ? " is-active" : "")}>
      <figure>
        <figcaption><span>Signal</span><small>POST · TIME</small></figcaption>
        <canvas ref={waveformRef} aria-label="Audio input waveform" />
      </figure>
      <figure>
        <figcaption><span>Spectrum</span><small>POST · FFT · LOG Hz</small></figcaption>
        <canvas ref={spectrumRef} aria-label="Audio frequency spectrum" />
      </figure>
    </div>
  )
}
