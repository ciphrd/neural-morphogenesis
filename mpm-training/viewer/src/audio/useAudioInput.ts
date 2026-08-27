import { useEffect, useRef, useState } from "react"

export type AudioInputStatus = "idle" | "requesting" | "active" | "error" | "unsupported"

export interface AudioInputDevice {
  deviceId: string
  label: string
}

export interface AudioAnalysisFrame {
  waveform: Float32Array
  spectrum: Float32Array
  sampleRate: number
}

interface AudioInputOptions {
  enabled: boolean
  deviceId: string
  gain: number
  threshold: number
  smoothing: number
}

export function useAudioInput({
  enabled,
  deviceId,
  gain,
  threshold,
  smoothing,
}: AudioInputOptions) {
  const [devices, setDevices] = useState<AudioInputDevice[]>([])
  const [energy, setEnergy] = useState(0)
  const [status, setStatus] = useState<AudioInputStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const smoothedEnergy = useRef(0)
  const analysis = useRef<AudioAnalysisFrame | null>(null)
  const processing = useRef({ gain, threshold, smoothing })
  processing.current = { gain, threshold, smoothing }

  useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setStatus("unsupported")
      return
    }
    let cancelled = false
    const refresh = async () => {
      const inputs = (await navigator.mediaDevices.enumerateDevices())
        .filter((device) => device.kind === "audioinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Audio input ${index + 1}`,
        }))
      if (!cancelled) setDevices(inputs)
    }
    void refresh()
    navigator.mediaDevices.addEventListener("devicechange", refresh)
    return () => {
      cancelled = true
      navigator.mediaDevices.removeEventListener("devicechange", refresh)
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      setStatus("idle")
      setEnergy(0)
      smoothedEnergy.current = 0
      analysis.current = null
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unsupported")
      setError("Audio capture is unavailable in this browser.")
      return
    }

    let cancelled = false
    let frame = 0
    let stream: MediaStream | null = null
    let context: AudioContext | null = null
    setStatus("requesting")
    setError(null)

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: deviceId ? { deviceId: { exact: deviceId } } : true,
        })
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        context = new AudioContext()
        await context.resume()
        const source = context.createMediaStreamSource(stream)
        const analyser = context.createAnalyser()
        analyser.fftSize = 2048
        analyser.minDecibels = -100
        analyser.maxDecibels = -20
        source.connect(analyser)
        const samples = new Float32Array(analyser.fftSize)
        const spectrum = new Float32Array(analyser.frequencyBinCount)
        analysis.current = { waveform: samples, spectrum, sampleRate: context.sampleRate }
        setStatus("active")

        const sample = () => {
          analyser.smoothingTimeConstant = processing.current.smoothing
          analyser.getFloatTimeDomainData(samples)
          analyser.getFloatFrequencyData(spectrum)
          let sumSquares = 0
          for (const value of samples) sumSquares += value * value
          const rms = Math.sqrt(sumSquares / samples.length)
          const current = processing.current
          const gated = Math.max(0, rms - current.threshold)
          const normalized = Math.min(1, gated * current.gain)
          smoothedEnergy.current +=
            (normalized - smoothedEnergy.current) * (1 - current.smoothing)
          setEnergy(smoothedEnergy.current)
          frame = requestAnimationFrame(sample)
        }
        sample()

        const inputs = (await navigator.mediaDevices.enumerateDevices())
          .filter((candidate) => candidate.kind === "audioinput")
          .map((candidate, index) => ({
            deviceId: candidate.deviceId,
            label: candidate.label || `Audio input ${index + 1}`,
          }))
        if (!cancelled) setDevices(inputs)
      } catch (cause) {
        if (cancelled) return
        setStatus("error")
        setError(cause instanceof Error ? cause.message : "Could not open the audio input.")
      }
    }
    void start()

    return () => {
      cancelled = true
      cancelAnimationFrame(frame)
      stream?.getTracks().forEach((track) => track.stop())
      analysis.current = null
      if (context) void context.close()
    }
  }, [deviceId, enabled])

  return { devices, energy, status, error, analysis }
}
