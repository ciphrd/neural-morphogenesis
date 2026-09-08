/** Active playback seconds used for a random-brain crossfade. */
export const BRAIN_TRANSITION_SECONDS = 2

export class WeightTransition {
  private current = new Float32Array(0)
  private source: Float32Array | null = null
  private target: Float32Array | null = null
  private elapsed = 0

  load(values: Float32Array): Float32Array {
    this.current = new Float32Array(values)
    this.source = this.target = null
    this.elapsed = 0
    return this.current
  }

  start(values: Float32Array): void {
    if (values.length !== this.current.length) throw Error("Brain transition weight dimensions differ")
    this.source = new Float32Array(this.current)
    this.target = new Float32Array(values)
    this.elapsed = 0
  }

  advance(seconds: number): Float32Array | null {
    if (!this.source || !this.target || !Number.isFinite(seconds) || seconds <= 0) return null
    this.elapsed = Math.min(BRAIN_TRANSITION_SECONDS, this.elapsed + seconds)
    const t = this.elapsed / BRAIN_TRANSITION_SECONDS
    if (t >= 1) {
      this.current.set(this.target)
      this.source = this.target = null
    } else {
      const amount = t * t * (3 - 2 * t)
      for (let i = 0; i < this.current.length; i++) {
        this.current[i] = this.source[i] + (this.target[i] - this.source[i]) * amount
      }
    }
    return this.current
  }
}
