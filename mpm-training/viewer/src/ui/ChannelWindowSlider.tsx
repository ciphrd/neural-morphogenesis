interface ChannelWindowSliderProps {
  channels: number
  value: number
  onChange: (value: number) => void
}

const RGB_WINDOW_SIZE = 3

/** A native range input with a channel-grid visualization painted beneath it.
 * The input stays real (but transparent), preserving drag, click-to-seek,
 * touch, keyboard, and screen-reader behavior. */
export function ChannelWindowSlider({
  channels,
  value,
  onChange,
}: ChannelWindowSliderProps) {
  const channelCount = Math.max(1, Math.floor(channels))
  const windowSize = Math.min(RGB_WINDOW_SIZE, channelCount)
  const maxStart = Math.max(0, channelCount - windowSize)
  const start = Math.min(maxStart, Math.max(0, Math.floor(value)))

  return (
    <div
      className="channel-window-slider"
      style={{ gridTemplateColumns: `repeat(${channelCount}, minmax(0, 1fr))` }}
    >
      {Array.from({ length: channelCount }, (_, channel) => {
        const offset = channel - start
        const rgbClass = offset === 0 ? " is-red" : offset === 1 ? " is-green" : offset === 2 ? " is-blue" : ""
        return (
          <span
            key={channel}
            className={`channel-window-cell${offset >= 0 && offset < windowSize ? ` is-selected${rgbClass}` : ""}`}
            style={{ gridColumn: channel + 1 }}
            title={offset >= 0 && offset < windowSize ? `Channel ${channel} → ${"RGB"[offset]}` : `Channel ${channel}`}
          >
            {channel}
          </span>
        )
      })}
      <span
        className="channel-window-selection"
        style={{ gridColumn: `${start + 1} / span ${windowSize}` }}
        aria-hidden="true"
      />
      <input
        className="channel-window-input"
        type="range"
        min={0}
        max={maxStart}
        step={1}
        value={start}
        aria-label="First substrate channel displayed as RGB"
        aria-valuetext={`Channels ${start} through ${start + windowSize - 1}`}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </div>
  )
}
