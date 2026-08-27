interface AtlasAxis {
  label: string
  values: readonly number[]
}

interface AtlasImage {
  filename: string
  blob: Blob
}

interface SampleAtlasOptions {
  images: readonly AtlasImage[]
  rows?: AtlasAxis
  columns: AtlasAxis
  signal: AbortSignal
}

const MAX_ATLAS_EDGE = 16_384
const MAX_ATLAS_PIXELS = 32_000_000
const MAX_TILE_EDGE = 256
const TOP_GUTTER = 58
const LEFT_GUTTER = 120

function abortIfRequested(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new DOMException("Sample collection cancelled", "AbortError")
  }
}

function valueLabel(value: number): string {
  return Number(value.toPrecision(12)).toString()
}

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob
          ? resolve(blob)
          : reject(new Error("Could not encode the sample atlas.")),
      "image/png"
    )
  })
}

/** Builds a labelled contact sheet from a one- or two-axis sweep.
 * Images must be in row-major order: every column of row 0, then row 1. */
export async function createSampleAtlas({
  images,
  rows,
  columns,
  signal,
}: SampleAtlasOptions): Promise<Blob> {
  const rowCount = rows?.values.length ?? 1
  const leftGutter = rows ? LEFT_GUTTER : 0
  const expected = rowCount * columns.values.length
  if (images.length !== expected || expected === 0) {
    throw new Error(
      `Cannot build sample atlas: expected ${expected} PNGs, received ${images.length}.`
    )
  }

  abortIfRequested(signal)
  let bitmap = await createImageBitmap(images[0].blob)
  const sourceWidth = bitmap.width
  const sourceHeight = bitmap.height
  const scale = Math.min(
    1,
    MAX_TILE_EDGE / sourceWidth,
    MAX_TILE_EDGE / sourceHeight,
    (MAX_ATLAS_EDGE - leftGutter) /
      (columns.values.length * sourceWidth),
    (MAX_ATLAS_EDGE - TOP_GUTTER) / (rowCount * sourceHeight),
    Math.sqrt(
      MAX_ATLAS_PIXELS /
        (expected * sourceWidth * sourceHeight)
    )
  )
  const tileWidth = Math.max(1, Math.floor(sourceWidth * scale))
  const tileHeight = Math.max(1, Math.floor(sourceHeight * scale))
  const canvas = document.createElement("canvas")
  canvas.width = leftGutter + columns.values.length * tileWidth
  canvas.height = TOP_GUTTER + rowCount * tileHeight
  const context = canvas.getContext("2d")
  if (!context) {
    bitmap.close()
    throw new Error("Could not create the sample atlas canvas.")
  }

  context.fillStyle = "#0d0d0d"
  context.fillRect(0, 0, canvas.width, canvas.height)
  context.fillStyle = "#e8e8e8"
  context.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
  context.textBaseline = "middle"
  context.textAlign = "center"
  context.fillText(
    columns.label,
    leftGutter + (columns.values.length * tileWidth) / 2,
    14
  )
  context.fillStyle = "#aaa"
  context.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
  for (let column = 0; column < columns.values.length; column += 1) {
    context.fillText(
      valueLabel(columns.values[column]),
      leftGutter + (column + 0.5) * tileWidth,
      39
    )
  }
  context.textAlign = "right"
  if (rows) {
    for (let row = 0; row < rows.values.length; row += 1) {
      context.fillText(
        valueLabel(rows.values[row]),
        leftGutter - 10,
        TOP_GUTTER + (row + 0.5) * tileHeight
      )
    }
    context.save()
    context.translate(
      18,
      TOP_GUTTER + (rowCount * tileHeight) / 2
    )
    context.rotate(-Math.PI / 2)
    context.fillStyle = "#e8e8e8"
    context.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    context.textAlign = "center"
    context.fillText(rows.label, 0, 0)
    context.restore()
  }

  for (let index = 0; index < images.length; index += 1) {
    abortIfRequested(signal)
    if (index > 0) bitmap = await createImageBitmap(images[index].blob)
    const row = Math.floor(index / columns.values.length)
    const column = index % columns.values.length
    const x = leftGutter + column * tileWidth
    const y = TOP_GUTTER + row * tileHeight
    context.drawImage(bitmap, x, y, tileWidth, tileHeight)
    bitmap.close()
    context.strokeStyle = "#2a2a2a"
    context.strokeRect(x + 0.5, y + 0.5, tileWidth - 1, tileHeight - 1)
  }

  abortIfRequested(signal)
  return canvasBlob(canvas)
}
