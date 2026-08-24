interface ZipEntry {
  filename: string
  blob: Blob
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
    }
  }
  return (crc ^ 0xffffffff) >>> 0
}

function writeU16(view: DataView, offset: number, value: number): void {
  view.setUint16(offset, value, true)
}

function writeU32(view: DataView, offset: number, value: number): void {
  view.setUint32(offset, value, true)
}

/** Creates a dependency-free, uncompressed ZIP. PNGs are already compressed,
 * so deflating them again would add time without materially reducing size. */
export async function createZip(entries: ZipEntry[]): Promise<Blob> {
  const encoder = new TextEncoder()
  const prepared = await Promise.all(entries.map(async ({ filename, blob }) => {
    const nameBytes = encoder.encode(filename)
    const bytes = new Uint8Array(await blob.arrayBuffer())
    return { nameBytes, bytes, crc: crc32(bytes) }
  }))

  const localParts: BlobPart[] = []
  const centralParts: BlobPart[] = []
  let offset = 0

  for (const entry of prepared) {
    const local = new ArrayBuffer(30)
    const localView = new DataView(local)
    writeU32(localView, 0, 0x04034b50)
    writeU16(localView, 4, 20)
    writeU16(localView, 6, 0x0800)
    writeU16(localView, 8, 0)
    writeU32(localView, 14, entry.crc)
    writeU32(localView, 18, entry.bytes.byteLength)
    writeU32(localView, 22, entry.bytes.byteLength)
    writeU16(localView, 26, entry.nameBytes.byteLength)
    localParts.push(local, entry.nameBytes, entry.bytes)

    const central = new ArrayBuffer(46)
    const centralView = new DataView(central)
    writeU32(centralView, 0, 0x02014b50)
    writeU16(centralView, 4, 20)
    writeU16(centralView, 6, 20)
    writeU16(centralView, 8, 0x0800)
    writeU16(centralView, 10, 0)
    writeU32(centralView, 16, entry.crc)
    writeU32(centralView, 20, entry.bytes.byteLength)
    writeU32(centralView, 24, entry.bytes.byteLength)
    writeU16(centralView, 28, entry.nameBytes.byteLength)
    writeU32(centralView, 42, offset)
    centralParts.push(central, entry.nameBytes)

    offset += 30 + entry.nameBytes.byteLength + entry.bytes.byteLength
  }

  const centralSize = centralParts.reduce((size, part) => {
    if (part instanceof ArrayBuffer) return size + part.byteLength
    if (ArrayBuffer.isView(part)) return size + part.byteLength
    return size
  }, 0)
  const end = new ArrayBuffer(22)
  const endView = new DataView(end)
  writeU32(endView, 0, 0x06054b50)
  writeU16(endView, 8, prepared.length)
  writeU16(endView, 10, prepared.length)
  writeU32(endView, 12, centralSize)
  writeU32(endView, 16, offset)
  return new Blob([...localParts, ...centralParts, end], { type: "application/zip" })
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
