export async function pingServer(): Promise<boolean> {
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(2000) })
    return r.ok
  } catch {
    return false
  }
}

export async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text().catch(() => '')}`)
  return (await r.json()) as T
}

export async function postJSON<T>(url: string, body: unknown): Promise<T> {
  return getJSON<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** Decode a base64 string into an ArrayBuffer. */
export function b64ToBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out.buffer
}

const F16_TABLE = (() => {
  const t = new Float32Array(65536)
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1
    const e = (h >> 10) & 0x1f
    const f = h & 0x3ff
    let v: number
    if (e === 0) v = s * Math.pow(2, -14) * (f / 1024)
    else if (e === 31) v = f ? NaN : s * Infinity
    else v = s * Math.pow(2, e - 15) * (1 + f / 1024)
    t[h] = v
  }
  return t
})()

/** Convert little-endian float16 data to Float32Array. */
export function f16ToF32(buf: ArrayBuffer, byteOffset = 0, count?: number): Float32Array {
  const n = count ?? (buf.byteLength - byteOffset) / 2
  const u16 = new Uint16Array(buf, byteOffset, n)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = F16_TABLE[u16[i]]
  return out
}
