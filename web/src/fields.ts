import { b64ToBuffer } from './api'
import { fanSpeed, S_MIN_REF, type BankLevel } from './scenario'
import type { BankPkg, Q8 } from './types'

/** A bank's 3D fields, decoded, per level: air velocity (m/s, [3, X, Y, Z] flattened), air temperature rise per
 * watt of CPU power and per watt of GPU power (K/W, [X, Y, Z]), dust deposits (mm). */
export interface Fields { u: Float32Array[]; kc: Float32Array[]; kg: Float32Array[]; dust: number[][][] }

export function decodeQ8(q: Q8): Float32Array {
  const bytes = new Uint8Array(b64ToBuffer(q.b64))
  const out = new Float32Array(bytes.length)
  if (q.signed) {
    const k = q.scale / 127
    for (let i = 0; i < bytes.length; i++) out[i] = (bytes[i] - 128) * k
  } else {
    const k = q.scale / 255
    for (let i = 0; i < bytes.length; i++) out[i] = bytes[i] * k
  }
  return out
}

const cache = new Map<string, Fields>()
/** Decoded fields of a bank (decoded once per bank). */
export function bankFields(b: BankPkg | null | undefined): Fields | null {
  if (!b?.fields) return null
  let f = cache.get(b.key)
  if (!f) {
    f = { u: b.fields.map((x) => decodeQ8(x.u)), kc: b.fields.map((x) => decodeQ8(x.kc)), kg: b.fields.map((x) => decodeQ8(x.kg)), dust: b.fields.map((x) => x.dust_points) }
    cache.set(b.key, f)
  }
  return f
}

/** The two levels and their weights at a case-fan duty, exactly as bankState (scenario.ts) interpolates the
 * per-watt rises: linear in 1 / speed, extrapolated past the ends. */
export function levelWeights(levels: BankLevel[], duty: number): { a: number; b: number; wa: number; wb: number } {
  const n = levels.length
  if (n === 1) return { a: 0, b: 0, wa: 1, wb: 0 }
  const inv = levels.map((L) => 1 / fanSpeed(L.duty ?? 1, S_MIN_REF)).reverse()      // increasing
  const x = 1 / fanSpeed(duty, S_MIN_REF)
  let i = 0
  while (i < n - 2 && x > inv[i + 1]) i++
  const f = (x - inv[i]) / (inv[i + 1] - inv[i])
  return { a: n - 1 - i, b: n - 2 - i, wa: 1 - f, wb: f }
}

/** The level whose case-fan speed is closest to the speed at this duty (for the air velocity the 3D view draws). */
export function nearestLevel(levels: BankLevel[], duty: number): number {
  const s = fanSpeed(duty, S_MIN_REF)
  let best = 0
  levels.forEach((L, i) => { if (Math.abs(fanSpeed(L.duty ?? 1, S_MIN_REF) - s) < Math.abs(fanSpeed(levels[best].duty ?? 1, S_MIN_REF) - s)) best = i })
  return best
}

/** Air temperature rise (K above the room) everywhere, for a CPU and GPU power at a case-fan duty. */
export function composeT(f: Fields, levels: BankLevel[], duty: number, pc: number, pg: number, out?: Float32Array): Float32Array {
  const { a, b, wa, wb } = levelWeights(levels, duty)
  const ca = f.kc[a], ga = f.kg[a], cb = f.kc[b], gb = f.kg[b]
  const n = ca.length
  const T = out && out.length === n ? out : new Float32Array(n)
  const ac = wa * pc, ag = wa * pg, bc = wb * pc, bg = wb * pg
  for (let i = 0; i < n; i++) {
    const t = ac * ca[i] + ag * ga[i] + bc * cb[i] + bg * gb[i]
    T[i] = t > 0 ? t : 0
  }
  return T
}
