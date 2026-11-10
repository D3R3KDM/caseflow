/** The neural surrogate on the web side: its model card, and the whole design space it predicted
 * (web/public/data/ai_space.json, written by scripts/ai_space.py).
 *
 * A layout code has one digit per mount (0 off, 1 in, 2 out) in the order of AiSpace.mounts; each mount carries the
 * fan model the network was trained with there. */
import type { Bank, BankLevel } from './scenario'
import type { Build, FanSlot, Mode } from './types'

export interface AiTest {
  layouts: number; cpu_mae_c: number; gpu_mae_c: number; cpu_max_c: number; gpu_max_c: number
  u_rmse_ms?: number; k_rmse_kw?: number; intake_mae_kw?: number; intake_rel?: number
}
export interface AiInfo {
  name: string; params: number
  config?: { base: number; depth: number }
  trained: { layouts: number; samples: number; epochs: number; minutes: number; date: string }
  test: AiTest | null
  baseline?: { cpu_mae_c: number; gpu_mae_c: number }
}
export interface AiRow { c: string; L: (number | number[])[][] }
export interface AiSpace {
  version: number; model: AiInfo; mounts: string[]; mount_fan: Record<string, string>; levels: number[]; seconds: number; rows: AiRow[]
}
/** Where a predicted bank's layout stands for the network. */
export type Seen = 'train' | 'test' | 'new'

const MODES: Mode[] = ['off', 'in', 'out']

export function buildOfCode(space: Pick<AiSpace, 'mounts' | 'mount_fan'>, base: Build, code: string): Build {
  const fans: FanSlot[] = []
  space.mounts.forEach((slot, i) => {
    const m = MODES[+code[i]]
    if (m !== 'off') fans.push({ slot, fan: space.mount_fan[slot], mode: m, control: 'curve', pct: 70 })
  })
  return { ...base, fans }
}

/** The layout code of a build, or null if the network was not trained on builds like it. */
export function codeOf(space: Pick<AiSpace, 'mounts' | 'mount_fan'>, b: Build): string | null {
  let code = ''
  for (const slot of space.mounts) {
    const f = b.fans.find((x) => x.slot === slot && x.mode !== 'off')
    if (!f) { code += '0'; continue }
    if (f.fan !== space.mount_fan[slot] || (f.control ?? 'curve') !== 'curve') return null
    code += f.mode === 'in' ? '1' : '2'
  }
  return b.fans.every((f) => f.mode === 'off' || space.mounts.includes(f.slot)) ? code : null
}

export const fanCount = (code: string) => [...code].filter((c) => c !== '0').length

/** A predicted row as a bank the scenario solver takes (no dust: the network does not predict it). */
export function bankOfRow(space: Pick<AiSpace, 'mounts' | 'levels'>, row: AiRow): Bank {
  const slots = space.mounts.filter((_, i) => row.c[i] !== '0')
  const levels: BankLevel[] = row.L.map((v, j) => {
    const n = v as number[], q = v[11] as number[]
    return {
      duty: slots.length ? space.levels[j] : null,
      k_cpu_in: [n[0], n[1]], k_gpu_in: [n[2], n[3]], k_air_mean: [n[4], n[5]], k_air_max: [n[6], n[7]],
      flow_in_m3s: n[8], flow_out_m3s: n[9], case_pa: n[10], fans: slots.map((slot, k) => ({ slot, q_m3s: q[k] })), dust: null,
    }
  })
  return { levels }
}

/** A code at a glance: "F▸▸  B▸▸  T·◂◂  R◂" (front low/high, bottom front/rear, top front/mid/rear, rear;
 * ▸ in, ◂ out, · off), or in words with long = true. */
export function codeWords(space: Pick<AiSpace, 'mounts'>, code: string, long = false): string {
  const groups: [string, string, string[]][] = [['F', 'front', ['front_low', 'front_high']], ['B', 'bottom', ['bottom_front', 'bottom_rear']],
    ['T', 'top', ['top_1', 'top_2', 'top_3']], ['R', 'rear', ['rear']]]
  const sym = (c: string) => (c === '1' ? '▸' : c === '2' ? '◂' : '·')
  return groups.map(([s, w, slots]) => `${long ? w + ' ' : s}${slots.map((x) => sym(code[space.mounts.indexOf(x)] ?? '0')).join('')}`).join(long ? ' · ' : '  ')
}
