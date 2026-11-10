/// <reference lib="webworker" />
/** Ranks the whole predicted design space (6561 layouts) for one scenario off the main thread, so the sliders stay
 * smooth. A newer request makes the running one stop at its next pause. */
import { bankOfRow, buildOfCode, type AiSpace } from './ai'
import { loadWatts, setupFor, solve, type Bank, type Curves, type ScenarioBuild, type ScenarioLib } from './scenario'
import type { Build } from './types'

export interface RankJob { id: number; lib: ScenarioLib; base: Build; ambient_c: number; cpu: number; gpu: number; curves: Curves; cpuId: string }
export interface RankOut { id: number; cpu: Float32Array; gpu: Float32Array; noise: Float32Array; throttled: Uint8Array; ms: number }

let space: AiSpace | null = null
let banks: Bank[] = []
let loading: Promise<void> | null = null
let latest = 0

async function load() {
  const r = await fetch('/data/ai_space.json')
  if (!r.ok) throw new Error(`ai_space.json: ${r.status}`)
  space = (await r.json()) as AiSpace
  banks = space.rows.map((row) => bankOfRow(space!, row))
}

self.onmessage = async (e: MessageEvent<RankJob>) => {
  const job = e.data
  latest = job.id
  try {
    loading ??= load()
    await loading
  } catch (err) {
    self.postMessage({ id: job.id, error: String(err) })
    return
  }
  const sp = space!
  const n = sp.rows.length
  const t0 = performance.now()
  const out: RankOut = { id: job.id, cpu: new Float32Array(n), gpu: new Float32Array(n), noise: new Float32Array(n), throttled: new Uint8Array(n), ms: 0 }
  for (let i = 0; i < n; i++) {
    if (i % 400 === 399) {
      await new Promise((r) => setTimeout(r, 0))            // let a newer request in
      if (latest !== job.id) return
    }
    const b = buildOfCode(sp, job.base, sp.rows[i].c)
    const su = setupFor({ ...b, cpu: job.cpuId, curves: job.curves } as ScenarioBuild, job.lib)
    const [pc, pg] = loadWatts(su, job.cpu, job.gpu)
    const r = solve(banks[i], su, job.ambient_c, pc, pg)
    out.cpu[i] = r.cpu.temp_c
    out.gpu[i] = r.gpu.temp_c
    out.noise[i] = r.noise_dba
    out.throttled[i] = r.cpu.throttled || r.gpu.throttled ? 1 : 0
  }
  out.ms = performance.now() - t0
  self.postMessage(out, [out.cpu.buffer, out.gpu.buffer, out.noise.buffer, out.throttled.buffer])
}
