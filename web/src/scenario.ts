/** Fan curves and the steady state of a build in a scenario (room temperature, CPU power, GPU power).
 * A line-by-line port of engine/scenario.py, where the model is explained; tests/test_scenario.py runs both on the
 * same inputs and checks that they agree. No imports, so Node can run this file directly.
 *
 *   part temperature = room + intake air rise + P / G(fan speed),   G = 1 / (r_base + r_air_full * speed^-0.7)
 *
 * The intake air rise comes from the build's bank: the rise per watt of CPU and of GPU power at a few case-fan
 * duties, interpolated in 1 / speed. Units: °C, W, m^3/s; duties and speeds are fractions 0..1. */

export const H_EXP = 0.7
export const S_MIN_REF = 0.2
export const CPU_FAN_FLOOR = 0.05
const BISECT_ITERS = 50
const OUTER_ITERS = 60

export type Pt = [number, number]
export type Source = 'cpu' | 'gpu' | 'max'
export interface Curve { source?: Source; points: Pt[] }
export interface Curves { case: Curve; cpu: Curve; gpu: Curve }
export type Pair = [number, number]
export interface DustStats { dust_entering_rel: number; deposited_frac: number; d_um: number; by_part: { name: string; share: number }[] }
export interface BankFan { slot: string; q_m3s: number; op_frac?: number; duty?: number; speed?: number; rpm?: number }
export interface BankLevel {
  duty: number | null
  k_cpu_in: Pair; k_gpu_in: Pair; k_air_mean: Pair; k_air_max: Pair
  flow_in_m3s: number; flow_out_m3s: number; case_pa: number
  fans: BankFan[]
  pressure?: string; dead_air_fraction?: number; dust?: DustStats | null
}
export interface Bank { levels: BankLevel[] }
export interface SetupFan { slot: string; curve: boolean; pct: number; s_min: number; noise: number; rpm_max: number }
export interface Setup {
  cpu: { r_base: number; t_max: number; idle_w: number; max_w: number }
  cooler: { r_air_full: number; s_min: number; fans: number; noise: number; rpm_max: number }
  gpu: { r_base: number; r_air_full: number; floor: number; t_max: number; idle_w: number; max_w: number; fans: number; noise: number; rpm_max: number }
  case_fans: SetupFan[]
  curves: Curves
}

// ----------------------------------------------------------------------------- fans and coolers
export function curveDuty(points: Pt[], t: number): number {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (t <= pts[0][0]) return pts[0][1] / 100
  for (let i = 0; i < pts.length - 1; i++) {
    const [t0, d0] = pts[i], [t1, d1] = pts[i + 1]
    if (t <= t1) {
      const f = t1 > t0 ? (t - t0) / (t1 - t0) : 1
      return (d0 + f * (d1 - d0)) / 100
    }
  }
  return pts[pts.length - 1][1] / 100
}

export const fanSpeed = (duty: number, sMin: number) => sMin + (1 - sMin) * Math.min(Math.max(duty, 0), 1)
export const gpuFanSpeed = (duty: number) => Math.min(Math.max(duty, 0), 1)
export const conductance = (rBase: number, rAirFull: number, speed: number, floor: number) =>
  1 / (rBase + rAirFull * Math.pow(Math.max(speed, floor), -H_EXP))

export function noiseDba(sources: [number, number, number][]): number {
  let e = 0
  for (const [lw, s, n] of sources) if (s > 1e-3 && n > 0) e += n * Math.pow(10, (lw + 50 * Math.log10(s)) / 10)
  return e > 0 ? 10 * Math.log10(e) : 0
}

// ----------------------------------------------------------------------------- bank interpolation
const RISE_KEYS = ['k_cpu_in', 'k_gpu_in', 'k_air_mean', 'k_air_max'] as const
const FLOW_KEYS = ['flow_in_m3s', 'flow_out_m3s', 'case_pa'] as const
export interface BankState {
  k_cpu_in: Pair; k_gpu_in: Pair; k_air_mean: Pair; k_air_max: Pair
  flow_in_m3s: number; flow_out_m3s: number; case_pa: number
  fans: { slot: string; q_m3s: number }[]
}

function interp(xs: number[], ys: number[], x: number): number {
  if (xs.length === 1) return ys[0]
  let i = 0
  while (i < xs.length - 2 && x > xs[i + 1]) i++
  const f = (x - xs[i]) / (xs[i + 1] - xs[i])
  return ys[i] + f * (ys[i + 1] - ys[i])
}

export function bankState(bank: Bank, duty: number): BankState {
  const lv = bank.levels
  if (lv.length === 1) return lv[0]
  const s = lv.map((L) => fanSpeed(L.duty ?? 1, S_MIN_REF))
  const sn = fanSpeed(duty, S_MIN_REF)
  const inv = s.map((v) => 1 / v).reverse()
  const out = {} as BankState
  for (const k of RISE_KEYS) {
    const pair = [0, 1].map((j) => Math.max(interp(inv, lv.map((L) => L[k][j]).reverse(), 1 / sn), 0))
    out[k] = [pair[0], pair[1]]
  }
  for (const k of FLOW_KEYS) out[k] = interp(s, lv.map((L) => L[k]), sn)
  out.fans = lv[0].fans.map((f, i) => ({ slot: f.slot, q_m3s: Math.max(interp(s, lv.map((L) => L.fans[i].q_m3s), sn), 0) }))
  return out
}

// ----------------------------------------------------------------------------- setup from a build
interface LibPart { id: string; [k: string]: unknown }
export interface ScenarioLib {
  cpus: LibPart[]; gpus: LibPart[]; coolers: LibPart[]; fans: LibPart[]
  defaults: { curves: { case: Curve; cpu: Curve } }
}
export interface ScenarioBuild {
  cpu?: string; gpu: string; cooler: string
  fans: { slot: string; fan: string; mode: string; control?: 'curve' | 'fixed'; pct?: number }[]
  curves?: Partial<Curves>
}

export function setupFor(build: ScenarioBuild, lib: ScenarioLib): Setup {
  const find = (cat: 'cpus' | 'gpus' | 'coolers' | 'fans', id: string) => {
    const p = lib[cat].find((x) => x.id === id)
    if (!p) throw new Error(`${cat}:${id} not in the library`)
    return p as Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
  }
  const cpu = find('cpus', build.cpu ?? lib.cpus[0].id)
  const cooler = find('coolers', build.cooler)
  const cfan = find('fans', cooler.fans[0].fan)
  const gpu = find('gpus', build.gpu)
  const gc = gpu.cooling
  const d = lib.defaults.curves
  const cv = build.curves ?? {}
  const caseC = cv.case ?? d.case
  const curves: Curves = {
    case: { source: caseC.source ?? 'cpu', points: caseC.points },
    cpu: { points: (cv.cpu ?? d.cpu).points },
    gpu: { points: (cv.gpu ?? { points: gc.fan_curve as Pt[] }).points },
  }
  const fans: SetupFan[] = []
  for (const f of build.fans) {
    if (f.mode !== 'in' && f.mode !== 'out') continue
    const p = find('fans', f.fan)
    fans.push({ slot: f.slot, curve: (f.control ?? 'curve') === 'curve', pct: f.pct ?? 100, s_min: p.s_min ?? S_MIN_REF, noise: p.noise_dba ?? 30, rpm_max: p.rpm_max })
  }
  return {
    cpu: { r_base: cpu.r_die_k_per_w, t_max: cpu.t_max_c, idle_w: cpu.idle_w, max_w: cpu.max_w },
    cooler: { r_air_full: cooler.r_air_full_k_per_w, s_min: cfan.s_min ?? S_MIN_REF,
      fans: (cooler.fans as { count?: number }[]).reduce((a, x) => a + (x.count ?? 1), 0), noise: cfan.noise_dba ?? 30, rpm_max: cfan.rpm_max },
    gpu: { r_base: gc.r_base_k_per_w, r_air_full: gc.r_air_full_k_per_w, floor: gc.passive_speed, t_max: gpu.t_max_c, idle_w: gpu.idle_w,
      max_w: gpu.tdp_w, fans: gc.fans, noise: gc.fan_noise_dba, rpm_max: gc.fan_rpm_max },
    case_fans: fans,
    curves,
  }
}

export function loadWatts(su: Setup, cpuFrac: number, gpuFrac: number): [number, number] {
  const c = su.cpu, g = su.gpu
  return [c.idle_w + cpuFrac * (c.max_w - c.idle_w), g.idle_w + gpuFrac * (g.max_w - g.idle_w)]
}

// ----------------------------------------------------------------------------- the coupled solve
interface Implied {
  tc: number; tg: number; b: BankState; d_case: number; d_cpu: number; s_cpu: number; d_gpu: number; s_gpu: number
  g_cpu: number; g_gpu: number; rise_c: number; rise_g: number
}

function implied(bank: Bank, su: Setup, amb: number, pc: number, pg: number, tc: number, tg: number): Implied {
  const cv = su.curves
  const src = cv.case.source ?? 'cpu'
  const tSrc = src === 'cpu' ? tc : src === 'gpu' ? tg : Math.max(tc, tg)
  const dCase = curveDuty(cv.case.points, tSrc)
  const b = bankState(bank, dCase)
  const dCpu = curveDuty(cv.cpu.points, tc)
  const sCpu = fanSpeed(dCpu, su.cooler.s_min)
  const dGpu = curveDuty(cv.gpu.points, tg)
  const sGpu = gpuFanSpeed(dGpu)
  const gCpu = conductance(su.cpu.r_base, su.cooler.r_air_full, sCpu, CPU_FAN_FLOOR)
  const gGpu = conductance(su.gpu.r_base, su.gpu.r_air_full, sGpu, su.gpu.floor)
  const riseC = b.k_cpu_in[0] * pc + b.k_cpu_in[1] * pg
  const riseG = b.k_gpu_in[0] * pc + b.k_gpu_in[1] * pg
  return { tc: amb + riseC + pc / gCpu, tg: amb + riseG + pg / gGpu, b, d_case: dCase, d_cpu: dCpu, s_cpu: sCpu, d_gpu: dGpu,
    s_gpu: sGpu, g_cpu: gCpu, g_gpu: gGpu, rise_c: riseC, rise_g: riseG }
}

function bisect(fn: (x: number) => number, lo: number, hi: number): number {
  if (fn(lo) <= 0) return lo
  if (fn(hi) >= 0) return hi
  for (let i = 0; i < BISECT_ITERS; i++) {
    const mid = 0.5 * (lo + hi)
    if (fn(mid) > 0) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}

export function equilibrium(bank: Bank, su: Setup, amb: number, pc: number, pg: number): [number, number] {
  let tc = amb + 20, tg = amb + 20
  const hi = amb + 400
  for (let i = 0; i < OUTER_ITERS; i++) {
    const tg0 = tg
    const tcN = bisect((t) => implied(bank, su, amb, pc, pg, t, tg0).tc - t, amb, hi)
    const tgN = bisect((t) => implied(bank, su, amb, pc, pg, tcN, t).tg - t, amb, hi)
    const done = Math.abs(tcN - tc) < 1e-5 && Math.abs(tgN - tg) < 1e-5
    tc = tcN
    tg = tgN
    if (done) break
  }
  return [tc, tg]
}

function throttle(bank: Bank, su: Setup, amb: number, pc: number, pg: number, part: 'cpu' | 'gpu'): number {
  const lim = su[part].t_max
  const idx = part === 'cpu' ? 0 : 1
  const over = (p: number) => lim - equilibrium(bank, su, amb, part === 'cpu' ? p : pc, part === 'gpu' ? p : pg)[idx]
  return bisect(over, 0, part === 'cpu' ? pc : pg)
}

export interface PartState {
  temp_c: number; watts: number; watts_set: number; throttled: boolean; intake_rise_c: number; g_w_per_k: number
  fan_duty: number; fan_speed: number; fan_rpm: number; t_max_c: number
}
export interface CaseFanState { slot: string; curve: boolean; duty: number; speed: number; rpm: number; q_m3s: number }
export interface ScenarioResult {
  ambient_c: number; cpu: PartState; gpu: PartState; case_duty: number; fans: CaseFanState[]
  flow_in_m3s: number; flow_out_m3s: number; case_pa: number; air_mean_rise_c: number; air_max_rise_c: number; noise_dba: number
}

export function solve(bank: Bank, su: Setup, ambientC: number, cpuW: number, gpuW: number): ScenarioResult {
  const amb = ambientC
  let pc = cpuW, pg = gpuW
  let [tc, tg] = equilibrium(bank, su, amb, pc, pg)
  const thr = { cpu: false, gpu: false }
  for (let r = 0; r < 3; r++) {
    let changed = false
    if (tc > su.cpu.t_max + 1e-3) {
      pc = throttle(bank, su, amb, pc, pg, 'cpu'); thr.cpu = true; changed = true;
      [tc, tg] = equilibrium(bank, su, amb, pc, pg)
    }
    if (tg > su.gpu.t_max + 1e-3) {
      pg = throttle(bank, su, amb, pc, pg, 'gpu'); thr.gpu = true; changed = true;
      [tc, tg] = equilibrium(bank, su, amb, pc, pg)
    }
    if (!changed) break
  }
  const s = implied(bank, su, amb, pc, pg, tc, tg)
  const b = s.b
  const fans: CaseFanState[] = su.case_fans.map((f) => {
    const duty = f.curve ? s.d_case : f.pct / 100
    const sp = fanSpeed(duty, f.s_min)
    const q = b.fans.find((x) => x.slot === f.slot)?.q_m3s ?? 0
    return { slot: f.slot, curve: f.curve, duty, speed: sp, rpm: sp * f.rpm_max, q_m3s: q }
  })
  const co = su.cooler, gp = su.gpu
  const noise = noiseDba([
    ...su.case_fans.map((f, i): [number, number, number] => [f.noise, fans[i].speed, 1]),
    [co.noise, s.s_cpu, co.fans], [gp.noise, s.s_gpu, gp.fans],
  ])
  return {
    ambient_c: amb,
    cpu: { temp_c: tc, watts: pc, watts_set: cpuW, throttled: thr.cpu, intake_rise_c: s.rise_c, g_w_per_k: s.g_cpu,
      fan_duty: s.d_cpu, fan_speed: s.s_cpu, fan_rpm: s.s_cpu * co.rpm_max, t_max_c: su.cpu.t_max },
    gpu: { temp_c: tg, watts: pg, watts_set: gpuW, throttled: thr.gpu, intake_rise_c: s.rise_g, g_w_per_k: s.g_gpu,
      fan_duty: s.d_gpu, fan_speed: s.s_gpu, fan_rpm: s.s_gpu * gp.rpm_max, t_max_c: su.gpu.t_max },
    case_duty: s.d_case, fans,
    flow_in_m3s: b.flow_in_m3s, flow_out_m3s: b.flow_out_m3s, case_pa: b.case_pa,
    air_mean_rise_c: b.k_air_mean[0] * pc + b.k_air_mean[1] * pg,
    air_max_rise_c: b.k_air_max[0] * pc + b.k_air_max[1] * pg,
    noise_dba: noise,
  }
}
