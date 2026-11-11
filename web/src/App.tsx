import { useDeferredValue, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { buildOfCode, codeOf, type AiDust, type AiInfo, type AiSpace } from './ai'
import { AiAllTab, AiHereTab, ModelCard, SEEN, type AiHere } from './AiWindow'
import type { RankJob, RankOut } from './aiWorker'
import { getJSON, pingServer, postJSON } from './api'
import CurveEditor from './CurveEditor'
import { bankFields, composeT, nearestLevel } from './fields'
import { AirPanel, CompareCard, DustPanel, Holo, PartPanel, SearchPanel, VerdictPanel, type CmpRow } from './Holo'
import HudWindow, { resetWindows } from './HudWindow'
import { THERMAL_CSS } from './palette'
import CaseScene, { CMP_SHIFT, resetHolos, T_MAX, type HoloSpec, type Layers, type Pick as FanPick, type SceneData, type SceneFan, type TwinData } from './Scene'
import TrendChart, { SERIES } from './TrendChart'
import { loadWatts, setupFor, solve, type Bank, type Curves, type ScenarioBuild, type ScenarioLib, type ScenarioResult, type Setup, type Source } from './scenario'
import { WHERE, type BankPkg, type Build, type Control, type FanSlot, type Geometry, type Lib, type Mode, type Opt, type RankRow, type Scenario, type ViewKey } from './types'

const fanKey = (fs: FanSlot[]) => fs.filter((f) => f.mode !== 'off').map((f) => `${f.slot}:${f.fan}:${f.mode}`).sort().join('|')
const partsKey = (b: Build) => `${b.case}|${b.gpu}|${b.cooler}`
/** Everything that shapes a build's airflow (and so its bank): parts, fans, and which fans follow the curve. */
const layoutId = (b: Build) => partsKey(b) + '#' + b.fans.filter((f) => f.mode !== 'off')
  .map((f) => `${f.slot}:${f.fan}:${f.mode}:${f.control ?? 'curve'}${(f.control ?? 'curve') === 'fixed' ? ':' + f.pct : ''}`).sort().join('|')
const asLib = (lib: Lib) => lib as unknown as ScenarioLib
/** Preset buttons, short enough for one row (the full name is the tooltip). */
const SHORT: Record<string, string> = { idle: 'Idle', office: 'Everyday', cpu_game: 'CPU game', gpu_game: 'GPU game', stress: 'Stress' }
const pct = (x: number) => `${Math.round(x * 100)}%`
const signed = (v: number) => {
  const r = Math.round(v * 10) / 10
  return `${r > 0 ? '+' : r < 0 ? '−' : '±'}${Math.abs(r).toFixed(1)}`
}
/** Short part names for the selects: "Stock 140 mm", "Generic 120 mm", "RTX 5080 Founders Edition". */
const fanName = (name: string) => name
  .replace(/^Lian Li LANCOOL 207 (front|bottom) fan \((\d+) mm.*\)$/, (_m, w: string, mm: string) => `Stock ${mm} mm${w === 'bottom' ? ' rev' : ''}`)
  .replace(/^(\d+) mm fan, model unknown.*$/, 'Generic $1 mm')
  .replace(/ \(\d+ mm\)$/, '')
const partName = (name: string) => name.replace(/^NVIDIA GeForce /, '').replace(/ ARGB$/, '')
/** Layouts whose full banks (3D fields, ~30-40 MB each in the page) stay in memory; older ones are dropped and
 * fetched again from the engine's cache if opened again, so browsing many layouts cannot exhaust the page's memory. */
const KEEP = 8
function keepRecent<T>(m: Record<string, T>, id: string, v: T, pinned: string | null): Record<string, T> {
  const entries = Object.entries(m).filter(([k]) => k !== id)
  entries.push([id, v])
  while (entries.length > KEEP) {
    const i = entries.findIndex(([k]) => k !== pinned && k !== id)
    if (i < 0) break
    entries.splice(i, 1)
  }
  return Object.fromEntries(entries)
}
/** The engine's reason for not predicting a build (FastAPI puts it in "detail"). */
const aiWhy = (e: unknown) => {
  const s = String(e)
  const m = s.match(/"detail":"([^"]+)"/)
  return m ? m[1] : s.includes(': 503') ? 'no trained network yet' : 'the engine did not answer'
}
/** The network's bank with the dust tracked in its airflow merged in (the dust comes from a separate request). */
const withDust = (p: BankPkg | undefined, d: AiDust | undefined): BankPkg | undefined =>
  p && d && d.levels.length === p.levels.length ? { ...p, levels: p.levels.map((L, i) => ({ ...L, dust: d.levels[i].dust })) } : p
/** How far apart the AI and the simulation are: under 0.5 °C reads as agreement (the network's test error is 0.3-0.4). */
const tempDiff = (a: number | null | undefined, s: number | null | undefined) =>
  a == null || s == null ? null : { text: `${signed(a - s)} °C`, cls: Math.abs(a - s) < 0.5 ? 'ok' : Math.abs(a - s) < 1.5 ? 'meh' : 'bad' }

/** Plain-language steps that turn one fan layout into another. */
function changes(from: FanSlot[], to: FanSlot[]): string[] {
  const A = new Map(from.map((f) => [f.slot, f])), B = new Map(to.map((f) => [f.slot, f]))
  const removed = from.filter((f) => !B.has(f.slot)), added = to.filter((f) => !A.has(f.slot))
  const verb = (m: Mode) => (m === 'in' ? 'blowing in' : 'blowing out')
  const out: string[] = []
  for (const f of to) {
    const a = A.get(f.slot)
    if (a && a.mode !== f.mode) out.push(`Flip the ${WHERE[f.slot] ?? f.slot} fan so it is ${verb(f.mode)}`)
  }
  added.forEach((f, i) => {
    const s = removed[i]
    out.push(s ? `Move the ${WHERE[s.slot] ?? s.slot} fan to ${WHERE[f.slot] ?? f.slot}, ${verb(f.mode)}` : `Add a fan at ${WHERE[f.slot] ?? f.slot}, ${verb(f.mode)}`)
  })
  removed.slice(added.length).forEach((f) => out.push(`Remove the ${WHERE[f.slot] ?? f.slot} fan`))
  return out
}

function defaultCurves(lib: Lib, gpuId: string): Curves {
  const g = lib.gpus.find((x) => x.id === gpuId) ?? lib.gpus[0]
  const d = lib.defaults.curves
  return { case: { source: d.case.source ?? 'cpu', points: d.case.points.map((p) => [...p] as [number, number]) },
    cpu: { points: d.cpu.points.map((p) => [...p] as [number, number]) }, gpu: { points: g.cooling.fan_curve.map((p) => [...p] as [number, number]) } }
}

/** Every searched layout's steady state in one scenario, best score first (engine/optimize.py ranks the same way). */
function rankLayouts(opt: Opt, lib: Lib, sc: Scenario, curves: Curves, cpu: string): RankRow[] {
  const rows = opt.layouts.map((L) => {
    const su = setupFor({ ...L.build, cpu, curves } as ScenarioBuild, asLib(lib))
    const [pc, pg] = loadWatts(su, sc.cpu, sc.gpu)
    const r = solve(L.bank, su, sc.ambient_c, pc, pg)
    return { name: L.name, build: L.build, current: L.current, cpu_c: r.cpu.temp_c, gpu_c: r.gpu.temp_c, throttled: r.cpu.throttled || r.gpu.throttled,
      noise_dba: r.noise_dba, case_duty: r.case_duty, cfm_in: r.flow_in_m3s * 2119, dust: r.dust_rate ?? 0, dust_rel: 1, score: 0, tied: false }
  })
  const ref = rows.find((x) => x.current)?.dust || Math.max(...rows.map((x) => x.dust)) || 1
  for (const x of rows) {
    x.dust_rel = x.dust / ref
    x.score = x.cpu_c + x.gpu_c + opt.dust_weight * x.dust_rel
  }
  rows.sort((a, b) => a.score - b.score)
  const top = rows[0]?.score ?? 0
  for (const x of rows) x.tied = x.score - top < opt.tie_margin
  return rows
}

/** The same setup with every fan held at its current duty (flat curves): what the fans would do if they did not react. */
function heldSetup(su: Setup, r: ScenarioResult): Setup {
  const flat = (d: number): [number, number][] => [[0, d * 100], [100, d * 100]]
  return { ...su, curves: { case: { source: 'cpu', points: flat(r.case_duty) }, cpu: { points: flat(r.cpu.fan_duty) }, gpu: { points: flat(r.gpu.fan_duty) } } }
}

function useNarrow(px = 900) {
  const q = `(max-width: ${px}px)`
  const [m, setM] = useState(() => window.matchMedia(q).matches)
  useEffect(() => {
    const mq = window.matchMedia(q)
    const on = () => setM(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [q])
  return m
}

/** A labelled slider with its value. */
function Slider({ label, value, min, max, step, show, onChange }: { label: string; value: number; min: number; max: number; step: number; show: ReactNode; onChange: (v: number) => void }) {
  return (
    <label className="sl">
      <span className="sl-h"><span className="lbl">{label}</span><span className="mono sl-v">{show}</span></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  )
}

export default function App() {
  const [online, setOnline] = useState(false)
  const [pinged, setPinged] = useState(false)                        // the first health check has answered
  const [lib, setLib] = useState<Lib | null>(null)
  const [opt, setOpt] = useState<Opt | null>(null)
  const [home, setHome] = useState<Build | null>(null)                // the user's own layout
  const [banks, setBanks] = useState<Record<string, BankPkg>>({})     // simulated banks (with 3D fields) by layoutId
  const [view, setView] = useState<ViewKey>('current')
  const [custom, setCustom] = useState<Build | null>(null)
  const [scn, setScn] = useState<Scenario>({ ambient_c: 23, cpu: 0.97, gpu: 0.38, preset: 'cpu_game' })
  const [curves, setCurves] = useState<Curves | null>(null)
  const [cpuId, setCpuId] = useState('amd-ryzen-7-9700x')
  const [layers, setLayers] = useState<Layers>({ flow: true, lines: false, dust: false, heat: false, panels: true })
  const [win, setWin] = useState({ scn: true, curve: false, trend: false, cfg: false, info: false, ai: false, rank: false })
  const [sweep, setSweep] = useState<'gpu' | 'cpu'>('gpu')
  const [curveTab, setCurveTab] = useState<'case' | 'cpu' | 'gpu'>('case')
  const [allRows, setAllRows] = useState(false)
  const [jobs, setJobs] = useState<Record<string, number>>({})        // simulations running on the GPU: progress by layoutId
  const [cmp, setCmp] = useState<string | null>(null)                  // the layout shown side by side with its simulation
  const [scanned, setScanned] = useState<string | null>(null)          // ... once the scan has drawn the simulation's case
  const [msg, setMsg] = useState<string | null>(null)
  const [layoutKey, setLayoutKey] = useState(0)
  const [menu, setMenu] = useState<FanPick | null>(null)
  const [ai, setAi] = useState<Record<string, BankPkg | { error: string; final: boolean }>>({})   // the network's banks by layoutId
  const [aiDust, setAiDust] = useState<Record<string, AiDust | { error: string }>>({})            // dust in the network's airflow
  const [aiInfo, setAiInfo] = useState<AiInfo | null>(null)
  const [space, setSpace] = useState<AiSpace | null>(null)                         // every layout of the 8 mounts, predicted
  const [ranks, setRanks] = useState<RankOut | null>(null)
  const [aiTab, setAiTab] = useState<'here' | 'all'>('here')
  const [aiFans, setAiFans] = useState<number | null>(null)
  const [aiHeld, setAiHeld] = useState(false)                                        // the AI list shows only the held-out layouts
  const [missing, setMissing] = useState<Record<string, true>>({})                  // layouts the engine has never simulated
  const worker = useRef<Worker | null>(null)
  const rankId = useRef(0)
  const asked = useRef(new Set<string>())          // requests on their way (AI, AI dust, banks, simulations): none is sent twice
  const narrow = useNarrow()
  const menuRef = useRef(menu)
  menuRef.current = menu
  useEffect(() => {
    const esc = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (menuRef.current) setMenu(null)
      else setCmp(null)                                            // Esc also ends the side-by-side comparison
    }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [])

  // ---------- data ----------
  useEffect(() => {
    let alive = true
    const ping = () => pingServer().then((ok) => { if (alive) { setOnline(ok); setPinged(true) } })
    ping()
    const id = window.setInterval(ping, 5000)
    return () => { alive = false; window.clearInterval(id) }
  }, [])
  useEffect(() => {
    getJSON<Opt>('/data/optimize.json').then((o) => { if (o.version >= 7 && o.layouts) setOpt(o) }).catch(() => setMsg('Layout search not found - run engine/optimize.py and scripts/precompute.py'))
    getJSON<Build>('/data/default_build.json').then((b) => setHome((h) => h ?? b)).catch(() => {})
  }, [])
  useEffect(() => {
    getJSON<Lib>(online ? '/api/library' : '/data/library.json').catch(() => getJSON<Lib>('/data/library.json')).then(setLib).catch(() => {})
    if (online) getJSON<Build>('/api/default_build').then(setHome).catch(() => {})
  }, [online])
  useEffect(() => {
    if (online) getJSON<AiInfo>('/api/ai/info').then(setAiInfo).catch(() => {})
  }, [online])
  useEffect(() => {
    if (!win.ai || aiTab !== 'all' || space) return
    getJSON<AiSpace>('/data/ai_space.json').then((sp) => { setSpace(sp); setAiInfo((i) => i ?? sp.model) })
      .catch(() => setMsg('No predicted design space yet - run scripts/ai_space.py'))
  }, [win.ai, aiTab, space])
  useEffect(() => () => worker.current?.terminate(), [])
  useEffect(() => {
    if (!lib || !home || curves) return
    setCurves(defaultCurves(lib, home.gpu))
    if (home.cpu) setCpuId(home.cpu)
  }, [lib, home, curves])
  useEffect(() => {
    if (!msg) return
    const id = window.setTimeout(() => setMsg(null), 6000)
    return () => window.clearTimeout(id)
  }, [msg])

  // ---------- the ranking for the current scenario (deferred: sliders stay smooth) ----------
  const scnD = useDeferredValue(scn)
  const curvesD = useDeferredValue(curves)
  const rows = useMemo(() => (opt && lib && curvesD ? rankLayouts(opt, lib, scnD, curvesD, cpuId) : []), [opt, lib, scnD, curvesD, cpuId])
  const n = rows.length
  const rankOf = (name?: string | null) => (name ? rows.findIndex((r) => r.name === name) + 1 : 0)
  const curRow = rows.find((r) => r.current)
  const topRow = rows[0]
  const recRow = curRow?.tied ? curRow : topRow                    // never recommend a change for less than the tie margin
  const bestIsCur = !!curRow && recRow?.name === curRow.name
  const cpuRow = rows.length ? rows.reduce((a, b) => (b.cpu_c < a.cpu_c ? b : a)) : undefined
  const worstRow = rows[n - 1]
  const homeBuild = curRow?.build ?? home
  const buildOf = (k: ViewKey): Build | null | undefined =>
    k === 'custom' ? custom : k === 'current' ? homeBuild : k === 'best' ? recRow?.build : k === 'cpu' ? cpuRow?.build : worstRow?.build
  const viewBuild = buildOf(view) ?? homeBuild ?? null
  useEffect(() => {
    if (!win.ai || aiTab !== 'all' || !space || !lib || !curvesD || !homeBuild) return
    if (!worker.current) {
      worker.current = new Worker(new URL('./aiWorker.ts', import.meta.url), { type: 'module' })
      worker.current.onmessage = (e: MessageEvent<RankOut & { error?: string }>) => {
        if (e.data.error) setMsg(e.data.error)
        else if (e.data.id === rankId.current) setRanks(e.data)
      }
    }
    const job: RankJob = { id: ++rankId.current, lib: asLib(lib), base: homeBuild, ambient_c: scnD.ambient_c, cpu: scnD.cpu, gpu: scnD.gpu, curves: curvesD, cpuId }
    worker.current.postMessage(job)
  }, [win.ai, aiTab, space, lib, curvesD, scnD, cpuId, homeBuild])

  const viewId = viewBuild ? layoutId(viewBuild) : null
  const homeId = homeBuild ? layoutId(homeBuild) : null
  const aiOk = (id: string | null) => { const e = id ? ai[id] : undefined; return e && !('error' in e) ? e : undefined }
  const dustOk = (id: string | null) => { const e = id ? aiDust[id] : undefined; return e && !('error' in e) ? e : undefined }

  // ---------- the network's prediction: what the app shows, for the layout on screen (and yours, to compare against) ----------
  useEffect(() => {
    if (!online) return
    for (const [id, b] of [[viewId, viewBuild], [homeId, homeBuild]] as const) {
      const had = id ? ai[id] : undefined
      if (!id || !b || (had && (!('error' in had) || had.final)) || asked.current.has('ai:' + id)) continue
      asked.current.add('ai:' + id)
      postJSON<BankPkg>('/api/ai', { build: { ...b, cpu: cpuId } })
        .then((p) => setAi((m) => keepRecent(m, id, p, homeId)))
        .catch((e) => setAi((m) => keepRecent(m, id, { error: aiWhy(e), final: String(e).includes(': 422') }, homeId)))   // only a refusal is final
        .finally(() => asked.current.delete('ai:' + id))
    }
  }, [viewId, homeId, online]) // eslint-disable-line react-hooks/exhaustive-deps
  // the dust in the network's airflow takes a few seconds on the GPU: asked for only while the dust is shown
  useEffect(() => {
    if (!online || !layers.dust) return
    for (const [id, b] of [[viewId, viewBuild], [homeId, homeBuild]] as const) {
      if (!id || !b || !aiOk(id) || aiDust[id] || asked.current.has('dust:' + id)) continue
      asked.current.add('dust:' + id)
      postJSON<AiDust>('/api/ai/dust', { build: { ...b, cpu: cpuId } })
        .then((d) => setAiDust((m) => keepRecent(m, id, d, homeId)))
        .catch((e) => setAiDust((m) => keepRecent(m, id, { error: aiWhy(e) }, homeId)))
        .finally(() => asked.current.delete('dust:' + id))
    }
  }, [viewId, homeId, online, layers.dust, ai]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- the simulation: a bank from the engine's cache, or computed on the GPU (about 90 s) when asked ----------
  const fetchBank = async (b: Build, simulate: boolean): Promise<BankPkg | null> => {
    const { job: id, cached } = await postJSON<{ job: string | null; cached: boolean }>('/api/bank', { build: { ...b, cpu: cpuId }, simulate })
    if (!id) return null                                         // never simulated, and not asked to
    const lid = layoutId(b)
    const progress = (p: number | null) => setJobs((m) => {
      const r = { ...m }
      if (p == null) delete r[lid]
      else r[lid] = p
      return r
    })
    if (!cached) progress(0)
    try {
      for (;;) {
        await new Promise((r) => setTimeout(r, cached ? 250 : 800))
        const j = await getJSON<{ status: string; progress: number; error?: string; result?: BankPkg }>(`/api/job/${id}`)
        if (j.status === 'error') throw new Error(j.error)
        if (!cached) progress(j.progress)
        if (j.status === 'done' && j.result) return j.result
      }
    } finally {
      if (!cached) progress(null)
    }
  }
  const aiEntry = viewId ? ai[viewId] : undefined
  const aiFailed = !!aiEntry && 'error' in aiEntry
  // the simulation's bank is loaded when the AI cannot answer (engine off: your stored layout; parts it was not trained
  // on), for the AI window's table, and for the side-by-side comparison (simulate, below)
  const needPhys = !!viewId && pinged && (!online || aiFailed || (win.ai && aiTab === 'here'))
  useEffect(() => {
    if (!needPhys || !viewBuild || !viewId || banks[viewId] || missing[viewId] || asked.current.has('bank:' + viewId)) return
    const id = viewId
    const store = (pkg: BankPkg | null) => {
      if (pkg) setBanks((m) => keepRecent(m, id, pkg, homeId))
      else setMissing((m) => ({ ...m, [id]: true }))
    }
    asked.current.add('bank:' + id)
    const done = () => asked.current.delete('bank:' + id)
    if (online) fetchBank(viewBuild, false).then(store).catch((e) => setMsg(String(e).slice(0, 160))).finally(done)
    else if (viewBuild === homeBuild) getJSON<BankPkg>('/data/bank_current.json').then(store).catch(() => {}).finally(done)
    else done()
  }, [viewId, online, needPhys]) // eslint-disable-line react-hooks/exhaustive-deps
  /** The simulation of a layout: from the cache in a second or two, or run on the GPU (about 90 s), kept when done. */
  const simulate = (b: Build, id: string) => {
    if (asked.current.has('sim:' + id)) return
    asked.current.add('sim:' + id)
    fetchBank(b, true).then((p) => {
      if (!p) return
      setBanks((m) => keepRecent(m, id, p, homeId))
      setMissing((m) => { const r = { ...m }; delete r[id]; return r })
    }).catch((e) => setMsg(String(e).slice(0, 160))).finally(() => asked.current.delete('sim:' + id))
  }

  // ---------- the steady state of the viewed build: the AI's answer whenever there is one ----------
  const phys = viewId ? banks[viewId] : undefined
  const optBank = opt?.layouts.find((L) => viewBuild && layoutId(L.build) === layoutId(viewBuild))?.bank ?? null
  const aiPkg = aiOk(viewId)
  const aiSum = useMemo(() => withDust(aiOk(viewId), dustOk(viewId)), [ai, aiDust, viewId]) // eslint-disable-line react-hooks/exhaustive-deps
  const showAi = !!aiPkg
  const aiPending = (online || !pinged) && !!viewId && !aiEntry   // asked (or about to be), not answered yet: the last answer stays up
  const pkg = showAi ? aiSum : phys
  const summary: Bank | null = pkg ?? (aiPending ? null : optBank)
  const fullBuild = (b: Build): ScenarioBuild => ({ ...b, cpu: cpuId, curves: curves ?? undefined })
  const setup = useMemo(() => (viewBuild && lib && curves ? setupFor(fullBuild(viewBuild), asLib(lib)) : null), [viewBuild, lib, curves, cpuId]) // eslint-disable-line react-hooks/exhaustive-deps
  const watts = setup ? loadWatts(setup, scn.cpu, scn.gpu) : [0, 0]
  const res: ScenarioResult | null = useMemo(() => (setup && summary ? solve(summary, setup, scn.ambient_c, watts[0], watts[1]) : null),
    [setup, summary, scn]) // eslint-disable-line react-hooks/exhaustive-deps
  const held = useRef<{ res: ScenarioResult | null; flow: Pick<SceneData, 'flowKey' | 'u' | 'T' | 'dust'> | null }>({ res: null, flow: null })
  if (res) held.current.res = res
  const resShow = res ?? (aiPending ? held.current.res : null)
  const resD = useDeferredValue(res)
  const physSummary: Bank | null = phys ?? optBank
  const resPhys = useMemo(() => (win.ai && setup && physSummary ? solve(physSummary, setup, scn.ambient_c, watts[0], watts[1]) : null),
    [win.ai, setup, physSummary, scn]) // eslint-disable-line react-hooks/exhaustive-deps
  const homeLayout = opt?.layouts.find((L) => L.current)
  const homeAi = useMemo(() => withDust(aiOk(homeId), dustOk(homeId)), [ai, aiDust, homeId]) // eslint-disable-line react-hooks/exhaustive-deps
  // your layout, as the same source sees it (the AI's against the AI's, the simulation's against the simulation's)
  const refRes: ScenarioResult | null = useMemo(() => {
    const bank = showAi ? homeAi : homeLayout?.bank
    if (view === 'current' || !lib || !curves || !homeBuild || !bank) return null
    const su = setupFor(fullBuild(homeBuild), asLib(lib))
    const [pc, pg] = loadWatts(su, scn.cpu, scn.gpu)
    return solve(bank, su, scn.ambient_c, pc, pg)
  }, [view, lib, curves, homeBuild, homeAi, homeLayout, showAi, scn, cpuId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- side by side: the AI's case on the left, the simulation's on the right ----------
  const comparing = !!cmp && cmp === viewId
  useEffect(() => { if (cmp && cmp !== viewId) setCmp(null) }, [viewId, cmp])     // another layout: back to the AI view
  const runP = viewId != null && jobs[viewId] != null ? jobs[viewId] : null       // this layout's simulation is running
  const simReady = !!phys || !!aiPkg?.simulated
  const startCompare = () => {
    if (!viewBuild || !viewId || !online || !aiPkg) return
    setScanned(null)
    setCmp(viewId)
    if (!phys) simulate(viewBuild, viewId)
  }
  const twinPkg = comparing ? phys : undefined
  const resTwin = useMemo(() => (twinPkg && setup ? solve(twinPkg, setup, scn.ambient_c, watts[0], watts[1]) : null),
    [twinPkg, setup, scn]) // eslint-disable-line react-hooks/exhaustive-deps
  const resTwinD = useDeferredValue(resTwin)
  const twinFields = bankFields(twinPkg)
  const lvlTwin = twinPkg && resTwin ? nearestLevel(twinPkg.levels, resTwin.case_duty) : 0
  const Ttwin = useMemo(() => (twinFields && twinPkg && resTwinD ? composeT(twinFields, twinPkg.levels, resTwinD.case_duty, resTwinD.cpu.watts, resTwinD.gpu.watts) : null),
    [twinFields, twinPkg, resTwinD])
  const twinShown = comparing && scanned === viewId                // the scan has drawn it: its numbers appear
  const simRes = twinShown ? resTwin : null

  // ---------- trends: the steady state across room temperature and across load (solid: fans follow their curves;
  // dashed: fans held at the speed they have now) ----------
  const trends = useMemo(() => {
    if (!win.trend || !summary || !setup || !resD) return null
    const hold = heldSetup(setup, resD)
    const at = (su: Setup, room: number, cf: number, gf: number) => { const [pc, pg] = loadWatts(su, cf, gf); return solve(summary, su, room, pc, pg) }
    const rooms = Array.from({ length: 31 }, (_, i) => 10 + i)
    const loads = Array.from({ length: 21 }, (_, i) => i * 5)
    const room = [setup, hold].map((su) => rooms.map((t) => at(su, t, scnD.cpu, scnD.gpu)))
    const load = [setup, hold].map((su) => loads.map((l) => (sweep === 'gpu' ? at(su, scnD.ambient_c, scnD.cpu, l / 100) : at(su, scnD.ambient_c, l / 100, scnD.gpu))))
    const series = (rs: ScenarioResult[][]) => [
      { id: 'cpu', label: 'CPU', color: SERIES.cpu, ys: rs[0].map((r) => r.cpu.temp_c) },
      { id: 'gpu', label: 'GPU', color: SERIES.gpu, ys: rs[0].map((r) => r.gpu.temp_c) },
      { id: 'cpu-h', label: 'CPU', color: SERIES.cpu, dashed: true, ys: rs[1].map((r) => r.cpu.temp_c) },
      { id: 'gpu-h', label: 'GPU', color: SERIES.gpu, dashed: true, ys: rs[1].map((r) => r.gpu.temp_c) },
    ]
    return { rooms, loads, room: series(room), load: series(load) }
  }, [win.trend, summary, setup, resD, scnD, sweep]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- what the 3D view draws ----------
  // a layout without its own bank yet still shows its case, parts and fans: geometry from any bank with the same parts
  const geo: Geometry | null = pkg?.geometry ?? (viewBuild
    ? [...Object.entries(banks), ...Object.entries(ai)].map(([id, p]) => (id.split('#')[0] === partsKey(viewBuild) && !('error' in p) ? p.geometry : null)).find(Boolean)
    : undefined) ?? null
  const fields = bankFields(pkg)
  const T = useMemo(() => (fields && pkg && resD ? composeT(fields, pkg.levels, resD.case_duty, resD.cpu.watts, resD.gpu.watts) : null), [fields, pkg, resD])
  const lvl = pkg && res ? nearestLevel(pkg.levels, res.case_duty) : 0
  const dustPts = showAi ? dustOk(viewId)?.levels[lvl]?.dust_points ?? null : fields?.dust[lvl] ?? null
  const twin: TwinData | null = comparing ? {
    flowKey: twinPkg && twinFields ? `${twinPkg.key}:sim:${lvlTwin}` : null,
    u: twinFields?.u[lvlTwin] ?? null, T: Ttwin, dust: twinFields?.dust[lvlTwin] ?? null,
    // the scan follows the simulation; a cached one is drawn in a few seconds
    progress: twinPkg ? 1 : runP ?? (simReady ? 0.6 : 0), ready: !!twinPkg && !!twinFields,
    labels: ['AI PREDICTION', runP != null ? `SIMULATING ${Math.round(runP * 100)}%` : 'SIMULATION'],
    // whether the network has seen this layout: on one it was trained on, the two agree almost exactly
    subs: [aiPkg?.seen ? SEEN[aiPkg.seen].toUpperCase() : '', ''],
  } : null
  const sceneData: SceneData | null = useMemo(() => {
    if (!geo || !viewBuild || !lib) return null
    const speed = new Map((resShow?.fans ?? []).map((f) => [f.slot, f.speed]))
    const fans: SceneFan[] = []
    for (const f of viewBuild.fans) {
      const m = geo.mounts.find((x) => x.slot === f.slot)
      if (!m || f.mode === 'off') continue
      const part = lib.fans.find((x) => x.id === f.fan)
      fans.push({ slot: f.slot, mode: f.mode, wall: m.wall, a: m.a, b: m.b, size: part && m.sizes.includes(part.size_mm) ? part.size_mm : Math.max(...m.sizes), speed: speed.get(f.slot) ?? 0.5 })
    }
    let flow: Pick<SceneData, 'flowKey' | 'u' | 'T' | 'dust'> | null = fields && pkg
      ? { flowKey: `${pkg.key}:${pkg.source ?? 'sim'}:${lvl}`, u: fields.u[lvl] ?? null, T, dust: dustPts } : null
    if (flow) held.current.flow = flow
    else if (aiPending) flow = held.current.flow                  // the last airflow stays until the network answers
    return { geoKey: partsKey(viewBuild) + '#' + fanKey(viewBuild.fans), geometry: geo, fans, flowKey: flow?.flowKey ?? null, u: flow?.u ?? null,
      T: flow?.T ?? null, dust: flow?.dust ?? null, cpuFan: resShow?.cpu.fan_speed ?? 0.7, gpuFan: resShow?.gpu.fan_speed ?? 0.4, twin }
  }, [geo, viewBuild, lib, resShow, fields, pkg, lvl, T, dustPts, aiPending, twin?.flowKey, twin?.T, twin?.dust, twin?.progress, twin?.ready, twin?.labels[1], twin?.subs?.[0], comparing]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- actions ----------
  const cs = lib && viewBuild ? lib.cases.find((c) => c.id === viewBuild.case) ?? null : null
  const gpu = lib && viewBuild ? lib.gpus.find((c) => c.id === viewBuild.gpu) ?? null : null
  const cpu = lib?.cpus.find((c) => c.id === cpuId) ?? null
  /** The fan model the case comes with on a mount (what the network was trained with there), if any. */
  const stockFan = (slot: string) => cs?.preinstalled.find((p) => p.slot === slot)?.fan
  const viewOf = (r: RankRow): ViewKey | null =>
    r.current ? 'current' : r.name === recRow?.name ? 'best' : r.name === cpuRow?.name ? 'cpu' : r.name === worstRow?.name ? 'worst' : null
  const openBuild = (b: Build) => {
    const row = rows.find((r) => layoutId(r.build) === layoutId(b))
    const k = row ? viewOf(row) : null
    if (k) return setView(k)
    setCustom(b)
    setView('custom')
  }
  /** Click on a fan or an empty mount in 3D: flip it, switch it off, or add one. */
  const editFan = (slot: string, action: 'flip' | 'off' | 'in' | 'out') => {
    setMenu(null)
    if (!viewBuild) return
    const b: Build = { ...viewBuild, fans: viewBuild.fans.map((f) => ({ ...f })) }
    const i = b.fans.findIndex((f) => f.slot === slot)
    if (action === 'flip' && i >= 0) b.fans[i].mode = b.fans[i].mode === 'in' ? 'out' : 'in'
    else if (action === 'off' && i >= 0) b.fans.splice(i, 1)
    else if (action === 'in' || action === 'out') {
      const m = cs?.mounts.find((x) => x.slot === slot)
      b.fans.push({ slot, fan: stockFan(slot) ?? (m && !m.sizes.includes(120) ? 'generic-140' : 'generic-120'), mode: action, control: 'curve', pct: 80 })
    } else return
    const row = rows.find((r) => layoutId(r.build) === layoutId(b))
    if (!row && !online) return setMsg('Start the engine to open new layouts.')
    openBuild(b)
  }
  const setFan = (slot: string, patch: Partial<FanSlot>) => {
    if (!viewBuild || !cs) return
    const m = cs.mounts.find((x) => x.slot === slot)!
    const ex = viewBuild.fans.find((f) => f.slot === slot)
    const fallback = stockFan(slot) ?? (m.sizes.includes(120) ? 'generic-120' : 'generic-140')
    const next: FanSlot = { ...(ex ?? { slot, fan: fallback, mode: 'off' as Mode, control: 'curve' as Control, pct: 80 }), ...patch }
    const fans = viewBuild.fans.filter((f) => f.slot !== slot)
    if (next.mode !== 'off') fans.push(next)
    openBuild({ ...viewBuild, fans })
  }
  const setPart = (k: 'case' | 'gpu' | 'cooler', id: string) => {
    if (!viewBuild) return
    openBuild({ ...viewBuild, [k]: id })
    if (k === 'gpu' && lib && curves) setCurves({ ...curves, gpu: defaultCurves(lib, id).gpu })
  }
  const codeNow = space && viewBuild ? codeOf(space, viewBuild) : null
  const openCode = (code: string) => {
    if (!space || !homeBuild) return
    openBuild(buildOfCode(space, homeBuild, code))
  }
  // a build the network was not trained on and never simulated: the simulation can be run (no comparison: no AI)
  const canSimulate = online && !showAi && !aiPending && !phys && !optBank && !!viewId && !!missing[viewId] && runP == null
  const verify = () => { if (viewBuild && viewId && online) simulate(viewBuild, viewId) }
  const bgRun = Object.entries(jobs).find(([id]) => !(comparing && id === viewId))?.[1]   // a simulation running for another layout
  const aiHere: AiHere = {
    ai: showAi ? res : null, phys: resPhys, seen: aiPkg?.seen ?? null,
    why: !online ? 'start the engine' : aiEntry && 'error' in aiEntry ? aiEntry.error : null,
    progress: runP, simReady, comparing,
  }
  const aiTag = showAi && !comparing ? <b className="ai-tag">AI</b> : null
  const steps = homeBuild && recRow && !bestIsCur ? changes(homeBuild.fans, recRow.build.fans) : []
  const menuFan = menu && viewBuild ? viewBuild.fans.find((f) => f.slot === menu.slot) : undefined
  const menuMount = menu && cs ? cs.mounts.find((m) => m.slot === menu.slot) : undefined
  const menuQ = menu && resShow ? resShow.fans.find((f) => f.slot === menu.slot) : undefined
  const scnName = `${SHORT[scn.preset ?? ''] ?? 'custom load'}, ${scn.ambient_c.toFixed(scn.ambient_c % 1 ? 1 : 0)} °C`
  const isBankless = !!viewBuild && !pkg && !summary && !aiPending && runP == null

  // ---------- tabs ----------
  const tabs = (['current', 'best', 'cpu', 'worst', 'custom'] as ViewKey[]).filter((k) =>
    k === 'current' || (k === 'custom' ? !!custom : !!buildOf(k) && !(k === 'best' && bestIsCur) && !(k === 'cpu' && cpuRow && (cpuRow.current || cpuRow.name === recRow?.name))))
  const tabLabel: Record<ViewKey, string> = { current: 'Your layout', best: 'Recommended', cpu: 'Lowest CPU', worst: 'Worst', custom: 'Custom' }
  const tagOf = (r: RankRow) => [
    r.current && 'YOURS', r.name === recRow?.name && !r.current && 'BEST', r.name === worstRow?.name && 'WORST', r.throttled && 'THROTTLES',
  ].filter(Boolean) as string[]
  const shown = rows.map((r, i) => ({ r, i })).filter(({ r, i }) => allRows || i < 6 || tagOf(r).length > 0)
  const [lo, hi] = rows.length ? [rows[0].score, rows[n - 1].score] : [0, 1]
  const vsHome = view !== 'current'

  // ---------- floating screens ----------
  const flash = `${view}:${viewId ?? ''}`
  const screens: (HoloSpec & { title: string })[] = []
  const r0 = resShow
  if (r0 && viewBuild) {
    const caseFans = viewBuild.fans.filter((f) => f.mode !== 'off').map((f) => ({ slot: f.slot, mode: f.mode as 'in' | 'out' }))
    const aiDustNow = dustOk(viewId)
    const dustWait = showAi && !aiDustNow
    const dustErr = viewId ? aiDust[viewId] : undefined
    const dustNote = !dustWait ? undefined : dustErr && 'error' in dustErr ? `No AI dust: ${dustErr.error}.` : 'Tracking dust in the AI airflow…'
    const dustRef = vsHome ? refRes?.dust_rate ?? null : r0.dust_rate
    const dustRel = r0.dust_rate != null && dustRef ? r0.dust_rate / dustRef : null
    if (comparing) {
      // between the two cases, each card wired to the part in both; the fans' flows are the same in both (they come from
      // the fans' operating points, not from the network), so the airflow screen rests while comparing
      const part = (k: 'cpu' | 'gpu'): CmpRow[] => [
        { label: '°C', ai: r0[k].temp_c, sim: simRes?.[k].temp_c ?? null, fmt: (v) => v.toFixed(1) },
        { label: 'air in', ai: r0[k].intake_rise_c, sim: simRes?.[k].intake_rise_c ?? null, fmt: (v) => `+${v.toFixed(1)}°` },
      ]
      const cmpCard = (k: 'cpu' | 'gpu', y: number) => ({ id: 'cmp-' + k, title: k.toUpperCase(), pos: [CMP_SHIFT, y, 0.5] as [number, number, number], yaw: 0,
        anchor: k, to: 'both' as const, scale: 1.35, content: (
          <Holo tag={k.toUpperCase()} sub={k === 'cpu' ? cpu?.name.replace('AMD ', '').replace('Intel ', '') : gpu?.name.replace('NVIDIA GeForce ', '').replace(' Founders Edition', ' FE')} width={210}>
            <CompareCard rows={part(k)} progress={twinPkg ? null : runP} diff={tempDiff(r0[k].temp_c, simRes?.[k].temp_c)} />
          </Holo>) })
      screens.push(cmpCard('cpu', 0.8), cmpCard('gpu', -0.2))
      if (layers.dust) {
        const dA = aiDustNow?.levels[lvl]?.dust ?? null, dS = twinShown ? twinPkg?.levels[lvlTwin]?.dust ?? null : null
        const top = (dS ?? dA)?.by_part[0]?.name
        const share = (d: typeof dA) => d?.by_part.find((p) => p.name === top)?.share ?? null
        const rowsD: CmpRow[] = [
          { label: 'settles', ai: dA?.deposited_frac ?? null, sim: dS?.deposited_frac ?? null, fmt: (v) => `${Math.round(v * 100)}%` },
          ...(top ? [{ label: top, ai: share(dA), sim: share(dS), fmt: (v: number) => `${Math.round(v * 100)}%` }] : []),
        ]
        const dd = dA && dS ? (dA.deposited_frac - dS.deposited_frac) * 100 : null
        screens.push({ id: 'cmp-dust', title: 'Dust', pos: [CMP_SHIFT, -1.2, 0.5], yaw: 0, anchor: 'glass', to: 'both', scale: 1.35, content: (
          <Holo tag="DUST" width={210}>
            <CompareCard rows={rowsD} progress={twinPkg ? null : runP} diff={dd != null ? { text: `${signed(dd)} pts`, cls: 'meh' } : null} />
          </Holo>) })
      }
    } else {
      screens.push(
        { id: 'cpu', title: 'CPU', pos: [-1.6, 1.08, 0.5], yaw: 0.06, anchor: 'cpu', content: (
          <Holo tag="CPU" badge={aiTag} sub={cpu?.name.replace('AMD ', '').replace('Intel ', '') ?? 'CPU'} width={262} flashKey={'cpu' + flash}>
            <PartPanel part="cpu" st={r0.cpu} refSt={vsHome ? refRes?.cpu ?? null : null} />
          </Holo>) },
        { id: 'gpu', title: 'GPU', pos: [-1.6, -0.78, 0.6], yaw: 0.06, anchor: 'gpu', content: (
          <Holo tag="GPU" badge={aiTag} sub={gpu?.name.replace('NVIDIA GeForce ', '').replace(' Founders Edition', ' FE') ?? 'GPU'} width={262} flashKey={'gpu' + flash}>
            <PartPanel part="gpu" st={r0.gpu} refSt={vsHome ? refRes?.gpu ?? null : null} />
          </Holo>) },
        { id: 'air', title: 'Airflow', pos: [1.6, 1.08, 0.5], yaw: -0.06, anchor: 'intake', content: (
          <Holo tag="AIRFLOW" badge={aiTag} width={268} flashKey={'air' + flash}>
            <AirPanel res={r0} fans={caseFans} refRes={vsHome ? refRes : null} />
          </Holo>) },
      )
      if (layers.dust)
        screens.push({ id: 'dust', title: 'Dust', pos: [1.6, -0.78, 0.6], yaw: -0.06, anchor: 'glass', content: (
          <Holo tag="DUST" badge={dustWait ? null : aiTag} width={250} flashKey={'dust' + flash}>
            <DustPanel rel={dustRel} stats={pkg?.levels[lvl]?.dust ?? summary?.levels[0]?.dust} compare={vsHome} note={dustNote} />
          </Holo>) })
      if (win.rank) {
        screens.push({ id: 'search', title: 'Layout search', pos: [-3.35, -0.7, -0.12], yaw: 0.2, content: (
          <Holo tag="LAYOUT SEARCH" width={300} flashKey="search">
            <SearchPanel rows={rows} shown={shown} tagOf={tagOf} onOpen={(r) => openBuild(r.build)} allRows={allRows} setAllRows={setAllRows} lo={lo} hi={hi}
              isOn={(r) => !!viewBuild && layoutId(r.build) === layoutId(viewBuild)} />
          </Holo>) })
        const other = bestIsCur ? worstRow : recRow
        if (curRow && other)
          screens.push({ id: 'verdict', title: 'Verdict', pos: [3.35, -0.04, -0.12], yaw: -0.2, content: (
            <Holo tag="VERDICT" width={290} flashKey="verdict">
              <VerdictPanel cur={curRow} other={other} bestIsCur={bestIsCur} n={n} rankCur={rankOf(curRow.name)} rankOther={rankOf(other.name)} geo={geo} steps={steps}
                onOpen={(k) => setView(k === 'current' ? 'current' : bestIsCur ? 'worst' : 'best')}
                tie={bestIsCur && topRow && !topRow.current ? { name: topRow.name, delta: curRow.score - topRow.score, margin: opt?.tie_margin ?? 0.5 } : null} />
            </Holo>) })
      }
    }
  }

  // ---------- windows ----------
  const curveNow = (k: 'case' | 'cpu' | 'gpu') => {
    if (!resShow || !curves) return null
    if (k === 'cpu') return { t: resShow.cpu.temp_c, d: resShow.cpu.fan_duty }
    if (k === 'gpu') return { t: resShow.gpu.temp_c, d: resShow.gpu.fan_duty }
    const src = curves.case.source ?? 'cpu'
    return { t: src === 'cpu' ? resShow.cpu.temp_c : src === 'gpu' ? resShow.gpu.temp_c : Math.max(resShow.cpu.temp_c, resShow.gpu.temp_c), d: resShow.case_duty }
  }

  return (
    <div className="cf">
      <CaseScene scene={sceneData} layers={layers} panels={narrow ? [] : screens} resetKey={layoutKey} onPick={setMenu} selected={menu?.slot ?? null}
        onScanned={() => setScanned(cmp)} />
      <div className="vignette" />

      <header className="top">
        <div className="brand">
          <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
            <rect x="1.5" y="1.5" width="17" height="17" rx="3" fill="none" stroke="rgba(255,255,255,.45)" />
            <path d="M4.5 14 C8.5 14 9 6.5 15.5 6.5" fill="none" stroke="#ececec" strokeWidth="1.4" />
            <circle cx="15.5" cy="6.5" r="1.6" fill="#d9773a" />
          </svg>
          <span className="brand-name">CASEFLOW</span>
        </div>
        <nav className="views">
          {tabs.map((k) => (
            <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>
              {tabLabel[k]}
            </button>
          ))}
        </nav>
        <div className="status">
          {comparing ? (
            <>
              <span className="mono src-chip"><b>AI</b> vs SIMULATION{runP != null ? ` · ${Math.round(runP * 100)}%` : ''}</span>
              <button className="mono verify" title="Back to the AI view (Esc)" onClick={() => setCmp(null)}>BACK</button>
            </>
          ) : showAi ? (
            <>
              <button className="mono ai-chip" title="The 3D view and the screens show the neural network's prediction" onClick={() => setWin({ ...win, ai: true })}>AI PREDICTION</button>
              {online && (
                <button className="mono verify" onClick={startCompare}
                  title={simReady ? 'Show the simulation next to the AI, side by side' : 'Run the simulation (about 90 s on the GPU) and show it next to the AI'}>
                  {runP != null ? `SIMULATION ${Math.round(runP * 100)}%` : simReady ? 'SIMULATION' : 'SIMULATION · 90 s'}
                </button>
              )}
            </>
          ) : pkg ? <span className="mono src-chip">SIMULATION</span> : null}
          {canSimulate && <button className="mono verify" title="Simulate this layout (about 90 s on the GPU)" onClick={verify}>SIMULATE · 90 s</button>}
          {bgRun != null && !(showAi && runP != null) && <span className="mono sim-now">SIMULATING {Math.round(bgRun * 100)}%</span>}
          <span className="mono"><i className={'led' + (online ? ' on' : '')} />{online ? 'ONLINE' : 'OFFLINE'}</span>
        </div>
      </header>

      {menu && viewBuild && (
        <div className="fanmenu" style={{ left: Math.min(menu.x + 16, window.innerWidth - 250), top: Math.min(Math.max(menu.y - 24, 64), window.innerHeight - 220) }}>
          <i className="hc tl" /><i className="hc br" />
          <div className="fm-h">
            <span className="h-tag">{menu.empty ? 'MOUNT' : 'FAN'}</span>
            <span className="fm-where">{WHERE[menu.slot] ?? menu.slot}</span>
            {menuFan && <span className={'fm-mode m-' + menuFan.mode}>{menuFan.mode === 'in' ? '▸ in' : '◂ out'}</span>}
            <button className="win-btn" title="Close" onClick={() => setMenu(null)}>×</button>
          </div>
          <div className="fm-sub">{menuFan
            ? `${lib?.fans.find((x) => x.id === menuFan.fan)?.size_mm ?? ''} mm${menuQ ? ` · ${(menuQ.q_m3s * 2119).toFixed(0)} CFM` : ''}`
            : `empty ${menuMount?.sizes.join('/') ?? ''} mm mount`}</div>
          {menuFan ? (
            <div className="fm-acts">
              <button onClick={() => editFan(menu.slot, 'flip')}>Flip → blow {menuFan.mode === 'in' ? 'out' : 'in'}</button>
              <button onClick={() => editFan(menu.slot, 'off')}>Turn off</button>
            </div>
          ) : (
            <div className="fm-acts">
              <button onClick={() => editFan(menu.slot, 'in')}>Add fan · blow in</button>
              <button onClick={() => editFan(menu.slot, 'out')}>Add fan · blow out</button>
            </div>
          )}
        </div>
      )}

      {msg && <div className="toast"><span>{msg}</span></div>}

      {narrow && layers.panels && (
        <div className="holo-column">{screens.map((s) => <div key={s.id}>{s.content as ReactNode}</div>)}</div>
      )}

      <div className="hud">
        {win.scn && lib && setup && (
          <HudWindow id="scenario" code="01" title="Scenario" side="left" x={20} y={72} w={340} layoutKey={layoutKey} onClose={() => setWin({ ...win, scn: false })}>
            <div className="presets">
              {lib.defaults.scenarios.map((p) => (
                <button key={p.id} className={scn.preset === p.id ? 'on' : ''} title={p.name + (p.note ? ` - ${p.note}` : '')}
                  onClick={() => setScn({ ...scn, cpu: p.cpu, gpu: p.gpu, preset: p.id })}>{SHORT[p.id] ?? p.name}</button>
              ))}
            </div>
            <Slider label="Room" value={scn.ambient_c} min={10} max={40} step={0.5} show={`${scn.ambient_c.toFixed(1)} °C`} onChange={(v) => setScn({ ...scn, ambient_c: v })} />
            <Slider label="CPU load" value={Math.round(scn.cpu * 100)} min={0} max={100} step={1} show={<>{pct(scn.cpu)} · {watts[0].toFixed(0)} W</>}
              onChange={(v) => setScn({ ...scn, cpu: v / 100, preset: null })} />
            <Slider label="GPU load" value={Math.round(scn.gpu * 100)} min={0} max={100} step={1} show={<>{pct(scn.gpu)} · {watts[1].toFixed(0)} W</>}
              onChange={(v) => setScn({ ...scn, gpu: v / 100, preset: null })} />
            {resShow && (
              <div className="scn-out">
                <div><span className="lbl">case fans</span><b>{pct(resShow.case_duty)}</b></div>
                <div><span className="lbl">CPU fans</span><b>{pct(resShow.cpu.fan_duty)}</b></div>
                <div><span className="lbl">GPU fans</span><b>{resShow.gpu.fan_speed === 0 ? 'off' : pct(resShow.gpu.fan_duty)}</b></div>
                <div><span className="lbl">noise</span><b>{resShow.noise_dba.toFixed(0)}<u>dBA</u></b></div>
              </div>
            )}
          </HudWindow>
        )}

        {win.curve && curves && (
          <HudWindow id="curves" code="02" title="Fan curves" side="right" x={20} y={72} w={350} layoutKey={layoutKey} onClose={() => setWin({ ...win, curve: false })}>
            <div className="seg3 tabs3">
              {([['case', 'Case fans'], ['cpu', 'CPU fans'], ['gpu', 'GPU fans']] as const).map(([k, l]) => (
                <button key={k} className={curveTab === k ? 'on' : ''} onClick={() => setCurveTab(k)}>{l}</button>
              ))}
            </div>
            {curveTab === 'case' && (
              <label className="cv-src"><span className="lbl">follows</span>
                <select className="field sm" value={curves.case.source ?? 'cpu'} onChange={(e) => setCurves({ ...curves, case: { ...curves.case, source: e.target.value as Source } })}>
                  <option value="cpu">CPU temperature</option><option value="gpu">GPU temperature</option><option value="max">the hotter of the two</option>
                </select>
              </label>
            )}
            <CurveEditor points={curves[curveTab].points} now={curveNow(curveTab)} color={curveTab === 'gpu' ? '#d9773a' : '#2fa8a9'}
              onChange={(p) => setCurves({ ...curves, [curveTab]: { ...curves[curveTab], points: p } })} />
            {(() => {
              const now = curveNow(curveTab)
              return now && <div className="cv-foot mono">now {now.t.toFixed(1)} °C → {pct(now.d)}{curveTab === 'gpu' && resShow?.gpu.fan_speed === 0 ? ' (stopped)' : ''}</div>
            })()}
            <button className="textbtn" onClick={() => lib && viewBuild && setCurves({ ...curves, [curveTab]: defaultCurves(lib, viewBuild.gpu)[curveTab] })}>Reset this curve</button>
          </HudWindow>
        )}

        {win.trend && trends && (
          <HudWindow id="trends" code="03" title="Trends" side="right" x={win.curve ? 384 : 20} y={72} w={340} layoutKey={layoutKey} onClose={() => setWin({ ...win, trend: false })}>
            <TrendChart title="vs room temperature" xs={trends.rooms} series={trends.room} xFmt={(x) => `${x}`} xUnit="room °C" now={scn.ambient_c} />
            <div className="tr-sw">
              <span className="lbl">sweep</span>
              <span className="seg3">{(['gpu', 'cpu'] as const).map((k) => <button key={k} className={sweep === k ? 'on' : ''} onClick={() => setSweep(k)}>{k} load</button>)}</span>
            </div>
            <TrendChart title={`vs ${sweep.toUpperCase()} load`} xs={trends.loads} series={trends.load}
              xFmt={(x) => `${x}%`} xUnit={`${sweep} load`} now={(sweep === 'gpu' ? scn.gpu : scn.cpu) * 100} />
          </HudWindow>
        )}

        {win.cfg && viewBuild && cs && lib && (
          <HudWindow id="configure" code="05" title="Configure" side="left" x={win.scn ? 374 : 20} y={72} w={330} layoutKey={layoutKey} onClose={() => setWin({ ...win, cfg: false })}>
            <div className="cfg-parts">
              {([['Case', 'case', lib.cases], ['GPU', 'gpu', lib.gpus], ['Cooler', 'cooler', lib.coolers]] as const).map(([l, k, list]) => (
                <label key={k}><span className="lbl">{l}</span>
                  <select className="field" value={viewBuild[k]} onChange={(e) => setPart(k, e.target.value)}>
                    {(list as { id: string; name: string }[]).map((p) => <option key={p.id} value={p.id}>{partName(p.name)}</option>)}
                  </select>
                </label>
              ))}
              <label><span className="lbl">CPU</span>
                <select className="field" value={cpuId} onChange={(e) => setCpuId(e.target.value)}>
                  {lib.cpus.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </label>
            </div>
            <span className="lbl">Case fans</span>
            <div className="mounts">
              {cs.mounts.map((m) => {
                const f = viewBuild.fans.find((x) => x.slot === m.slot)
                const mode: Mode = f?.mode ?? 'off'
                const live = resShow?.fans.find((x) => x.slot === m.slot)
                return (
                  <div key={m.slot} className="mt">
                    <div className="mt-1">
                      <span className="mt-n">{WHERE[m.slot] ?? m.slot}</span>
                      <span className="seg3">{(['in', 'off', 'out'] as Mode[]).map((md) => <button key={md} className={mode === md ? 'on ' + md : ''} onClick={() => setFan(m.slot, { mode: md })}>{md}</button>)}</span>
                    </div>
                    {f && (
                      <div className="mt-2">
                        <select className="field sm" value={f.fan} onChange={(e) => setFan(m.slot, { fan: e.target.value })}>
                          {lib.fans.filter((x) => m.sizes.includes(x.size_mm)).map((x) => <option key={x.id} value={x.id}>{fanName(x.name)}</option>)}
                        </select>
                        <span className="seg3 ctl">{(['curve', 'fixed'] as Control[]).map((c) => <button key={c} className={(f.control ?? 'curve') === c ? 'on' : ''} onClick={() => setFan(m.slot, { control: c })}>{c}</button>)}</span>
                        <span className="mono pct">{live ? `${Math.round(live.rpm)}` : '—'}<small>rpm</small></span>
                        {(f.control ?? 'curve') === 'fixed' && (
                          <label className="mt-fix"><input type="range" min={0} max={100} step={10} value={f.pct} onChange={(e) => setFan(m.slot, { pct: +e.target.value })} /><span className="mono">{f.pct}%</span></label>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
            {showAi && online && !comparing && (
              <button className="run" onClick={startCompare}>{runP != null ? `Simulating ${Math.round(runP * 100)}% · show` : simReady ? 'Compare with the simulation' : 'Simulate and compare · ~90 s'}</button>
            )}
            {!showAi && (canSimulate || runP != null) && (
              <button className="run" disabled={!canSimulate} onClick={verify}>{runP != null ? `Simulating ${Math.round(runP * 100)}%` : 'Simulate · ~90 s'}</button>
            )}
          </HudWindow>
        )}

        {win.ai && (
          <HudWindow id="ai" code="07" title="AI" side="left" x={20 + (win.scn ? 354 : 0) + (win.cfg ? 344 : 0)} y={72} w={372} layoutKey={layoutKey}
            onClose={() => setWin({ ...win, ai: false })} meta="3D U-NET">
            <div className="seg3 tabs3">
              {([['here', 'This layout'], ['all', `All ${space?.rows.length ?? 6561}`]] as const).map(([k, l]) => (
                <button key={k} className={aiTab === k ? 'on' : ''} onClick={() => setAiTab(k)}>{l}</button>
              ))}
            </div>
            {aiTab === 'here'
              ? <><AiHereTab here={aiHere} onCompare={startCompare} /><ModelCard info={aiInfo} /></>
              : <AiAllTab space={space} ranks={ranks} codeNow={codeNow} scenario={scnName} fans={aiFans} setFans={setAiFans} heldOut={aiHeld} setHeldOut={setAiHeld} onOpen={openCode} />}
          </HudWindow>
        )}

        {win.info && (
          <HudWindow id="method" code="06" title="Method" side="right" x={win.curve ? 384 : 20} y={72} w={380} layoutKey={layoutKey} onClose={() => setWin({ ...win, info: false })}>
            <div className="method">
              <p><span className="lbl">AI</span>A 3D U-Net trained on 150 simulated layouts predicts the airflow and heat of a layout in 0.1 s{aiInfo?.test ? ` (held-out layouts: ${aiInfo.test.cpu_mae_c.toFixed(1)} / ${aiInfo.test.gpu_mae_c.toFixed(1)} °C mean error)` : ''}. This is what you see.</p>
              <p><span className="lbl">Simulation</span>Our own lattice-Boltzmann solver (PyTorch, GPU) on a 6 mm grid of the case, at three case-fan speeds, about 90 s. SIMULATION shows it next to the AI.</p>
              <p><span className="lbl">Heat</span>CPU and GPU heat carried by that air: the rise per watt, so any load is instant.</p>
              <p><span className="lbl">Fans</span>Each fan at its operating point against the case pressure; its speed follows its curve.</p>
              <p><span className="lbl">Steady state</span>Part = room + intake air rise + power ÷ cooler conductance, solved in your browser; a part past its limit throttles.</p>
              <p><span className="lbl">Dust</span>40 000 particles of 20 µm drift with the air and stick where they land (in the AI's airflow or the simulation's).</p>
              <p><span className="lbl">Check</span>The reference PC measured CPU 65.3 / GPU 49.3 °C; the model gives 65.6 / 49.7. Trust differences more than absolute °C.</p>
            </div>
          </HudWindow>
        )}
      </div>

      {isBankless && (
        <div className="toast"><span>{online ? 'Not simulated yet' : 'Start the engine to see this layout'}</span>{canSimulate && <button className="textbtn" onClick={verify}>Simulate · 90 s</button>}</div>
      )}

      <nav className="dock">
        {([['panels', 'Screens'], ['flow', 'Flow'], ['lines', 'Lines'], ['dust', 'Dust'], ['heat', 'Heat']] as const).map(([k, l]) => (
          <button key={k} className={layers[k] ? 'on' : ''} onClick={() => setLayers({ ...layers, [k]: !layers[k] })}>{l}</button>
        ))}
        <span className="sep" />
        <span className="legend"><span className="lbl">Air ΔT</span><span>0</span><i style={{ background: THERMAL_CSS }} /><span>+{T_MAX} °C</span></span>
        <span className="sep" />
        <button className={win.scn ? 'on' : ''} onClick={() => setWin({ ...win, scn: !win.scn })}>Scenario</button>
        <button className={win.curve ? 'on' : ''} onClick={() => setWin({ ...win, curve: !win.curve })}>Curves</button>
        <button className={win.trend ? 'on' : ''} onClick={() => setWin({ ...win, trend: !win.trend })}>Trends</button>
        <button className={win.cfg ? 'on' : ''} onClick={() => setWin({ ...win, cfg: !win.cfg })}>Configure</button>
        <button className={win.rank ? 'on' : ''} title="The 32 searched layouts, ranked, and the verdict on yours" onClick={() => setWin({ ...win, rank: !win.rank })}>Ranking</button>
        <button className={win.ai ? 'on' : ''} onClick={() => setWin({ ...win, ai: !win.ai })}>AI</button>
        <button className={win.info ? 'on' : ''} onClick={() => setWin({ ...win, info: !win.info })}>Method</button>
        <button title="Reset the view, the windows and the screens (drag a screen by its title; double-click the title to put it back)"
          onClick={() => { resetWindows(); resetHolos(); setLayoutKey((k) => k + 1) }}>⟲</button>
      </nav>
    </div>
  )
}
