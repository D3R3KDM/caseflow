import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getJSON, pingServer, postJSON } from './api'
import CurveEditor from './CurveEditor'
import { bankFields, composeT, nearestLevel } from './fields'
import { AirPanel, DustPanel, Holo, PartPanel, SearchPanel, VerdictPanel } from './Holo'
import HudWindow, { resetWindows } from './HudWindow'
import { THERMAL_CSS } from './palette'
import CaseScene, { resetHolos, T_MAX, type HoloSpec, type Layers, type Pick, type SceneData, type SceneFan } from './Scene'
import TrendChart, { SERIES } from './TrendChart'
import { loadWatts, setupFor, solve, type Bank, type Curves, type ScenarioBuild, type ScenarioLib, type ScenarioResult, type Setup, type Source } from './scenario'
import { WHERE, type BankPkg, type Build, type Control, type FanSlot, type Geometry, type Lib, type Mode, type Opt, type RankRow, type Scenario, type ViewKey } from './types'

const amazon = (q: string) => `https://www.amazon.com/s?k=${encodeURIComponent(q)}`
const fanKey = (fs: FanSlot[]) => fs.filter((f) => f.mode !== 'off').map((f) => `${f.slot}:${f.fan}:${f.mode}`).sort().join('|')
const partsKey = (b: Build) => `${b.case}|${b.gpu}|${b.cooler}`
/** Everything that shapes a build's airflow (and so its bank): parts, fans, and which fans follow the curve. */
const layoutId = (b: Build) => partsKey(b) + '#' + b.fans.filter((f) => f.mode !== 'off')
  .map((f) => `${f.slot}:${f.fan}:${f.mode}:${f.control ?? 'curve'}${(f.control ?? 'curve') === 'fixed' ? ':' + f.pct : ''}`).sort().join('|')
const asLib = (lib: Lib) => lib as unknown as ScenarioLib
/** Preset buttons, short enough for one row (the full name is the tooltip). */
const SHORT: Record<string, string> = { idle: 'Idle', office: 'Everyday', cpu_game: 'CPU game', gpu_game: 'GPU game', stress: 'Stress' }
const pct = (x: number) => `${Math.round(x * 100)}%`

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
  const [lib, setLib] = useState<Lib | null>(null)
  const [opt, setOpt] = useState<Opt | null>(null)
  const [home, setHome] = useState<Build | null>(null)                // the user's own layout
  const [banks, setBanks] = useState<Record<string, BankPkg>>({})     // full banks (with 3D fields) by layoutId
  const [view, setView] = useState<ViewKey>('current')
  const [custom, setCustom] = useState<Build | null>(null)
  const [scn, setScn] = useState<Scenario>({ ambient_c: 23, cpu: 0.97, gpu: 0.38, preset: 'cpu_game' })
  const [curves, setCurves] = useState<Curves | null>(null)
  const [cpuId, setCpuId] = useState('amd-ryzen-7-9700x')
  const [layers, setLayers] = useState<Layers>({ flow: true, lines: false, dust: true, heat: false, panels: true })
  const [win, setWin] = useState({ scn: true, curve: false, trend: false, cfg: false, info: false })
  const [sweep, setSweep] = useState<'gpu' | 'cpu'>('gpu')
  const [curveTab, setCurveTab] = useState<'case' | 'cpu' | 'gpu'>('case')
  const [allRows, setAllRows] = useState(false)
  const [job, setJob] = useState<{ p: number; label: string } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [layoutKey, setLayoutKey] = useState(0)
  const [menu, setMenu] = useState<Pick | null>(null)
  const narrow = useNarrow()
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setMenu(null)
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [])

  // ---------- data ----------
  useEffect(() => {
    let alive = true
    const ping = () => pingServer().then((ok) => alive && setOnline(ok))
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

  // ---------- banks: the engine's cache (or a fresh run), or the stored one for your layout offline ----------
  const fetchBank = async (b: Build, label: string): Promise<BankPkg | null> => {
    const { job: id, cached } = await postJSON<{ job: string; cached: boolean }>('/api/bank', { build: { ...b, cpu: cpuId } })
    if (!cached) setJob({ p: 0, label })
    try {
      for (;;) {
        await new Promise((r) => setTimeout(r, cached ? 250 : 800))
        const j = await getJSON<{ status: string; progress: number; error?: string; result?: BankPkg }>(`/api/job/${id}`)
        if (j.status === 'error') throw new Error(j.error)
        if (!cached) setJob({ p: j.progress, label })
        if (j.status === 'done' && j.result) return j.result
      }
    } finally {
      if (!cached) setJob(null)
    }
  }
  const viewId = viewBuild ? layoutId(viewBuild) : null
  useEffect(() => {
    if (!viewBuild || !viewId || banks[viewId]) return
    let alive = true
    const store = (pkg: BankPkg | null) => { if (alive && pkg) setBanks((m) => ({ ...m, [viewId]: pkg })) }
    if (online) fetchBank(viewBuild, view === 'custom' ? 'your edited layout' : 'layout').then(store).catch((e) => setMsg(String(e).slice(0, 220)))
    else if (viewBuild === homeBuild) getJSON<BankPkg>('/data/bank_current.json').then(store).catch(() => {})
    return () => { alive = false }
  }, [viewId, online]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- the steady state of the viewed build ----------
  const pkg = viewId ? banks[viewId] : undefined
  const summary: Bank | null = pkg ?? (opt?.layouts.find((L) => viewBuild && layoutId(L.build) === layoutId(viewBuild))?.bank ?? null)
  const fullBuild = (b: Build): ScenarioBuild => ({ ...b, cpu: cpuId, curves: curves ?? undefined })
  const setup = useMemo(() => (viewBuild && lib && curves ? setupFor(fullBuild(viewBuild), asLib(lib)) : null), [viewBuild, lib, curves, cpuId]) // eslint-disable-line react-hooks/exhaustive-deps
  const watts = setup ? loadWatts(setup, scn.cpu, scn.gpu) : [0, 0]
  const res: ScenarioResult | null = useMemo(() => (setup && summary ? solve(summary, setup, scn.ambient_c, watts[0], watts[1]) : null),
    [setup, summary, scn]) // eslint-disable-line react-hooks/exhaustive-deps
  const resD = useDeferredValue(res)
  const homeLayout = opt?.layouts.find((L) => L.current)
  const refRes: ScenarioResult | null = useMemo(() => {
    if (view === 'current' || !lib || !curves || !homeLayout) return null
    const su = setupFor(fullBuild(homeLayout.build), asLib(lib))
    const [pc, pg] = loadWatts(su, scn.cpu, scn.gpu)
    return solve(homeLayout.bank, su, scn.ambient_c, pc, pg)
  }, [view, lib, curves, homeLayout, scn, cpuId]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- trends: the steady state across room temperature and across load (solid: fans follow their curves;
  // dashed: fans held at the speed they have now) ----------
  const trends = useMemo(() => {
    if (!win.trend || !summary || !setup || !resD) return null
    const held = heldSetup(setup, resD)
    const at = (su: Setup, room: number, cf: number, gf: number) => { const [pc, pg] = loadWatts(su, cf, gf); return solve(summary, su, room, pc, pg) }
    const rooms = Array.from({ length: 31 }, (_, i) => 10 + i)
    const loads = Array.from({ length: 21 }, (_, i) => i * 5)
    const room = [setup, held].map((su) => rooms.map((t) => at(su, t, scnD.cpu, scnD.gpu)))
    const load = [setup, held].map((su) => loads.map((l) => (sweep === 'gpu' ? at(su, scnD.ambient_c, scnD.cpu, l / 100) : at(su, scnD.ambient_c, l / 100, scnD.gpu))))
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
  const geo: Geometry | null = pkg?.geometry
    ?? (viewBuild ? Object.entries(banks).find(([id]) => id.split('#')[0] === partsKey(viewBuild))?.[1].geometry : undefined) ?? null
  const fields = bankFields(pkg)
  const T = useMemo(() => (fields && pkg && resD ? composeT(fields, pkg.levels, resD.case_duty, resD.cpu.watts, resD.gpu.watts) : null), [fields, pkg, resD])
  const lvl = pkg && res ? nearestLevel(pkg.levels, res.case_duty) : 0
  const sceneData: SceneData | null = useMemo(() => {
    if (!geo || !viewBuild || !lib) return null
    const speed = new Map((res?.fans ?? []).map((f) => [f.slot, f.speed]))
    const fans: SceneFan[] = []
    for (const f of viewBuild.fans) {
      const m = geo.mounts.find((x) => x.slot === f.slot)
      if (!m || f.mode === 'off') continue
      const part = lib.fans.find((x) => x.id === f.fan)
      fans.push({ slot: f.slot, mode: f.mode, wall: m.wall, a: m.a, b: m.b, size: part && m.sizes.includes(part.size_mm) ? part.size_mm : Math.max(...m.sizes), speed: speed.get(f.slot) ?? 0.5 })
    }
    return { geoKey: partsKey(viewBuild) + '#' + fanKey(viewBuild.fans), geometry: geo, fans, flowKey: fields && pkg ? `${pkg.key}:${lvl}` : null,
      u: fields?.u[lvl] ?? null, T, dust: fields?.dust[lvl] ?? null, cpuFan: res?.cpu.fan_speed ?? 0.7, gpuFan: res?.gpu.fan_speed ?? 0.4 }
  }, [geo, viewBuild, lib, res, fields, pkg, lvl, T])

  // ---------- actions ----------
  const cs = lib && viewBuild ? lib.cases.find((c) => c.id === viewBuild.case) ?? null : null
  const gpu = lib && viewBuild ? lib.gpus.find((c) => c.id === viewBuild.gpu) ?? null : null
  const cooler = lib && viewBuild ? lib.coolers.find((c) => c.id === viewBuild.cooler) ?? null : null
  const cpu = lib?.cpus.find((c) => c.id === cpuId) ?? null
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
    if (job) return setMsg('A simulation is still running - wait for it to finish.')
    const b: Build = { ...viewBuild, fans: viewBuild.fans.map((f) => ({ ...f })) }
    const i = b.fans.findIndex((f) => f.slot === slot)
    if (action === 'flip' && i >= 0) b.fans[i].mode = b.fans[i].mode === 'in' ? 'out' : 'in'
    else if (action === 'off' && i >= 0) b.fans.splice(i, 1)
    else if (action === 'in' || action === 'out') {
      const m = cs?.mounts.find((x) => x.slot === slot)
      b.fans.push({ slot, fan: m && !m.sizes.includes(120) ? 'generic-140' : 'generic-120', mode: action, control: 'curve', pct: 80 })
    } else return
    const row = rows.find((r) => layoutId(r.build) === layoutId(b))
    if (!row && !online) return setMsg('Not one of the searched layouts: start the engine (start.bat) to simulate it.')
    openBuild(b)
  }
  const setFan = (slot: string, patch: Partial<FanSlot>) => {
    if (!viewBuild || !cs) return
    const m = cs.mounts.find((x) => x.slot === slot)!
    const ex = viewBuild.fans.find((f) => f.slot === slot)
    const fallback = m.sizes.includes(120) ? 'generic-120' : 'generic-140'
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
  const steps = homeBuild && recRow && !bestIsCur ? changes(homeBuild.fans, recRow.build.fans) : []
  const menuFan = menu && viewBuild ? viewBuild.fans.find((f) => f.slot === menu.slot) : undefined
  const menuMount = menu && cs ? cs.mounts.find((m) => m.slot === menu.slot) : undefined
  const menuQ = menu && res ? res.fans.find((f) => f.slot === menu.slot) : undefined
  const preset = lib?.defaults.scenarios.find((p) => p.id === scn.preset)
  const scnName = `${preset?.name ?? 'custom load'}, ${scn.ambient_c.toFixed(scn.ambient_c % 1 ? 1 : 0)} °C room`
  const isBankless = !!viewBuild && !pkg && !summary

  // ---------- tabs ----------
  const tabs = (['current', 'best', 'cpu', 'worst', 'custom'] as ViewKey[]).filter((k) =>
    k === 'current' || (k === 'custom' ? !!custom : !!buildOf(k) && !(k === 'best' && bestIsCur) && !(k === 'cpu' && cpuRow && (cpuRow.current || cpuRow.name === recRow?.name))))
  const tabLabel: Record<ViewKey, [string, string]> = {
    current: ['Your layout', curRow ? (curRow.tied && n > 1 && topRow && !topRow.current ? 'tied #1' : `#${rankOf(curRow.name)}/${n}`) : ''],
    best: ['Recommended', `#1/${n}`], cpu: ['Lowest CPU', `#${rankOf(cpuRow?.name)}/${n}`], worst: ['Worst', `#${n}/${n}`], custom: ['Custom', ''],
  }
  const tagOf = (r: RankRow) => [
    r.current && 'YOURS', r.tied && rows.filter((x) => x.tied).length > 1 && 'TIED #1', r.name === recRow?.name && !r.current && 'RECOMMENDED',
    r.name === cpuRow?.name && 'LOWEST CPU', r.name === worstRow?.name && 'WORST', r.throttled && 'THROTTLES',
  ].filter(Boolean) as string[]
  const shown = rows.map((r, i) => ({ r, i })).filter(({ r, i }) => allRows || i < 6 || tagOf(r).length > 0)
  const [lo, hi] = rows.length ? [rows[0].score, rows[n - 1].score] : [0, 1]
  const compare = view !== 'current'

  // ---------- floating screens ----------
  const flash = `${view}:${viewId ?? ''}`
  const screens: (HoloSpec & { title: string })[] = []
  if (res && viewBuild) {
    const caseFans = viewBuild.fans.filter((f) => f.mode !== 'off').map((f) => ({ slot: f.slot, mode: f.mode as 'in' | 'out' }))
    const lv = pkg?.levels[lvl]
    const dustRel = res.dust_rate != null && (refRes?.dust_rate ?? res.dust_rate) ? res.dust_rate / (refRes?.dust_rate ?? res.dust_rate) : null
    screens.push(
      { id: 'search', title: 'Layout search', pos: [-3.35, -0.7, -0.12], yaw: 0.2, content: (
        <Holo tag="LAYOUT SEARCH" sub={`${n} layouts · live`} width={300} flashKey="search">
          <SearchPanel rows={rows} shown={shown} tagOf={tagOf} onOpen={(r) => openBuild(r.build)} allRows={allRows} setAllRows={setAllRows} lo={lo} hi={hi}
            isOn={(r) => !!viewBuild && layoutId(r.build) === layoutId(viewBuild)} scenario={scnName} dustWeight={opt?.dust_weight ?? 2} />
        </Holo>) },
      { id: 'cpu', title: 'CPU', pos: [-1.6, 1.08, 0.5], yaw: 0.06, anchor: 'cpu', content: (
        <Holo tag="CPU" sub={cpu?.name.replace('AMD ', '').replace('Intel ', '') ?? 'CPU'} width={250} flashKey={'cpu' + flash}>
          <PartPanel part="cpu" st={res.cpu} refSt={compare ? refRes?.cpu ?? null : null} name={cooler?.name.replace('Thermalright ', '').replace(' ARGB', '') ?? ''} />
        </Holo>) },
      { id: 'gpu', title: 'GPU', pos: [-1.6, -0.78, 0.6], yaw: 0.06, anchor: 'gpu', content: (
        <Holo tag="GPU" sub={gpu?.name.replace('NVIDIA GeForce ', '') ?? 'GPU'} width={250} flashKey={'gpu' + flash}>
          <PartPanel part="gpu" st={res.gpu} refSt={compare ? refRes?.gpu ?? null : null} name={gpu?.cooler.replace(/_/g, '-') ?? ''} />
        </Holo>) },
      { id: 'air', title: 'Airflow', pos: [1.6, 1.08, 0.5], yaw: -0.06, anchor: 'intake', content: (
        <Holo tag="AIRFLOW" sub={view === 'custom' ? 'custom layout' : (rows.find((r) => layoutId(r.build) === viewId)?.name ?? '')} width={268} flashKey={'air' + flash}>
          <AirPanel res={res} fans={caseFans} refRes={compare ? refRes : null} />
        </Holo>) },
      { id: 'dust', title: 'Dust', pos: [1.6, -0.78, 0.6], yaw: -0.06, anchor: 'glass', content: (
        <Holo tag="DUST" sub="20 µm particles" width={250} flashKey={'dust' + flash}>
          <DustPanel rel={dustRel} stats={lv?.dust ?? summary?.levels[0]?.dust} compare={compare} />
        </Holo>) },
    )
    const other = bestIsCur ? worstRow : recRow
    if (curRow && other)
      screens.push({ id: 'verdict', title: 'Verdict', pos: [3.35, -0.04, -0.12], yaw: -0.2, content: (
        <Holo tag="VERDICT" sub={bestIsCur ? 'yours vs worst' : 'yours vs best'} width={290} flashKey="verdict">
          <VerdictPanel cur={curRow} other={other} bestIsCur={bestIsCur} n={n} rankCur={rankOf(curRow.name)} rankOther={rankOf(other.name)} geo={geo} steps={steps}
            onOpen={(k) => setView(k === 'current' ? 'current' : bestIsCur ? 'worst' : 'best')}
            tie={bestIsCur && topRow && !topRow.current ? { name: topRow.name, delta: curRow.score - topRow.score, margin: opt?.tie_margin ?? 0.5 } : null} />
        </Holo>) })
  }

  // ---------- windows ----------
  const curveNow = (k: 'case' | 'cpu' | 'gpu') => {
    if (!res || !curves) return null
    if (k === 'cpu') return { t: res.cpu.temp_c, d: res.cpu.fan_duty }
    if (k === 'gpu') return { t: res.gpu.temp_c, d: res.gpu.fan_duty }
    const src = curves.case.source ?? 'cpu'
    return { t: src === 'cpu' ? res.cpu.temp_c : src === 'gpu' ? res.gpu.temp_c : Math.max(res.cpu.temp_c, res.gpu.temp_c), d: res.case_duty }
  }

  return (
    <div className="cf">
      <CaseScene scene={sceneData} layers={layers} panels={narrow ? [] : screens} resetKey={layoutKey} onPick={setMenu} selected={menu?.slot ?? null} />
      <div className="vignette" />

      <header className="top">
        <div className="brand">
          <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
            <rect x="1.5" y="1.5" width="17" height="17" fill="none" stroke="rgba(140,200,255,.55)" />
            <path d="M4.5 14 C8.5 14 9 6.5 15.5 6.5" fill="none" stroke="#63e0ff" strokeWidth="1.4" />
            <circle cx="15.5" cy="6.5" r="1.5" fill="#63e0ff" />
          </svg>
          <div className="brand-txt">
            <span className="brand-name">CASEFLOW</span>
            <span className="brand-sub">{cs ? `${cs.name.replace('Lian Li ', '')} · ${scnName}` : 'airflow · heat · dust'}
              {gpu && <> · <a href={amazon(gpu.amazon_query ?? gpu.name)} target="_blank" rel="noreferrer">parts ↗</a></>}</span>
          </div>
        </div>
        <nav className="views">
          {tabs.map((k) => (
            <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>
              {tabLabel[k][0]}{tabLabel[k][1] && <span className="v-sub">{tabLabel[k][1]}</span>}
            </button>
          ))}
        </nav>
        <div className="status">
          <span className="mono"><i className={'led' + (online ? ' on' : '')} />{online ? 'ENGINE ONLINE' : 'OFFLINE · PRECOMPUTED'}</span>
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
            ? `${lib?.fans.find((x) => x.id === menuFan.fan)?.name.replace(/^Lian Li LANCOOL 207 /, 'Stock ').replace(/, model unknown.*$/, '') ?? menuFan.fan}${menuQ ? ` · ${(menuQ.q_m3s * 2119).toFixed(0)} CFM · ${Math.round(menuQ.rpm)} rpm now` : ''}`
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
          <div className="fm-note">{online ? 'A stored layout opens at once; a new one is simulated (~1.5 min).' : 'Engine offline: only the searched layouts.'}</div>
        </div>
      )}

      {(job || msg) && (
        <div className="toast">
          {job ? <><span>SIMULATING · {job.label} · 3 fan speeds</span><span className="mono">{Math.round(job.p * 100)}%</span><i style={{ width: `${job.p * 100}%` }} /></> : <span>{msg}</span>}
        </div>
      )}

      {narrow && layers.panels && (
        <div className="holo-column">{screens.map((s) => <div key={s.id}>{s.content as ReactNode}</div>)}</div>
      )}

      <div className="hud">
        {win.scn && lib && setup && (
          <HudWindow id="scenario" code="01" title="Scenario" side="left" x={20} y={72} w={340} layoutKey={layoutKey} onClose={() => setWin({ ...win, scn: false })}
            meta={res ? `${res.noise_dba.toFixed(0)} dBA` : ''}>
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
            {res && (
              <div className="scn-out">
                <div><span className="lbl">case fans</span><b>{pct(res.case_duty)}</b></div>
                <div><span className="lbl">CPU fans</span><b>{pct(res.cpu.fan_duty)}</b></div>
                <div><span className="lbl">GPU fans</span><b>{res.gpu.fan_speed === 0 ? 'off' : pct(res.gpu.fan_duty)}</b></div>
                <div><span className="lbl">noise</span><b>{res.noise_dba.toFixed(0)}<u>dBA</u></b></div>
              </div>
            )}
            <p className="fine" title={`Load runs from idle (${setup.cpu.idle_w} W / ${setup.gpu.idle_w} W) to the power limit (${setup.cpu.max_w} W / ${setup.gpu.max_w} W).`}>
              Fans follow their <button className="linkbtn" onClick={() => setWin({ ...win, curve: true })}>curves</button>: a warmer room or heavier load spins them up (<button className="linkbtn" onClick={() => setWin({ ...win, trend: true })}>trends</button>).</p>
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
            <CurveEditor points={curves[curveTab].points} now={curveNow(curveTab)} color={curveTab === 'gpu' ? '#ffa860' : '#63e0ff'}
              onChange={(p) => setCurves({ ...curves, [curveTab]: { ...curves[curveTab], points: p } })} />
            {(() => {
              const now = curveNow(curveTab)
              return now && <div className="cv-foot mono">now {now.t.toFixed(1)} °C → {pct(now.d)}{curveTab === 'gpu' && res?.gpu.fan_speed === 0 ? ' (stopped)' : ''}</div>
            })()}
            <button className="textbtn" onClick={() => lib && viewBuild && setCurves({ ...curves, [curveTab]: defaultCurves(lib, viewBuild.gpu)[curveTab] })}>Reset this curve</button>
            <p className="fine">{curveTab === 'gpu'
              ? 'The card\'s own curve: below about 40 °C the fans stop (zero-RPM idle).'
              : 'The board\'s curve, as ASUS Fan Xpert sets it (the reference PC\'s smart mode). Drag the points.'} Curves change instantly; no new simulation is needed.</p>
          </HudWindow>
        )}

        {win.trend && trends && (
          <HudWindow id="trends" code="03" title="Trends" side="right" x={win.curve ? 384 : 20} y={72} w={340} layoutKey={layoutKey} onClose={() => setWin({ ...win, trend: false })}>
            <TrendChart title="Temperature against room temperature" xs={trends.rooms} series={trends.room} xFmt={(x) => `${x}`} xUnit="room °C" now={scn.ambient_c} />
            <div className="tr-sw">
              <span className="lbl">sweep</span>
              <span className="seg3">{(['gpu', 'cpu'] as const).map((k) => <button key={k} className={sweep === k ? 'on' : ''} onClick={() => setSweep(k)}>{k} load</button>)}</span>
            </div>
            <TrendChart title={`Temperature against ${sweep.toUpperCase()} load (the other at ${pct(sweep === 'gpu' ? scn.cpu : scn.gpu)})`} xs={trends.loads} series={trends.load}
              xFmt={(x) => `${x}%`} xUnit={`${sweep} load`} now={(sweep === 'gpu' ? scn.gpu : scn.cpu) * 100} />
            <p className="fine">Solid: the fans follow their curves. Dashed: the same fans held at the speed they run at now. With fans that do not react, a room 5 °C warmer makes everything exactly 5 °C hotter.</p>
          </HudWindow>
        )}

        {win.cfg && viewBuild && cs && lib && (
          <HudWindow id="configure" code="05" title="Configure" side="left" x={win.scn ? 374 : 20} y={72} w={330} layoutKey={layoutKey} onClose={() => setWin({ ...win, cfg: false })} meta={online ? 'LIVE' : 'ENGINE OFF'}>
            <div className="cfg-parts">
              {([['Case', 'case', lib.cases], ['GPU', 'gpu', lib.gpus], ['Cooler', 'cooler', lib.coolers]] as const).map(([l, k, list]) => (
                <label key={k}><span className="lbl">{l}</span>
                  <select className="field" value={viewBuild[k]} onChange={(e) => setPart(k, e.target.value)}>
                    {(list as { id: string; name: string }[]).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </label>
              ))}
              <label><span className="lbl">CPU · no new simulation needed</span>
                <select className="field" value={cpuId} onChange={(e) => setCpuId(e.target.value)}>
                  {lib.cpus.map((c) => <option key={c.id} value={c.id}>{c.name} · {c.max_w} W, Tj {c.t_max_c} °C</option>)}
                </select>
              </label>
            </div>
            <span className="lbl">Case fans</span>
            <div className="mounts">
              {cs.mounts.map((m) => {
                const f = viewBuild.fans.find((x) => x.slot === m.slot)
                const mode: Mode = f?.mode ?? 'off'
                const live = res?.fans.find((x) => x.slot === m.slot)
                return (
                  <div key={m.slot} className="mt">
                    <div className="mt-1">
                      <span className="mt-n">{WHERE[m.slot] ?? m.slot}<small>{m.sizes.join('/')}</small></span>
                      <span className="seg3">{(['in', 'off', 'out'] as Mode[]).map((md) => <button key={md} className={mode === md ? 'on ' + md : ''} onClick={() => setFan(m.slot, { mode: md })}>{md}</button>)}</span>
                    </div>
                    {f && (
                      <div className="mt-2">
                        <select className="field sm" value={f.fan} onChange={(e) => setFan(m.slot, { fan: e.target.value })}>
                          {lib.fans.filter((x) => m.sizes.includes(x.size_mm)).map((x) => <option key={x.id} value={x.id}>{x.name.replace(/^Lian Li LANCOOL 207 /, 'Stock ')} · {x.cfm} CFM</option>)}
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
            {view === 'custom' && custom && !banks[layoutId(custom)] && (
              <button className="run" disabled={!online || !!job} onClick={() => fetchBank(custom, 'your edited layout').then((p) => p && setBanks((m) => ({ ...m, [layoutId(custom)]: p })))}>
                {online ? 'Simulate this layout' : 'Engine offline'}</button>
            )}
            <p className="fine">Changing parts or fans changes the airflow: CaseFlow simulates it at 3 case-fan speeds (about 1.5 min on the GPU), then room, load and curves update instantly. "curve" fans follow the case-fan curve; "fixed" fans hold their duty.</p>
          </HudWindow>
        )}

        {win.info && (
          <HudWindow id="method" code="06" title="Method" side="right" x={win.curve ? 384 : 20} y={72} w={380} layoutKey={layoutKey} onClose={() => setWin({ ...win, info: false })}>
            <div className="method">
              <p><span className="lbl">Airflow</span>Our own D3Q19 lattice-Boltzmann solver (PyTorch, GPU, Smagorinsky turbulence) on a 6 mm grid of the case. Walls, PSU, board, GPU and cooler are solid; case fans and the GPU / CPU cooler fans are cells with a set air velocity; empty mesh mounts and slot covers are open to the room.</p>
              <p><span className="lbl">Fans</span>Each case fan runs at its operating point: a linear fan curve (rated flow and static pressure, scaled by the fan laws) against one case pressure, balanced by the passive openings. Its speed comes from its temperature curve: speed = s_min + (1 − s_min) × duty.</p>
              <p><span className="lbl">Bank</span>Each layout is simulated at three case-fan duties (12, 45, 100 %), all in one GPU batch. For a fixed airflow the air's temperature rise is proportional to power, so the bank keeps the rise per watt of CPU and of GPU power; any load is a weighted sum.</p>
              <p><span className="lbl">Steady state</span>Part temperature = room + intake air rise + P / G, with the cooler conductance G = 1 / (r_base + r_air × speed^-0.7) rising with its own fans. The solver finds where temperatures and fan speeds agree (milliseconds, in your browser); a part past its limit is throttled. Checked against full simulations at the solved speeds: within 0.5 °C.</p>
              <p><span className="lbl">Dust</span>40 000 particles of 20 µm enter where air comes in (the bottom filter stops 85%), drift with the flow plus turbulent diffusion and settling, and stick where they touch. Faster fans pull in more.</p>
              <p><span className="lbl">Inputs</span>CaseFlow is a simulator: room, load, curves and parts are settings. Defaults are the reference PC: its Fan Xpert curves, a 23 °C room, and its CPU-heavy game (86 W CPU, 146 W GPU) - the model gives CPU 65.6 / GPU 49.7 °C there; the PC measured 65.3 / 49.3 °C.</p>
              <p><span className="lbl">Search</span>Every placement and direction of your three fans (4 mounts × in / out = {n || 32} layouts) has its own bank; the ranking is redone for whatever scenario you set.</p>
              <p><span className="lbl">Limits</span>Mount positions, opening porosity and cooler conductance are estimates; the cooler and GPU fans stir the case air at one representative speed; no buoyancy. Light loads read a few °C low on the CPU (boost spikes). Trust differences more than absolute °C.</p>
            </div>
          </HudWindow>
        )}
      </div>

      {isBankless && <div className="toast"><span>No simulation of this layout yet{online ? '' : ' - start the engine to simulate it'}.</span></div>}

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
        <button className={win.info ? 'on' : ''} onClick={() => setWin({ ...win, info: !win.info })}>Method</button>
        <button title="Reset the view, the windows and the screens (drag a screen by its title; double-click the title to put it back)"
          onClick={() => { resetWindows(); resetHolos(); setLayoutKey((k) => k + 1) }}>⟲</button>
      </nav>
    </div>
  )
}
