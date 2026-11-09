import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { getJSON, pingServer, postJSON } from './api'
import { AirPanel, DustPanel, Holo, PartPanel, SearchPanel, VerdictPanel } from './Holo'
import HudWindow, { resetWindows } from './HudWindow'
import { THERMAL_CSS } from './palette'
import CaseScene, { T_MAX, type HoloSpec, type Layers, type Pick } from './Scene'
import { WHERE, type Build, type FanSlot, type Lib, type Mode, type Opt, type RankRow, type SimRes, type ViewKey } from './types'

const amazon = (q: string) => `https://www.amazon.com/s?k=${encodeURIComponent(q)}`
const layoutName = (r?: SimRes | null) => (r ? r.label.split(': ').slice(1).join(': ') || r.label : '')

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

const fanKey = (fs: FanSlot[]) => fs.map((f) => `${f.slot}:${f.fan}:${f.mode}`).sort().join('|')

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

export default function App() {
  const [online, setOnline] = useState(false)
  const [lib, setLib] = useState<Lib | null>(null)
  const [opt, setOpt] = useState<Opt | null>(null)
  const [views, setViews] = useState<Partial<Record<ViewKey, SimRes>>>({})
  const [view, setView] = useState<ViewKey>('current')
  const [layers, setLayers] = useState<Layers>({ flow: true, lines: false, dust: true, heat: false, panels: true })
  const [build, setBuild] = useState<Build | null>(null)
  const [cfgOpen, setCfgOpen] = useState(false)
  const [infoOpen, setInfoOpen] = useState(false)
  const [allRows, setAllRows] = useState(false)
  const [job, setJob] = useState<{ p: number; label: string } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [layoutKey, setLayoutKey] = useState(0)
  const narrow = useNarrow()
  const [menu, setMenu] = useState<Pick | null>(null)
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
    getJSON<Opt>('/data/optimize.json').then(setOpt).catch(() => setMsg('Precomputed results not found - run scripts/precompute.py'))
    const v: Partial<Record<ViewKey, SimRes>> = {}
    Promise.all((['current', 'best', 'cpu', 'worst'] as ViewKey[]).map((k) => getJSON<SimRes>(`/data/${k}.json`).then((r) => { v[k] = r }).catch(() => {})))
      .then(() => { setViews({ ...v }); if (v.current?.build) setBuild(v.current.build) })
  }, [])
  useEffect(() => {
    getJSON<Lib>(online ? '/api/library' : '/data/library.json').catch(() => getJSON<Lib>('/data/library.json')).then(setLib).catch(() => {})
  }, [online])
  useEffect(() => {
    if (!msg) return
    const id = window.setTimeout(() => setMsg(null), 6000)
    return () => window.clearTimeout(id)
  }, [msg])

  // ---------- derived ----------
  const rows = opt?.ranking ?? []
  const n = rows.length
  const rankOf = (name?: string | null) => (name ? rows.findIndex((r) => r.name === name) + 1 : 0)
  const curRow = rows.find((r) => r.current)
  const bestIsCur = !!curRow && curRow.name === opt?.best
  const topRow = rows.find((r) => r.name === opt?.top) ?? rows[0]
  const curTied = !!curRow && (opt?.tied?.includes(curRow.name) ?? false) && (opt?.tied?.length ?? 0) > 1
  const cur = views.current ?? null
  const res = views[view] ?? cur
  const other = bestIsCur ? views.worst : views.best
  const otherKey: ViewKey = bestIsCur ? 'worst' : 'best'
  const cs = lib && build ? lib.cases.find((c) => c.id === build.case) ?? null : null
  const gpu = lib && build ? lib.gpus.find((c) => c.id === build.gpu) ?? null : null
  const cooler = lib && build ? lib.coolers.find((c) => c.id === build.cooler) ?? null : null
  const key = (r: RankRow) => r.combined ?? r.score
  const [lo, hi] = useMemo(() => (rows.length ? [Math.min(...rows.map(key)), Math.max(...rows.map(key))] : [0, 1]), [rows])
  const tagOf = (r: RankRow) => [
    r.current && 'YOURS', (opt?.tied?.length ?? 0) > 1 && opt?.tied?.includes(r.name) && 'TIED #1', r.name === opt?.best && !r.current && 'RECOMMENDED', r.name === opt?.cpu && 'LOWEST CPU', r.name === opt?.worst && 'WORST',
  ].filter(Boolean) as string[]
  const shown = rows.map((r, i) => ({ r, i })).filter(({ r, i }) => allRows || i < 6 || tagOf(r).length > 0)
  const stock = useMemo(() => {
    if (!cs || !build) return { stock: 0, own: 0 }
    const pre = new Set(cs.preinstalled.map((p) => `${p.slot}:${p.fan}`))
    const s = build.fans.filter((f) => pre.has(`${f.slot}:${f.fan}`)).length
    return { stock: s, own: build.fans.length - s }
  }, [cs, build])
  const viewOf = (r: RankRow): ViewKey | null =>
    r.current ? 'current' : r.name === opt?.best ? 'best' : r.name === opt?.cpu ? 'cpu' : r.name === opt?.worst ? 'worst' : null
  const tabs = (['current', 'best', 'cpu', 'worst', 'custom'] as ViewKey[]).filter((k) => views[k] && !(k === 'best' && bestIsCur))
  const tabLabel: Record<ViewKey, [string, string]> = {
    current: ['Your layout', curTied ? 'tied #1' : `#${rankOf(curRow?.name)}/${n}`], best: ['Recommended', `#1/${n}`],
    cpu: ['Lowest CPU', `#${rankOf(opt?.cpu)}/${n}`], worst: ['Worst', `#${n}/${n}`], custom: ['Custom', ''],
  }
  const compare = view !== 'current'

  // ---------- actions ----------
  const simulate = async (b: Build, label: string) => {
    setMsg(null)
    setJob({ p: 0, label })
    try {
      const { job: id } = await postJSON<{ job: string }>('/api/simulate', { build: b })
      for (;;) {
        await new Promise((r) => setTimeout(r, 700))
        const j = await getJSON<{ status: string; progress: number; error?: string; result?: SimRes }>(`/api/job/${id}`)
        if (j.status === 'error') throw new Error(j.error)
        setJob({ p: j.progress, label })
        if (j.status === 'done' && j.result) {
          setViews((v) => ({ ...v, custom: { ...j.result!, label: `Custom: ${label}`, build: b } }))
          setView('custom')
          break
        }
      }
    } catch (e) {
      setMsg(String(e).slice(0, 220))
    } finally {
      setJob(null)
    }
  }
  const openRow = (r: RankRow) => {
    const k = viewOf(r)
    if (k && views[k]) return setView(k)
    if (!online) return setMsg('Engine offline - start it with start.bat to simulate this layout live.')
    setBuild(r.build)
    simulate(r.build, r.name)
  }
  const setFan = (slot: string, patch: Partial<FanSlot>) => {
    if (!build || !cs) return
    const m = cs.mounts.find((x) => x.slot === slot)!
    const ex = build.fans.find((f) => f.slot === slot)
    const fallback = m.sizes.includes(120) ? 'generic-120' : 'generic-140'
    const next = { ...(ex ?? { slot, fan: fallback, mode: 'off' as Mode, pct: 80 }), ...patch }
    const fans = build.fans.filter((f) => f.slot !== slot)
    if (next.mode !== 'off') fans.push(next)
    setBuild({ ...build, fans })
  }
  const steps = cur?.build && views.best?.build && !bestIsCur ? changes(cur.build.fans, views.best.build.fans) : []

  /** Click on a fan or an empty mount in 3D: flip it, switch it off, or add one; then show the result. */
  const editFan = (slot: string, action: 'flip' | 'off' | 'in' | 'out') => {
    const base = res?.build ?? build
    setMenu(null)
    if (!base) return
    if (job) return setMsg('A simulation is still running - wait for it to finish.')
    const b: Build = { ...base, fans: base.fans.map((f) => ({ ...f })) }
    const where = WHERE[slot] ?? slot
    let desc = ''
    const i = b.fans.findIndex((f) => f.slot === slot)
    if (action === 'flip' && i >= 0) {
      b.fans[i].mode = b.fans[i].mode === 'in' ? 'out' : 'in'
      desc = `${where} fan flipped to ${b.fans[i].mode}`
    } else if (action === 'off' && i >= 0) {
      b.fans.splice(i, 1)
      desc = `${where} fan off`
    } else if (action === 'in' || action === 'out') {
      const m = cs?.mounts.find((x) => x.slot === slot)
      b.fans.push({ slot, fan: m && !m.sizes.includes(120) ? 'generic-140' : 'generic-120', mode: action, pct: 80 })
      desc = `fan added at ${where}, blowing ${action}`
    } else return
    setBuild(b)
    const row = rows.find((r) => fanKey(r.build.fans) === fanKey(b.fans))
    const rank = row ? rankOf(row.name) : 0
    const k = row ? viewOf(row) : null
    if (k && views[k]) {
      setView(k)
      setMsg(`${desc} → layout #${rank}/${n}`)
    } else if (online) {
      simulate(b, desc + (row ? ` · #${rank}/${n} in the search` : ''))
    } else if (row) {
      setMsg(`${desc} → layout #${rank}/${n}: CPU ${row.cpu_c.toFixed(1)} °C, GPU ${row.gpu_c.toFixed(1)} °C, dust ×${(row.dust_rel ?? 1).toFixed(2)}. Start the engine to see its airflow.`)
    } else {
      setMsg(`${desc}: not one of the ${n} searched layouts. Start the engine (start.bat) to simulate it.`)
    }
  }
  const menuFan = menu && res ? res.fans.find((f) => f.slot === menu.slot) : undefined
  const menuMount = menu && res ? res.mounts.find((m) => m.slot === menu.slot) : undefined

  // ---------- floating screens ----------
  const flash = `${view}:${res?.label ?? ''}`
  const screens: (HoloSpec & { title: string })[] = []
  if (res) {
    screens.push(
      { id: 'search', title: 'Layout search', pos: [-3.35, -0.04, -0.12], yaw: 0.2, content: (
        <Holo tag="LAYOUT SEARCH" sub={`${n}/${n} simulated`} width={300} flashKey="search">
          <SearchPanel opt={opt} rows={rows} shown={shown} tagOf={tagOf} onOpen={openRow} allRows={allRows} setAllRows={setAllRows} lo={lo} hi={hi}
            isOn={(r) => { const v = viewOf(r); return (!!v && v === view) || (view === 'custom' && views.custom?.label === `Custom: ${r.name}`) }} />
        </Holo>) },
      { id: 'cpu', title: 'CPU', pos: [-1.6, 1.08, 0.5], yaw: 0.06, anchor: 'cpu', content: (
        <Holo tag="CPU" sub={cooler?.name.replace('Thermalright ', '').replace(' ARGB', '') ?? 'CPU cooler'} width={250} flashKey={'cpu' + flash}>
          <PartPanel part="cpu" res={res} cur={cur} compare={compare} name={cooler?.type.replace(/_/g, '-') ?? ''}
            extra={[['cooler', `${(res.parts.cpu as { g_w_per_k?: number }).g_w_per_k?.toFixed(1) ?? '2.0'} W/K`]]} />
        </Holo>) },
      { id: 'gpu', title: 'GPU', pos: [-1.6, -0.78, 0.6], yaw: 0.06, anchor: 'gpu', content: (
        <Holo tag="GPU" sub={gpu?.name.replace('NVIDIA GeForce ', '') ?? 'GPU'} width={250} flashKey={'gpu' + flash}>
          <PartPanel part="gpu" res={res} cur={cur} compare={compare} name={gpu?.cooler.replace(/_/g, '-') ?? ''}
            extra={[['cooler', `${(res.parts.gpu as { g_w_per_k?: number }).g_w_per_k?.toFixed(1) ?? '8.5'} W/K`]]} />
        </Holo>) },
      { id: 'air', title: 'Airflow', pos: [1.6, 1.08, 0.5], yaw: -0.06, anchor: 'intake', content: (
        <Holo tag="AIRFLOW" sub={layoutName(res)} width={262} flashKey={'air' + flash}>
          <AirPanel res={res} cur={cur} compare={compare} />
        </Holo>) },
      { id: 'dust', title: 'Dust', pos: [1.6, -0.78, 0.6], yaw: -0.06, anchor: 'glass', content: (
        <Holo tag="DUST" sub="20 µm particles" width={250} flashKey={'dust' + flash}>
          <DustPanel res={res} cur={cur} compare={compare} />
        </Holo>) },
    )
    if (cur && other)
      screens.push({ id: 'verdict', title: 'Verdict', pos: [3.35, -0.04, -0.12], yaw: -0.2, content: (
        <Holo tag="VERDICT" sub={bestIsCur ? 'yours vs worst' : 'yours vs best'} width={290} flashKey="verdict">
          <VerdictPanel cur={cur} other={other} bestIsCur={bestIsCur} n={n} rankCur={rankOf(curRow?.name)} rankOther={rankOf(bestIsCur ? opt?.worst : opt?.best)}
            steps={steps} onOpen={(k) => setView(k === 'current' ? 'current' : otherKey)}
            tie={curTied && topRow && !topRow.current && curRow ? { name: topRow.name, delta: (curRow.combined ?? curRow.score) - (topRow.combined ?? topRow.score), margin: opt?.tie_margin ?? 0.5 } : null} />
        </Holo>) })
  }

  return (
    <div className="cf">
      <CaseScene res={res} layers={layers} panels={narrow ? [] : screens} resetKey={layoutKey} onPick={setMenu} selected={menu?.slot ?? null} />
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
            <span className="brand-sub">{cs ? `${cs.name.replace('Lian Li ', '')} · ${stock.stock} stock + ${stock.own} own fans` : 'airflow · heat · dust'}
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

      {menu && res && (
        <div className="fanmenu" style={{ left: Math.min(menu.x + 16, window.innerWidth - 250), top: Math.min(Math.max(menu.y - 24, 64), window.innerHeight - 220) }}>
          <i className="hc tl" /><i className="hc br" />
          <div className="fm-h">
            <span className="h-tag">{menu.empty ? 'MOUNT' : 'FAN'}</span>
            <span className="fm-where">{WHERE[menu.slot] ?? menu.slot}</span>
            {menuFan && <span className={'fm-mode m-' + menuFan.mode}>{menuFan.mode === 'in' ? '▸ in' : '◂ out'}</span>}
            <button className="win-btn" title="Close" onClick={() => setMenu(null)}>×</button>
          </div>
          <div className="fm-sub">{menuFan
            ? `${menuFan.name.replace(/^Lian Li LANCOOL 207 /, 'Stock ').replace(/, model unknown.*$/, '')} · ${(menuFan.q_m3s * 2119).toFixed(0)} CFM now`
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
          <div className="fm-note">{online ? 'Runs a new simulation (~40 s) unless the layout is already stored.' : 'Engine offline: stored results only.'}</div>
        </div>
      )}

      {(job || msg) && (
        <div className="toast">
          {job ? <><span>SIMULATING · {job.label}</span><span className="mono">{Math.round(job.p * 100)}%</span><i style={{ width: `${job.p * 100}%` }} /></> : <span>{msg}</span>}
        </div>
      )}

      {narrow && layers.panels && (
        <div className="holo-column">{screens.map((s) => <div key={s.id}>{s.content as ReactNode}</div>)}</div>
      )}

      <div className="hud">
        {cfgOpen && build && cs && (
          <HudWindow id="configure" code="05" title="Configure" side="left" x={20} y={72} w={330} layoutKey={layoutKey} onClose={() => setCfgOpen(false)} meta={online ? 'LIVE' : 'ENGINE OFF'}>
            <div className="cfg-parts">
              {([['Case', 'case', lib?.cases], ['GPU', 'gpu', lib?.gpus], ['Cooler', 'cooler', lib?.coolers]] as const).map(([l, k, list]) => (
                <label key={k}><span className="lbl">{l}</span>
                  <select className="field" value={build[k]} onChange={(e) => setBuild({ ...build, [k]: e.target.value })}>
                    {(list ?? []).map((p: { id: string; name: string }) => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </label>
              ))}
            </div>
            <span className="lbl">Case fans</span>
            <div className="mounts">
              {cs.mounts.map((m) => {
                const f = build.fans.find((x) => x.slot === m.slot)
                const mode: Mode = f?.mode ?? 'off'
                return (
                  <div key={m.slot} className="mt">
                    <div className="mt-1">
                      <span className="mt-n">{WHERE[m.slot] ?? m.slot}<small>{m.sizes.join('/')}</small></span>
                      <span className="seg3">{(['in', 'off', 'out'] as Mode[]).map((md) => <button key={md} className={mode === md ? 'on ' + md : ''} onClick={() => setFan(m.slot, { mode: md })}>{md}</button>)}</span>
                    </div>
                    {f && (
                      <div className="mt-2">
                        <select className="field sm" value={f.fan} onChange={(e) => setFan(m.slot, { fan: e.target.value })}>
                          {lib?.fans.filter((x) => m.sizes.includes(x.size_mm)).map((x) => <option key={x.id} value={x.id}>{x.name.replace(/^Lian Li LANCOOL 207 /, 'Stock ')} · {x.cfm} CFM</option>)}
                        </select>
                        <input type="range" min={30} max={100} step={10} value={f.pct} onChange={(e) => setFan(m.slot, { pct: +e.target.value })} />
                        <span className="mono pct" title="% of this fan's top speed. Set it to the rpm Armoury Crate (Fan Xpert) or the BIOS shows while you game.">{f.pct}%<small>{(() => { const p = lib?.fans.find((x) => x.id === f.fan); return p ? `${Math.round(p.rpm_max * f.pct / 100)} rpm` : '' })()}</small></span>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
            <div className="cfg-row">
              <label><span className="lbl">CPU W</span><input className="field" value={build.load.cpu_w} onChange={(e) => setBuild({ ...build, load: { ...build.load, cpu_w: +e.target.value || 0 } })} /></label>
              <label><span className="lbl">GPU W</span><input className="field" value={build.load.gpu_w} onChange={(e) => setBuild({ ...build, load: { ...build.load, gpu_w: +e.target.value || 0 } })} /></label>
              <label><span className="lbl">Room °C</span><input className="field" value={build.load.ambient_c} onChange={(e) => setBuild({ ...build, load: { ...build.load, ambient_c: +e.target.value || 0 } })} /></label>
            </div>
            {build.load.gpu_w < 200 && <p className="fine warnline">Below about 200 W a real GPU slows or stops its fans (0 % at idle); this model keeps them at a fixed speed, so it shows the GPU cooler than it would be.</p>}
            <button className="run" disabled={!online || !!job} onClick={() => simulate(build, 'edited layout')}>{online ? 'Run simulation' : 'Engine offline'}</button>
            <p className="fine">About 45 s on the GPU: airflow, then heat, then dust.</p>
          </HudWindow>
        )}

        {infoOpen && (
          <HudWindow id="method" code="06" title="Method" side="right" x={20} y={72} w={380} layoutKey={layoutKey} onClose={() => setInfoOpen(false)}>
            <div className="method">
              <p><span className="lbl">Airflow</span>Our own D3Q19 lattice-Boltzmann solver (PyTorch, GPU, Smagorinsky turbulence) on a 6 mm grid of the case. Walls, PSU, board, GPU and cooler are solid; case fans and the GPU / CPU cooler fans are cells with a set air velocity; empty mesh mounts and slot covers are open to the room, an open boundary at room pressure, so fans deliver their set flow.</p>
              <p><span className="lbl">Fans</span>Each case fan runs at its operating point: a linear fan curve (rated flow and static pressure, scaled by fan laws) against one case pressure, which the passive openings balance as orifices. This is why "all intake, no exhaust" does not win: its fans stall against +7.8 Pa.</p>
              <p><span className="lbl">Heat</span>CPU and GPU power enters the air in their fin stacks and is carried by the solved flow (conservative advection–diffusion, iterated to convergence; all 388 W leave through the openings). Part temperature = intake air rise + power / cooler conductance.</p>
              <p><span className="lbl">Dust</span>40 000 particles of 20 µm enter where air comes in (the bottom filter stops 85%), drift with the flow plus turbulent diffusion and settling, and stick where they touch. Dust × is the rate at which outside dust settles inside (unfiltered intake × share that settles), relative to your layout.</p>
              <p><span className="lbl">Inputs</span>CaseFlow is a simulator: fan speeds, CPU W, GPU W and room °C are settings, nothing is read from this PC. Defaults: stock fans 70 %, your fans 80 % of top speed; CPU 88 W (the 9700X's power limit, all cores busy; games use about 70 W); GPU 300 W (typical in games). Utilisation is not used: heat follows watts, not % busy. Room °C and cooler conductance shift every layout equally. Fan speed and the CPU / GPU split can reorder them, so 3 loads and 5 fan-speed settings were tested: the worst layouts never moved, and the recommended one was never meaningfully worse than yours (see the guide).</p>
              <p><span className="lbl">Search</span>Every placement and direction of your three fans (4 mounts × in / out = {n || 32} layouts) is simulated in full, about 45 s each.</p>
              <p><span className="lbl">Limits</span>Mount positions, opening porosity and cooler conductance are estimates; trust the differences between layouts more than the absolute °C.</p>
            </div>
          </HudWindow>
        )}
      </div>

      <nav className="dock">
        {([['panels', 'Screens'], ['flow', 'Flow'], ['lines', 'Lines'], ['dust', 'Dust'], ['heat', 'Heat']] as const).map(([k, l]) => (
          <button key={k} className={layers[k] ? 'on' : ''} onClick={() => setLayers({ ...layers, [k]: !layers[k] })}>{l}</button>
        ))}
        <span className="sep" />
        <span className="legend"><span className="lbl">Air ΔT</span><span>0</span><i style={{ background: THERMAL_CSS }} /><span>+{T_MAX} °C</span></span>
        <span className="sep" />
        <button className={cfgOpen ? 'on' : ''} onClick={() => setCfgOpen(!cfgOpen)}>Configure</button>
        <button className={infoOpen ? 'on' : ''} onClick={() => setInfoOpen(!infoOpen)}>Method</button>
        <button title="Reset the view and the windows" onClick={() => { resetWindows(); setLayoutKey((k) => k + 1) }}>⟲</button>
      </nav>
    </div>
  )
}
