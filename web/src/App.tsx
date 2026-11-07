import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { getJSON, pingServer, postJSON } from './api'
import HudWindow, { resetWindows } from './HudWindow'
import { THERMAL_CSS } from './palette'
import CaseScene, { T_MAX, type Layers } from './Scene'
import Schematic from './Schematic'
import { WHERE, type Build, type FanSlot, type HW, type Lib, type Mode, type Opt, type RankRow, type SimRes, type ViewKey } from './types'

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

function Delta({ v, unit = '°' }: { v: number | null; unit?: string }) {
  if (v == null) return null
  const cls = v < -0.05 ? 'good' : v > 0.05 ? 'bad' : 'flat'
  return <span className={'d ' + cls}>{v > 0 ? '+' : v < 0 ? '−' : '±'}{Math.abs(v).toFixed(1)}{unit}</span>
}

export default function App() {
  const [online, setOnline] = useState(false)
  const [hw, setHw] = useState<HW | null>(null)
  const [lib, setLib] = useState<Lib | null>(null)
  const [opt, setOpt] = useState<Opt | null>(null)
  const [views, setViews] = useState<Partial<Record<ViewKey, SimRes>>>({})
  const [view, setView] = useState<ViewKey>('current')
  const [layers, setLayers] = useState<Layers>({ flow: true, lines: false, dust: true, heat: false })
  const [build, setBuild] = useState<Build | null>(null)
  const [cfgOpen, setCfgOpen] = useState(false)
  const [infoOpen, setInfoOpen] = useState(false)
  const [allRows, setAllRows] = useState(false)
  const [cal, setCal] = useState({ temp: '', watts: '' })
  const [job, setJob] = useState<{ p: number; label: string } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [layoutKey, setLayoutKey] = useState(0)
  const [hs, setHs] = useState<Record<string, number>>({})
  const onH = useMemo(() => {
    const m: Record<string, (h: number) => void> = {}
    for (const k of ['system', 'telemetry']) m[k] = (h) => setHs((s) => (s[k] === h ? s : { ...s, [k]: h }))
    return m
  }, [])
  const searchY = 72 + (hs.system ?? 230) + 12
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => { listRef.current?.querySelector('.rk.on')?.scrollIntoView({ block: 'nearest' }) }, [view, allRows, opt])
  const compareY = 72 + (hs.telemetry ?? 370) + 12

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
    if (!online) return
    let alive = true
    getJSON<HW>('/api/hardware?refresh=true').then((h) => alive && setHw(h)).catch(() => {})
    const id = window.setInterval(() => getJSON<HW>('/api/hardware').then((h) => alive && setHw(h)).catch(() => {}), 5000)
    return () => { alive = false; window.clearInterval(id) }
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
  const cur = views.current ?? null
  const res = views[view] ?? cur
  const other = bestIsCur ? views.worst : views.best
  const otherKey: ViewKey = bestIsCur ? 'worst' : 'best'
  const cs = lib && build ? lib.cases.find((c) => c.id === build.case) ?? null : null
  const gpu = lib && build ? lib.gpus.find((c) => c.id === build.gpu) ?? null : null
  const cooler = lib && build ? lib.coolers.find((c) => c.id === build.cooler) ?? null : null
  const g0 = hw?.gpus[0]
  const key = (r: RankRow) => r.combined ?? r.score
  const [lo, hi] = useMemo(() => (rows.length ? [Math.min(...rows.map(key)), Math.max(...rows.map(key))] : [0, 1]), [rows])
  const tagOf = (r: RankRow) => [
    r.current && 'YOURS', r.name === opt?.best && !r.current && 'RECOMMENDED', r.name === opt?.cpu && 'LOWEST CPU', r.name === opt?.worst && 'WORST',
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
    current: ['Your layout', `#${rankOf(curRow?.name)}/${n}`], best: ['Recommended', `#1/${n}`],
    cpu: ['Lowest CPU', `#${rankOf(opt?.cpu)}/${n}`], worst: ['Worst', `#${n}/${n}`], custom: ['Custom', ''],
  }
  const dd = (a?: number, b?: number) => (a == null || b == null || view === 'current' ? null : a - b)

  // ---------- actions ----------
  const simulate = async (b: Build, label: string) => {
    setMsg(null)
    setJob({ p: 0, label })
    try {
      const calib = cal.temp && cal.watts ? { temp_c: +cal.temp, watts: +cal.watts } : null
      const { job: id } = await postJSON<{ job: string }>('/api/simulate', { build: b, calib_gpu: calib })
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
  const worstD = views.worst && cur ? { cpu: views.worst.parts.cpu.temp_c - cur.parts.cpu.temp_c, gpu: views.worst.parts.gpu.temp_c - cur.parts.gpu.temp_c } : null

  return (
    <div className="cf">
      <CaseScene res={res} layers={layers} />
      <div className="vignette" />

      <header className="top">
        <div className="brand">
          <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
            <rect x="1.5" y="1.5" width="17" height="17" fill="none" stroke="rgba(140,200,255,.55)" />
            <path d="M4.5 14 C8.5 14 9 6.5 15.5 6.5" fill="none" stroke="#63e0ff" strokeWidth="1.4" />
            <circle cx="15.5" cy="6.5" r="1.5" fill="#63e0ff" />
          </svg>
          <span className="brand-name">CASEFLOW</span>
          <span className="brand-sub">airflow · heat · dust</span>
        </div>
        <nav className="views">
          {tabs.map((k) => (
            <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>
              {tabLabel[k][0]}{tabLabel[k][1] && <span className="v-sub">{tabLabel[k][1]}</span>}
            </button>
          ))}
        </nav>
        <div className="status">
          {g0 && <span className="mono">GPU {g0.temp_c?.toFixed(0)}°C · {g0.power_w?.toFixed(0)} W</span>}
          <span className="mono"><i className={'led' + (online ? ' on' : '')} />{online ? 'ENGINE ONLINE' : 'OFFLINE · PRECOMPUTED'}</span>
        </div>
      </header>

      {(job || msg) && (
        <div className="toast">
          {job ? <><span>SIMULATING · {job.label}</span><span className="mono">{Math.round(job.p * 100)}%</span><i style={{ width: `${job.p * 100}%` }} /></> : <span>{msg}</span>}
        </div>
      )}

      <div className="hud">
        <HudWindow id="system" code="01" title="System" side="left" x={20} y={72} w={300} layoutKey={layoutKey} onHeight={onH.system}
          meta={cs ? `${cs.dims_mm.D}×${cs.dims_mm.W}×${cs.dims_mm.H}` : undefined}>
          <div className="sys-name">{cs?.name ?? '—'}</div>
          <dl className="kv">
            <dt>GPU</dt>
            <dd>{gpu?.name.replace('NVIDIA GeForce ', '') ?? '—'}<small>{gpu ? `${gpu.tdp_w} W · ${gpu.cooler.replace(/_/g, '-')}` : ''}{gpu && <span className="links"><a href={gpu.sources[0]} target="_blank" rel="noreferrer">spec ↗</a><a href={amazon(gpu.amazon_query ?? gpu.name)} target="_blank" rel="noreferrer">buy ↗</a></span>}</small></dd>
            <dt>Cooler</dt>
            <dd>{cooler?.name.replace(' ARGB', '').replace('Thermalright ', '') ?? '—'}<small>{cooler ? `${cooler.type.replace('_', ' ')} · ${cooler.height_mm} mm` : ''}{cooler && <span className="links"><a href={cooler.sources[0]} target="_blank" rel="noreferrer">spec ↗</a><a href={amazon(cooler.amazon_query ?? cooler.name)} target="_blank" rel="noreferrer">buy ↗</a></span>}</small></dd>
            <dt>CPU</dt>
            <dd>{hw ? hw.cpu.name.replace(/ \d+-Core Processor/, '') : 'detected when the engine runs'}<small>{hw ? `${hw.ram_gb} GB RAM` : ''}</small></dd>
            <dt>Fans</dt>
            <dd>{stock.stock} stock + {stock.own} own<small>{build ? `${build.fans.length} case fans · cooler 2 × 120` : ''}</small></dd>
          </dl>
        </HudWindow>

        <HudWindow id="search" code="02" title="Layout search" side="left" x={20} y={searchY} w={300} layoutKey={layoutKey} meta={n ? `${n}/${n} RUN` : undefined}>
          <p className="lede">Your three 120 mm fans on every free mount, each blowing in or out. The four stock fans stay put.</p>
          <div className="rk-h"><span>#</span><span>Layout</span><span>CPU</span><span>GPU</span><span>Dust</span></div>
          <div className="rk-list" ref={listRef} style={{ maxHeight: `max(170px, calc(100vh - ${searchY + 225}px))` }}>
            {shown.map(({ r, i }, k) => {
              const gap = k > 0 && shown[k - 1].i !== i - 1
              const v = viewOf(r)
              const on = (v && v === view) || (view === 'custom' && views.custom?.label === `Custom: ${r.name}`)
              const tags = tagOf(r)
              return (
                <div key={r.name}>
                  {gap && <div className="rk-gap">· · ·</div>}
                  <button className={'rk' + (on ? ' on' : '')} style={{ '--w': 1 - (key(r) - lo) / Math.max(hi - lo, 1e-6) } as CSSProperties} onClick={() => openRow(r)}>
                    <span className="n">{String(i + 1).padStart(2, '0')}</span>
                    <span className="nm">{r.name}{tags.map((t) => <em key={t} className={t === 'WORST' ? 'warn' : ''}>{t}</em>)}</span>
                    <span className="v">{r.cpu_c.toFixed(1)}</span>
                    <span className="v">{r.gpu_c.toFixed(1)}</span>
                    <span className="v">{Math.round(r.dust_in * 100)}%</span>
                  </button>
                </div>
              )
            })}
          </div>
          {n > 0 && <button className="textbtn" onClick={() => setAllRows(!allRows)}>{allRows ? 'Show fewer' : `Show all ${n}`}</button>}
          <p className="fine">°C at CPU 88 W / GPU 300 W, room 25 °C. Ranked by CPU + GPU + {opt?.dust_weight ?? 2} °C × dust let in.</p>
        </HudWindow>

        {res && (
          <HudWindow id="telemetry" code="03" title="Telemetry" side="right" x={20} y={72} w={310} layoutKey={layoutKey} onHeight={onH.telemetry} meta={tabLabel[view][0].toUpperCase()}>
            <div className="tm-name">{layoutName(res)}</div>
            <div className="big">
              {(['cpu', 'gpu'] as const).map((p) => (
                <div key={p}>
                  <span className="lbl">{p}</span>
                  <b>{res.parts[p].temp_c.toFixed(1)}<u>°C</u></b>
                  <span className="sub">air in +{res.parts[p].intake_rise_c.toFixed(1)}° <Delta v={dd(res.parts[p].temp_c, cur?.parts[p].temp_c)} /></span>
                </div>
              ))}
            </div>
            <div className="rows">
              <div><span className="lbl">Airflow</span><span>{(res.flow_in_m3s * 2119).toFixed(0)} in · {(res.flow_out_m3s * 2119).toFixed(0)} out <u>CFM</u></span></div>
              {res.case_pa != null && <div><span className="lbl">Case</span><span>{res.case_pa >= 0 ? '+' : '−'}{Math.abs(res.case_pa).toFixed(1)} <u>Pa</u> · {res.pressure}</span></div>}
              {res.dust && <div><span className="lbl">Dust in</span><span>{(res.dust.dust_entering_rel * 100).toFixed(0)}<u>%</u> of outside dust <Delta v={dd(res.dust.dust_entering_rel * 100, cur?.dust ? cur.dust.dust_entering_rel * 100 : undefined)} unit=" pt" /></span></div>}
            </div>
            {res.dust && (
              <div className="dep">
                <span className="lbl">Where it settles</span>
                {res.dust.by_part.slice(0, 5).map((p) => (
                  <div key={p.name} className="dep-r"><span>{p.name}</span><i><b style={{ width: `${Math.min(100, p.share * 250)}%` }} /></i><span className="mono">{(p.share * 100).toFixed(0)}%</span></div>
                ))}
              </div>
            )}
            {res.notes?.length > 0 && <p className="fine">{res.notes.join(' · ')}</p>}
          </HudWindow>
        )}

        {cur && other && (
          <HudWindow id="compare" code="04" title="Compare" side="right" x={20} y={compareY} w={310} layoutKey={layoutKey} meta={bestIsCur ? 'YOURS / WORST' : 'YOURS / RECOMMENDED'}>
            {bestIsCur
              ? <p className="verdict"><b>Keep your layout.</b> It ranks #1 of {n}.{worstD && <> The worst one runs the CPU {worstD.cpu.toFixed(1)}° and the GPU {worstD.gpu.toFixed(1)}° hotter.</>}</p>
              : <p className="verdict"><b>One change helps.</b> The recommended layout ranks #1 of {n}.</p>}
            <div className="cmp">
              {([['current', cur, 'Yours'], [otherKey, other, bestIsCur ? 'Worst' : 'Recommended']] as const).map(([k, r, t]) => (
                <button key={k} className={'cmp-c' + (view === k ? ' on' : '')} onClick={() => setView(k)}>
                  <span className="lbl">{t} · #{k === 'current' ? rankOf(curRow?.name) : rankOf(k === 'worst' ? opt?.worst : opt?.best)}</span>
                  <Schematic dims={r.case} boxes={r.boxes} mounts={r.mounts} fans={r.fans} width={112} />
                  <span className="cmp-n"><span>CPU</span><b>{r.parts.cpu.temp_c.toFixed(1)}°</b></span>
                  <span className="cmp-n"><span>GPU</span><b>{r.parts.gpu.temp_c.toFixed(1)}°</b></span>
                  {r.dust && <span className="cmp-n"><span>Dust</span><b>{(r.dust.dust_entering_rel * 100).toFixed(0)}%</b></span>}
                </button>
              ))}
            </div>
            {steps.length > 0 && <ol className="steps">{steps.map((s) => <li key={s}>{s}</li>)}</ol>}
          </HudWindow>
        )}

        {cfgOpen && build && cs && (
          <HudWindow id="configure" code="05" title="Configure" side="left" x={340} y={72} w={330} layoutKey={layoutKey} onClose={() => setCfgOpen(false)} meta={online ? 'LIVE' : 'ENGINE OFF'}>
            <div className="cfg-parts">
              {([['Case', 'case', lib?.cases], ['GPU', 'gpu', lib?.gpus], ['Cooler', 'cooler', lib?.coolers]] as const).map(([l, k, list]) => (
                <label key={k}><span className="lbl">{l}</span>
                  <select className="in" value={build[k]} onChange={(e) => setBuild({ ...build, [k]: e.target.value })}>
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
                        <select className="in sm" value={f.fan} onChange={(e) => setFan(m.slot, { fan: e.target.value })}>
                          {lib?.fans.filter((x) => m.sizes.includes(x.size_mm)).map((x) => <option key={x.id} value={x.id}>{x.name.replace(/^Lian Li LANCOOL 207 /, 'Stock ')} · {x.cfm} CFM</option>)}
                        </select>
                        <input type="range" min={30} max={100} step={10} value={f.pct} onChange={(e) => setFan(m.slot, { pct: +e.target.value })} />
                        <span className="mono">{f.pct}%</span>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
            <div className="cfg-row">
              <label><span className="lbl">CPU W</span><input className="in" value={build.load.cpu_w} onChange={(e) => setBuild({ ...build, load: { ...build.load, cpu_w: +e.target.value || 0 } })} /></label>
              <label><span className="lbl">GPU W</span><input className="in" value={build.load.gpu_w} onChange={(e) => setBuild({ ...build, load: { ...build.load, gpu_w: +e.target.value || 0 } })} /></label>
              <label><span className="lbl">Room °C</span><input className="in" value={build.load.ambient_c} onChange={(e) => setBuild({ ...build, load: { ...build.load, ambient_c: +e.target.value || 0 } })} /></label>
            </div>
            <div className="cfg-row">
              <label><span className="lbl">GPU measured °C</span><input className="in" placeholder="—" value={cal.temp} onChange={(e) => setCal({ ...cal, temp: e.target.value })} /></label>
              <label><span className="lbl">at W</span><input className="in" placeholder="—" value={cal.watts} onChange={(e) => setCal({ ...cal, watts: e.target.value })} /></label>
              <button className="textbtn" disabled={!g0?.temp_c} onClick={() => g0?.temp_c != null && g0.power_w != null && setCal({ temp: g0.temp_c.toFixed(0), watts: g0.power_w.toFixed(0) })}>use live</button>
            </div>
            <button className="run" disabled={!online || !!job} onClick={() => simulate(build, 'edited layout')}>{online ? 'Run simulation' : 'Engine offline'}</button>
            <p className="fine">About 40 s on the GPU: airflow, then heat, then dust. Calibrating with a measured GPU temperature under load fixes the cooler's conductance.</p>
          </HudWindow>
        )}

        {infoOpen && (
          <HudWindow id="method" code="06" title="Method" side="right" x={340} y={72} w={380} layoutKey={layoutKey} onClose={() => setInfoOpen(false)}>
            <div className="method">
              <p><span className="lbl">Airflow</span>Our own D3Q19 lattice-Boltzmann solver (PyTorch, GPU, Smagorinsky turbulence) on a 6 mm grid of the case. Walls, PSU, board, GPU and cooler are solid; case fans and the GPU / CPU cooler fans are cells with a set air velocity; empty mesh mounts and slot covers are open to the room.</p>
              <p><span className="lbl">Fans</span>Each case fan runs at its operating point: a linear fan curve (rated flow and static pressure, scaled by fan laws) against one case pressure, which the passive openings balance as orifices. This is why "all intake, no exhaust" does not win: its fans stall against +7.8 Pa.</p>
              <p><span className="lbl">Heat</span>CPU and GPU power enters the air in their fin stacks and is carried by the solved flow (advection–diffusion). Part temperature = intake air rise + power / cooler conductance.</p>
              <p><span className="lbl">Dust</span>40 000 particles of 20 µm enter where air comes in (the bottom filter stops 85%), drift with the flow plus turbulent diffusion and settling, and stick where they touch.</p>
              <p><span className="lbl">Search</span>Every placement and direction of your three fans (4 mounts × in / out = {n || 32} layouts) is simulated in full, about 33 s each.</p>
              <p><span className="lbl">Limits</span>Mount positions, opening porosity and cooler conductance are estimates; trust the differences between layouts more than the absolute °C. A measured GPU temperature calibrates the cooler.</p>
            </div>
          </HudWindow>
        )}
      </div>

      <nav className="dock">
        {([['flow', 'Flow'], ['lines', 'Lines'], ['dust', 'Dust'], ['heat', 'Heat']] as const).map(([k, l]) => (
          <button key={k} className={layers[k] ? 'on' : ''} onClick={() => setLayers({ ...layers, [k]: !layers[k] })}>{l}</button>
        ))}
        <span className="sep" />
        <span className="legend"><span className="lbl">Air ΔT</span><span>0</span><i style={{ background: THERMAL_CSS }} /><span>+{T_MAX} °C</span></span>
        <span className="sep" />
        <button className={cfgOpen ? 'on' : ''} onClick={() => setCfgOpen(!cfgOpen)}>Configure</button>
        <button className={infoOpen ? 'on' : ''} onClick={() => setInfoOpen(!infoOpen)}>Method</button>
        <button title="Put the windows back" onClick={() => { resetWindows(); setLayoutKey((k) => k + 1) }}>⟲</button>
      </nav>
    </div>
  )
}
