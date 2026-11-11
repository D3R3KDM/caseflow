import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react'
import Schematic, { type SchFan } from './Schematic'
import type { CaseFanState, DustStats, PartState, ScenarioResult } from './scenario'
import { WHERE, type Geometry, type RankRow } from './types'

/** Frame of a floating holographic screen: glow border, corner brackets, scanlines, sweep. */
export function Holo({ tag, sub, badge, width, flashKey, children }: { tag: string; sub?: ReactNode; badge?: ReactNode; width: number; flashKey?: string; children: ReactNode }) {
  return (
    <div className="holo" style={{ width } as CSSProperties} key={flashKey}>
      <i className="hc tl" /><i className="hc tr" /><i className="hc bl" /><i className="hc br" />
      <div className="h-head">
        <span className="h-tag">{tag}</span>
        {badge}
        {sub && <span className="h-sub">{sub}</span>}
      </div>
      <div className="h-body">{children}</div>
    </div>
  )
}

const mix = (a: string, b: string, t: number) => {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16)), pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(',')})`
}

/** Semicircle temperature gauge with ticks every 10 °C. */
export function Gauge({ value, min = 30, max = 90 }: { value: number; min?: number; max?: number }) {
  const R = 54, cx = 66, cy = 64
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  const pt = (f: number, r: number) => [cx - r * Math.cos(Math.PI * f), cy - r * Math.sin(Math.PI * f)]
  const arc = (f0: number, f1: number) => {
    const [x0, y0] = pt(f0, R), [x1, y1] = pt(f1, R)
    return `M${x0},${y0} A${R},${R} 0 0 1 ${x1},${y1}`   // a fraction of a semicircle: never the large arc
  }
  const col = mix('#63e0ff', '#ffa860', Math.max(0, Math.min(1, (value - 55) / 25)))
  const ticks = []
  for (let v = Math.ceil(min / 10) * 10; v <= max; v += 10) {
    const f = (v - min) / (max - min)
    const [a, b] = pt(f, R + 6), [c, d] = pt(f, R + (v % 20 === 0 ? 12 : 9))
    ticks.push(<line key={v} x1={a} y1={b} x2={c} y2={d} stroke="rgba(140,200,255,.45)" strokeWidth={1} />)
    if (v % 20 === 0) {
      const [tx, ty] = pt(f, R + 21)
      ticks.push(<text key={'t' + v} x={tx} y={ty + 3} textAnchor="middle" className="g-tick">{v}</text>)
    }
  }
  return (
    <svg width={132} height={84} viewBox="0 -10 132 84" className="gauge" aria-label={`${value.toFixed(1)} °C`}>
      <path d={arc(0, 1)} stroke="rgba(140,200,255,.14)" strokeWidth={7} fill="none" />
      <path d={arc(0, Math.max(t, 0.001))} stroke={col} strokeWidth={7} fill="none" style={{ filter: `drop-shadow(0 0 6px ${col})` }} />
      {ticks}
      <text x={cx} y={cy - 8} textAnchor="middle" className="g-val" fill={col}>{value.toFixed(1)}</text>
      <text x={cx} y={cy + 7} textAnchor="middle" className="g-unit">°C</text>
    </svg>
  )
}

/** Change against your layout. better: which direction is good (temperatures and dust: lower; airflow: higher). */
export function Delta({ v, unit = '°', better = 'lower' }: { v: number | null; unit?: string; better?: 'lower' | 'higher' }) {
  if (v == null) return null
  const g = better === 'lower' ? -v : v
  const cls = g > 0.05 ? 'good' : g < -0.05 ? 'bad' : 'flat'
  return <span className={'d ' + cls}>{v > 0 ? '+' : v < 0 ? '−' : '±'}{Math.abs(v).toFixed(1)}{unit}</span>
}

const pctOf = (x: number) => `${Math.round(x * 100)}%`
const rpm = (x: number) => `${Math.round(x / 10) * 10}`

/** CPU or GPU: temperature gauge, intake air, power (and throttling), its own fans. */
export function PartPanel({ part, st, refSt }: { part: 'cpu' | 'gpu'; st: PartState; refSt: PartState | null }) {
  const stopped = part === 'gpu' && st.fan_speed === 0
  return (
    <div className="pp">
      <div className="pp-row">
        <Gauge value={st.temp_c} min={part === 'cpu' ? 30 : 20} max={part === 'cpu' ? 100 : 90} />
        <dl className="pp-kv">
          <dt>air in</dt><dd>+{st.intake_rise_c.toFixed(1)}°</dd>
          <dt>power</dt><dd className={st.throttled ? 'warnline' : ''}>{st.watts.toFixed(0)} W</dd>
          <dt>fan rpm</dt><dd>{stopped ? 'off' : rpm(st.fan_rpm)}</dd>
        </dl>
      </div>
      {(st.throttled || refSt) && (
        <div className="pp-foot">
          {st.throttled ? <span className="warnline">throttled at {st.t_max_c} °C</span> : refSt && <span>vs yours <Delta v={st.temp_c - refSt.temp_c} /></span>}
        </div>
      )}
    </div>
  )
}

/** Airflow: in / out, case pressure, and every case fan at its operating point. */
export function AirPanel({ res, fans, refRes }: { res: ScenarioResult; fans: { slot: string; mode: 'in' | 'out' }[]; refRes: ScenarioResult | null }) {
  const cfm = (q: number) => q * 2119
  const pa = res.case_pa
  const pressure = res.flow_in_m3s > res.flow_out_m3s * 1.05 ? 'positive' : res.flow_out_m3s > res.flow_in_m3s * 1.05 ? 'negative' : 'balanced'
  const mode = new Map(fans.map((f) => [f.slot, f.mode]))
  const rows = [...res.fans].sort((a, b) => (mode.get(a.slot) === mode.get(b.slot) ? 0 : mode.get(a.slot) === 'in' ? -1 : 1))
  return (
    <div className="ap">
      <div className="ap-big">
        <div><span className="lbl">in</span><b>{cfm(res.flow_in_m3s).toFixed(0)}</b></div>
        <div><span className="lbl">out</span><b>{cfm(res.flow_out_m3s).toFixed(0)}</b></div>
        <div><span className="lbl">case</span><b>{pa >= 0 ? '+' : '−'}{Math.abs(pa).toFixed(1)}<u>Pa</u></b></div>
      </div>
      <div className="ap-unit">CFM · {pressure} pressure{refRes && <> · in <Delta v={cfm(res.flow_in_m3s) - cfm(refRes.flow_in_m3s)} unit="" better="higher" /></>}</div>
      <div className="ap-fans">
        {rows.map((f: CaseFanState) => {
          const m = mode.get(f.slot) ?? 'in'
          return (
            <div key={f.slot} className={'af m-' + m}>
              <span className="af-n">{WHERE[f.slot] ?? f.slot}{!f.curve && <small> fixed</small>}</span>
              <span className="af-d">{m === 'in' ? '▸ in' : '◂ out'}</span>
              <span className="af-q">{cfm(f.q_m3s).toFixed(0)}</span>
              <i className="af-bar"><b style={{ width: `${Math.min(100, f.speed * 100)}%` }} /></i>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** Dust: how fast outside dust settles inside, relative to your layout, and where it lands. */
export function DustPanel({ rel, stats, compare, note }: { rel: number | null; stats: DustStats | null | undefined; compare: boolean; note?: string }) {
  if (note) return <div className="fine">{note}</div>
  if (!stats || rel == null) return <div className="fine">No dust run for this layout.</div>
  return (
    <div className="dp">
      <div className="dp-big"><b>×{rel.toFixed(2)}</b><span>dust{compare ? ' vs your layout' : ''}</span></div>
      {stats.by_part.slice(0, 3).map((p) => (
        <div key={p.name} className="dp-r"><span>{p.name}</span><i><b style={{ width: `${Math.min(100, p.share * 250)}%` }} /></i><span className="mono">{(p.share * 100).toFixed(0)}%</span></div>
      ))}
    </div>
  )
}

/** Your layout against the best (or, if yours is best, the worst) in the current scenario. */
export function VerdictPanel({ cur, other, rankCur, rankOther, n, bestIsCur, tie, geo, steps, onOpen }: {
  cur: RankRow; other: RankRow; rankCur: number; rankOther: number; n: number; bestIsCur: boolean
  tie: { name: string; delta: number; margin: number } | null; geo: Geometry | null; steps: string[]; onOpen: (k: 'current' | 'other') => void
}) {
  const dC = other.cpu_c - cur.cpu_c, dG = other.gpu_c - cur.gpu_c
  const schFans = (r: RankRow): SchFan[] => r.build.fans.filter((f) => f.mode !== 'off').map((f) => ({ slot: f.slot, mode: f.mode as 'in' | 'out' }))
  return (
    <div className="vp">
      <div className="vp-head">{bestIsCur ? 'Keep your layout' : 'One change helps'}</div>
      <p className="vp-text">{bestIsCur ? <>{tie ? 'Tied for best' : `#${rankCur}`} of {n} · worst <b>+{dC.toFixed(1)}°</b> CPU</> : <>Best of {n}</>}</p>
      <div className="vp-cmp">
        {([['current', cur, `Yours · #${rankCur}`], ['other', other, `${bestIsCur ? 'Worst' : 'Best'} · #${rankOther}`]] as const).map(([k, r, t]) => (
          <button key={k} className={'vp-c ' + k} onClick={() => onOpen(k)}>
            <span className="lbl">{t}</span>
            {geo && <Schematic dims={geo.case} boxes={geo.boxes} mounts={geo.mounts} fans={schFans(r)} width={104} />}
            <span className="vp-n"><span>CPU</span><b>{r.cpu_c.toFixed(1)}°</b></span>
            <span className="vp-n"><span>GPU</span><b>{r.gpu_c.toFixed(1)}°</b></span>
            <span className="vp-n"><span>Dust</span><b>×{r.dust_rel.toFixed(2)}</b></span>
          </button>
        ))}
      </div>
      {steps.length > 0 && <ol className="steps">{steps.map((s) => <li key={s}>{s}</li>)}</ol>}
    </div>
  )
}

/** All 32 layouts ranked for the current scenario (recomputed in the browser whenever it changes). */
export function SearchPanel({ rows, shown, tagOf, isOn, onOpen, allRows, setAllRows, lo, hi }: {
  rows: RankRow[]; shown: { r: RankRow; i: number }[]; tagOf: (r: RankRow) => string[]; isOn: (r: RankRow) => boolean
  onOpen: (r: RankRow) => void; allRows: boolean; setAllRows: (v: boolean) => void; lo: number; hi: number
}) {
  const list = useRef<HTMLDivElement>(null)
  const onName = shown.find(({ r }) => isOn(r))?.r.name
  useEffect(() => {
    // keep the selected row in sight; scroll only the list (scrollIntoView would also move the 3D layer)
    const l = list.current, el = l?.querySelector<HTMLElement>('.rk.on')
    if (l && el) l.scrollTop = Math.max(0, el.offsetTop - l.clientHeight / 2 + el.offsetHeight / 2)
  }, [onName, allRows])
  return (
    <div className="sp">
      <div className="rk-h"><span>#</span><span>Layout</span><span>CPU</span><span>GPU</span><span>Dust ×</span></div>
      <div className="rk-list" ref={list}>
        {shown.map(({ r, i }, k) => (
          <div key={r.name}>
            {k > 0 && shown[k - 1].i !== i - 1 && <div className="rk-gap">· · ·</div>}
            <button className={'rk' + (isOn(r) ? ' on' : '')} style={{ '--w': 1 - (r.score - lo) / Math.max(hi - lo, 1e-6) } as CSSProperties} onClick={() => onOpen(r)}>
              <span className="n">{String(i + 1).padStart(2, '0')}</span>
              <span className="nm">{r.name}{tagOf(r).map((t) => <em key={t} className={t === 'WORST' ? 'warn' : ''}>{t}</em>)}</span>
              <span className="v">{r.cpu_c.toFixed(1)}</span>
              <span className="v">{r.gpu_c.toFixed(1)}</span>
              <span className="v">{r.dust_rel.toFixed(2)}</span>
            </button>
          </div>
        ))}
      </div>
      {rows.length > 0 && <button className="textbtn" onClick={() => setAllRows(!allRows)}>{allRows ? 'Show fewer' : `Show all ${rows.length}`}</button>}
    </div>
  )
}

