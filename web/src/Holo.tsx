import { Fragment, useEffect, useRef, type CSSProperties, type ReactNode } from 'react'
import Schematic from './Schematic'
import { WHERE, type Opt, type RankRow, type SimRes } from './types'

/** Frame of a floating holographic screen: glow border, corner brackets, scanlines, sweep. */
export function Holo({ tag, sub, live, width, flashKey, children }: { tag: string; sub?: ReactNode; live?: boolean; width: number; flashKey?: string; children: ReactNode }) {
  return (
    <div className="holo" style={{ width } as CSSProperties} key={flashKey}>
      <i className="hc tl" /><i className="hc tr" /><i className="hc bl" /><i className="hc br" />
      <div className="h-head">
        <span className="h-tag">{tag}</span>
        {sub && <span className="h-sub">{sub}</span>}
        {live && <i className="h-live" title="live" />}
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
/** Rate at which outside dust settles inside: unfiltered intake x share that settles. */
export const dustRate = (r: SimRes | null | undefined) =>
  r?.dust ? r.flow_in_m3s * r.dust.dust_entering_rel * r.dust.deposited_frac : null
export const dustRel = (r: SimRes | null | undefined, ref: SimRes | null | undefined) => {
  const a = dustRate(r), b = dustRate(ref)
  return a != null && b ? a / b : null
}

export function Delta({ v, unit = '°', better = 'lower' }: { v: number | null; unit?: string; better?: 'lower' | 'higher' }) {
  if (v == null) return null
  const g = better === 'lower' ? -v : v
  const cls = g > 0.05 ? 'good' : g < -0.05 ? 'bad' : 'flat'
  return <span className={'d ' + cls}>{v > 0 ? '+' : v < 0 ? '−' : '±'}{Math.abs(v).toFixed(1)}{unit}</span>
}

interface PartProps { res: SimRes; cur: SimRes | null; compare: boolean; name: string; extra: [string, string][] }
export function PartPanel({ part, res, cur, compare, name, extra }: PartProps & { part: 'cpu' | 'gpu' }) {
  const p = res.parts[part]
  return (
    <div className="pp">
      <div className="pp-row">
        <Gauge value={p.temp_c} min={part === 'cpu' ? 40 : 30} max={part === 'cpu' ? 100 : 90} />
        <dl className="pp-kv">
          <dt>air in</dt><dd>+{p.intake_rise_c.toFixed(1)}°</dd>
          <dt>load</dt><dd>{p.watts.toFixed(0)} W</dd>
          {extra.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}
        </dl>
      </div>
      <div className="pp-foot">
        <span>{name}</span>
        {compare && <span>vs yours <Delta v={cur ? p.temp_c - cur.parts[part].temp_c : null} /></span>}
        {p.calibrated && <span className="cal">calibrated</span>}
      </div>
    </div>
  )
}

export function AirPanel({ res, cur, compare }: { res: SimRes; cur: SimRes | null; compare: boolean }) {
  const cfm = (q: number) => q * 2119
  const pa = res.case_pa ?? 0
  return (
    <div className="ap">
      <div className="ap-big">
        <div><span className="lbl">in</span><b>{cfm(res.flow_in_m3s).toFixed(0)}</b></div>
        <div><span className="lbl">out</span><b>{cfm(res.flow_out_m3s).toFixed(0)}</b></div>
        <div><span className="lbl">case</span><b>{pa >= 0 ? '+' : '−'}{Math.abs(pa).toFixed(1)}<u>Pa</u></b></div>
      </div>
      <div className="ap-unit">CFM · {res.pressure} pressure{compare && cur && <> · in <Delta v={cfm(res.flow_in_m3s) - cfm(cur.flow_in_m3s)} unit="" better="higher" /></>}</div>
      <div className="ap-fans">
        {[...res.fans].sort((a, b) => (a.mode === b.mode ? 0 : a.mode === 'in' ? -1 : 1)).map((f) => (
          <div key={f.slot} className={'af m-' + f.mode}>
            <span className="af-n">{WHERE[f.slot] ?? f.slot}</span>
            <span className="af-d">{f.mode === 'in' ? '▸ in' : '◂ out'}</span>
            <span className="af-q">{cfm(f.q_m3s).toFixed(0)}</span>
            <i className="af-bar"><b style={{ width: `${Math.min(100, (f.op_frac ?? 1) * 80)}%` }} /></i>
          </div>
        ))}
      </div>
      <div className="ap-note">bar = share of the fan's free flow it actually delivers</div>
    </div>
  )
}

export function DustPanel({ res, cur, compare }: { res: SimRes; cur: SimRes | null; compare: boolean }) {
  if (!res.dust) return <div className="fine">No dust run for this view.</div>
  const d = res.dust
  return (
    <div className="dp">
      <div className="dp-big"><b>×{(dustRel(res, cur) ?? 1).toFixed(2)}</b><span>dust settling{compare ? ' vs your layout' : ' (your layout = 1)'}</span></div>
      <div className="dp-sub">{(d.dust_entering_rel * 100).toFixed(0)}% of the intake air is unfiltered (only the bottom has a filter) · {(d.deposited_frac * 100).toFixed(0)}% of what enters settles</div>
      {d.by_part.slice(0, 4).map((p) => (
        <div key={p.name} className="dp-r"><span>{p.name}</span><i><b style={{ width: `${Math.min(100, p.share * 250)}%` }} /></i><span className="mono">{(p.share * 100).toFixed(0)}%</span></div>
      ))}
    </div>
  )
}

export function VerdictPanel({ cur, other, bestIsCur, n, rankCur, rankOther, steps, onOpen, tie }: {
  cur: SimRes; other: SimRes; bestIsCur: boolean; n: number; rankCur: number; rankOther: number; steps: string[]; onOpen: (k: 'current' | 'other') => void
  tie?: { name: string; delta: number; margin: number } | null
}) {
  const dC = other.parts.cpu.temp_c - cur.parts.cpu.temp_c, dG = other.parts.gpu.temp_c - cur.parts.gpu.temp_c
  return (
    <div className="vp">
      <div className="vp-head">{bestIsCur ? 'Keep your layout' : 'One change helps'}</div>
      <p className="vp-text">{bestIsCur
        ? <>{tie
            ? <>It ties for <b>first of {n}</b>: #1 ({tie.name}) scores only {tie.delta.toFixed(2)} °C better, inside the model's ±{tie.margin} °C.</>
            : <>It ranks <b>#{rankCur} of {n}</b>.</>} The worst layout runs the CPU {dC.toFixed(1)}° and the GPU {dG.toFixed(1)}° hotter.</>
        : <>The recommended layout ranks <b>#1 of {n}</b>.</>}</p>
      <div className="vp-cmp">
        {([['current', cur, `Yours · #${rankCur}`], ['other', other, `${bestIsCur ? 'Worst' : 'Best'} · #${rankOther}`]] as const).map(([k, r, t]) => (
          <button key={k} className={'vp-c ' + k} onClick={() => onOpen(k)}>
            <span className="lbl">{t}</span>
            <Schematic dims={r.case} boxes={r.boxes} mounts={r.mounts} fans={r.fans} width={104} />
            <span className="vp-n"><span>CPU</span><b>{r.parts.cpu.temp_c.toFixed(1)}°</b></span>
            <span className="vp-n"><span>GPU</span><b>{r.parts.gpu.temp_c.toFixed(1)}°</b></span>
            {r.dust && <span className="vp-n"><span>Dust</span><b>×{(dustRel(r, cur) ?? 1).toFixed(2)}</b></span>}
          </button>
        ))}
      </div>
      {steps.length > 0 && <ol className="steps">{steps.map((s) => <li key={s}>{s}</li>)}</ol>}
    </div>
  )
}

export function SearchPanel({ opt, rows, shown, tagOf, isOn, onOpen, allRows, setAllRows, lo, hi }: {
  opt: Opt | null; rows: RankRow[]; shown: { r: RankRow; i: number }[]; tagOf: (r: RankRow) => string[]; isOn: (r: RankRow) => boolean
  onOpen: (r: RankRow) => void; allRows: boolean; setAllRows: (v: boolean) => void; lo: number; hi: number
}) {
  const key = (r: RankRow) => r.combined ?? r.score
  const list = useRef<HTMLDivElement>(null)
  const onName = shown.find(({ r }) => isOn(r))?.r.name
  useEffect(() => {
    // keep the selected row in sight; scroll only the list (scrollIntoView would also move the 3D layer)
    const l = list.current, el = l?.querySelector<HTMLElement>('.rk.on')
    if (l && el) l.scrollTop = Math.max(0, el.offsetTop - l.clientHeight / 2 + el.offsetHeight / 2)
  }, [onName, allRows])
  return (
    <div className="sp">
      <p className="sp-lede">Your three 120 mm fans on every free mount, each in or out. The four stock fans stay put.</p>
      <div className="rk-h"><span>#</span><span>Layout</span><span>CPU</span><span>GPU</span><span>Dust ×</span></div>
      <div className="rk-list" ref={list}>
        {shown.map(({ r, i }, k) => (
          <div key={r.name}>
            {k > 0 && shown[k - 1].i !== i - 1 && <div className="rk-gap">· · ·</div>}
            <button className={'rk' + (isOn(r) ? ' on' : '')} style={{ '--w': 1 - (key(r) - lo) / Math.max(hi - lo, 1e-6) } as CSSProperties} onClick={() => onOpen(r)}>
              <span className="n">{String(i + 1).padStart(2, '0')}</span>
              <span className="nm">{r.name}{tagOf(r).map((t) => <em key={t} className={t === 'WORST' ? 'warn' : ''}>{t}</em>)}</span>
              <span className="v">{r.cpu_c.toFixed(1)}</span>
              <span className="v">{r.gpu_c.toFixed(1)}</span>
              <span className="v">{r.dust_rel != null ? r.dust_rel.toFixed(2) : `${Math.round(r.dust_in * 100)}%`}</span>
            </button>
          </div>
        ))}
      </div>
      {rows.length > 0 && <button className="textbtn" onClick={() => setAllRows(!allRows)}>{allRows ? 'Show fewer' : `Show all ${rows.length}`}</button>}
      <p className="fine">°C at CPU 88 W / GPU 300 W, room 25 °C · Dust × = settling rate vs your layout · score = CPU + GPU + {opt?.dust_weight ?? 2} °C × Dust ×</p>
    </div>
  )
}
