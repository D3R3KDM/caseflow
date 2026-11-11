import { useState } from 'react'

/** Series colours: the dataviz reference palette's dark slots 1-2, validated on the window surface (#1e1e1e):
 * lightness band, chroma, CVD separation (ΔE 26.8) and contrast all pass. */
export const SERIES = { cpu: '#3987e5', gpu: '#d95926' }

export interface Series { id: string; label: string; color: string; dashed?: boolean; ys: number[] }

/** Part temperatures against one scenario variable. One °C axis; solid = fans follow their curves, dashed = the
 * fans held at their current speed. Crosshair + tooltip on hover, legend, end labels, and a table view. */
export default function TrendChart({ title, xs, series, xFmt, xUnit, now, width = 300, height = 150, rel = false }: {
  title: string; xs: number[]; series: Series[]; xFmt: (x: number) => string; xUnit: string; now: number; width?: number; height?: number
  /** the values are differences in °C (small range, signed labels) */
  rel?: boolean
}) {
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const pad = { l: 30, r: 58, t: 8, b: 22 }
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b
  const all = series.flatMap((s) => s.ys).filter(Number.isFinite)
  const unit = rel ? (Math.max(...all) - Math.min(...all) > 4 ? 1 : 0.5) : 10
  const lo = Math.floor(Math.min(...all) / unit) * unit, hi = Math.ceil(Math.max(...all) / unit) * unit
  const fmt = (v: number, d: number) => (rel && v >= 0.05 ? '+' : '') + v.toFixed(d)
  const x0 = xs[0], x1 = xs[xs.length - 1]
  const X = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * w
  const Y = (y: number) => pad.t + (1 - (y - lo) / Math.max(hi - lo, 1)) * h
  const path = (ys: number[]) => ys.map((y, i) => `${i ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(y).toFixed(1)}`).join('')
  const yTicks: number[] = []
  const stepY = rel ? unit : hi - lo > 40 ? 20 : 10
  for (let y = lo; y <= hi; y += stepY) yTicks.push(y)
  const xTicks = xs.filter((x, i) => i % Math.ceil(xs.length / 6) === 0 || i === xs.length - 1)
  const iNow = xs.reduce((b, x, i) => (Math.abs(x - now) < Math.abs(xs[b] - now) ? i : b), 0)
  const solid = series.filter((s) => !s.dashed)
  // end labels on the solid lines, unless they would collide (then the legend and tooltip carry identity)
  const ends = solid.map((s) => ({ s, y: Y(s.ys[s.ys.length - 1]) }))
  const collide = ends.length > 1 && Math.abs(ends[0].y - ends[1].y) < 12
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    const f = (px - pad.l) / w
    setHover(Math.max(0, Math.min(xs.length - 1, Math.round(f * (xs.length - 1)))))
  }
  const hi_ = hover ?? null
  return (
    <figure className="trend">
      <figcaption className="tr-h">
        <span className="tr-t">{title}</span>
        <button className="textbtn tr-tb" onClick={() => setTable(!table)}>{table ? 'chart' : 'table'}</button>
      </figcaption>
      <div className="tr-legend">
        {series.filter((s) => !s.dashed).map((s) => (
          <span key={s.id}><svg width="16" height="6" aria-hidden="true"><line x1="0" y1="3" x2="16" y2="3" stroke={s.color} strokeWidth="2" strokeLinecap="round" /></svg>{s.label}</span>
        ))}
        {series.some((s) => s.dashed) && (
          <span><svg width="16" height="6" aria-hidden="true"><line x1="0" y1="3" x2="16" y2="3" stroke="rgba(236,236,236,0.55)" strokeWidth="2" strokeDasharray="4 3" /></svg>fans fixed</span>
        )}
      </div>
      {table ? (
        <table className="tr-table">
          <thead><tr><th>{xUnit}</th>{series.map((s) => <th key={s.id}>{s.label}{s.dashed ? ' (held)' : ''}</th>)}</tr></thead>
          <tbody>{xs.map((x, i) => (i % Math.max(1, Math.round(xs.length / 8)) === 0 || i === xs.length - 1) && (
            <tr key={x} className={i === iNow ? 'on' : ''}><td>{xFmt(x)}</td>{series.map((s) => <td key={s.id}>{s.ys[i].toFixed(1)}</td>)}</tr>
          ))}</tbody>
        </table>
      ) : (
        <div className="tr-plot">
          <svg width={width} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(null)} role="img" aria-label={title}>
            {yTicks.map((y) => <g key={y}><line x1={pad.l} x2={pad.l + w} y1={Y(y)} y2={Y(y)} className="tr-grid" /><text x={pad.l - 5} y={Y(y) + 3} textAnchor="end" className="tr-tick">{Number.isInteger(y) ? y : y.toFixed(1)}</text></g>)}
            {xTicks.map((x) => <text key={x} x={X(x)} y={height - 6} textAnchor="middle" className="tr-tick">{xFmt(x)}</text>)}
            <line x1={X(xs[iNow])} x2={X(xs[iNow])} y1={pad.t} y2={pad.t + h} className="tr-now" />
            {series.map((s) => <path key={s.id} d={path(s.ys)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray={s.dashed ? '4 3' : undefined} opacity={s.dashed ? 0.75 : 1} />)}
            {solid.map((s) => <circle key={s.id} cx={X(xs[iNow])} cy={Y(s.ys[iNow])} r={4} fill={s.color} stroke="#1e1e1e" strokeWidth={2} />)}
            {!collide && ends.map(({ s, y }) => <text key={s.id} x={pad.l + w + 6} y={y + 3} className="tr-end">{s.label} {fmt(s.ys[s.ys.length - 1], rel ? 1 : 0)}°</text>)}
            {hi_ != null && <line x1={X(xs[hi_])} x2={X(xs[hi_])} y1={pad.t} y2={pad.t + h} className="tr-cross" />}
          </svg>
          {hi_ != null && (
            <div className="tr-tip" style={{ left: Math.min(X(xs[hi_]) + 10, width - 150), top: 6 }}>
              <div className="tr-tip-x">{xFmt(xs[hi_])}</div>
              {series.map((s) => (
                <div key={s.id} className="tr-tip-r">
                  <svg width="12" height="6" aria-hidden="true"><line x1="0" y1="3" x2="12" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '3 2' : undefined} /></svg>
                  <b>{fmt(s.ys[hi_], 1)} °C</b><span>{s.label}{s.dashed ? ', held' : ''}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </figure>
  )
}
