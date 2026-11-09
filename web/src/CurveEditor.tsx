import { useRef, useState } from 'react'
import type { Pt } from './scenario'

/** A fan curve in the style of ASUS Fan Xpert: temperature (°C, x) against duty (%, y). Drag a point to change
 * the curve; points keep their order. The ring shows where the fan runs now (current temperature, duty). */
export default function CurveEditor({ points, onChange, now, color = '#63e0ff', width = 316, height = 176, xMin = 20, xMax = 100 }: {
  points: Pt[]; onChange: (p: Pt[]) => void; now?: { t: number; d: number } | null; color?: string
  width?: number; height?: number; xMin?: number; xMax?: number
}) {
  const pad = { l: 30, r: 10, t: 10, b: 22 }
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b
  const X = (t: number) => pad.l + ((t - xMin) / (xMax - xMin)) * w
  const Y = (d: number) => pad.t + (1 - d / 100) * h
  const svg = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<number | null>(null)
  const pts = [...points].sort((a, b) => a[0] - b[0])
  const toData = (e: React.PointerEvent): Pt => {
    const r = svg.current!.getBoundingClientRect()
    const t = xMin + ((e.clientX - r.left - pad.l) / w) * (xMax - xMin)
    const d = (1 - (e.clientY - r.top - pad.t) / h) * 100
    return [t, d]
  }
  const move = (e: React.PointerEvent) => {
    if (drag == null) return
    const [t, d] = toData(e)
    const lo = drag > 0 ? pts[drag - 1][0] + 1 : xMin, hi = drag < pts.length - 1 ? pts[drag + 1][0] - 1 : xMax
    const next = pts.map((p) => [...p] as Pt)
    next[drag] = [Math.round(Math.min(hi, Math.max(lo, t))), Math.round(Math.min(100, Math.max(0, d)))]
    onChange(next)
  }
  const line = [[xMin, pts[0][1]], ...pts, [xMax, pts[pts.length - 1][1]]].map(([t, d]) => `${X(t)},${Y(d)}`).join(' ')
  const grid = []
  for (let t = Math.ceil(xMin / 10) * 10; t <= xMax; t += 10)
    grid.push(<g key={'x' + t}><line x1={X(t)} x2={X(t)} y1={pad.t} y2={pad.t + h} className="cv-grid" /><text x={X(t)} y={height - 7} textAnchor="middle" className="cv-tick">{t}</text></g>)
  for (let d = 0; d <= 100; d += 20)
    grid.push(<g key={'y' + d}><line x1={pad.l} x2={pad.l + w} y1={Y(d)} y2={Y(d)} className="cv-grid" /><text x={pad.l - 5} y={Y(d) + 3} textAnchor="end" className="cv-tick">{d}</text></g>)
  return (
    <svg ref={svg} width={width} height={height} className="curve" onPointerMove={move} onPointerUp={() => setDrag(null)}
      role="img" aria-label="fan curve: duty against temperature">
      {grid}
      <text x={pad.l + w} y={pad.t + h - 4} textAnchor="end" className="cv-axis">°C</text>
      <text x={pad.l + 4} y={pad.t + 9} className="cv-axis">%</text>
      <polyline points={line} fill="none" stroke={color} strokeWidth={1.6} style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
      {now && now.t >= xMin && now.t <= xMax && (
        <g>
          <line x1={X(now.t)} x2={X(now.t)} y1={Y(now.d * 100)} y2={pad.t + h} className="cv-now-l" />
          <circle cx={X(now.t)} cy={Y(now.d * 100)} r={5.5} className="cv-now" />
        </g>
      )}
      {pts.map(([t, d], i) => (
        <circle key={i} cx={X(t)} cy={Y(d)} r={drag === i ? 7 : 5.5} className={'cv-pt' + (drag === i ? ' on' : '')}
          onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); setDrag(i) }}>
          <title>{`${t} °C → ${d} %`}</title>
        </circle>
      ))}
    </svg>
  )
}
