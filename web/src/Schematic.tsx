import { CSS_IN, CSS_OUT } from './palette'
import type { Mount, SimFan } from './types'

/** Side view through the glass (rear on the left, front on the right): shell, PSU, board, GPU, cooler,
 * and every mount with its fan direction. */
interface Props {
  dims: { D: number; W: number; H: number }
  boxes: { name: string; kind: string; b: number[] }[]
  mounts: Mount[]
  fans: SimFan[]
  width?: number
}

export default function Schematic({ dims, boxes, mounts, fans, width = 124 }: Props) {
  const { D, H } = dims
  const pad = 11
  const sc = (width - 2 * pad) / D
  const h = H * sc + 2 * pad + 8
  const X = (x: number) => pad + (D - x) * sc
  const Y = (z: number) => pad + (H - z) * sc
  const rect = (b: number[], fill: string, stroke: string, key: string) => (
    <rect key={key} x={X(b[1])} y={Y(b[5])} width={(b[1] - b[0]) * sc} height={(b[5] - b[4]) * sc} fill={fill} stroke={stroke} strokeWidth={0.6} />
  )
  const mountMark = (m: Mount, f: SimFan | undefined) => {
    const mode = f?.mode ?? null
    const col = mode === 'in' ? CSS_IN : mode === 'out' ? CSS_OUT : 'rgba(140,200,255,0.25)'
    const half = ((f?.size ?? Math.max(...m.sizes)) / 2) * sc
    let x1 = 0, y1 = 0, x2 = 0, y2 = 0, ax = 0, ay = 0, nx = 0, ny = 0
    if (m.wall === 'front') { x1 = x2 = ax = X(0); ay = Y(m.b); y1 = ay - half; y2 = ay + half; nx = -1 }
    else if (m.wall === 'rear') { x1 = x2 = ax = X(D); ay = Y(m.b); y1 = ay - half; y2 = ay + half; nx = 1 }
    else if (m.wall === 'top') { y1 = y2 = ay = Y(H); ax = X(m.a); x1 = ax - half; x2 = ax + half; ny = 1 }
    else { y1 = y2 = ay = Y(0); ax = X(m.a); x1 = ax - half; x2 = ax + half; ny = -1 }
    const inside = 11, outside = 5
    const [tx, ty, hx, hy] = mode === 'in'
      ? [ax - nx * outside, ay - ny * outside, ax + nx * inside, ay + ny * inside]
      : [ax + nx * inside, ay + ny * inside, ax - nx * outside, ay - ny * outside]
    return (
      <g key={m.slot}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={mode ? 2.4 : 1.2} />
        {mode && <line x1={tx} y1={ty} x2={hx} y2={hy} stroke={col} strokeWidth={1.1} markerEnd={`url(#sch-${mode})`} />}
      </g>
    )
  }
  return (
    <svg width={width} height={h} className="schematic" role="img" aria-label="Side view of the fan layout">
      <defs>
        <marker id="sch-in" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill={CSS_IN} /></marker>
        <marker id="sch-out" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 z" fill={CSS_OUT} /></marker>
      </defs>
      <rect x={X(D)} y={Y(H)} width={D * sc} height={H * sc} fill="rgba(8,14,22,0.5)" stroke="rgba(140,200,255,0.4)" strokeWidth={0.8} />
      {boxes.filter((b) => b.kind === 'psu').map((b, i) => rect(b.b, 'rgba(140,200,255,0.06)', 'rgba(140,200,255,0.3)', 'p' + i))}
      {boxes.filter((b) => b.kind === 'board').map((b, i) => rect(b.b, 'rgba(60,170,130,0.12)', 'rgba(90,200,160,0.35)', 'm' + i))}
      {boxes.filter((b) => b.kind === 'gpu' || b.kind === 'gpu_hs').map((b, i) => rect(b.b, 'rgba(200,215,230,0.16)', 'rgba(200,215,230,0.45)', 'g' + i))}
      {boxes.filter((b) => b.kind === 'cpu_hs').map((b, i) => rect(b.b, 'rgba(200,215,230,0.22)', 'rgba(200,215,230,0.55)', 'c' + i))}
      {mounts.map((m) => mountMark(m, fans.find((x) => x.slot === m.slot)))}
      <text x={X(D)} y={h - 2} className="sch-l">REAR</text>
      <text x={X(0)} y={h - 2} textAnchor="end" className="sch-l">FRONT</text>
    </svg>
  )
}
