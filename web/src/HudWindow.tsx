import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'

let zTop = 10
const KEY = (id: string) => `caseflow.win.${id}`
export const resetWindows = () => {
  try {
    Object.keys(localStorage).filter((k) => k.startsWith('caseflow.win.')).forEach((k) => localStorage.removeItem(k))
  } catch { /* storage unavailable */ }
}

interface Saved { x: number; y: number; min: boolean; moved: boolean }
interface Props {
  id: string
  code: string
  title: string
  side: 'left' | 'right'
  /** default position; used until the window is dragged */
  x: number
  y: number
  w: number
  meta?: ReactNode
  onClose?: () => void
  /** reports the rendered height so the next window in the column can sit below it */
  onHeight?: (h: number) => void
  layoutKey?: number
  children: ReactNode
}

/** A floating, draggable, collapsible glass window. A dragged position is kept per window in localStorage. */
export default function HudWindow({ id, code, title, side, x, y, w, meta, onClose, onHeight, layoutKey = 0, children }: Props) {
  const fresh = (): Saved => ({ x, y, min: false, moved: false })
  const [st, setSt] = useState<Saved>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(KEY(id)) ?? 'null')
      if (s && typeof s.x === 'number') return { ...fresh(), ...s }
    } catch { /* ignore */ }
    return fresh()
  })
  const [z, setZ] = useState(() => ++zTop)
  const ref = useRef<HTMLElement>(null)
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null)
  useEffect(() => { if (layoutKey) setSt(fresh()) }, [layoutKey]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    try {
      if (st.moved || st.min) localStorage.setItem(KEY(id), JSON.stringify(st))
      else localStorage.removeItem(KEY(id))
    } catch { /* ignore */ }
  }, [id, st])
  useLayoutEffect(() => {
    if (!onHeight || !ref.current) return
    const el = ref.current
    const ro = new ResizeObserver(() => onHeight(el.offsetHeight))
    ro.observe(el)
    onHeight(el.offsetHeight)
    return () => ro.disconnect()
  }, [onHeight])

  const px = st.moved ? st.x : x
  const py = st.moved ? st.y : y
  const down = (e: React.PointerEvent) => {
    setZ(++zTop)
    if ((e.target as HTMLElement).closest('button')) return
    drag.current = { px: e.clientX, py: e.clientY, x: px, y: py }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }
  const move = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.px
    const nx = Math.max(6, Math.min(window.innerWidth - 120, d.x + (side === 'left' ? dx : -dx)))
    const ny = Math.max(56, Math.min(window.innerHeight - 40, d.y + e.clientY - d.py))
    setSt((s) => ({ ...s, x: nx, y: ny, moved: true }))
  }
  const up = () => { drag.current = null }

  return (
    <section ref={ref} className={'win' + (st.min ? ' min' : '')} style={{ [side]: px, top: py, width: w, zIndex: z }} onPointerDown={() => setZ(++zTop)}>
      <header className="win-h" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        <span className="win-code">{code}</span>
        <span className="win-title">{title}</span>
        {meta && <span className="win-meta">{meta}</span>}
        <button className="win-btn" title={st.min ? 'Expand' : 'Collapse'} onClick={() => setSt((s) => ({ ...s, min: !s.min }))}>{st.min ? '+' : '–'}</button>
        {onClose && <button className="win-btn" title="Close" onClick={onClose}>×</button>}
      </header>
      {!st.min && <div className="win-b">{children}</div>}
    </section>
  )
}
