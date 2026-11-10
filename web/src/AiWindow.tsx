import { useMemo } from 'react'
import { codeWords, fanCount, type AiInfo, type AiSpace, type Seen } from './ai'
import type { RankOut } from './aiWorker'
import TrendChart, { SERIES } from './TrendChart'
import type { ScenarioResult } from './scenario'

/** The network's view of the layout on screen, next to the simulation's. */
export interface AiHere {
  ai: ScenarioResult | null; phys: ScenarioResult | null; seen: Seen | null; ms: number | null
  why: string | null                       // why there is no prediction (other parts, the engine is off, ...)
  progress: number | null                  // the simulation's progress while it runs
  physLabel: string                        // where the simulated numbers come from
}

const SEEN: Record<Seen, string> = {
  train: 'This layout was one of the training layouts.',
  test: 'Held-out test layout: the network never trained on it.',
  new: 'New layout: the network never trained on it.',
}

const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(1)}`

export function ModelCard({ info }: { info: AiInfo | null }) {
  if (!info) return <p className="fine">No trained network found (scripts/train_surrogate.py).</p>
  const t = info.test, tr = info.trained
  return (
    <dl className="ai-card">
      <dt>network</dt><dd>3D U-Net (CNN) · {(info.params / 1e6).toFixed(1)} M weights · {info.config?.depth ?? 4} levels down and up</dd>
      <dt>input</dt><dd>80 × 40 × 80 voxels of 6 mm: walls and openings, fans with their air velocity, CPU and GPU heat zones</dd>
      <dt>output</dt><dd>air velocity and the air's temperature rise per watt of CPU and of GPU power, in every voxel</dd>
      <dt>trained</dt><dd>{tr.layouts} simulated layouts × 3 fan speeds = {tr.samples} samples · {tr.epochs} epochs · {tr.minutes.toFixed(0)} min on the GPU</dd>
      {t && <><dt>test</dt><dd>{t.layouts} held-out layouts × 6 scenarios: mean error CPU {t.cpu_mae_c.toFixed(2)} °C, GPU {t.gpu_mae_c.toFixed(2)} °C
        (worst {t.cpu_max_c.toFixed(1)} / {t.gpu_max_c.toFixed(1)}){info.baseline && <> · guessing the average layout: {info.baseline.cpu_mae_c.toFixed(2)} / {info.baseline.gpu_mae_c.toFixed(2)} °C</>}</dd></>}
    </dl>
  )
}

/** This layout: the network's prediction against the simulation, and which one the 3D view shows. */
export function AiHereTab({ here, src, setSrc }: { here: AiHere; src: 'physics' | 'ai'; setSrc: (s: 'physics' | 'ai') => void }) {
  const { ai, phys } = here
  if (here.why) return <p className="fine">No prediction: {here.why}.</p>
  if (!ai) return <p className="fine">Asking the network…</p>
  const row = (label: string, a: number, p: number | null) => (
    <tr key={label}>
      <th>{label}</th><td className="mono">{a.toFixed(1)}</td>
      <td className="mono">{p != null ? p.toFixed(1) : here.progress != null ? `${Math.round(here.progress * 100)}%` : '—'}</td>
      <td className={'mono ' + (p == null ? '' : Math.abs(a - p) < 0.5 ? 'ok' : Math.abs(a - p) < 1.5 ? 'meh' : 'bad')}>{p != null ? signed(a - p) : ''}</td>
    </tr>
  )
  return (
    <div className="ai-here">
      <table className="ai-cmp">
        <thead><tr><th /><th>AI</th><th>physics</th><th>AI − physics</th></tr></thead>
        <tbody>
          {row('CPU °C', ai.cpu.temp_c, phys?.cpu.temp_c ?? null)}
          {row('GPU °C', ai.gpu.temp_c, phys?.gpu.temp_c ?? null)}
          {row('air in, CPU', ai.cpu.intake_rise_c, phys?.cpu.intake_rise_c ?? null)}
          {row('air in, GPU', ai.gpu.intake_rise_c, phys?.gpu.intake_rise_c ?? null)}
        </tbody>
      </table>
      <p className="fine">AI: {here.ms != null ? `${Math.round(here.ms)} ms` : 'instant'} · physics: {here.physLabel}. {here.seen ? SEEN[here.seen] : ''}</p>
      {phys && (
        <div className="ai-src">
          <span className="lbl">3D view</span>
          <span className="seg3">{(['physics', 'ai'] as const).map((k) => <button key={k} className={src === k ? 'on' : ''} onClick={() => setSrc(k)}>{k === 'ai' ? 'AI' : 'Physics'}</button>)}</span>
        </div>
      )}
    </div>
  )
}

/** All 3^8 layouts as the network predicts them, ranked for the scenario on screen. */
export function AiAllTab({ space, ranks, codeNow, scenario, fans, setFans, onOpen }: {
  space: AiSpace | null; ranks: RankOut | null; codeNow: string | null; scenario: string
  fans: number | null; setFans: (n: number | null) => void; onOpen: (code: string) => void
}) {
  const view = useMemo(() => {
    if (!space || !ranks || ranks.cpu.length !== space.rows.length) return null
    const n = space.rows.length
    const score = (i: number) => ranks.cpu[i] + ranks.gpu[i] + (ranks.throttled[i] ? 1000 : 0)
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => score(a) - score(b))
    const counts = space.rows.map((r) => fanCount(r.c))
    const best: (number | null)[] = Array.from({ length: space.mounts.length + 1 }, () => null)
    for (const i of order) if (best[counts[i]] == null) best[counts[i]] = i
    const iNow = codeNow ? space.rows.findIndex((r) => r.c === codeNow) : -1
    return { order, counts, best, iNow, rankNow: iNow >= 0 ? order.indexOf(iNow) + 1 : 0 }
  }, [space, ranks, codeNow])
  if (!space) return <p className="fine">Loading the predicted design space…</p>
  if (!view || !ranks) return <p className="fine">Ranking {space.rows.length} layouts…</p>
  const list = view.order.filter((i) => fans == null || view.counts[i] === fans).slice(0, 8)
  const xs = view.best.map((_, k) => k).filter((k) => view.best[k] != null)
  const days = (space.rows.length * 90) / 86400
  return (
    <div className="ai-all">
      <p className="sp-lede">The network predicted all <b>{space.rows.length}</b> ways to use the 8 mounts (off / in / out, the stock fan
        types) in {space.seconds.toFixed(0)} s; simulating them would take about {days.toFixed(0)} days. Ranked by CPU + GPU °C for <b>{scenario}</b>.</p>
      <div className="ai-filter">
        <span className="lbl">fans</span>
        <span className="seg3">
          <button className={fans == null ? 'on' : ''} onClick={() => setFans(null)}>any</button>
          {[3, 4, 5, 6, 7, 8].map((k) => <button key={k} className={fans === k ? 'on' : ''} onClick={() => setFans(k)}>{k}</button>)}
        </span>
      </div>
      <div className="rk-h ai-h"><span>#</span><span>Layout</span><span>CPU</span><span>GPU</span><span>dBA</span></div>
      <div className="ai-list">
        {list.map((i) => {
          const c = space.rows[i].c
          return (
            <button key={c} className={'rk ai-rk' + (c === codeNow ? ' on' : '')} onClick={() => onOpen(c)} title={`${codeWords(space, c, true)} - open: the AI shows it at once, then the simulation checks it`}>
              <span className="n">{view.order.indexOf(i) + 1}</span>
              <span className="nm mono">{codeWords(space, c)}{ranks.throttled[i] ? <em className="warn">THROTTLES</em> : null}</span>
              <span className="v">{ranks.cpu[i].toFixed(1)}</span>
              <span className="v">{ranks.gpu[i].toFixed(1)}</span>
              <span className="v">{ranks.noise[i].toFixed(0)}</span>
            </button>
          )
        })}
      </div>
      <p className="fine ai-key">F front (low, high) · B bottom (front, rear) · T top (front, mid, rear) · R rear · ▸ in · ◂ out · off</p>
      {view.iNow >= 0 && (
        <p className="fine">On screen: #{view.rankNow} of {space.rows.length} · CPU {ranks.cpu[view.iNow].toFixed(1)}, GPU {ranks.gpu[view.iNow].toFixed(1)} °C (AI)
          {codeNow && view.counts[view.iNow] < space.mounts.length && <> · best with {view.counts[view.iNow]} fans: CPU {ranks.cpu[view.best[view.counts[view.iNow]]!].toFixed(1)}, GPU {ranks.gpu[view.best[view.counts[view.iNow]]!].toFixed(1)}</>}</p>
      )}
      <TrendChart title="Best layout for each number of fans (AI)" xs={xs}
        series={[{ id: 'cpu', label: 'CPU', color: SERIES.cpu, ys: xs.map((k) => ranks.cpu[view.best[k]!]) },
          { id: 'gpu', label: 'GPU', color: SERIES.gpu, ys: xs.map((k) => ranks.gpu[view.best[k]!]) }]}
        xFmt={(x) => `${x}`} xUnit="fans" now={view.iNow >= 0 ? view.counts[view.iNow] : -1} />
      <p className="fine">Ranked in your browser in {Math.round(ranks.ms)} ms. Open a layout to see the AI's 3D fields at once and have the simulation check them (~1.5 min).</p>
    </div>
  )
}
