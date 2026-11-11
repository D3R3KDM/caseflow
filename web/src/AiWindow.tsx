import { useMemo } from 'react'
import { codeWords, fanCount, type AiInfo, type AiSpace, type Seen } from './ai'
import type { RankOut } from './aiWorker'
import TrendChart, { SERIES } from './TrendChart'
import type { ScenarioResult } from './scenario'

/** The network's view of the layout on screen, next to the simulation's. */
export interface AiHere {
  ai: ScenarioResult | null; phys: ScenarioResult | null; seen: Seen | null
  why: string | null                       // why there is no prediction (other parts, the engine is off, ...)
  progress: number | null                  // the simulation's progress while it runs
  simReady: boolean                        // the simulation is cached: comparing takes seconds, not about 90 s
  comparing: boolean                       // the 3D view shows the AI's case and the simulation's side by side
}

const SEEN: Record<Seen, string> = { train: 'In the training set', test: 'Held-out test layout', new: 'New to the network' }

const signed = (v: number) => {
  const r = Math.round(v * 10) / 10
  return `${r > 0 ? '+' : r < 0 ? '−' : '±'}${Math.abs(r).toFixed(1)}`
}

export function ModelCard({ info }: { info: AiInfo | null }) {
  if (!info) return null
  const t = info.test
  return (
    <dl className="ai-card">
      <dt>model</dt><dd>3D U-Net · {(info.params / 1e6).toFixed(1)} M weights</dd>
      <dt>trained</dt><dd>{info.trained.layouts} simulated layouts</dd>
      {t && <><dt>test</dt><dd>{t.layouts} new layouts · CPU ±{t.cpu_mae_c.toFixed(2)} °C · GPU ±{t.gpu_mae_c.toFixed(2)} °C</dd></>}
    </dl>
  )
}

/** This layout: the network's prediction against the simulation; the button puts the two side by side in 3D. */
export function AiHereTab({ here, onCompare }: { here: AiHere; onCompare: () => void }) {
  const { ai, phys } = here
  if (here.why) return <p className="fine">No AI prediction: {here.why}.</p>
  if (!ai) return <p className="fine">Asking the network…</p>
  const row = (label: string, a: number, p: number | null) => (
    <tr key={label}>
      <th>{label}</th><td className="mono">{a.toFixed(1)}</td><td className="mono">{p != null ? p.toFixed(1) : '—'}</td>
      <td className={'mono ' + (p == null ? '' : Math.abs(a - p) < 0.5 ? 'ok' : Math.abs(a - p) < 1.5 ? 'meh' : 'bad')}>{p != null ? signed(a - p) : ''}</td>
    </tr>
  )
  return (
    <div className="ai-here">
      <table className="ai-cmp">
        <thead><tr><th>{here.seen ? SEEN[here.seen] : ''}</th><th>AI</th><th>simulation</th><th>diff</th></tr></thead>
        <tbody>
          {row('CPU °C', ai.cpu.temp_c, phys?.cpu.temp_c ?? null)}
          {row('GPU °C', ai.gpu.temp_c, phys?.gpu.temp_c ?? null)}
        </tbody>
      </table>
      {here.comparing ? <button className="run" disabled>{here.progress != null ? `Simulating ${Math.round(here.progress * 100)}%` : 'Side by side in 3D'}</button>
        : <button className="run" onClick={onCompare}>{here.progress != null ? `Simulating ${Math.round(here.progress * 100)}% · show` : here.simReady || phys ? 'Compare in 3D' : 'Simulate and compare · ~90 s'}</button>}
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
      <p className="sp-lede">{space.rows.length} layouts · predicted in {space.seconds.toFixed(0)} s · ranked for {scenario}</p>
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
      <p className="fine ai-key">F front · B bottom · T top · R rear · ▸ in · ◂ out</p>
      {view.iNow >= 0 && <p className="fine">On screen: #{view.rankNow}</p>}
      <TrendChart title="Best per fan count (°C above the best)" xs={xs} rel
        series={[{ id: 'cpu', label: 'CPU', color: SERIES.cpu, ys: xs.map((k) => ranks.cpu[view.best[k]!] - ranks.cpu[view.order[0]]) },
          { id: 'gpu', label: 'GPU', color: SERIES.gpu, ys: xs.map((k) => ranks.gpu[view.best[k]!] - ranks.gpu[view.order[0]]) }]}
        xFmt={(x) => `${x}`} xUnit="fans" now={view.iNow >= 0 ? view.counts[view.iNow] : -1} />
    </div>
  )
}
