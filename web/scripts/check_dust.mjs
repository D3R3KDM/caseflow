// Run the browser's dust particles (src/dust.ts) the way the engine runs them - n particles released at once and
// followed to the end - on your layout's bank (public/data/bank_current.json) and print, per level, the share that
// settles and the settled particles per cell. tests/test_dust_web.py compares that with the engine's dust.
// Node 22.18+ runs the .ts file directly (type stripping).
import { readFileSync } from 'node:fs'
import { DustSim, voxelsOf } from '../src/dust.ts'

const n = +(process.argv[2] ?? 40000)
const pkg = JSON.parse(readFileSync(new URL('../public/data/bank_current.json', import.meta.url), 'utf8'))
const decode = (q) => {
  const b = Buffer.from(q.b64, 'base64')
  const out = new Float32Array(b.length)
  const k = q.signed ? q.scale / 127 : q.scale / 255
  for (let i = 0; i < b.length; i++) out[i] = q.signed ? (b[i] - 128) * k : b[i] * k
  return out
}
const vox = voxelsOf(pkg.key, pkg.voxels)
if (!vox) throw new Error('bank_current.json has no voxels: run scripts/precompute.py')
const levels = pkg.fields.map((f, li) => {
  const u = decode(f.u)
  const t0 = performance.now()
  const r = DustSim.batch(u, vox, pkg.geometry.dx_mm, n, 7 + li)
  const cells = []
  r.grid.forEach((c, i) => { if (c) cells.push([i, c]) })
  return { duty: pkg.levels[li].duty, settled_frac: r.settled / r.released, left_frac: r.left / r.released, airborne_frac: r.airborne / r.released,
    inflow_w: new DustSim(u, vox, pkg.geometry.dx_mm).inflowW, ms: Math.round(performance.now() - t0), cells }
})
process.stdout.write(JSON.stringify({ shape: vox.shape, levels }))
