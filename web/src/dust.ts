/** Dust in the air the 3D view shows: the particles of engine/sim.simulate_dust, run live in the browser as a slow-motion
 * time-lapse, in whichever air velocity the case shows (the network's or the simulation's).
 *
 * Dust comes in through the openings in proportion to the inflow there (a dust filter stops 85 %), drifts with the air
 * velocity of the cell it is in, settles (Stokes, 20 µm), is mixed by the case's turbulence (a random walk), sticks to
 * the first surface it touches and leaves through the openings. The constants and the step are the engine's: a step
 * moves the fastest air half a cell. The numbers on the screens stay those of the engine's 40 000 particles; this shows
 * the same physics happening. No runtime imports, so node runs it too (web/scripts/check_dust.mjs). */

/** As the engine sends it (engine/bank.voxels): bit-packed masks, most significant bit first, [X, Y, Z] in C order. */
export interface VoxelPkg {
  shape: [number, number, number]; solid: string; outside: string
  openings: { box: number[]; wall: string; filtered: boolean }[]; filter_eff: number
}
export interface Voxels {
  shape: [number, number, number]; solid: Uint8Array; outside: Uint8Array
  openings: { box: number[]; wall: string; filtered: boolean }[]; filterEff: number
}

const RHO_P = 1500                  // kg/m3, mineral dust
const MU = 1.8e-5                   // Pa s, air
const D_UM = 20                     // particle diameter
const V_SETTLE = (RHO_P * 9.81 * (D_UM * 1e-6) ** 2) / (18 * MU)   // m/s, Stokes settling (about 1.8 cm/s)
const D_EDDY = 2e-3                 // m2/s, turbulent mixing in the case
const MAX_STEPS = 2500              // the engine follows a particle this many steps; one still flying is dropped
const INWARD: Record<string, [number, number]> = { front: [0, 1], rear: [0, -1], top: [2, -1], bottom: [2, 1] }
const PILE = 30                     // settled particles in a cell (6 mm) at which new ones are drawn fully opaque
/** Released per physical second per unit of weighted inflow (m/s summed over the source cells, about 2100 for your
 * layout at the middle fan speed): about 2400 particles per second of animation there, of which about 40 % settle.
 * A layout that draws in more unfiltered air gathers dust faster. */
const RATE = 6.8
/** Physical seconds per second of animation: the white air particles move at 0.168 x the air speed (Scene.tsx), so the
 * dust drifts with them. */
export const DUST_TIME = 0.168

function bits(b64: string, n: number): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1
  return out
}

const voxelCache = new Map<string, Voxels>()
/** Decoded voxels of a bank (keyed by its bank key: the network's and the simulation's bank of a build share them). */
export function voxelsOf(key: string, v: VoxelPkg | undefined | null): Voxels | null {
  if (!v) return null
  let d = voxelCache.get(key)
  if (!d) {
    const n = v.shape[0] * v.shape[1] * v.shape[2]
    d = { shape: v.shape, solid: bits(v.solid, n), outside: bits(v.outside, n), openings: v.openings, filterEff: v.filter_eff }
    voxelCache.set(key, d)
    while (voxelCache.size > 8) voxelCache.delete(voxelCache.keys().next().value as string)
  }
  return d
}

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
/** Standard normal samples (Box-Muller), drawn from a table: three per particle and step must be cheap. */
const GAUSS = (() => {
  const r = mulberry32(12345)
  const g = new Float32Array(1 << 16)
  for (let i = 0; i < g.length; i += 2) {
    const u1 = Math.max(r(), 1e-12), u2 = r()
    const m = Math.sqrt(-2 * Math.log(u1))
    g[i] = m * Math.cos(2 * Math.PI * u2)
    g[i + 1] = m * Math.sin(2 * Math.PI * u2)
  }
  return g
})()

export interface DustOpts { seed?: number; maxFly?: number; maxDep?: number }

export class DustSim {
  readonly X: number; readonly Y: number; readonly Z: number
  /** dust in the air: positions in cells (x front -> rear, y glass -> tray, z up), nFly of them */
  readonly fly: Float32Array
  nFly = 0
  /** settled dust: positions in cells, on the surface it touched (a third of a cell off it), nDep of them */
  readonly dep: Float32Array
  nDep = 0
  /** how strongly to draw each settled particle, 0..1: by how much has settled in its cell so far (a pile shows as a
   * dense, opaque coat; a stray speck stays faint), and faint on the glass, where dust is a haze in front of the view */
  readonly depW: Float32Array
  private readonly pile: Uint16Array
  /** released, settled and left so far (the share that settles, like the engine's deposited_frac) */
  released = 0; settled = 0; left = 0
  private age: Uint16Array
  private readonly u: Float32Array
  private readonly vox: Voxels
  private readonly dt: number        // s per step: the fastest air moves half a cell
  private readonly mv: number        // cells per (m/s) per step
  private readonly sink: number      // settling, cells per step
  private readonly sig: number       // random walk, cells per step (standard deviation)
  private readonly src: { cells: Int32Array; w: number }[] = []
  private readonly cum: number[] = []
  readonly inflowW: number           // weighted inflow: what comes in, after the filters
  private carry = 0                  // physical time not yet stepped
  private owed = 0                   // particles due but not yet released
  private gi = 0
  private readonly rand: () => number

  constructor(u: Float32Array, vox: Voxels, dxMm: number, opts: DustOpts = {}) {
    ;[this.X, this.Y, this.Z] = vox.shape
    this.u = u
    this.vox = vox
    this.rand = mulberry32(opts.seed ?? 1)
    this.fly = new Float32Array((opts.maxFly ?? 8000) * 3)
    this.age = new Uint16Array(opts.maxFly ?? 8000)
    this.dep = new Float32Array((opts.maxDep ?? 60000) * 3)
    this.depW = new Float32Array(opts.maxDep ?? 60000)
    this.pile = new Uint16Array(opts.maxDep ? this.X * this.Y * this.Z : 0)
    const dx = dxMm / 1000
    let umax = 0
    for (let i = 0; i < u.length; i++) { const a = Math.abs(u[i]); if (a > umax) umax = a }
    this.dt = (0.5 * dx) / (umax + 1e-6)
    this.mv = this.dt / dx
    this.sink = (V_SETTLE * this.dt) / dx
    this.sig = Math.sqrt(2 * D_EDDY * this.dt) / dx
    // sources: in each opening, the cells with inflow above 2 % of its largest, weighted by the opening's inflow
    const N = this.X * this.Y * this.Z
    let total = 0
    for (const o of vox.openings) {
      const [c, sgn] = INWARD[o.wall] ?? [0, 1]
      const [i0, i1, j0, j1, k0, k1] = o.box
      let flux = 0, peak = 0
      for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) for (let k = k0; k < k1; k++) {
        const q = Math.max(0, u[c * N + (i * this.Y + j) * this.Z + k] * sgn)
        flux += q
        if (q > peak) peak = q
      }
      if (flux <= 1e-6) continue
      const cells: number[] = []
      for (let i = i0; i < i1; i++) for (let j = j0; j < j1; j++) for (let k = k0; k < k1; k++)
        if (Math.max(0, u[c * N + (i * this.Y + j) * this.Z + k] * sgn) > 0.02 * peak) cells.push((i * this.Y + j) * this.Z + k)
      const w = flux * (o.filtered ? 1 - vox.filterEff : 1)
      this.src.push({ cells: Int32Array.from(cells), w })
      total += w
      this.cum.push(total)
    }
    this.inflowW = total
  }

  private gauss() {
    this.gi = (this.gi + 1 + ((this.rand() * 64) | 0)) & 0xffff
    return GAUSS[this.gi]
  }

  /** One new particle at a source cell, uniformly within it. */
  private release(): boolean {
    if (!this.src.length || this.nFly * 3 >= this.fly.length) return false
    const r = this.rand() * this.inflowW
    let s = 0
    while (s < this.cum.length - 1 && r > this.cum[s]) s++
    const cells = this.src[s].cells
    const c = cells[(this.rand() * cells.length) | 0]
    const k = c % this.Z, j = ((c - k) / this.Z) % this.Y, i = (c - k - j * this.Z) / (this.Z * this.Y)
    const o = this.nFly * 3
    this.fly[o] = i + this.rand(); this.fly[o + 1] = j + this.rand(); this.fly[o + 2] = k + this.rand()
    this.age[this.nFly] = 0
    this.nFly++
    this.released++
    return true
  }

  /** Move every particle one step. depGrid (optional): count settled particles per cell, as the engine does. */
  private step(depGrid?: Uint32Array) {
    const { X, Y, Z, u, fly } = this
    const N = X * Y * Z, solid = this.vox.solid, outside = this.vox.outside
    const mv = this.mv, sink = this.sink, sig = this.sig
    for (let p = 0; p < this.nFly; ) {
      const o = p * 3
      const x = fly[o], y = fly[o + 1], z = fly[o + 2]
      const i = Math.min(X - 1, Math.max(0, Math.floor(x))), j = Math.min(Y - 1, Math.max(0, Math.floor(y))), k = Math.min(Z - 1, Math.max(0, Math.floor(z)))
      const c = (i * Y + j) * Z + k
      const qx = x + u[c] * mv + this.gauss() * sig
      const qy = y + u[N + c] * mv + this.gauss() * sig
      const qz = z + u[2 * N + c] * mv - sink + this.gauss() * sig
      const qi = Math.min(X - 1, Math.max(0, Math.floor(qx))), qj = Math.min(Y - 1, Math.max(0, Math.floor(qy))), qk = Math.min(Z - 1, Math.max(0, Math.floor(qz)))
      const q = (qi * Y + qj) * Z + qk
      const hit = solid[q] === 1
      if (hit || outside[q] === 1 || ++this.age[p] >= MAX_STEPS) {
        if (hit) {
          this.settled++
          if (depGrid) depGrid[c]++
          if (this.nDep * 3 < this.dep.length) {
            // on the face it crossed into the solid, a third of a cell off it
            let sx = x, sy = y, sz = z
            if (qi !== i && solid[(qi * Y + j) * Z + k]) sx = qi > i ? i + 0.67 : i + 0.33
            else if (qj !== j && solid[(i * Y + qj) * Z + k]) sy = qj > j ? j + 0.67 : j + 0.33
            else if (qk !== k && solid[(i * Y + j) * Z + qk]) sz = qk > k ? k + 0.67 : k + 0.33
            const d = this.nDep * 3
            this.dep[d] = sx; this.dep[d + 1] = sy; this.dep[d + 2] = sz
            const n = this.pile[c] = Math.min(65535, this.pile[c] + 1)
            const glass = qj < j && qj <= 1                       // the glass panel is the shell at y = 1
            this.depW[this.nDep] = (0.3 + 0.7 * Math.min(1, Math.log1p(n) / Math.log1p(PILE))) * (glass ? 0.4 : 1)
            this.nDep++
          }
        } else if (outside[q] === 1) this.left++
        // drop it: the last particle takes its place
        this.nFly--
        const l = this.nFly * 3
        fly[o] = fly[l]; fly[o + 1] = fly[l + 1]; fly[o + 2] = fly[l + 2]
        this.age[p] = this.age[this.nFly]
        continue
      }
      fly[o] = qx; fly[o + 1] = qy; fly[o + 2] = qz
      p++
    }
  }

  /** Advance by dtPhys physical seconds: dust keeps coming in at the steady rate, every particle moves in steps of dt. */
  advance(dtPhys: number) {
    this.carry += dtPhys
    const n = Math.floor(this.carry / this.dt)
    this.carry -= n * this.dt
    for (let s = 0; s < n; s++) {
      this.owed += RATE * this.inflowW * this.dt
      while (this.owed >= 1) {
        this.owed -= 1
        if (!this.release()) { this.owed = 0; break }
      }
      this.step()
    }
  }

  /** As the engine runs it: n particles released at once and followed to the end. Settled particles per cell. */
  static batch(u: Float32Array, vox: Voxels, dxMm: number, n: number, seed = 1) {
    const sim = new DustSim(u, vox, dxMm, { seed, maxFly: n, maxDep: 0 })
    const grid = new Uint32Array(sim.X * sim.Y * sim.Z)
    for (let i = 0; i < n; i++) sim.release()
    for (let s = 0; s < MAX_STEPS && sim.nFly > 0; s++) sim.step(grid)
    return { grid, released: sim.released, settled: sim.settled, left: sim.left, airborne: sim.released - sim.settled - sim.left }
  }
}
