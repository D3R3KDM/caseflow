/** Velocity-field sampling and streamline tracing on the solver grid (cell units). */
export interface Field {
  dims: [number, number, number]
  u: Float32Array // 3 * N, layout [c][x][y][z]
  cp: Float32Array // N
  /** full-grid cells per field cell (1 for solver results, 2 for the 64x32x32 network output) */
  scale: number
}


export function sampleU(f: Field, px: number, py: number, pz: number, out: Float32Array): boolean {
  const [X, Y, Z] = f.dims
  const s = f.scale
  let x = px / s - 0.5
  let y = py / s - 0.5
  let z = pz / s - 0.5
  if (x < 0 || x > X - 1 || y < -0.5 || y > Y - 0.5 || z < -0.5 || z > Z - 0.5) return false
  // periodic in y and z
  y = (y + Y) % Y
  z = (z + Z) % Z
  const x0 = Math.min(Math.floor(x), X - 2)
  const y0 = Math.floor(y)
  const z0 = Math.floor(z)
  const fx = x - x0
  const fy = y - y0
  const fz = z - z0
  const y1 = (y0 + 1) % Y
  const z1 = (z0 + 1) % Z
  const N = X * Y * Z
  const i000 = (x0 * Y + y0) * Z + z0
  const i001 = (x0 * Y + y0) * Z + z1
  const i010 = (x0 * Y + y1) * Z + z0
  const i011 = (x0 * Y + y1) * Z + z1
  const i100 = ((x0 + 1) * Y + y0) * Z + z0
  const i101 = ((x0 + 1) * Y + y0) * Z + z1
  const i110 = ((x0 + 1) * Y + y1) * Z + z0
  const i111 = ((x0 + 1) * Y + y1) * Z + z1
  const w000 = (1 - fx) * (1 - fy) * (1 - fz)
  const w001 = (1 - fx) * (1 - fy) * fz
  const w010 = (1 - fx) * fy * (1 - fz)
  const w011 = (1 - fx) * fy * fz
  const w100 = fx * (1 - fy) * (1 - fz)
  const w101 = fx * (1 - fy) * fz
  const w110 = fx * fy * (1 - fz)
  const w111 = fx * fy * fz
  const u = f.u
  for (let c = 0; c < 3; c++) {
    const o = c * N
    out[c] =
      u[o + i000] * w000 + u[o + i001] * w001 + u[o + i010] * w010 + u[o + i011] * w011 +
      u[o + i100] * w100 + u[o + i101] * w101 + u[o + i110] * w110 + u[o + i111] * w111
  }
  return true
}

export interface StreamlineSet {
  positions: Float32Array // segment endpoints (full-grid cell units)
  speeds: Float32Array // speed per vertex (u/U)
  paths: { start: number; count: number }[] // vertex ranges per streamline (polyline points)
  polyline: Float32Array // polyline points x,y,z per point
  polySpeed: Float32Array
}

/** Trace streamlines with a midpoint (RK2) integrator. */
export function traceStreamlines(
  f: Field,
  seeds: [number, number, number][],
  opts: { h?: number; maxSteps?: number; full?: [number, number, number] } = {},
): StreamlineSet {
  const h = opts.h ?? 0.7
  const maxSteps = opts.maxSteps ?? 420
  const full = opts.full ?? f.dims
  const seg: number[] = []
  const spd: number[] = []
  const poly: number[] = []
  const polyS: number[] = []
  const paths: { start: number; count: number }[] = []
  const k1 = new Float32Array(3)
  const k2 = new Float32Array(3)
  for (const s of seeds) {
    let [x, y, z] = s
    const start = poly.length / 3
    let count = 0
    for (let i = 0; i < maxSteps; i++) {
      if (!sampleU(f, x, y, z, k1)) break
      const sp = Math.hypot(k1[0], k1[1], k1[2])
      if (sp < 0.015) break
      poly.push(x, y, z)
      polyS.push(sp)
      count++
      const mx = x + (0.5 * h * k1[0]) / sp
      const my = y + (0.5 * h * k1[1]) / sp
      const mz = z + (0.5 * h * k1[2]) / sp
      if (!sampleU(f, mx, my, mz, k2)) break
      const sp2 = Math.hypot(k2[0], k2[1], k2[2])
      if (sp2 < 0.015) break
      const nx = x + (h * k2[0]) / sp2
      const ny = y + (h * k2[1]) / sp2
      const nz = z + (h * k2[2]) / sp2
      seg.push(x, y, z, nx, ny, nz)
      spd.push(sp, sp2)
      x = nx
      y = ny
      z = nz
      if (x < 0 || x >= full[0] - 1) break
    }
    if (count > 1) paths.push({ start, count })
  }
  return {
    positions: new Float32Array(seg),
    speeds: new Float32Array(spd),
    paths,
    polyline: new Float32Array(poly),
    polySpeed: new Float32Array(polyS),
  }
}

/** Bounding box of mesh positions (cell units). */
