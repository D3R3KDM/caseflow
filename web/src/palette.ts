import * as THREE from 'three'

/** Interface colours (also used by CSS through the same values). */
export const C = {
  bg: 0x05080c,
  line: 0x8cc8ff,
  intake: 0x63e0ff,
  exhaust: 0xffa860,
  dust: 0xd9b77c,
}
export const CSS_IN = '#63e0ff'
export const CSS_OUT = '#ffa860'

/** Air temperature above room, t in [0, 1]: blue -> cyan -> near-white -> amber -> red (sRGB stops). */
const STOPS: [number, string][] = [
  [0.0, '#2a62ff'],
  [0.25, '#35d2ff'],
  [0.5, '#e6f7ff'],
  [0.74, '#ffc26a'],
  [1.0, '#ff553d'],
]
const LIN = STOPS.map(([t, hex]) => [t, new THREE.Color(hex)] as const)   // THREE.Color parses hex into linear space

export function thermal(t: number, out: THREE.Color = new THREE.Color()): THREE.Color {
  t = Math.max(0, Math.min(1, t))
  for (let i = 1; i < LIN.length; i++) {
    if (t <= LIN[i][0]) {
      const [t0, a] = LIN[i - 1]
      const [t1, b] = LIN[i]
      return out.copy(a).lerp(b, (t - t0) / (t1 - t0))
    }
  }
  return out.copy(LIN[LIN.length - 1][1])
}

export const THERMAL_CSS = `linear-gradient(90deg, ${STOPS.map(([t, c]) => `${c} ${t * 100}%`).join(', ')})`
