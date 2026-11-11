import * as THREE from 'three'

/** Interface colours (the same values as the CSS tokens in styles.css). Air in / out: a categorical pair that passes
 * the dataviz palette checks on the scene surface #161616 (lightness band, chroma, CVD and normal-vision separation,
 * contrast). */
export const C = {
  bg: 0x161616,
  line: 0xcfcfcf,
  lead: 0xd8d8d8,          // leader lines and anchors of the floating screens
  intake: 0x2fa8a9,
  exhaust: 0xd9773a,
  dust: 0xc49a5a,          // ochre: settled dust, apart from the air (heat ramp) and the fans (teal / copper)
}
export const CSS_IN = '#2fa8a9'
export const CSS_OUT = '#d9773a'

/** Air temperature above room, t in [0, 1]: a sequential ramp whose lightness rises with heat - cool air a quiet light
 * slate that still reads on the dark scene, hot air glowing: slate -> warm grey -> orange -> amber -> pale yellow. */
const STOPS: [number, string][] = [
  [0.0, '#7f93ad'],
  [0.35, '#b7a28e'],
  [0.6, '#e39552'],
  [0.82, '#f6bd6c'],
  [1.0, '#fff3d1'],
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
