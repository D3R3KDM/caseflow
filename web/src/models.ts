import * as THREE from 'three'

/** A case / cooler fan: square frame, ring light, hub and swept blades. Axis = local +Y.
 * userData.spin holds the angular speed (rad/s) used by the frame loop. */
export function makeFan(sizeMm: number, S: number, color: number, pct = 80, thicknessMm = 25): THREE.Group {
  const g = new THREE.Group()
  const s = sizeMm * S
  const t = thicknessMm * S
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x1b1b1b, roughness: 0.5, metalness: 0.35 })
  const bar = (w: number, d: number, x: number, z: number) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, t, d), frameMat)
    m.position.set(x, 0, z)
    g.add(m)
  }
  const bw = s * 0.07
  bar(s, bw, 0, (s - bw) / 2)
  bar(s, bw, 0, -(s - bw) / 2)
  bar(bw, s, (s - bw) / 2, 0)
  bar(bw, s, -(s - bw) / 2, 0)
  // glowing ring (ARGB look)
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(s * 0.43, s * 0.018, 8, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 }),
  )
  ring.rotation.x = Math.PI / 2
  ring.position.y = t * 0.45
  g.add(ring)
  // rotor
  const rotor = new THREE.Group()
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(s * 0.14, s * 0.14, t * 0.8, 24), new THREE.MeshStandardMaterial({ color: 0x2f2f2f, roughness: 0.4, metalness: 0.3 }))
  rotor.add(hub)
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0xe4e4e4, roughness: 0.35, metalness: 0.05, transparent: true, opacity: 0.88, side: THREE.DoubleSide })
  const nb = 7
  for (let i = 0; i < nb; i++) {
    const shape = new THREE.Shape()
    shape.moveTo(0, 0)
    shape.quadraticCurveTo(s * 0.12, s * 0.06, s * 0.27, s * 0.02)
    shape.quadraticCurveTo(s * 0.2, -s * 0.08, 0, -s * 0.05)
    const blade = new THREE.Mesh(new THREE.ShapeGeometry(shape, 8), bladeMat)
    blade.rotation.x = -Math.PI / 2 + 0.35
    const holder = new THREE.Group()
    blade.position.x = s * 0.12
    holder.add(blade)
    holder.rotation.y = (i / nb) * Math.PI * 2
    rotor.add(holder)
  }
  g.add(rotor)
  g.userData.rotor = rotor
  g.userData.spin = 4 + (pct / 100) * 14
  return g
}

/** A stack of thin fins (plates in the x-z plane... caller rotates) filling a box. */
export function makeFinStack(w: number, h: number, d: number, plates: number, color = 0xbdbdbd): THREE.Group {
  const g = new THREE.Group()
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.75 })
  const gap = h / plates
  for (let i = 0; i < plates; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, gap * 0.35, d), mat)
    m.position.y = -h / 2 + gap * (i + 0.5)
    g.add(m)
  }
  return g
}

let meshTex: THREE.Texture | null = null
/** Perforated-mesh texture for mesh panels. */
export function meshTexture(): THREE.Texture {
  if (meshTex) return meshTex
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const ctx = c.getContext('2d')!
  ctx.fillStyle = 'rgba(255,255,255,0.12)'
  ctx.fillRect(0, 0, 64, 64)
  ctx.globalCompositeOperation = 'destination-out'
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++) {
      ctx.beginPath()
      ctx.arc(8 + x * 16 + (y % 2) * 8, 8 + y * 16, 5.2, 0, Math.PI * 2)
      ctx.fill()
    }
  meshTex = new THREE.CanvasTexture(c)
  meshTex.wrapS = meshTex.wrapT = THREE.RepeatWrapping
  meshTex.colorSpace = THREE.SRGBColorSpace
  return meshTex
}
