import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import * as THREE from 'three'
import { CSS3DObject, CSS3DRenderer } from 'three/examples/jsm/renderers/CSS3DRenderer.js'
import { b64ToBuffer, f16ToF32 } from './api'
import { makeFan, makeFinStack, meshTexture } from './models'
import { C, thermal } from './palette'
import { traceStreamlines, type Field } from './streamlines'
import type { SimRes } from './types'
import { createViewer, disposeObject, type Viewer } from './viewer'

export const T_MAX = 12                      // °C above room at the top of the colour scale
export interface Layers { flow: boolean; lines: boolean; dust: boolean; heat: boolean; panels: boolean }
export type Anchor = 'cpu' | 'gpu' | 'intake' | 'glass' | 'exhaust'
/** A floating screen around the case. pos is in the "ring frame": x right, y up, z towards the viewer for the
 * default view; the ring turns with the camera so the screens stay arranged around the case and readable. */
export interface HoloSpec { id: string; pos: [number, number, number]; yaw: number; anchor?: Anchor; content: ReactNode }

const SCALE = 1 / 250                        // world units per mm of the case
const PX = 0.0039                            // world units per CSS pixel of a screen
const FLOOR = -0.95
const AZ0 = 0.42, EL0 = 0.14                 // default view: a little from the front, slightly above
const FIT_W = 9.1, FIT_H = 4.7               // world extent the default view must show (screens included)

interface HoloObj {
  obj: CSS3DObject; w: number; h: number; ro: ResizeObserver
  line: THREE.Line; marker: THREE.Group; packet: THREE.Mesh; stand: THREE.Line; base: THREE.Mesh; phase: number
}

let dotTex: THREE.Texture | null = null
/** Soft round sprite for particles. */
function dot(): THREE.Texture {
  if (dotTex) return dotTex
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32)
  r.addColorStop(0, 'rgba(255,255,255,1)')
  r.addColorStop(0.35, 'rgba(255,255,255,0.75)')
  r.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = r
  g.fillRect(0, 0, 64, 64)
  dotTex = new THREE.CanvasTexture(c)
  return dotTex
}

function glowDisc(radius: number): THREE.Mesh {
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const g = c.getContext('2d')!
  const r = g.createRadialGradient(128, 128, 0, 128, 128, 128)
  r.addColorStop(0, 'rgba(70,150,230,0.22)')
  r.addColorStop(0.45, 'rgba(40,110,190,0.08)')
  r.addColorStop(1, 'rgba(0,0,0,0)')
  g.fillStyle = r
  g.fillRect(0, 0, 256, 256)
  const m = new THREE.Mesh(new THREE.CircleGeometry(radius, 96), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }))
  m.rotation.x = -Math.PI / 2
  return m
}

/** Thin ring with tick marks (the turntable under the case). */
function tickRing(radius: number, ticks: number, color: number, opacity: number): THREE.LineSegments {
  const pts: number[] = []
  const n = 192
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * Math.PI * 2, a1 = ((i + 1) / n) * Math.PI * 2
    pts.push(Math.cos(a0) * radius, 0, Math.sin(a0) * radius, Math.cos(a1) * radius, 0, Math.sin(a1) * radius)
  }
  for (let i = 0; i < ticks; i++) {
    const a = (i / ticks) * Math.PI * 2, l = i % 6 === 0 ? 0.09 : 0.04
    pts.push(Math.cos(a) * radius, 0, Math.sin(a) * radius, Math.cos(a) * (radius + l), 0, Math.sin(a) * (radius + l))
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }))
}

interface Anim {
  fans: THREE.Object3D[]
  ring: THREE.Object3D | null
  parts: null | {
    pts: THREE.Points; paths: { start: number; count: number }[]; poly: Float32Array; speed: Float32Array; temp: Float32Array
    phase: Float32Array; P: (x: number, y: number, z: number) => THREE.Vector3; toMm: (p: number) => number
  }
}

export interface Pick { slot: string; empty: boolean; x: number; y: number }

export default function CaseScene({ res, layers, panels, resetKey, onPick, selected }: {
  res: SimRes | null; layers: Layers; panels: HoloSpec[]; resetKey: number; onPick?: (p: Pick | null) => void; selected?: string | null
}) {
  const host = useRef<HTMLDivElement>(null)
  const viewer = useRef<Viewer | null>(null)
  const group = useRef<THREE.Group | null>(null)
  const anim = useRef<Anim>({ fans: [], ring: null, parts: null })
  // floating screens
  const els = useRef(new Map<string, HTMLDivElement>())
  const holos = useRef(new Map<string, HoloObj>())
  const specs = useRef<HoloSpec[]>(panels)
  specs.current = panels
  const anchors = useRef<Partial<Record<Anchor, THREE.Vector3>>>({})
  const css = useRef<{ ring: THREE.Group; gl: THREE.Group; fit: () => void } | null>(null)
  // clickable case fans and empty mounts
  const pickRoots = useRef<THREE.Object3D[]>([])
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  const elFor = (id: string) => {
    let d = els.current.get(id)
    if (!d) {
      d = document.createElement('div')
      d.className = 'holo-wrap'
      els.current.set(id, d)
    }
    return d
  }

  // ---------- viewer + environment ----------
  useEffect(() => {
    const v = createViewer(host.current!, C.bg)
    const floorY = FLOOR
    // CSS3D layer for the screens, on top of the WebGL canvas; it shares the camera
    const cssR = new CSS3DRenderer()
    cssR.domElement.className = 'holo-layer'
    host.current!.appendChild(cssR.domElement)
    const cssScene = new THREE.Scene()
    const ringFrame = new THREE.Group()
    cssScene.add(ringFrame)
    const holoGL = new THREE.Group()
    // pick a case fan / empty mount with a click (a drag still orbits)
    const ray = new THREE.Raycaster()
    const ndc = new THREE.Vector2()
    const hoverRing = new THREE.Mesh(new THREE.TorusGeometry(1, 0.03, 8, 72), new THREE.MeshBasicMaterial({ color: 0xe8f8ff, transparent: true, opacity: 0.9, depthTest: false }))
    hoverRing.renderOrder = 30
    hoverRing.visible = false
    v.scene.add(hoverRing)
    let hovered: THREE.Object3D | null = null
    const hitAt = (e: PointerEvent) => {
      const r = v.renderer.domElement.getBoundingClientRect()
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
      ray.setFromCamera(ndc, v.camera)
      for (const h of ray.intersectObjects(pickRoots.current, true)) {
        let o: THREE.Object3D | null = h.object
        while (o && !o.userData.pick) o = o.parent
        if (o) return o
      }
      return null
    }
    const cv = v.renderer.domElement
    let down: { x: number; y: number; t: number } | null = null
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY, t: performance.now() } }
    const onUp = (e: PointerEvent) => {
      const d = down
      down = null
      if (!d || e.button !== 0 || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5 || performance.now() - d.t > 450) return
      const o = hitAt(e)
      onPickRef.current?.(o ? { slot: o.userData.pick.slot, empty: o.userData.pick.empty, x: e.clientX, y: e.clientY } : null)
    }
    const onMove = (e: PointerEvent) => {
      if (e.buttons) return
      hovered = hitAt(e)
      cv.style.cursor = hovered ? 'pointer' : ''
    }
    cv.addEventListener('pointerdown', onDown)
    cv.addEventListener('pointerup', onUp)
    cv.addEventListener('pointermove', onMove)
    const placeRing = () => {
      const sel = selectedRef.current ? pickRoots.current.find((o) => o.userData.pick.slot === selectedRef.current) : undefined
      const o = hovered && pickRoots.current.includes(hovered) ? hovered : sel
      hoverRing.visible = !!o
      if (!o) return
      const pk = o.userData.pick as { pos: THREE.Vector3; normal: THREE.Vector3; size: number }
      hoverRing.position.copy(pk.pos).addScaledVector(pk.normal, 0.06)
      hoverRing.lookAt(hoverRing.position.clone().add(pk.normal))
      const pulse = o === sel ? 1 + 0.05 * Math.sin(performance.now() / 160) : 1
      hoverRing.scale.setScalar((pk.size / 2) * SCALE * 1.08 * pulse)
    }
    v.scene.add(holoGL)
    const sizeCss = () => cssR.setSize(host.current!.clientWidth || 1, host.current!.clientHeight || 1)
    const ro = new ResizeObserver(sizeCss)
    ro.observe(host.current!)
    sizeCss()
    const fit = () => {
      const w = host.current!.clientWidth || 1, h = host.current!.clientHeight || 1
      const vf = (v.camera.fov * Math.PI) / 180
      const hf = 2 * Math.atan(Math.tan(vf / 2) * (w / h))
      const d = Math.max(FIT_W / 2 / Math.tan(hf / 2), FIT_H / 2 / Math.tan(vf / 2))
      v.controls.target.set(0, -0.06, 0)
      v.camera.position.set(Math.sin(AZ0) * Math.cos(EL0) * d, -0.06 + Math.sin(EL0) * d, Math.cos(AZ0) * Math.cos(EL0) * d)
      v.controls.maxDistance = d * 1.6
      v.controls.update()
    }
    fit()
    css.current = { ring: ringFrame, gl: holoGL, fit }
    const grid = new THREE.PolarGridHelper(3.2, 24, 12, 128, 0x2c5f86, 0x16324a)
    for (const m of [grid.material].flat()) Object.assign(m, { transparent: true, opacity: 0.35 })
    grid.position.y = floorY
    v.scene.add(grid)
    const disc = glowDisc(2.4)
    disc.position.y = floorY + 0.002
    v.scene.add(disc)
    const ring = tickRing(1.05, 72, C.line, 0.45)
    ring.position.y = floorY + 0.004
    v.scene.add(ring)
    anim.current.ring = ring
    const g = new THREE.Group()
    v.scene.add(g)
    viewer.current = v
    group.current = g
    const tmp = new THREE.Vector3()
    const c = new THREE.Color()
    const qa = new THREE.Vector3(), qb = new THREE.Vector3(), qe = new THREE.Vector3(), qk = new THREE.Vector3()
    let clock = 0
    const updateScreens = (dt: number) => {
      clock += dt
      const cam = v.camera, tgt = v.controls.target
      ringFrame.position.copy(tgt)
      ringFrame.rotation.y = Math.atan2(cam.position.x - tgt.x, cam.position.z - tgt.z)   // ring faces the camera
      ringFrame.updateMatrixWorld(true)
      const byId = new Map(specs.current.map((s) => [s.id, s]))
      for (const [id, hobj] of holos.current) {
        const s = byId.get(id)
        if (!s) continue
        const { obj } = hobj
        obj.position.set(s.pos[0], s.pos[1], s.pos[2])
        obj.lookAt(cam.position)
        obj.rotateY(s.yaw)
        obj.updateMatrixWorld(true)
        // projector beam from the floor to the bottom of the screen
        obj.localToWorld(qa.set(0, -hobj.h / 2, 0))
        const sp = hobj.stand.geometry.attributes.position as THREE.BufferAttribute
        sp.setXYZ(0, qa.x, floorY + 0.005, qa.z)
        sp.setXYZ(1, qa.x, qa.y, qa.z)
        sp.needsUpdate = true
        hobj.stand.computeLineDistances()
        hobj.base.position.set(qa.x, floorY + 0.006, qa.z)
        // leader line: screen edge -> elbow -> part
        const anc = s.anchor ? anchors.current[s.anchor] : undefined
        hobj.line.visible = hobj.marker.visible = hobj.packet.visible = !!anc
        if (!anc) continue
        obj.worldToLocal(qk.copy(anc))
        const side = qk.x >= 0 ? 1 : -1
        obj.localToWorld(qe.set(side * hobj.w / 2, hobj.h / 2 - 24, 0))
        obj.localToWorld(qb.set(side * (hobj.w / 2 + 44), hobj.h / 2 - 24, 0))
        const lp = hobj.line.geometry.attributes.position as THREE.BufferAttribute
        lp.setXYZ(0, qe.x, qe.y, qe.z); lp.setXYZ(1, qb.x, qb.y, qb.z); lp.setXYZ(2, anc.x, anc.y, anc.z)
        lp.needsUpdate = true
        hobj.line.geometry.computeBoundingSphere()
        hobj.marker.position.copy(anc)
        hobj.marker.quaternion.copy(cam.quaternion)
        hobj.marker.children[0].scale.setScalar(1 + 0.45 * (0.5 + 0.5 * Math.sin(clock * 3 + hobj.phase)))
        // a light packet runs from the part to the screen
        const l1 = anc.distanceTo(qb), l2 = qb.distanceTo(qe)
        const u = ((clock * 0.45 + hobj.phase) % 1) * (l1 + l2)
        if (u < l1) hobj.packet.position.lerpVectors(anc, qb, u / l1)
        else hobj.packet.position.lerpVectors(qb, qe, (u - l1) / Math.max(l2, 1e-6))
        hobj.packet.quaternion.copy(cam.quaternion)
      }
      cssR.render(cssScene, cam)
    }
    const off = v.onFrame((dt) => {
      const a = anim.current
      if (a.ring) a.ring.rotation.y += dt * 0.05
      for (const f of a.fans) {
        const rotor = f.userData.rotor as THREE.Object3D | undefined
        if (rotor) rotor.rotation.y += dt * (f.userData.spin as number)
      }
      updateScreens(dt)
      placeRing()
      // fog follows the camera: only the far floor fades, the case never goes dark when zoomed out
      const fog = v.scene.fog as THREE.Fog
      const dist = v.camera.position.distanceTo(v.controls.target)
      fog.near = dist + 2.5
      fog.far = dist + 10
      const ps = a.parts
      if (!ps || !ps.pts.visible) return
      const pos = ps.pts.geometry.attributes.position as THREE.BufferAttribute
      const col = ps.pts.geometry.attributes.color as THREE.BufferAttribute
      for (let k = 0; k < ps.phase.length; k++) {
        const path = ps.paths[k % ps.paths.length]
        const idx = Math.min(path.count - 1, Math.floor(ps.phase[k]))
        ps.phase[k] += dt * 40 * Math.max(ps.speed[path.start + idx], 0.15)
        if (ps.phase[k] >= path.count - 1) ps.phase[k] = 0
        const j = path.start + Math.floor(ps.phase[k])
        tmp.copy(ps.P(ps.toMm(ps.poly[3 * j]), ps.toMm(ps.poly[3 * j + 1]), ps.toMm(ps.poly[3 * j + 2])))
        pos.setXYZ(k, tmp.x, tmp.y, tmp.z)
        thermal(ps.temp[j] / T_MAX, c)
        col.setXYZ(k, c.r, c.g, c.b)
      }
      pos.needsUpdate = true
      col.needsUpdate = true
    })
    return () => {
      off()
      ro.disconnect()
      cv.removeEventListener('pointerdown', onDown)
      cv.removeEventListener('pointerup', onUp)
      cv.removeEventListener('pointermove', onMove)
      for (const h of holos.current.values()) h.ro.disconnect()
      holos.current.clear()
      cssR.domElement.remove()
      css.current = null
      v.dispose()
    }
  }, [])

  // ---------- screens: create / remove CSS3D objects and their leader lines ----------
  const idKey = panels.map((p) => p.id).join('|')
  useEffect(() => {
    const k = css.current
    if (!k) return
    const want = new Set(panels.map((p) => p.id))
    for (const [id, h] of holos.current) {
      if (want.has(id)) continue
      k.ring.remove(h.obj)
      for (const o of [h.line, h.marker, h.packet, h.stand, h.base]) { k.gl.remove(o); disposeObject(o) }
      h.ro.disconnect()
      holos.current.delete(id)
    }
    panels.forEach((p, i) => {
      if (holos.current.has(p.id)) return
      const el = elFor(p.id)
      const obj = new CSS3DObject(el)
      obj.scale.setScalar(PX)
      k.ring.add(obj)
      const lineMat = new THREE.LineBasicMaterial({ color: C.intake, transparent: true, opacity: 0.7, depthTest: false })
      const line = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(9), 3)), lineMat)
      const flat = (geo: THREE.BufferGeometry, opacity: number) =>
        new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: C.intake, transparent: true, opacity, depthTest: false, side: THREE.DoubleSide }))
      const marker = new THREE.Group()
      marker.add(flat(new THREE.RingGeometry(0.03, 0.038, 40), 0.85))
      marker.add(flat(new THREE.CircleGeometry(0.011, 20), 1))
      const packet = flat(new THREE.CircleGeometry(0.013, 16), 1)
      const stand = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3)),
        new THREE.LineDashedMaterial({ color: C.intake, transparent: true, opacity: 0.28, dashSize: 0.04, gapSize: 0.035 }))
      const base = flat(new THREE.RingGeometry(0.05, 0.062, 40), 0.35)
      base.rotation.x = -Math.PI / 2
      for (const o of [line, marker, packet]) o.renderOrder = 20
      for (const o of [line, marker, packet, stand, base]) { o.frustumCulled = false; k.gl.add(o) }
      const h: HoloObj = { obj, w: 260, h: 200, ro: new ResizeObserver(() => { h.w = el.offsetWidth || h.w; h.h = el.offsetHeight || h.h }), line, marker, packet, stand, base, phase: i * 0.37 }
      h.ro.observe(el)
      holos.current.set(p.id, h)
    })
  }, [idKey]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { css.current?.fit() }, [resetKey])

  // ---------- case, parts and fields ----------
  useEffect(() => {
    const g = group.current
    if (!g) return
    g.children.forEach(disposeObject)
    g.clear()
    anim.current.fans = []
    anim.current.parts = null
    pickRoots.current = []
    if (!res) return
    const { D, W, H } = res.case
    const S = SCALE
    // case frame (x front->rear, y glass->tray, z up) is left-handed: mirror it so that, through the glass,
    // the front is on the right and the rear I/O on the left - as the PC stands on a desk
    const P = (x: number, y: number, z: number) => new THREE.Vector3((D / 2 - x) * S, (z - H / 2) * S, (W / 2 - y) * S)
    const add = <T extends THREE.Object3D>(o: T) => { g.add(o); return o }
    const lineMat = (opacity: number, color = C.line) => new THREE.LineBasicMaterial({ color, transparent: true, opacity })
    const outline = (m: THREE.Mesh, opacity: number) => { m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 20), lineMat(opacity))); return m }
    const boxGeo = (b: number[]) => new THREE.BoxGeometry(Math.max(b[1] - b[0], 1) * S, Math.max(b[5] - b[4], 1) * S, Math.max(b[3] - b[2], 1) * S)
    const centre = (b: number[]) => P((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
    const boxMesh = (b: number[], mat: THREE.Material, edge = 0) => {
      const m = new THREE.Mesh(boxGeo(b), mat)
      m.position.copy(centre(b))
      if (edge) outline(m, edge)
      return add(m)
    }
    const solid = (color: number, opacity = 1, metal = 0.45) =>
      new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: metal, transparent: opacity < 1, opacity })

    // shell: two outlines (outer bright, inner faint) + panels
    for (const [e, o] of [[0, 0.6], [7, 0.16]] as const) {
      const l = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry((D + 2 * e) * S, (H + 2 * e) * S, (W + 2 * e) * S)), lineMat(o))
      l.position.copy(P(D / 2, W / 2, H / 2))
      add(l)
    }
    const meshMat = (w: number, h: number) => {
      const t = meshTexture().clone()
      t.needsUpdate = true
      t.repeat.set(w / 14, h / 14)
      return new THREE.MeshBasicMaterial({ map: t, transparent: true, side: THREE.DoubleSide, depthWrite: false })
    }
    const panel = (w: number, h: number, pos: THREE.Vector3, rot: [number, number, number], mat: THREE.Material) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w * S, h * S), mat)
      m.position.copy(pos)
      m.rotation.set(...rot)
      return add(m)
    }
    panel(W, H, P(0, W / 2, H / 2), [0, Math.PI / 2, 0], meshMat(W, H))
    panel(D, W, P(D / 2, W / 2, H), [-Math.PI / 2, 0, 0], meshMat(D, W))
    panel(D, W, P(D / 2, W / 2, 0), [-Math.PI / 2, 0, 0], meshMat(D, W))
    panel(D, H, P(D / 2, W, H / 2), [0, 0, 0], new THREE.MeshBasicMaterial({ color: 0x070b11, transparent: true, opacity: 0.72, side: THREE.DoubleSide, depthWrite: false }))
    panel(D, H, P(D / 2, 0, H / 2), [0, 0, 0], new THREE.MeshPhysicalMaterial({ color: 0xa8dcff, transparent: true, opacity: 0.045, roughness: 0.04, side: THREE.DoubleSide, depthWrite: false }))
    {
      // corner brackets on the glass
      const k = 34, pts: number[] = []
      const corner = (x: number, z: number, sx: number, sz: number) => {
        const a = P(x, -2, z), b = P(x + sx * k, -2, z), c2 = P(x, -2, z + sz * k)
        pts.push(a.x, a.y, a.z, b.x, b.y, b.z, a.x, a.y, a.z, c2.x, c2.y, c2.z)
      }
      corner(8, 8, 1, 1); corner(D - 8, 8, -1, 1); corner(8, H - 8, 1, -1); corner(D - 8, H - 8, -1, -1)
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
      add(new THREE.LineSegments(geo, lineMat(0.9, C.intake)))
    }

    // components
    for (const bx of res.boxes) {
      const b = bx.b
      if (bx.kind === 'psu') boxMesh(b, solid(0x0d1218), 0.28)
      else if (bx.kind === 'board') {
        boxMesh(b, solid(0x0c201b, 0.96, 0.2), 0.22)
        const ram = solid(0x161c24, 1, 0.6)
        boxMesh([b[1] - 150, b[1] - 143, b[2] - 34, b[2], b[5] - 140, b[5] - 10], ram, 0.3)
        boxMesh([b[1] - 140, b[1] - 133, b[2] - 34, b[2], b[5] - 140, b[5] - 10], ram, 0.3)
        boxMesh([b[1] - 45, b[1], b[2] - 30, b[2], b[5] - 90, b[5] - 5], solid(0x131820, 1, 0.6), 0.22)
      } else if (bx.kind === 'gpu') boxMesh(b, solid(0x232932, 1, 0.85), 0.4)
      else if (bx.kind === 'gpu_hs') {
        const fins = makeFinStack((b[1] - b[0]) * S, (b[5] - b[4]) * S, (b[3] - b[2]) * S, 14, 0x6f7b88)
        fins.rotation.z = Math.PI / 2
        fins.scale.set((b[5] - b[4]) / (b[1] - b[0]), (b[1] - b[0]) / (b[5] - b[4]), 1)
        fins.position.copy(centre(b))
        add(fins)
      } else if (bx.kind === 'gpu_fan') {
        const f = makeFan(Math.min(b[1] - b[0], b[3] - b[2]) * 0.92, S, 0x9fb6cc, 70, 10)
        f.rotation.x = Math.PI
        f.position.copy(centre(b))
        anim.current.fans.push(add(f))
      } else if (bx.kind === 'cpu_hs') {
        const fins = makeFinStack((b[1] - b[0]) * S, (b[5] - b[4]) * S, (b[3] - b[2]) * S, 22, 0x75808d)
        fins.position.copy(centre(b))
        add(fins)
        boxMesh([b[0] + 4, b[1] - 4, b[2] - 6, b[2], b[4], b[5]], solid(0x161b22, 1, 0.6))
      } else if (bx.kind === 'cpu_fan') {
        const f = makeFan(Math.min(b[3] - b[2], b[5] - b[4]), S, 0xd6e6ff, 70, b[1] - b[0])
        f.rotation.z = Math.PI / 2                                  // airflow front -> rear = -X
        f.position.copy(centre(b))
        anim.current.fans.push(add(f))
      } else if (bx.kind === 'vent') boxMesh(b, new THREE.MeshBasicMaterial({ color: 0x4f7396, transparent: true, opacity: 0.22 }))
    }

    // case fans + empty mounts
    const mountPos = (m: { wall: string; a: number; b: number }) =>
      m.wall === 'front' ? P(0, m.a, m.b) : m.wall === 'rear' ? P(D, m.a, m.b) : m.wall === 'top' ? P(m.a, m.b, H) : P(m.a, m.b, 0)
    const inward: Record<string, THREE.Vector3> = { front: new THREE.Vector3(-1, 0, 0), rear: new THREE.Vector3(1, 0, 0), top: new THREE.Vector3(0, -1, 0), bottom: new THREE.Vector3(0, 1, 0) }
    for (const m of res.mounts) {
      const f = res.fans.find((x) => x.slot === m.slot)
      const pos = mountPos(m)
      if (!f) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry((Math.max(...m.sizes) / 2) * S * 0.86, 0.0025, 6, 64), new THREE.MeshBasicMaterial({ color: 0x35506b }))
        ring.lookAt(inward[m.wall])
        ring.position.copy(pos)
        add(ring)
        // invisible disc so an empty mount is easy to click
        const s0 = Math.max(...m.sizes)
        const hit = new THREE.Mesh(new THREE.CircleGeometry((s0 / 2) * S, 32), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }))
        hit.lookAt(inward[m.wall])
        hit.position.copy(pos)
        hit.userData.pick = { slot: m.slot, empty: true, pos: pos.clone(), normal: inward[m.wall].clone(), size: s0 }
        pickRoots.current.push(add(hit))
        continue
      }
      const color = f.mode === 'in' ? C.intake : C.exhaust
      const fan = makeFan(f.size, S, color, f.pct, f.size >= 140 ? 30 : 25)
      const flow = inward[m.wall].clone().multiplyScalar(f.mode === 'in' ? 1 : -1)
      fan.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), flow)   // local +Y = airflow
      fan.position.copy(pos).addScaledVector(inward[m.wall], 0.05)
      anim.current.fans.push(add(fan))
      fan.userData.pick = { slot: m.slot, empty: false, pos: pos.clone(), normal: inward[m.wall].clone(), size: f.size }
      pickRoots.current.push(fan)
      const arrow = new THREE.ArrowHelper(flow, pos.clone().addScaledVector(flow, -0.08), 0.17, color, 0.045, 0.026)
      ;(arrow.line.material as THREE.LineBasicMaterial).transparent = true
      ;(arrow.line.material as THREE.LineBasicMaterial).opacity = 0.75
      add(arrow)
    }

    // anchor points the screens point at
    {
      const cen = (kinds: string[]) => {
        const bs = res.boxes.filter((b) => kinds.includes(b.kind)).map((b) => b.b)
        if (!bs.length) return undefined
        const m = [0, 1, 2, 3, 4, 5].map((i) => (i % 2 ? Math.max : Math.min)(...bs.map((b) => b[i])))
        return P((m[0] + m[1]) / 2, m[2] + 4, (m[4] + m[5]) / 2)       // on the glass-facing side of the part
      }
      const fanC = (pred: (f: SimRes['fans'][number]) => boolean) => {
        const fs = res.fans.filter(pred)
        if (!fs.length) return undefined
        const v = new THREE.Vector3()
        fs.forEach((f) => v.add(mountPos(f)))
        return v.multiplyScalar(1 / fs.length)
      }
      anchors.current = {
        cpu: cen(['cpu_hs', 'cpu_fan']),
        gpu: cen(['gpu', 'gpu_hs']),
        intake: fanC((f) => f.mode === 'in' && f.wall === 'front') ?? fanC((f) => f.mode === 'in'),
        exhaust: fanC((f) => f.mode === 'out'),
        glass: P(D * 0.14, 0, H * 0.2),                             // lower front of the glass, where dust lands
      }
    }

    // airflow field -> streamlines and moving particles
    const [X, Y, Z] = res.shape
    const N = X * Y * Z
    const u = f16ToF32(b64ToBuffer(res.u))
    const T = f16ToF32(b64ToBuffer(res.T))
    const field: Field = { dims: [X, Y, Z], u, cp: new Float32Array(N), scale: 1 }
    const offc = res.off, dx = res.dx_mm
    const cell = (mm: number) => mm / dx + offc
    const toMm = (p: number) => (p - offc) * dx
    const seeds: [number, number, number][] = []
    for (const f of res.fans.filter((q) => q.mode === 'in'))
      for (let a = -45; a <= 45; a += 15)
        for (let b = -45; b <= 45; b += 15) {
          if (f.wall === 'front') seeds.push([cell(14), cell(f.a + a), cell(f.b + b)])
          else if (f.wall === 'rear') seeds.push([cell(D - 14), cell(f.a + a), cell(f.b + b)])
          else if (f.wall === 'top') seeds.push([cell(f.a + a), cell(f.b + b), cell(H - 14)])
          else seeds.push([cell(f.a + a), cell(f.b + b), cell(14)])
        }
    for (let x = 40; x < D; x += 70) for (let z = 40; z < H; z += 70) seeds.push([cell(x), cell(W * 0.45), cell(z)])
    const sl = traceStreamlines(field, seeds, { h: 0.7, maxSteps: 1100, full: [X, Y, Z] })
    const tempAt = (px: number, py: number, pz: number) => {
      const i = Math.min(X - 1, Math.max(0, Math.floor(px))), j = Math.min(Y - 1, Math.max(0, Math.floor(py))), k = Math.min(Z - 1, Math.max(0, Math.floor(pz)))
      return T[(i * Y + j) * Z + k]
    }
    const c = new THREE.Color()
    {
      const n = sl.positions.length / 3
      const pos = new Float32Array(sl.positions.length), col = new Float32Array(sl.positions.length)
      for (let k = 0; k < n; k++) {
        const v = P(toMm(sl.positions[3 * k]), toMm(sl.positions[3 * k + 1]), toMm(sl.positions[3 * k + 2]))
        pos.set([v.x, v.y, v.z], 3 * k)
        thermal(tempAt(sl.positions[3 * k], sl.positions[3 * k + 1], sl.positions[3 * k + 2]) / T_MAX, c)
        col.set([c.r, c.g, c.b], 3 * k)
      }
      const lg = new THREE.BufferGeometry()
      lg.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      lg.setAttribute('color', new THREE.BufferAttribute(col, 3))
      const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false }))
      lines.name = 'lines'
      add(lines)
    }
    if (sl.paths.length) {
      const temp = new Float32Array(sl.polyline.length / 3)
      for (let k = 0; k < temp.length; k++) temp[k] = tempAt(sl.polyline[3 * k], sl.polyline[3 * k + 1], sl.polyline[3 * k + 2])
      const count = Math.min(1700, sl.paths.length * 8)
      const pg = new THREE.BufferGeometry()
      pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
      pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
      const pts = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.022, map: dot(), vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }))
      pts.name = 'flow'
      pts.frustumCulled = false
      add(pts)
      const phase = new Float32Array(count)
      for (let k = 0; k < count; k++) phase[k] = Math.random() * (sl.paths[k % sl.paths.length].count - 1)
      anim.current.parts = { pts, paths: sl.paths, poly: sl.polyline, speed: sl.polySpeed, temp, phase, P, toMm }
    }

    // heat slice through the GPU: colour and opacity both follow the air temperature
    const gb = res.boxes.find((b) => b.kind === 'gpu')
    const ySlice = gb ? (gb.b[2] + gb.b[3]) / 2 : W / 2
    const jy = Math.round(cell(ySlice))
    const tex = new Uint8Array(X * Z * 4)
    const sc = new THREE.Color()
    for (let i = 0; i < X; i++)
      for (let k = 0; k < Z; k++) {
        const t = T[(i * Y + jy) * Z + k] / T_MAX
        thermal(t, c)
        sc.copy(c).convertLinearToSRGB()
        const o = (k * X + (X - 1 - i)) * 4                         // texture u runs along +X = rear -> front
        tex[o] = Math.round(sc.r * 255); tex[o + 1] = Math.round(sc.g * 255); tex[o + 2] = Math.round(sc.b * 255)
        tex[o + 3] = Math.round(255 * Math.min(0.8, 0.12 + 0.9 * Math.max(0, t)))
      }
    const dt = new THREE.DataTexture(tex, X, Z, THREE.RGBAFormat)
    dt.magFilter = THREE.LinearFilter
    dt.colorSpace = THREE.SRGBColorSpace
    dt.needsUpdate = true
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(X * dx * S, Z * dx * S), new THREE.MeshBasicMaterial({ map: dt, transparent: true, side: THREE.DoubleSide, depthWrite: false }))
    plane.position.copy(P((X * dx) / 2 - offc * dx, ySlice, (Z * dx) / 2 - offc * dx))
    plane.name = 'heat'
    add(plane)

    if (res.dust?.points.length) {
      const dp = new Float32Array(res.dust.points.length * 3)
      res.dust.points.forEach((p, i) => { const v = P(p[0], p[1], p[2]); dp.set([v.x, v.y, v.z], 3 * i) })
      const dg = new THREE.BufferGeometry()
      dg.setAttribute('position', new THREE.BufferAttribute(dp, 3))
      const dust = new THREE.Points(dg, new THREE.PointsMaterial({ color: C.dust, size: 0.009, map: dot(), transparent: true, opacity: 0.85, depthWrite: false }))
      dust.name = 'dust'
      add(dust)
    }
  }, [res])

  // ---------- layers ----------
  useEffect(() => {
    const g = group.current
    if (!g) return
    for (const [name, on] of Object.entries(layers)) {
      const o = g.getObjectByName(name)
      if (o) o.visible = on
    }
    if (css.current) css.current.gl.visible = layers.panels
    for (const h of holos.current.values()) h.obj.visible = layers.panels
  }, [layers, res, idKey])

  return (
    <div className="scene" ref={host}>
      {panels.map((p) => createPortal(p.content, elFor(p.id), p.id))}
    </div>
  )
}
