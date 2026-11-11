import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export interface Viewer {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  controls: OrbitControls
  onFrame: (fn: (dt: number) => void) => () => void
  dispose: () => void
}

/** three.js viewer bound to a container: dark scene with distance fog, orbit controls, frame callbacks. */
export function createViewer(container: HTMLElement, bg: number): Viewer {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(bg)
  scene.fog = new THREE.Fog(bg, 6.5, 11)
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100)
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  renderer.localClippingEnabled = true          // the scan that draws the simulated twin (Scene.tsx)
  container.appendChild(renderer.domElement)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.1
  controls.minDistance = 1.6
  controls.maxDistance = 9

  scene.add(new THREE.HemisphereLight(0xf2f2f2, 0x121212, 0.55))
  const key = new THREE.DirectionalLight(0xffffff, 1.2)
  key.position.set(3, 5, 4)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xffffff, 0.35)
  rim.position.set(-3, 2.5, -4)
  scene.add(rim)

  const callbacks = new Set<(dt: number) => void>()
  let raf = 0
  let last = performance.now()
  const resize = () => {
    const w = container.clientWidth || 1
    const h = container.clientHeight || 1
    renderer.setSize(w, h, false)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    camera.aspect = w / h
    camera.updateProjectionMatrix()
  }
  const ro = new ResizeObserver(resize)
  ro.observe(container)
  resize()
  // render at full rate while the user interacts, at ~30 fps otherwise: the app should not heat the GPU it measures
  let lastInput = performance.now()
  const poke = () => { lastInput = performance.now() }
  const inputs = ['pointermove', 'pointerdown', 'wheel', 'keydown']
  for (const ev of inputs) container.addEventListener(ev, poke, { passive: true })
  const loop = () => {
    raf = requestAnimationFrame(loop)
    const now = performance.now()
    if (now - lastInput > 5000 && now - last < 31) return
    const dt = Math.min(0.1, (now - last) / 1000)
    last = now
    controls.update()
    callbacks.forEach((fn) => fn(dt))
    renderer.render(scene, camera)
  }
  loop()

  return {
    scene, camera, renderer, controls,
    onFrame(fn) {
      callbacks.add(fn)
      return () => callbacks.delete(fn)
    },
    dispose() {
      cancelAnimationFrame(raf)
      ro.disconnect()
      for (const ev of inputs) container.removeEventListener(ev, poke)
      controls.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    },
  }
}

/** Dispose geometries and materials (and their textures) of an object tree. */
export function disposeObject(obj: THREE.Object3D) {
  obj.traverse((o) => {
    const m = o as THREE.Mesh
    if (m.geometry) m.geometry.dispose()
    const mat = (m as unknown as { material?: THREE.Material | THREE.Material[] }).material
    for (const x of Array.isArray(mat) ? mat : mat ? [mat] : []) {
      const map = (x as THREE.MeshBasicMaterial).map
      if (map) map.dispose()
      x.dispose()
    }
  })
}
