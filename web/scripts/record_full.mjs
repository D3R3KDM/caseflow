// Records the full CaseFlow feature tour (about 4 min, 1920 x 1080, captioned) for the class presentation, in the
// style of Assignment 01's tour: headless Edge screencast -> ffmpeg -> MP4, with a caption bar, a cursor dot and a title
// card drawn on top for the recording only (the app itself is not changed).
// Needs the web app (localhost:5190) and the engine (127.0.0.1:8600) running. Nothing is simulated: every layout shown
// is cached; the GPU only draws the 3D view and runs the network (0.1 s per layout). From caseflow/web:
//   node scripts/record_full.mjs            -> ../docs/presentation/caseflow_full_demo.mp4
// The page is viewed like a 1600 x 900 browser window at 120 % (1920 x 1080 frames); the app gives the bottom 64 px to
// the caption bar.
import { execFileSync } from 'node:child_process'
import { mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(HERE, '../../docs/presentation/caseflow_full_demo.mp4')
const FR = join(tmpdir(), 'caseflow-rec-frames')
const FFDIR = resolve(HERE, '../../../.venv/Lib/site-packages/imageio_ffmpeg/binaries')
const FFMPEG = join(FFDIR, readdirSync(FFDIR).find((f) => f.startsWith('ffmpeg') && f.endsWith('.exe')))
const VW = 1600, VH = 900, DSF = 1.2, BAR = 64
const CHAPTERS = 13
const FRONT_HIGH_FAN = [933, 385]                 // where the upper front fan is in the default view (found by hovering)
const HELD_OUT = /F··\s+B·▸\s+T▸·◂\s+R▸/            // held-out test layout 00011021, as the AI window writes it
rmSync(FR, { recursive: true, force: true })
mkdirSync(FR, { recursive: true })
mkdirSync(dirname(OUT), { recursive: true })

// ---------------------------------------------------------------- page-side overlay (caption bar, cursor, title card)
function overlay(bar) {
  if (document.getElementById('rec-bar')) return
  const st = document.createElement('style')
  st.textContent = `
    .scene, .vignette { bottom: ${bar}px !important; }
    .dock { bottom: ${bar + 18}px !important; }
    #rec-bar { position: fixed; left: 0; right: 0; bottom: 0; height: ${bar}px; z-index: 99998; pointer-events: none; box-sizing: border-box;
      background: #131517; color: #EEF0EB; display: flex; align-items: center; gap: 18px; padding: 0 28px; border-top: 3px solid #6E3CE6;
      font-family: 'IBM Plex Sans', Arial, sans-serif; transition: opacity .3s; }
    #rec-bar .n { font-family: 'IBM Plex Mono', monospace; font-size: 14px; letter-spacing: .14em; color: #A98BFF; white-space: nowrap; }
    #rec-bar .t { font-size: 18px; font-weight: 600; white-space: nowrap; }
    #rec-bar .d { font-size: 17px; color: #C9CEC4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #rec-cur { position: fixed; left: 800px; top: 450px; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
      border: 2px solid #131517; background: rgba(255,255,255,.55); box-shadow: 0 0 0 2px rgba(255,255,255,.85); z-index: 99999;
      pointer-events: none; transition: left .45s ease, top .45s ease, transform .15s, background .15s; }
    #rec-cur.fast { transition: transform .15s, background .15s; }
    #rec-cur.down { transform: scale(.7); background: rgba(110,60,230,.75); }
    #rec-title { position: fixed; inset: 0; z-index: 99997; pointer-events: none; background: #131517; color: #EEF0EB; display: flex;
      flex-direction: column; justify-content: center; padding: 0 140px; gap: 16px; font-family: 'IBM Plex Sans', Arial, sans-serif; transition: opacity .7s; }
    #rec-title .k { font-family: 'IBM Plex Mono', monospace; font-size: 16px; letter-spacing: .18em; color: #A98BFF; }
    #rec-title .h { font-size: 84px; font-weight: 600; letter-spacing: -2px; line-height: 1; }
    #rec-title .s { font-size: 28px; color: #C9CEC4; }
    #rec-title .m { font-family: 'IBM Plex Mono', monospace; font-size: 17px; color: #8A8F94; }`
  document.head.appendChild(st)
  const b = document.createElement('div')
  b.id = 'rec-bar'
  b.innerHTML = '<span class="n"></span><span class="t"></span><span class="d"></span>'
  document.body.appendChild(b)
  const cur = document.createElement('div')
  cur.id = 'rec-cur'
  document.body.appendChild(cur)
  window.__cap = (n, t, d) => {
    b.querySelector('.n').textContent = n
    b.querySelector('.t').textContent = t
    b.querySelector('.d').innerHTML = d
  }
  window.__cur = (x, y, fast) => { cur.classList.toggle('fast', !!fast); cur.style.left = x + 'px'; cur.style.top = y + 'px' }
  window.__down = (on) => cur.classList.toggle('down', on)
  window.__title = (show, k, h, s, m) => {
    let el = document.getElementById('rec-title')
    if (show) {
      if (!el) { el = document.createElement('div'); el.id = 'rec-title'; document.body.appendChild(el) }
      el.innerHTML = `<div class="k">${k}</div><div class="h">${h}</div><div class="s">${s}</div><div class="m">${m}</div>`
      el.style.opacity = '1'
    } else if (el) el.style.opacity = '0'
  }
}

// ---------------------------------------------------------------- browser and helpers
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'] })
const ctx = await browser.newContext({ viewport: { width: VW, height: VH }, deviceScaleFactor: DSF })
await ctx.addInitScript(() => { try { localStorage.clear() } catch { /* ignore */ } })
const page = await ctx.newPage()
page.on('pageerror', (e) => console.log('pageerror:', e.message))
const sleep = (ms) => page.waitForTimeout(ms)
const frames = []
let chapter = 0
const capRaw = (n, t, d) => page.evaluate(([n, t, d]) => window.__cap(n, t, d), [n, t, d])
const cap = (t, d) => capRaw(`${String(chapter).padStart(2, '0')} / ${CHAPTERS}`, t, d)
const next = (t, d) => { chapter++; return cap(t, d) }
async function moveTo(x, y) {
  await page.evaluate(([x, y]) => window.__cur(x, y), [x, y])
  await page.mouse.move(x, y, { steps: 12 })
  await sleep(480)
}
async function clickXY(x, y) {
  await moveTo(x, y)
  await page.evaluate(() => window.__down(true))
  await page.mouse.down()
  await sleep(90)
  await page.mouse.up()
  await page.evaluate(() => window.__down(false))
  await sleep(250)
}
async function click(loc) {
  const b = await loc.first().boundingBox()
  if (!b) throw new Error('not on screen: ' + loc)
  await clickXY(b.x + b.width / 2, b.y + b.height / 2)
}
const dock = (t) => click(page.locator('.dock button', { hasText: t }))
const tab = (t) => click(page.locator('.views button', { hasText: t }))
const header = (t) => click(page.locator('.status button', { hasText: t }))
const preset = (t) => click(page.locator('.presets button', { hasText: t }))
async function drag(x1, y1, x2, y2, n = 30, ms = 30) {
  await moveTo(x1, y1)
  await page.evaluate(() => window.__down(true))
  await page.mouse.down()
  for (let i = 1; i <= n; i++) {
    const x = x1 + ((x2 - x1) * i) / n, y = y1 + ((y2 - y1) * i) / n
    await page.mouse.move(x, y)
    await page.evaluate(([x, y]) => window.__cur(x, y, true), [x, y])
    await sleep(ms)
  }
  await page.mouse.up()
  await page.evaluate(() => window.__down(false))
}
/** Walk a range input through values, the cursor riding its thumb. */
async function slide(loc, values, ms = 120) {
  const el = loc.first()
  const b = await el.boundingBox()
  const [min, max, now] = await el.evaluate((e) => [+e.min, +e.max, +e.value])
  const xOf = (v) => b.x + 8 + ((v - min) / (max - min)) * (b.width - 16)
  const y = b.y + b.height / 2
  await moveTo(xOf(now), y)
  await page.evaluate(() => window.__down(true))
  for (const v of values) {
    await el.evaluate((e, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, String(v))
      e.dispatchEvent(new Event('input', { bubbles: true }))
      e.dispatchEvent(new Event('change', { bubbles: true }))
    }, v)
    await page.evaluate(([x, y]) => window.__cur(x, y, true), [xOf(v), y])
    await sleep(ms)
  }
  await page.evaluate(() => window.__down(false))
}
const room = () => page.locator('.sl input[type=range]').first()
const range = (a, b) => Array.from({ length: Math.abs(b - a) + 1 }, (_, i) => a + Math.sign(b - a) * i)

// ---------------------------------------------------------------- the tour
try {
  await page.goto('http://localhost:5190/', { waitUntil: 'networkidle' })
  await page.evaluate(overlay, BAR)
  await page.evaluate(() => window.__title(true, 'CSE 3201 · Assignment 02', 'CaseFlow', 'Full feature tour',
    'Lian Li LANCOOL 207 · RTX 5080 FE · Ryzen 7 9700X · a 3D U-Net trained on 150 simulated layouts'))
  await capRaw('', 'CaseFlow', 'Airflow, heat and dust inside my own PC case')
  for (let i = 0; i < 60; i++) { await sleep(250); if ((await page.locator('.status').innerText()).includes('AI PREDICTION')) break }
  // before recording: the engine tracks the dust in the network's air for the held-out layout once (~8 s), so its card
  // has numbers when chapter 9 gets there
  await page.evaluate(async () => {
    const [space, base] = await Promise.all([fetch('/data/ai_space.json').then((r) => r.json()), fetch('/api/default_build').then((r) => r.json())])
    const code = '00011021', modes = ['off', 'in', 'out']
    const fans = space.mounts.flatMap((slot, i) => code[i] === '0' ? [] : [{ slot, fan: space.mount_fan[slot], mode: modes[+code[i]], control: 'curve', pct: 70 }])
    await fetch('/api/ai/dust', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ build: { ...base, fans } }) })
  })
  await sleep(2500)
  const cdp = await ctx.newCDPSession(page)
  cdp.on('Page.screencastFrame', async (f) => {
    const file = join(FR, `f${String(frames.length).padStart(5, '0')}.jpg`)
    writeFileSync(file, Buffer.from(f.data, 'base64'))
    frames.push({ file, t: f.metadata.timestamp })
    await cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId }).catch(() => {})
  })
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 85, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 })
  await sleep(3200)
  await page.evaluate(() => window.__title(false))
  await sleep(1000)

  // 01 the AI's answer in 3D
  await next("The AI's answer, in 3D", 'A 3D U-Net predicts the airflow and heat of this layout in 0.1 s; each screen is wired to its part')
  await moveTo(1000, 560)
  await sleep(2600)
  await cap("The AI's answer, in 3D", 'Drag to orbit, scroll to zoom, ⟲ puts the view back')
  await drag(1100, 640, 900, 625, 36, 35)
  await sleep(900)
  await page.mouse.wheel(0, -320)
  await sleep(1600)
  await dock('⟲')
  await sleep(2200)

  // 02 scenario
  await next('Scenario: room and load', 'Presets from idle to full stress, or the sliders: the fans follow their curves, the parts settle in milliseconds')
  for (const [p, ms] of [['Idle', 2000], ['GPU game', 2000], ['Stress', 2200], ['CPU game', 1400]]) { await preset(p); await sleep(ms) }
  await cap('Scenario: room and load', 'A warmer room: the fans speed up, so the CPU warms by less than the room does')
  await slide(room(), range(23, 35), 110)
  await sleep(2200)
  await slide(room(), range(35, 23), 70)
  await sleep(1000)

  // 03 fan curves
  await next('Fan curves', 'Like ASUS Fan Xpert: drag a point and every case fan speeds up; the ring shows where the fan runs now')
  await dock('Curves')
  await sleep(1200)
  const pt = await page.locator('.curve circle.cv-pt').nth(1).boundingBox()
  await drag(pt.x + pt.width / 2, pt.y + pt.height / 2, pt.x + pt.width / 2, pt.y + pt.height / 2 - 58, 30, 40)
  await sleep(2800)
  await click(page.locator('.textbtn', { hasText: 'Reset this curve' }))
  await sleep(1400)
  await dock('Curves')
  await sleep(500)

  // 04 trends
  await next('Trends', 'Across room temperature and load. Solid: fans on their curves; dashed: fans held at their speed now')
  await dock('Trends')
  await sleep(2600)
  const ch = await page.locator('.hud svg[role=img]').first().boundingBox()
  if (ch) { await moveTo(ch.x + ch.width * 0.35, ch.y + ch.height * 0.5); await sleep(700); await moveTo(ch.x + ch.width * 0.75, ch.y + ch.height * 0.5); await sleep(1200) }
  await click(page.locator('.tr-sw button', { hasText: 'cpu load' }))
  await sleep(2200)
  await dock('Trends')
  await sleep(500)

  // 05 layers
  await next('Layers', 'Streamlines, a heat slice through the GPU, and air particles colored by how much warmer than the room they are')
  await dock('Lines')
  await sleep(2200)
  await dock('Heat')
  await sleep(2600)
  await dock('Flow')
  await sleep(1800)
  await dock('Flow')
  await dock('Lines')
  await dock('Heat')
  await sleep(900)

  // 06 change the layout
  await next('Change the layout', 'Click a fan to flip it or switch it off, or an empty mount to add one')
  await clickXY(...FRONT_HIGH_FAN)
  await sleep(1300)
  await click(page.locator('.fanmenu button', { hasText: 'Turn off' }))
  await sleep(1800)
  await cap('Change the layout', 'Never simulated: the header offers SIMULATION · 90 s, but the network has answered already')
  await moveTo(1450, 40)
  await sleep(3000)
  await tab('Your layout')
  await sleep(1600)

  // 07 the AI against the simulation
  await next('AI against simulation', 'SIMULATION puts the simulated case beside the AI’s (lattice Boltzmann on the GPU; cached here, so it is quick)')
  await header('SIMULATION')
  await sleep(5200)
  await cap('AI against simulation', 'Trained on this layout, so the two agree to 0.0 °C: that is memory, not a test')
  await moveTo(1300, 300)
  await sleep(3800)
  await header('BACK')
  await sleep(1700)

  // 08 a layout the network never saw
  await next('A layout it never saw', 'The 16 held-out test layouts were kept out of training; the simulation has them')
  await dock(/^AI$/)
  await sleep(900)
  await click(page.locator('.tabs3 button', { hasText: /^All/ }))
  await page.locator('.ai-list .ai-rk').first().waitFor({ timeout: 60000 })
  await sleep(1300)
  await click(page.locator('.ai-filter button', { hasText: 'held-out' }))
  await sleep(1600)
  const row = page.locator('.ai-list .ai-rk', { hasText: HELD_OUT }).first()
  await row.evaluate((e) => e.scrollIntoView({ block: 'center', behavior: 'smooth' }))      // the list scrolls; the page does not
  await sleep(900)
  await click(row)
  await sleep(2200)
  await dock(/^AI$/)
  await sleep(500)
  await header('SIMULATION')
  await sleep(7800)
  await cap('A layout it never saw', 'Held-out layout 00011021: AI 67.1 °C, simulation 68.0 °C. That is the network’s real error')
  await moveTo(1300, 330)
  await sleep(3200)
  await cap('A layout it never saw', 'Full stress in a 35 °C room: the gap grows to 2.5 °C, the largest on any held-out layout')
  await preset('Stress')
  await sleep(900)
  await slide(room(), range(23, 35), 110)
  await sleep(3600)
  await preset('CPU game')
  await slide(room(), range(35, 23), 60)
  await sleep(1200)

  // 09 dust
  await next('Dust', 'Dust comes in through the openings, drifts with the air and piles up where it touches (time-lapse)')
  await dock('Dust')
  await sleep(7000)
  await cap('Dust', 'Left in the network’s air, right in the simulation’s: the same particles and physics, run live')
  await sleep(7500)
  await cap('Dust', 'The card compares the two: how much of the dust that gets in settles, and on which surface most of it lands')
  await sleep(7000)
  await header('BACK')
  await sleep(1200)
  await dock('Dust')
  await tab('Your layout')
  await sleep(1400)

  // 10 the layout search
  await next('The layout search', 'All 32 ways to place my three spare fans, ranked for the scenario on screen, and the verdict on mine')
  await dock('Ranking')
  await sleep(3400)
  await tab('Recommended')
  await sleep(2400)
  await cap('The layout search', 'Recommended: move the rear fan to the empty top-front mount, still blowing out')
  await sleep(3000)
  await tab('Your layout')
  await sleep(900)
  await dock('Ranking')
  await sleep(500)

  // 11 all 6561 layouts
  await next('All 6561 layouts', 'The network predicts every way to fill the 8 mounts in 4.5 minutes; simulating them all would take a week')
  await dock(/^AI$/)
  await sleep(900)
  await click(page.locator('.ai-filter button', { hasText: 'any' }))
  await sleep(2200)
  await click(page.locator('.ai-filter button', { hasText: /^4$/ }))
  await sleep(2000)
  await cap('All 6561 layouts', 'Best layout for each fan count: four well-placed fans already reach the best')
  const tc = await page.locator('.ai-all svg[role=img]').first().boundingBox()
  if (tc) { await moveTo(tc.x + tc.width * 0.3, tc.y + tc.height * 0.55); await sleep(900); await moveTo(tc.x + tc.width * 0.55, tc.y + tc.height * 0.55); await sleep(1500) }
  await click(page.locator('.ai-filter button', { hasText: 'any' }))
  await dock(/^AI$/)
  await sleep(600)

  // 12 configure
  await next('Configure', 'Parts and every mount: in / off / out, fan model, curve or fixed. Other parts fall back to the simulation')
  await dock('Configure')
  await sleep(4200)
  await dock('Configure')
  await sleep(500)

  // 13 method
  await next('Method', 'How it is computed; the real PC measured CPU 65.3 / GPU 49.3 °C, the model gives 65.6 / 49.7 °C')
  await dock('Method')
  await sleep(4800)
  await dock('Method')
  await sleep(600)

  // end
  await capRaw('', 'CaseFlow', 'CSE 3201 · Assignment 02 · Derek')
  await moveTo(800, 300)
  await sleep(2800)
  await cdp.send('Page.stopScreencast')
  await sleep(300)
} finally {
  await browser.close()
}

// frame durations from the screencast timestamps -> constant 30 fps MP4 (as in Assignment 01)
const lines = []
frames.forEach((f, i) => {
  const d = i + 1 < frames.length ? Math.max(0.001, frames[i + 1].t - f.t) : 1.0
  lines.push(`file '${f.file.replace(/\\/g, '/')}'`, `duration ${d.toFixed(4)}`)
})
lines.push(`file '${frames[frames.length - 1].file.replace(/\\/g, '/')}'`)
writeFileSync(join(FR, 'list.txt'), lines.join('\n'))
execFileSync(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', join(FR, 'list.txt'),
  '-vf', 'fps=30,scale=1920:1080:flags=lanczos:out_range=tv,format=yuv420p', '-color_range', 'tv', '-colorspace', 'bt709',
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-movflags', '+faststart', OUT])
rmSync(FR, { recursive: true, force: true })
console.log(`${frames.length} frames, ${(frames.at(-1).t - frames[0].t).toFixed(1)} s -> ${OUT}`)
