// Records a short guided tour of the running app (http://localhost:5190) as JPEG frames from the
// browser's screencast: <outDir>/frames/*.jpg plus <outDir>/frames.txt (ffmpeg concat list with real timing).
// Usage: node scripts/tour.mjs <outDir>      (encode with ffmpeg; see docs/make_video.py)
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright-core'

const outDir = path.resolve(process.argv[2] ?? '.')
const W = 1440, H = 810
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'] })
const ctx = await browser.newContext({ viewport: { width: W, height: H } })
await ctx.addInitScript(() => { try { localStorage.clear() } catch { /* ignore */ } })
const page = await ctx.newPage()
const wait = (ms) => page.waitForTimeout(ms)
const drag = async (x0, y0, x1, y1, ms) => {
  const n = Math.max(10, Math.round(ms / 30))
  await page.mouse.move(x0, y0)
  await page.mouse.down()
  for (let i = 1; i <= n; i++) {
    await page.mouse.move(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n)
    await wait(ms / n)
  }
  await page.mouse.up()
}
const dock = (t) => page.locator('.dock button', { hasText: t }).first().click()
const tab = (t) => page.locator('.views button', { hasText: t }).first().click()

await page.goto('http://localhost:5190/', { waitUntil: 'networkidle' })
await wait(4500)
const fdir = path.join(outDir, 'frames')
fs.rmSync(fdir, { recursive: true, force: true })
fs.mkdirSync(fdir, { recursive: true })
const frames = []
const cdp = await ctx.newCDPSession(page)
cdp.on('Page.screencastFrame', async ({ data, metadata, sessionId }) => {
  const name = `f${String(frames.length).padStart(5, '0')}.jpg`
  fs.writeFileSync(path.join(fdir, name), Buffer.from(data, 'base64'))
  frames.push({ name, t: metadata.timestamp })
  try { await cdp.send('Page.screencastFrameAck', { sessionId }) } catch { /* closing */ }
})
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 90, maxWidth: W, maxHeight: H, everyNthFrame: 1 })
const box = async (loc) => (await loc.boundingBox())
/** Drag a range input's thumb from its current value to a new one, slowly enough to watch. */
const slide = async (loc, lo, hi, from, to, ms) => {
  const r = await box(loc)
  const x = (v) => r.x + 8 + ((v - lo) / (hi - lo)) * (r.width - 16)
  await drag(x(from), r.y + r.height / 2, x(to), r.y + r.height / 2, ms)
}
await wait(1000)
await dock('Curves'); await wait(1200)                                                   // the case-fan curve, fans at 62 %
{
  const c = await box(page.locator('svg.curve').first())
  const X = (t) => c.x + 30 + ((t - 20) / 80) * (c.width - 40), Y = (d) => c.y + 10 + (1 - d / 100) * (c.height - 32)
  await drag(X(60), Y(40), X(60), Y(75), 1800); await wait(1600)                         // a steeper curve: fans speed up, CPU cools
}
await page.getByText('Reset this curve', { exact: true }).click(); await wait(900)
await dock('Curves')
await page.getByText('Stress', { exact: true }).click(); await wait(1800)            // a heavier load: fans spin up
await slide(page.locator('.sl input[type=range]').first(), 10, 40, 23, 35, 2400); await wait(1000)   // a hot room
await dock('Heat'); await wait(1800)
await dock('Trends'); await wait(1500)
{
  const t = await box(page.locator('.tr-plot svg').first())
  await page.mouse.move(t.x + 40, t.y + t.height / 2)
  for (let i = 0; i <= 24; i++) { await page.mouse.move(t.x + 40 + (i / 24) * (t.width - 110), t.y + t.height / 2); await wait(70) }
  await wait(1200)
}
await dock('Trends')
await tab('Worst'); await wait(2600)                                                    // the worst layout
await tab('Your layout'); await wait(1600)
await dock('Heat')
await drag(720, 430, 672, 426, 2600)                                                    // orbit a little
await page.mouse.move(720, 430)
for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, -60); await wait(120) }
await wait(1600)

// ---------- AI: the network answers at once, the simulation then checks it (its wait shown as a time-lapse) ----------
await dock('AI'); await wait(2400)                                                      // your layout: AI against physics
await page.locator('.ai-src button', { hasText: 'AI' }).click(); await wait(2600)       // the 3D view shows the network's airflow
await page.locator('.ai-src button', { hasText: 'Physics' }).click(); await wait(1400)
await page.locator('.tabs3 button', { hasText: 'layouts' }).click(); await wait(4200)   // all 6561 layouts, ranked for this scenario
{
  const t = await box(page.locator('.ai-all .tr-plot svg').first())
  for (let i = 0; i <= 16; i++) { await page.mouse.move(t.x + 40 + (i / 16) * (t.width - 110), t.y + t.height / 2); await wait(80) }
  await wait(800)
}
await page.locator('.tabs3 button', { hasText: 'This layout' }).click(); await wait(600)
await dock('Configure'); await wait(1000)
await page.locator('.mt', { hasText: 'top-mid' }).locator('.seg3 button', { hasText: 'off' }).click()   // a layout never simulated
await wait(3000)                                                                        // the network's answer, at once
const SPEED = 10
const lapse = [frames.length]
await page.evaluate((k) => {
  const d = document.createElement('div')
  d.id = 'lapse'
  d.textContent = `▶▶ ${k}× · the simulation checks the AI`
  Object.assign(d.style, { position: 'fixed', right: '24px', bottom: '84px', zIndex: 3000, padding: '6px 12px', font: '12px IBM Plex Mono, monospace',
    letterSpacing: '0.1em', color: '#c3b2ff', border: '1px solid rgba(195,178,255,.6)', background: 'rgba(7,12,19,.75)' })
  document.body.appendChild(d)
}, SPEED)
await page.locator('.toast', { hasText: 'SIMULATING' }).waitFor({ state: 'detached', timeout: 300000 })
lapse.push(frames.length)
await page.evaluate(() => document.getElementById('lapse')?.remove())
await wait(3500)                                                                        // physics in place, the difference in the window
await dock('Configure'); await wait(1200)

await cdp.send('Page.stopScreencast')
await wait(300)
const NL = String.fromCharCode(10)
const fast = (i) => (i >= lapse[0] && i < (lapse[1] ?? lapse[0]) ? SPEED : 1)
const lines = frames.map((f, i) => `file 'frames/${f.name}'${NL}duration ${Math.max(0.002, ((frames[i + 1]?.t ?? f.t + 0.04) - f.t) / fast(i)).toFixed(4)}`)
lines.push(`file 'frames/${frames[frames.length - 1].name}'`)
fs.writeFileSync(path.join(outDir, 'frames.txt'), lines.join(NL))
await browser.close()
const span = frames[frames.length - 1].t - frames[0].t
console.log(`frames ${frames.length} over ${span.toFixed(1)} s (${(frames.length / span).toFixed(1)} fps)`)
