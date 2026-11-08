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
await drag(720, 430, 672, 426, 3000)          // orbit a little: look along the case
await wait(700)
await dock('Lines'); await wait(2200); await dock('Lines')
await dock('Heat'); await dock('Flow'); await wait(2600)
await tab('Worst'); await wait(3000)          // the worst layout: rear fan blowing in
await tab('Your layout'); await wait(2200)
await dock('Heat'); await dock('Flow'); await wait(900)
await dock('Screens'); await wait(1600); await dock('Screens'); await wait(1300)   // hide / show the floating screens
await dock('Configure'); await wait(2600); await dock('Configure'); await wait(600)
await drag(672, 426, 790, 436, 3200)          // orbit to the other side
await page.mouse.move(720, 430)
for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, -60); await wait(120) }
await wait(1800)

await cdp.send('Page.stopScreencast')
await wait(300)
const NL = String.fromCharCode(10)
const lines = frames.map((f, i) => `file 'frames/${f.name}'${NL}duration ${Math.max(0.01, (frames[i + 1]?.t ?? f.t + 0.04) - f.t).toFixed(4)}`)
lines.push(`file 'frames/${frames[frames.length - 1].name}'`)
fs.writeFileSync(path.join(outDir, 'frames.txt'), lines.join(NL))
await browser.close()
const span = frames[frames.length - 1].t - frames[0].t
console.log(`frames ${frames.length} over ${span.toFixed(1)} s (${(frames.length / span).toFixed(1)} fps)`)
