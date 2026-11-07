// Usage: node scripts/shot.mjs <url> <out.png> [waitMs] [actionsJSON]
import { chromium } from 'playwright-core'
const [url, out, waitMs = '4000', actions = '[]'] = process.argv.slice(2)
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 })
const logs = []
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`) })
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))
await page.goto(url, { waitUntil: 'networkidle' })
await page.waitForTimeout(+waitMs)
for (const a of JSON.parse(actions)) {
  if (a.click) await page.getByText(a.click, { exact: a.exact ?? false }).first().click()
  if (a.css) await page.locator(a.css).first().click()
  if (a.wait) await page.waitForTimeout(a.wait)
  if (a.drag) { const [x0, y0, x1, y1] = a.drag; await page.mouse.move(x0, y0); await page.mouse.down(); await page.mouse.move(x1, y1, { steps: 10 }); await page.mouse.up() }
  if (a.eval) await page.evaluate(a.eval)
  if (a.fill) await page.locator(a.fill[0]).nth(a.fill[2] ?? 0).fill(String(a.fill[1]))
  if (a.upload) await page.locator(a.upload[0]).nth(a.upload[2] ?? 0).setInputFiles(a.upload[1])
}
await page.screenshot({ path: out })
console.log(logs.length ? logs.join('\n') : 'no console errors')
await browser.close()
