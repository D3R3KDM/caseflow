// An "editor" picture of the real codebase for the presentation (as in Assignment 01): the file tree with line counts,
// and two excerpts with their real line numbers - the lattice-Boltzmann step and the 3D U-Net.
//   node scripts/codeview.mjs        -> ../docs/presentation/img/codeview.png
import { mkdirSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '../..')
const OUT = join(ROOT, 'docs/presentation/img/codeview.png')
mkdirSync(dirname(OUT), { recursive: true })
const lines = (f) => readFileSync(join(ROOT, f), 'utf8').split(/\r?\n/)
const count = (dir, exts) => {
  const fs = readdirSync(join(ROOT, dir)).filter((f) => exts.some((e) => f.endsWith(e)))
  return [fs.length, fs.reduce((s, f) => s + lines(join(dir, f)).length - 1, 0)]
}
const KW = new Set('def class return import from if else elif for in not and or None True False with as lambda self super'.split(' '))
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
function hl(line) {
  let out = '', i = 0
  while (i < line.length) {
    const rest = line.slice(i)
    let m
    if (rest.startsWith('"""')) { const j = line.indexOf('"""', i + 3); const e = j < 0 ? line.length : j + 3; out += `<span class="s">${esc(line.slice(i, e))}</span>`; i = e; continue }
    if (rest[0] === '"') { const j = line.indexOf('"', i + 1); const e = j < 0 ? line.length : j + 1; out += `<span class="s">${esc(line.slice(i, e))}</span>`; i = e; continue }
    if (rest[0] === '#') { out += `<span class="c">${esc(rest)}</span>`; break }
    if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))) {
      const w = m[0]
      out += KW.has(w) ? `<span class="k">${w}</span>` : /^[A-Z]/.test(w) ? `<span class="t">${w}</span>`
        : line.slice(i + w.length).startsWith('(') ? `<span class="f">${w}</span>` : w
      i += w.length
      continue
    }
    if ((m = rest.match(/^\d+(\.\d+)?/))) { out += `<span class="n">${m[0]}</span>`; i += m[0].length; continue }
    out += esc(rest[0]); i++
  }
  return out
}
function block(file, ranges) {
  const L = lines(file)
  let h = ''
  ranges.forEach(([a, b], ri) => {
    if (ri) h += '<div class="fold">⋯</div>'
    for (let n = a; n <= b; n++) h += `<div class="ln"><i>${n}</i><code>${hl(L[n - 1]) || ' '}</code></div>`
  })
  return h
}
const n = (x) => x.toLocaleString('en-US')
const parts = [
  ['engine/', 'PyTorch on the GPU: LBM airflow, heat, dust, banks, the 3D U-Net', count('engine', ['.py']), 1],
  ['server/', 'FastAPI: simulations as jobs, /api/ai, /api/ai/dust', count('server', ['.py']), 1],
  ['scripts/', 'training data, training, design space, checks', count('scripts', ['.py']), 1],
  ['tests/', 'no GPU; the browser’s solver and dust against Python', count('tests', ['.py']), 1],
  ['web/src/', 'React · TypeScript · three.js', count('web/src', ['.ts', '.tsx', '.css']), 1],
  ['web/scripts/', 'node checks, screenshots, this recording', count('web/scripts', ['.mjs']), 1],
  ['docs/', 'guide, figures, video', count('docs', ['.py']), 1],
]
const total = parts.reduce((s, p) => s + p[2][1], 0)
const tree = [['caseflow/', '', '', 0], ...parts.map(([nm, d, [f, l], lv]) => [nm, d, `${f} files · ${n(l)}`, lv])]
const treeHtml = tree.map(([nm, d, c, lv]) => `<div class="tr" style="padding-left:${lv * 22}px"><b>${nm}</b><span class="d">${d}</span><span class="cnt">${c}</span></div>`).join('')
const font = (w) => `<link href="file:///${join(ROOT, 'web/node_modules/@fontsource/ibm-plex-mono', `${w}.css`).replace(/\\/g, '/')}" rel="stylesheet">`
const html = `<!doctype html><html><head><meta charset="utf-8">${font(400)}${font(500)}${font(600)}
<style>
body{margin:0;background:#eef0eb;font:14px/1.5 "IBM Plex Mono",monospace;color:#131517}
.win{width:1600px;height:900px;display:grid;grid-template-columns:500px 1fr;background:#fafbf8;border:1px solid #d2d6ce;box-sizing:border-box}
.side{border-right:1px solid #d2d6ce;padding:22px 20px}
.side h4,.pane h4{margin:0 0 14px;font:500 12px/1 "IBM Plex Mono";letter-spacing:.12em;text-transform:uppercase;color:#6f747a}
.tr{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;padding:7px 0;border-bottom:1px solid #e3e6df}
.tr b{font-weight:600}.tr .d{grid-column:1/-1;color:#6f747a;font-size:12.5px}.tr .cnt{grid-column:2;grid-row:1;justify-self:end;font-size:12.5px}
.tot{margin-top:18px;font-size:13px;color:#464b50;line-height:1.7}.tot b{color:#131517}
.main{display:grid;grid-template-rows:auto 1fr;min-width:0}
.pane{padding:16px 22px 8px;border-bottom:1px solid #d2d6ce;min-width:0;overflow:hidden}
.tab{display:inline-block;border:1px solid #131517;padding:3px 9px;font-size:12.5px;margin-bottom:8px}
.ln{display:grid;grid-template-columns:44px 1fr;font-size:13.2px;line-height:1.58;white-space:pre}
code{font-family:"IBM Plex Mono",monospace}.ln i{font-style:normal;color:#a3a8ad;text-align:right;padding-right:14px}
.fold{color:#a3a8ad;padding-left:58px;line-height:1.2}
.k{color:#6e3ce6;font-weight:500}.s{color:#0a8f84}.c{color:#8a8f94;font-style:italic}.n{color:#c98300}.t{color:#3150d8}.f{color:#131517;font-weight:500}
</style></head><body><div class="win">
<div class="side"><h4>Repository</h4>${treeHtml}
<div class="tot">≈ <b>${n(Math.round(total / 100) * 100)}</b> lines of Python + TypeScript<br><b>25</b> tests · <b>44</b> commits · Nov 7–11</div></div>
<div class="main">
<div class="pane"><span class="tab">engine/lbm.py · lattice-Boltzmann step</span>${block('engine/lbm.py', [[182, 202]])}</div>
<div class="pane"><span class="tab">engine/surrogate.py · 3D U-Net</span>${block('engine/surrogate.py', [[123, 128], [136, 146]])}</div>
</div></div></body></html>`
const browser = await chromium.launch({ channel: 'msedge', headless: true })
const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1.25 })
await page.setContent(html, { waitUntil: 'load' })
await page.evaluate(() => document.fonts.ready)
await page.screenshot({ path: OUT })
await browser.close()
console.log('wrote', OUT, '| total lines', total)
