// Run the browser's scenario solver (src/scenario.ts) on cases read from stdin as JSON and print the results as
// JSON. tests/test_scenario.py uses it to check the TypeScript port against engine/scenario.py.
// Node 22.18+ runs the .ts file directly (type stripping).
import { setupFor, solve } from '../src/scenario.ts'

let input = ''
process.stdin.setEncoding('utf8')
for await (const chunk of process.stdin) input += chunk
const { cases, setups } = JSON.parse(input)
process.stdout.write(JSON.stringify({
  results: cases.map((c) => solve(c.bank, c.setup, c.ambient_c, c.cpu_w, c.gpu_w)),
  setups: setups.map((s) => setupFor(s.build, s.lib)),
}))
