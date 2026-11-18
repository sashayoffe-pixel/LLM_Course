#!/usr/bin/env node
// Checks Hebrew dictionaries: each value must keep the key's placeholders — the same set of <n>…</n> / <n/> tags
// and the same number of {#} — and must load without errors. Usage: node tools/i18n-check.js [i18n/he/x.js ...]
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const dir = path.join(__dirname, '..', 'i18n', 'he');
let files = process.argv.slice(2);
if (!files.length) files = fs.readdirSync(dir).filter(f => f.endsWith('.js')).map(f => path.join(dir, f));
const tags = s => (s.match(/<\/?\d+\/?>/g) || []).sort().join(' ');
const nums = s => (s.match(/\{#\}/g) || []).length;
let bad = 0, total = 0;
const seen = {};
for (const f of files) {
  const entries = [];
  const sandbox = { Course: { i18n: { add: (l, a, b) => { const m = typeof a === 'string' ? b : a; Object.keys(m).forEach(k => entries.push([k, m[k]])); } } } };
  try { vm.runInNewContext(fs.readFileSync(f, 'utf8'), sandbox, { filename: f }); }
  catch (e) { console.log('✗ ' + path.basename(f) + ': load error: ' + e.message); bad++; continue; }
  let fb = 0;
  for (const [k, v] of entries) {
    total++;
    if (typeof v !== 'string') { console.log('✗ ' + path.basename(f) + ': non-string value for ' + JSON.stringify(k)); fb++; continue; }
    if (tags(k) !== tags(v)) { console.log('✗ ' + path.basename(f) + ': tags differ\n    key: ' + k + '\n    val: ' + v); fb++; }
    else if (nums(k) !== nums(v)) { console.log('✗ ' + path.basename(f) + ': {#} count ' + nums(k) + ' vs ' + nums(v) + '\n    key: ' + k + '\n    val: ' + v); fb++; }
    if (seen[k] && seen[k][1] !== v && !/core\.js$/.test(f)) { /* duplicate across files: last loaded wins (fine) */ }
    seen[k] = [f, v];
  }
  bad += fb;
  console.log((fb ? '✗ ' : '✓ ') + path.basename(f) + ': ' + entries.length + ' entries' + (fb ? ', ' + fb + ' problem(s)' : ''));
}
console.log('\n' + total + ' entries, ' + bad + ' problem(s)');
process.exit(bad ? 1 : 0);
