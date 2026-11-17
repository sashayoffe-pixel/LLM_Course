#!/usr/bin/env node
// Stub registration test: loads section files in a sandbox with a stub Course and checks the schema.
// Usage: node tools/validate.js [sections/foo.js ...]   (default: all files in sections/)
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const TYPES = ['text', 'token', 'integer', 'float', 'probability', 'vector', 'matrix', 'tensor', 'mask', 'boolean', 'json', 'function'];
const PARTS = ['I', 'II', 'A'];
const root = path.join(__dirname, '..');
let files = process.argv.slice(2);
if (!files.length) files = fs.readdirSync(path.join(root, 'sections')).filter(f => f.endsWith('.js')).map(f => path.join(root, 'sections', f));
let problems = 0;
const all = [];
for (const f of files) {
  const registered = [];
  const sandbox = { window: {}, console, Math, JSON, Date };
  sandbox.window.Course = { register: s => registered.push(s), TYPES: Object.fromEntries(TYPES.map(t => [t, {}])), EX: {}, REF: {}, math: {} };
  sandbox.Course = sandbox.window.Course;
  sandbox.document = { createElement: () => ({}) };
  try { vm.runInNewContext(fs.readFileSync(f, 'utf8'), sandbox, { filename: f }); }
  catch (e) { console.log('✗ ' + f + ': load error: ' + e.message); problems++; continue; }
  if (!registered.length) { console.log('✗ ' + f + ': did not call Course.register'); problems++; }
  for (const s of registered) {
    const errs = [];
    if (!s.id) errs.push('no id');
    if (!PARTS.includes(s.part)) errs.push('part must be I | II | A');
    if (typeof s.order !== 'number') errs.push('order must be a number');
    if (!s.title) errs.push('no title');
    if (!Array.isArray(s.slides) || !s.slides.length) errs.push('no slides');
    const ids = new Set();
    (s.slides || []).forEach((sl, i) => {
      const w = m => errs.push('slide ' + (sl.id || i) + ': ' + m);
      if (!sl.id) w('no id'); if (ids.has(sl.id)) w('duplicate id'); ids.add(sl.id);
      if (!/^[a-z0-9-]+$/.test(sl.id || '')) w('id must be kebab-case');
      if (!sl.title) w('no title');
      if (!sl.critical || !Array.isArray(sl.critical.understand) || !sl.critical.understand.length) w('critical.understand missing');
      if (!sl.critical || !Array.isArray(sl.critical.details) || !sl.critical.details.length) w('critical.details missing');
      if (!Array.isArray(sl.dataTypes) || !sl.dataTypes.length) w('dataTypes missing');
      (sl.dataTypes || []).forEach(d => { if (!TYPES.includes(d.type)) w('unknown type "' + d.type + '" for ' + d.name); if (!d.name) w('dataType without name'); if (!d.shape) w('dataType "' + d.name + '" without shape'); });
      const n = (sl.levels || []).length;
      if (n < 2 || n > 4) w(n + ' levels (need 2–4)');
      (sl.levels || []).forEach((L, j) => { if (!L.name) w('level ' + j + ' has no name'); if (typeof L.render !== 'function') w('level ' + j + ' render is not a function'); });
    });
    const nLv = (s.slides || []).reduce((a, sl) => a + (sl.levels || []).length, 0);
    if (errs.length) { problems += errs.length; console.log('✗ ' + s.id + ' (' + path.basename(f) + ')\n   - ' + errs.join('\n   - ')); }
    else console.log('✓ ' + s.id + ' · part ' + s.part + ' · order ' + s.order + ' · ' + s.slides.length + ' slides · ' + nLv + ' levels');
    all.push(s);
  }
}
const seen = {};
all.forEach(s => { if (seen[s.id]) { console.log('✗ duplicate section id ' + s.id); problems++; } seen[s.id] = 1; });
console.log(problems ? '\n' + problems + ' problem(s)' : '\nAll good.');
process.exit(problems ? 1 : 0);
