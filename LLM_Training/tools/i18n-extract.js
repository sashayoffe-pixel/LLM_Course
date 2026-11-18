#!/usr/bin/env node
// Hebrew translation coverage: walks every slide × zoom level in Hebrew mode (same interactions as
// browser-test.js) and records every English key that has no translation yet.
// Usage: node tools/i18n-extract.js [--section=id[,id]] [--write] [--width=1440]
//   --write   writes i18n/todo/<section>.json  ({ "key": "" } for every missing key, grouped by section)
// Exit code 1 if anything is missing.
'use strict';
const fs = require('fs'), path = require('path');
const { chromium } = require('playwright');
const args = Object.fromEntries(process.argv.slice(2).map(a => { const m = a.match(/^--([^=]+)(=(.*))?$/); return m ? [m[1], m[3] == null ? true : m[3]] : [a, true]; }));
const only = args.section ? String(args.section).split(',') : null;
const root = path.join(__dirname, '..');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: Number(args.width) || 1440, height: 900 } });
  await page.addInitScript(() => { window.__i18nCollect = true; try { localStorage.setItem('llmtraining.lang', 'he'); } catch (e) {} });
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.goto('file://' + path.join(root, 'index.html'));
  await page.waitForFunction(() => window.Course && Course.slides && Course.slides.length > 0 && Course.i18n.lang === 'he');
  await page.addStyleTag({ content: '.level, .level * { animation: none !important; transition: none !important; }' });
  const slides = await page.evaluate(() => Course.slides.map(s => ({ key: s.key, sec: s.section.id, levels: s.slide.levels.length })));
  // TOC titles of every section (TOC is built once): visit the whole TOC
  await page.evaluate(() => document.querySelectorAll('.toc-sec').forEach(s => s.classList.add('open')));
  await page.waitForTimeout(50);
  const settle = () => page.waitForTimeout(40);
  for (let i = 0; i < slides.length; i++) {
    const s = slides[i];
    if (only && !only.includes(s.sec)) continue;
    for (let L = 0; L < s.levels; L++) {
      await page.evaluate(([i, L]) => Course.show(i, L), [i, L]); await settle();
      const nStep = await page.locator('.slide-content .stepper').count();
      for (let k = 0; k < nStep; k++) for (let g = 0; g < 40; g++) {
        const b = page.locator('.slide-content .stepper').nth(k).locator('.step-next');
        if (!(await b.count()) || await b.isDisabled()) break; await b.click();
      }
      await page.evaluate(() => {
        document.querySelectorAll('.slide-content input[type=range]').forEach(el => {
          const set = v => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); };
          const a = +el.min, b = +el.max, st = +el.step || 1, n = Math.round((b - a) / st);
          if (n <= 12) for (let j = 0; j <= n; j++) set(a + j * st); else { set(a); set(b); set(a + (b - a) / 2); set(+el.defaultValue || a); }
        });
        document.querySelectorAll('.slide-content select').forEach(el => { [...el.options].forEach(o => { el.value = o.value; el.dispatchEvent(new Event('change', { bubbles: true })); }); });
        document.querySelectorAll('.slide-content input[type=checkbox]').forEach(c => { c.click(); c.click(); });
      });
      await settle();
      const sel = '.slide-content .level button:not(.step-prev):not(.step-next):not(.step-play):not(.step-dot)';
      const nB = await page.locator(sel).count();
      for (let k = 0; k < Math.min(nB, 25); k++) {
        const b = page.locator(sel).nth(k);
        if (!(await b.count()) || !(await b.isVisible()) || await b.isDisabled()) continue;
        await b.click({ timeout: 2000 }).catch(() => {});
        await page.evaluate(([i, L]) => { if (Course.state.idx !== i || Course.state.level !== L) Course.show(i, L); }, [i, L]);
      }
      if (await page.locator('.slide-content .step-play').count()) { await page.locator('.slide-content .step-play').first().click(); await page.waitForTimeout(150); }
      await settle();
    }
  }
  await page.evaluate(() => { Course.openToy(); }); await settle(); await page.evaluate(() => Course.closeToy());
  const res = await page.evaluate(() => ({ missing: Course.i18n.missing, hits: Course.i18n.hits, misses: Course.i18n.misses, size: Object.keys(Course.i18n.dict.he).length, errors: Course.errors.slice() }));
  await browser.close();

  const bySec = {};
  Object.keys(res.missing).forEach(k => { const sec = res.missing[k].section || 'core'; (bySec[sec] = bySec[sec] || []).push(k); });
  const secs = Object.keys(bySec).sort();
  secs.forEach(sec => console.log((only && !only.includes(sec) && sec !== 'core' ? '  ' : '✗ ') + sec + ': ' + bySec[sec].length + ' untranslated'));
  if (args.write) {
    fs.mkdirSync(path.join(root, 'i18n', 'todo'), { recursive: true });
    secs.forEach(sec => {
      const p = path.join(root, 'i18n', 'todo', sec + '.json');
      let prev = {}; try { prev = JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) {}
      bySec[sec].forEach(k => { if (!(k in prev)) prev[k] = ''; });
      fs.writeFileSync(p, JSON.stringify(prev, null, 1));
    });
    console.log('wrote i18n/todo/*.json');
  }
  if (args.verbose) secs.forEach(sec => bySec[sec].forEach(k => console.log('  [' + sec + '] ' + k)));
  if (errs.length || res.errors.length) console.log('errors:', errs.concat(res.errors).slice(0, 10));
  const total = Object.keys(res.missing).length;
  console.log('\nDictionary: ' + res.size + ' entries · lookups hit ' + res.hits + ' / missed ' + res.misses + ' · ' + total + ' distinct untranslated keys' + (only ? ' (in ' + only.join(',') + ' + chrome)' : ''));
  process.exit(total ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
