#!/usr/bin/env node
// Real-browser walk of every slide and zoom level (Playwright + Chromium).
// Usage: node tools/browser-test.js [--section=id[,id]] [--width=1440] [--no-interact] [--verbose]
// Checks: render errors, console errors, NaN/undefined in visible text, missing critical/dataTypes,
// unknown types, unresolved cross-links, horizontal overflow. Interacts with steppers, sliders,
// selects, toggles and buttons. Exit code 1 if any issue was found.
'use strict';
const path = require('path');
const { chromium } = require('playwright');
const args = Object.fromEntries(process.argv.slice(2).map(a => { const m = a.match(/^--([^=]+)(=(.*))?$/); return m ? [m[1], m[3] == null ? true : m[3]] : [a, true]; }));
const only = args.section ? String(args.section).split(',') : null;
const width = Number(args.width) || 1440;
const url = 'file://' + path.join(__dirname, '..', 'index.html');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  const consoleErrs = [];
  page.on('console', m => { if (m.type() === 'error' && !/ERR_FILE_NOT_FOUND|Failed to load resource/.test(m.text())) consoleErrs.push(m.text()); });
  page.on('pageerror', e => consoleErrs.push('pageerror: ' + e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.Course && window.Course.slides && window.Course.slides.length > 0, null, { timeout: 10000 });
  await page.addStyleTag({ content: '.level, .level * { animation: none !important; transition: none !important; }' });
  const meta = await page.evaluate(() => ({
    warnings: Course.warnings.slice(), errors: Course.errors.slice(),
    slides: Course.slides.map(s => ({ key: s.key, sec: s.section.id, levels: s.slide.levels.length, title: s.slide.title }))
  }));
  const issues = [];
  const add = (where, msg) => issues.push(where + ': ' + msg);
  meta.warnings.forEach(w => { if (!only || only.some(o => w.indexOf('[' + o + ']') === 0)) add('register', w); });
  meta.errors.forEach(e => add('load', e));
  consoleErrs.splice(0).forEach(e => add('load console', e));
  let nLevels = 0, nInteractions = 0; const pendingLinks = new Set();
  if (args.strict) await page.evaluate(() => { window.__strict = true; });

  for (let i = 0; i < meta.slides.length; i++) {
    const s = meta.slides[i];
    if (only && !only.includes(s.sec)) continue;
    for (let L = 0; L < s.levels; L++) {
      const where = s.key + ' L' + L;
      nLevels++;
      await page.evaluate(([i, L]) => { Course.errors.length = 0; Course.show(i, L); }, [i, L]);
      await page.waitForTimeout(60);
      const check = async (phase) => {
        const r = await page.evaluate(() => {
          const content = document.querySelector('.slide-content'), panel = document.querySelector('.keypanel'), stage = document.querySelector('.stage');
          const txt = content.innerText + '\n' + panel.innerText;
          const bad = [];
          const re = /(\bNaN\b|\bundefined\b|\[object Object\])/g; let m;
          while ((m = re.exec(txt))) { bad.push(txt.slice(Math.max(0, m.index - 40), m.index + 30).replace(/\n/g, ' ')); if (bad.length > 3) break; }
          const links = [...document.querySelectorAll('.stage [data-goto], .stage [data-link]')].map(a => a.getAttribute('data-goto') || a.getAttribute('data-link'));
          const secIds = new Set(Course.sections.map(s => s.id));
          const unresolved = links.filter(t => !Course.resolve(t) && (window.__strict || secIds.has(String(t).replace(/^#?\/?/, '').split('/')[0])));
          const pending = links.filter(t => !Course.resolve(t) && unresolved.indexOf(t) < 0);
          const over = [];
          if (content.scrollWidth > content.clientWidth + 2) {
            const cr = content.getBoundingClientRect();
            content.querySelectorAll('*').forEach(el => {
              const r = el.getBoundingClientRect();
              if (r.right > cr.right + 2 && r.width > 0 && !el.closest('.mx-scroll,.tbl-wrap,pre,svg') && over.length < 3) over.push(el.tagName.toLowerCase() + '.' + String(el.className).split(' ')[0] + ' +' + Math.round(r.right - cr.right) + 'px');
            });
            if (!over.length) over.push('content.scrollWidth ' + content.scrollWidth + ' > ' + content.clientWidth);
          }
          if (stage.scrollWidth > stage.clientWidth + 2) over.push('stage overflow ' + stage.scrollWidth + ' > ' + stage.clientWidth);
          const renderErr = [...document.querySelectorAll('.render-error')].map(e => e.innerText.slice(0, 200));
          const unknown = document.querySelectorAll('.dt-unknown').length;
          return { bad, unresolved, pending, over, renderErr, unknown, errors: Course.errors.splice(0) };
        });
        r.bad.forEach(b => add(where + phase, 'suspicious text: "' + b + '"'));
        r.unresolved.forEach(u => add(where + phase, 'unresolved link: ' + u));
        r.pending.forEach(u => pendingLinks.add(u));
        r.over.forEach(o => add(where + phase, 'horizontal overflow: ' + o));
        r.renderErr.forEach(e => add(where + phase, 'render error: ' + e));
        if (r.unknown) add(where + phase, r.unknown + ' unknown type badge(s)');
        r.errors.forEach(e => add(where + phase, 'Course.errors: ' + e.split('\n').slice(0, 3).join(' | ')));
        consoleErrs.splice(0).forEach(e => add(where + phase, 'console: ' + e));
      };
      await check('');
      if (args['no-interact']) continue;
      const back = async () => page.evaluate(([i, L]) => { if (Course.state.idx !== i || Course.state.level !== L) Course.show(i, L); }, [i, L]);
      // steppers: click Next until the end
      const nStep = await page.locator('.slide-content .stepper').count();
      for (let k = 0; k < nStep; k++) {
        for (let guard = 0; guard < 40; guard++) {
          const btn = page.locator('.slide-content .stepper').nth(k).locator('.step-next');
          if (!(await btn.count()) || await btn.isDisabled()) break;
          await btn.click(); nInteractions++;
        }
      }
      if (nStep) await check(' [stepper]');
      // sliders: min, max, middle
      const nSl = await page.locator('.slide-content input[type=range]').count();
      for (let k = 0; k < nSl; k++) {
        await page.evaluate(k => {
          const el = document.querySelectorAll('.slide-content input[type=range]')[k]; if (!el) return;
          const set = v => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); };
          const a = +el.min, b = +el.max; set(a); set(b); set(a + (b - a) / 2);
        }, k); nInteractions += 3;
      }
      if (nSl) await check(' [sliders]');
      // selects: every option
      const nSe = await page.locator('.slide-content select').count();
      for (let k = 0; k < nSe; k++) {
        await page.evaluate(k => {
          const el = document.querySelectorAll('.slide-content select')[k]; if (!el) return;
          for (const o of [...el.options]) { el.value = o.value; el.dispatchEvent(new Event('change', { bubbles: true })); }
        }, k); nInteractions++;
      }
      if (nSe) await check(' [selects]');
      // toggles
      await page.evaluate(() => document.querySelectorAll('.slide-content input[type=checkbox]').forEach(c => { c.click(); c.click(); }));
      // other buttons (not stepper/nav); stop if one navigates away
      const nB = await page.locator('.slide-content .level button:not(.step-prev):not(.step-next):not(.step-play):not(.step-dot)').count();
      for (let k = 0; k < Math.min(nB, 25); k++) {
        const b = page.locator('.slide-content .level button:not(.step-prev):not(.step-next):not(.step-play):not(.step-dot)').nth(k);
        if (!(await b.count()) || !(await b.isVisible()) || await b.isDisabled()) continue;
        await b.click({ timeout: 2000 }).catch(() => {}); nInteractions++;
        await page.waitForTimeout(20);
        const moved = await page.evaluate(([i, L]) => Course.state.idx !== i || Course.state.level !== L, [i, L]);
        if (moved) { await check(' [after button→nav]'); await back(); }
      }
      // play button(s): let them run briefly
      if (await page.locator('.slide-content .step-play').count()) { await page.locator('.slide-content .step-play').first().click(); await page.waitForTimeout(200); }
      await page.waitForTimeout(80);
      await check(' [buttons]');
    }
  }
  await browser.close();
  const uniq = [...new Set(issues)];
  if (args.verbose || uniq.length) uniq.forEach(x => console.log('✗ ' + x));
  if (pendingLinks.size) console.log('(info) links to sections not yet loaded: ' + [...pendingLinks].join(', ') + '  — use --strict to count them');
  console.log('\nWalked ' + nLevels + ' zoom levels' + (only ? ' in ' + only.join(',') : '') + ' at ' + width + 'px, ' + nInteractions + ' interactions, ' + uniq.length + ' issue(s).');
  process.exit(uniq.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
