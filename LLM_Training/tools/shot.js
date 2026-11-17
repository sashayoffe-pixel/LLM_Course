#!/usr/bin/env node
// Screenshot helper: node tools/shot.js "<hash e.g. attention/worked-example/2>" out.png [width] [height] [scale]
'use strict';
const path = require('path'); const { chromium } = require('playwright');
(async () => {
  const [hash, out, w, hgt, sc] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: +w || 1440, height: +hgt || 900 }, deviceScaleFactor: +sc || 1 });
  await p.goto('file://' + path.join(__dirname, '..', 'index.html') + '#/' + (hash || ''));
  await p.waitForTimeout(700);
  await p.screenshot({ path: out || 'shot.png' });
  await b.close();
})();
