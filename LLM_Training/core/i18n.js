/* LLM Training — translation layer (English source → Hebrew).
 *
 * Content is written once, in English. In Hebrew mode this file translates the rendered page:
 *  - every "text block" (an element whose children are only text and inline tags) is turned into a KEY:
 *      inline child elements become numbered tags  <0>…</0>  (or <0/> for badges, tokens, empty elements)
 *      stand-alone numbers become {#}               (so slider-driven values reuse one translation)
 *    e.g.  'Step <0>{#}</0> of the <1>worked example</1>'
 *  - the key is looked up in the dictionary (i18n/he/*.js); the Hebrew value keeps the same tags and {#};
 *    tags may move (Hebrew word order) — the original elements, listeners and links are reused.
 *  - text that is not inside a text block (e.g. SVG labels) is translated node by node; title/placeholder too.
 * A MutationObserver re-translates whatever interactive code changes later.
 * Dictionaries:  Course.i18n.add('he', { 'English key': 'Hebrew value', … })
 */
(function () {
  'use strict';
  var C = window.Course;
  if (!C) return;
  var STORE = 'llmtraining.lang';
  var dict = { he: {} };
  var lang = 'en';
  try { lang = localStorage.getItem(STORE) || 'en'; } catch (e) { /* ignore */ }
  if (/[?&#]lang=he\b/.test(location.search + location.hash)) lang = 'he';
  if (/[?&#]lang=en\b/.test(location.search + location.hash)) lang = 'en';

  var INLINE = { B: 1, STRONG: 1, I: 1, EM: 1, CODE: 1, A: 1, SPAN: 1, SUB: 1, SUP: 1, SMALL: 1, KBD: 1, BR: 1, U: 1, MARK: 1, S: 1, ABBR: 1, Q: 1, VAR: 1, TSPAN: 1 };
  // never translated (left exactly as rendered)
  var SKIP = 'pre, script, style, textarea, input, .tokens, .mx-scroll, .no-i18n, [data-no-i18n], .counter, .ctl-val, .bar-val';
  // kept as opaque units inside a sentence (they are values, not words)
  var OPAQUE = '.tok, .dt, .shape, kbd, code, .no-i18n, [data-no-i18n], .math, .ctl-val';
  var NUM = /(?<![\w.\-−])[−-]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])/g;
  var HEB = /[֐-׿]/;
  var LETTERS = /[A-Za-z]{2}/;

  var I = C.i18n = {
    get lang() { return lang; },
    dict: dict,
    collect: !!window.__i18nCollect, // tools/i18n-extract.js turns this on to list untranslated keys
    missing: {},             // key -> { section, sample }
    hits: 0, misses: 0,
    add: function (l, a, b) { var map = typeof a === 'string' ? b : a; dict[l] = dict[l] || {}; Object.keys(map || {}).forEach(function (k) { dict[l][norm(k)] = map[k]; }); },
    set: function (l) {
      lang = l === 'he' ? 'he' : 'en';
      try { localStorage.setItem(STORE, lang); } catch (e) { /* ignore */ }
      applyDocDir();
      if (C.rebuild) C.rebuild();
      observe();
    },
    toggle: function () { I.set(lang === 'he' ? 'en' : 'he'); },
    keyOf: function (el) { var parts = []; return { key: numKey(skel(el, parts)).key, parts: parts }; },
    translateTree: function (root) { run(function () { visit(root); }); },
    t: function (s) { return tText(String(s), null).out; }
  };
  C.t = function (s) { return lang === 'he' ? I.t(s) : s; };

  function norm(s) { return String(s).replace(/\s+/g, ' ').trim(); }
  function isSkip(el) { return el.nodeType === 1 && el.matches && el.matches(SKIP); }
  function isOpaque(el) { return el.matches && el.matches(OPAQUE); }

  // Is every descendant text or an inline tag? (opaque units count as inline and are not inspected)
  function inlineLeaf(el) {
    var ch = el.childNodes;
    for (var i = 0; i < ch.length; i++) {
      var n = ch[i];
      if (n.nodeType === 3 || n.nodeType === 8) continue;
      if (n.nodeType !== 1) return false;
      if (isOpaque(n)) continue;
      if (!INLINE[n.tagName.toUpperCase()] || isSkip(n)) return false;
      if (!inlineLeaf(n)) return false;
    }
    return true;
  }
  function ownText(el) {
    var s = '';
    for (var n = el.firstChild; n; n = n.nextSibling) if (n.nodeType === 3) s += n.nodeValue;
    return s;
  }
  // letters that need translating (outside opaque units)
  function wordsOf(el) {
    var s = '';
    (function walk(e) {
      for (var n = e.firstChild; n; n = n.nextSibling) {
        if (n.nodeType === 3) s += n.nodeValue;
        else if (n.nodeType === 1 && !isOpaque(n) && !isSkip(n)) walk(n);
      }
    })(el);
    return s;
  }
  function skel(el, parts) {
    var s = '';
    for (var n = el.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3) s += n.nodeValue.replace(/</g, '‹');
      else if (n.nodeType === 1) {
        var i = parts.length; parts.push(n);
        if (isOpaque(n) || !n.firstChild) s += '<' + i + '/>';
        else s += '<' + i + '>' + skel(n, parts) + '</' + i + '>';
      }
    }
    return s;
  }
  function numKey(s) {
    var nums = [];
    var key = s.split(/(<\/?\d+\/?>)/).map(function (seg) {
      if (/^<\/?\d+\/?>$/.test(seg)) return seg;
      return seg.replace(NUM, function (m) { nums.push(m); return '{#}'; });
    }).join('');
    return { key: norm(key), nums: nums };
  }
  function fill(val, nums) { var i = 0; return val.replace(/\{#\}/g, function () { return i < nums.length ? nums[i++] : ''; }); }

  function record(key, el) {
    I.misses++;
    if (!I.collect || I.missing[key]) return;
    var sec = 'core';
    var t = el && el.closest ? el.closest('[data-sec],[data-key]') : null;
    if (t) sec = (t.getAttribute('data-sec') || t.getAttribute('data-key') || '').split('/')[0];
    else if (el && el.closest && el.closest('.stage, .keypanel, .nav-label') && C.slides && C.slides[C.state.idx]) sec = C.slides[C.state.idx].section.id;
    I.missing[key] = { section: sec };
  }

  // translate a plain string (text node, attribute, title)
  function tText(s, el, quiet) {
    if (!LETTERS.test(s) || HEB.test(s)) return { out: s, hit: false };
    var lead = s.match(/^\s*/)[0], trail = s.match(/\s*$/)[0];
    var k = numKey(s), v = dict.he[k.key];
    if (v == null) { if (!quiet) record(k.key, el); return { out: s, hit: false }; }
    I.hits++;
    return { out: lead + fill(v, k.nums) + trail, hit: true };
  }

  var applied = new WeakMap();   // element -> skeleton we produced (to skip re-translation)

  function translateBlock(el) {
    var parts = [], sk = skel(el, parts);
    if (applied.get(el) === sk || HEB.test(wordsOf(el))) return;
    var k = numKey(sk), v = dict.he[k.key];
    if (v == null) { record(k.key, el); fallback(el); return; }
    I.hits++;
    var out = fill(v, k.nums);
    rebuild(el, out, parts);
    applied.set(el, skel(el, []));
  }
  // rebuild el's children from a translated skeleton, reusing the original inline elements
  function rebuild(el, out, parts) {
    var frag = document.createDocumentFragment(), stack = [frag], used = {};
    var re = /<(\d+)\/>|<(\d+)>|<\/(\d+)>/g, last = 0, m;
    function text(t) { if (t) stack[stack.length - 1].appendChild(document.createTextNode(t.replace(/‹/g, '<'))); }
    function take(i, deep) {
      var p = parts[i]; if (!p) return null;
      if (used[i]) return p.cloneNode(deep);
      used[i] = 1;
      if (!deep) while (p.firstChild) p.removeChild(p.firstChild);
      return p;
    }
    while ((m = re.exec(out))) {
      text(out.slice(last, m.index)); last = re.lastIndex;
      if (m[1] != null) { var e1 = take(+m[1], true); if (e1) stack[stack.length - 1].appendChild(e1); }
      else if (m[2] != null) { var e2 = take(+m[2], false); if (e2) { stack[stack.length - 1].appendChild(e2); stack.push(e2); } }
      else if (stack.length > 1) stack.pop();
    }
    text(out.slice(last));
    while (el.firstChild) el.removeChild(el.firstChild);
    el.appendChild(frag);
  }
  function fallback(el) {
    for (var n = el.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3) { var r = tText(n.nodeValue, el, true); if (r.hit) n.nodeValue = r.out; }
      else if (n.nodeType === 1 && !isOpaque(n) && !isSkip(n)) fallback(n);
    }
  }
  function attrs(el) {
    ['title', 'placeholder', 'aria-label'].forEach(function (a) {
      var v = el.getAttribute && el.getAttribute(a);
      if (v && LETTERS.test(v) && !HEB.test(v)) { var r = tText(v, el); if (r.hit) el.setAttribute(a, r.out); }
    });
  }
  function isBlockCandidate(el) {
    if (!inlineLeaf(el) || !LETTERS.test(wordsOf(el))) return false;
    if (LETTERS.test(ownText(el))) return true;
    var tag = el.tagName.toUpperCase();
    if (/^(P|LI|H[1-6]|TD|TH|OPTION|LABEL|SUMMARY|FIGCAPTION|CAPTION|DT|DD|TEXT)$/.test(tag)) return true;
    return el.childNodes.length === 1;
  }
  function visit(el) {
    if (!el || el.nodeType !== 1) return;
    if (isSkip(el)) return;
    attrs(el);
    if (isOpaque(el)) return;
    if (isBlockCandidate(el)) { translateBlock(el); return; }
    for (var n = el.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 3) { var r = tText(n.nodeValue, el); if (r.hit) n.nodeValue = r.out; }
      else if (n.nodeType === 1) visit(n);
    }
    if (el.tagName === 'SELECT') [].forEach.call(el.options, function (o) { visit(o); });
  }

  // ---- observer: translate whatever changes
  var mo = null, busy = false;
  function run(fn) {
    if (lang !== 'he') return;
    busy = true;
    try { fn(); } catch (e) { (C.errors || []).push('i18n: ' + (e && e.message)); }
    if (mo) mo.takeRecords();
    busy = false;
  }
  function topBlock(el) {
    while (el && el.parentElement && el.parentElement.nodeType === 1 && el.parentElement !== document.body &&
           inlineLeaf(el.parentElement) && !isSkip(el.parentElement)) el = el.parentElement;
    return el;
  }
  function onMut(list) {
    if (busy || lang !== 'he') return;
    var targets = [];
    list.forEach(function (m) {
      var t = m.type === 'characterData' ? m.target.parentElement : m.target;
      if (m.type === 'attributes') t = m.target;
      if (t && t.nodeType === 1 && targets.indexOf(t) < 0) targets.push(t);
    });
    run(function () {
      targets.forEach(function (t) {
        if (!t.isConnected || (t.closest && t.closest(SKIP))) return;
        visit(topBlock(t));
      });
    });
  }
  function observe() {
    if (mo) { mo.disconnect(); mo = null; }
    if (lang !== 'he' || typeof MutationObserver === 'undefined') return;
    mo = new MutationObserver(onMut);
    mo.observe(document.body, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['title', 'placeholder'] });
    run(function () { visit(document.body); });
  }
  function applyDocDir() {
    var root = document.documentElement;
    root.setAttribute('lang', lang === 'he' ? 'he' : 'en');
    root.setAttribute('dir', lang === 'he' ? 'rtl' : 'ltr');
  }
  applyDocDir();
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { setTimeout(observe, 0); });
  else setTimeout(observe, 0);
})();
