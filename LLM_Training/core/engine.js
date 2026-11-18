/* LLM Training — core engine.
 * Plain script, no modules, no network. Sections call Course.register({...}).
 * See CONTRACT.md for the section schema and the helper API passed to render(el, api).
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------ data-type taxonomy
  var TYPES = {
    text:        { label: 'text',        desc: 'Human-readable string (prompt, document, message).' },
    token:       { label: 'token',       desc: 'A sub-word piece of text the model actually sees (e.g. " cat").' },
    integer:     { label: 'integer',     desc: 'Whole number: token IDs, counts, indices, dimensions.' },
    float:       { label: 'float',       desc: 'Real number: a logit, a score, a weight, a loss value.' },
    probability: { label: 'probability', desc: 'Number in [0,1]; a distribution sums to 1.' },
    vector:      { label: 'vector',      desc: '1-D array of floats, e.g. an embedding [d_model].' },
    matrix:      { label: 'matrix',      desc: '2-D array of floats, e.g. X [seq_len × d_model] or a weight W.' },
    tensor:      { label: 'tensor',      desc: '3-D+ array, e.g. per-head Q [n_heads × seq_len × d_head].' },
    mask:        { label: 'mask',        desc: 'Pattern that blocks positions (causal mask, loss mask, padding).' },
    boolean:     { label: 'boolean',     desc: 'True/false flag: stop?, is_error, needs_approval.' },
    json:        { label: 'json',        desc: 'Structured data: tool calls, tool schemas, API payloads, state.' },
    function:    { label: 'function',    desc: 'Executable code: a tool, a node, a scoring or activation function.' }
  };

  var PARTS = {
    'I':  'Part I — How LLMs Work',
    'II': 'Part II — Agentic AI',
    'A':  'Appendix'
  };

  // ------------------------------------------------------------------ running examples
  var REF = {
    name: 'Llama-3-8B-class reference model',
    vocab: 128256, d_model: 4096, n_layers: 32, n_heads: 32, n_kv_heads: 8, d_head: 128,
    d_ff: 14336, context: 8192, bytesPerParamBF16: 2
  };
  REF.paramBreakdown = function () {
    var r = REF, kvDim = r.n_kv_heads * r.d_head;
    var wq = r.d_model * r.n_heads * r.d_head, wk = r.d_model * kvDim, wv = r.d_model * kvDim,
        wo = r.n_heads * r.d_head * r.d_model;
    var attn = wq + wk + wv + wo;
    var ffn = 3 * r.d_model * r.d_ff;              // gate, up, down (SwiGLU)
    var norms = 2 * r.d_model;                     // two RMSNorm gains per block
    var perLayer = attn + ffn + norms;
    var embed = r.vocab * r.d_model, unembed = r.vocab * r.d_model, finalNorm = r.d_model;
    var total = embed + perLayer * r.n_layers + finalNorm + unembed;
    return { wq: wq, wk: wk, wv: wv, wo: wo, attn: attn, ffn: ffn, norms: norms, perLayer: perLayer,
             embed: embed, unembed: unembed, finalNorm: finalNorm, total: total,
             bytesBF16: total * 2 };
  };
  // K and V, every layer, every KV head, BF16 (2 bytes)
  REF.kvBytesPerToken = 2 * REF.n_layers * REF.n_kv_heads * REF.d_head * 2; // 131072 = 128 KiB

  var EX = {
    prompt: 'The cat sat on the',
    next: ' mat',
    tokens: [
      { text: 'The', id: 464 }, { text: ' cat', id: 3797 }, { text: ' sat', id: 3332 },
      { text: ' on', id: 319 }, { text: ' the', id: 262 }
    ],
    nextToken: { text: ' mat', id: 2603 },
    toy: { d_model: 4, n_heads: 2, d_head: 2, seq_len: 5 },
    // Illustrative toy input matrix X [5 × 4] (embedding + position already combined).
    toyX: [
      [ 0.2, -0.1,  0.4,  0.0],   // The
      [ 0.9,  0.3, -0.2,  0.5],   // ␣cat
      [-0.3,  0.8,  0.1,  0.2],   // ␣sat
      [ 0.1, -0.4,  0.6, -0.2],   // ␣on
      [ 0.3, -0.2,  0.5,  0.1]    // ␣the
    ],
    ref: REF,
    agent: {
      request: 'Book me a table for 2 in Paris tomorrow at 8pm and tell me if I need an umbrella',
      tools: ['get_weather', 'search_restaurants', 'book_table'],
      coding: 'A coding agent fixing a failing unit test'
    }
  };

  // ------------------------------------------------------------------ math helpers
  var M = {
    sum: function (a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; },
    max: function (a) { return Math.max.apply(null, a); },
    argmax: function (a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; },
    dot: function (a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; },
    norm: function (a) { return Math.sqrt(M.dot(a, a)); },
    cosine: function (a, b) { var d = M.norm(a) * M.norm(b); return d === 0 ? 0 : M.dot(a, b) / d; },
    add: function (a, b) { return a.map(function (x, i) { return Array.isArray(x) ? M.add(x, b[i]) : x + b[i]; }); },
    sub: function (a, b) { return a.map(function (x, i) { return Array.isArray(x) ? M.sub(x, b[i]) : x - b[i]; }); },
    scale: function (a, k) { return a.map(function (x) { return Array.isArray(x) ? M.scale(x, k) : x * k; }); },
    transpose: function (A) { return A[0].map(function (_, j) { return A.map(function (r) { return r[j]; }); }); },
    matmul: function (A, B) {
      return A.map(function (row) {
        return B[0].map(function (_, j) { var s = 0; for (var k = 0; k < row.length; k++) s += row[k] * B[k][j]; return s; });
      });
    },
    matvec: function (A, v) { return A.map(function (r) { return M.dot(r, v); }); },
    vecmat: function (v, A) { return A[0].map(function (_, j) { var s = 0; for (var k = 0; k < v.length; k++) s += v[k] * A[k][j]; return s; }); },
    softmax: function (a, T) {
      T = T == null ? 1 : T;
      var finite = a.filter(function (x) { return isFinite(x); });
      if (!finite.length) return a.map(function () { return 0; });
      var m = Math.max.apply(null, finite);
      var e = a.map(function (x) { return isFinite(x) ? Math.exp((x - m) / T) : 0; });
      var s = M.sum(e);
      return e.map(function (x) { return x / s; });
    },
    logSoftmax: function (a) { var p = M.softmax(a); return p.map(function (x) { return Math.log(x); }); },
    rmsnorm: function (x, g, eps) {
      eps = eps == null ? 1e-5 : eps;
      var ms = M.sum(x.map(function (v) { return v * v; })) / x.length, r = Math.sqrt(ms + eps);
      return x.map(function (v, i) { return (v / r) * (g ? g[i] : 1); });
    },
    sigmoid: function (x) { return 1 / (1 + Math.exp(-x)); },
    silu: function (x) { return x / (1 + Math.exp(-x)); },
    relu: function (x) { return Math.max(0, x); },
    gelu: function (x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); },
    round: function (x, d) { var k = Math.pow(10, d == null ? 2 : d); return Math.round(x * k) / k; },
    range: function (n) { var r = []; for (var i = 0; i < n; i++) r.push(i); return r; },
    linspace: function (a, b, n) { var r = []; for (var i = 0; i < n; i++) r.push(a + (b - a) * i / (n - 1)); return r; },
    rng: function (seed) { // mulberry32 deterministic RNG -> function returning [0,1)
      var s = seed >>> 0;
      return function () {
        s = (s + 0x6D2B79F5) >>> 0; var t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },
    sampleIndex: function (probs, r) { var c = 0; for (var i = 0; i < probs.length; i++) { c += probs[i]; if (r < c) return i; } return probs.length - 1; }
  };

  function fmt(x, d) {
    if (x === -Infinity) return '−∞';
    if (x === Infinity) return '∞';
    if (typeof x !== 'number') return String(x);
    if (isNaN(x)) return 'NaN';
    d = d == null ? 2 : d;
    var s = x.toFixed(d);
    if (/^-0\.?0*$/.test(s)) s = s.slice(1);
    return s.replace('-', '−');
  }
  function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
  function fmtBytes(b) {
    var u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'], i = 0;
    while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (i === 0 ? b : b.toFixed(b >= 100 ? 0 : b >= 10 ? 1 : 2)) + ' ' + u[i];
  }
  function fmtBig(n) {
    var a = Math.abs(n);
    if (a >= 1e12) return (n / 1e12).toFixed(2) + ' T';
    if (a >= 1e9) return (n / 1e9).toFixed(2) + ' B';
    if (a >= 1e6) return (n / 1e6).toFixed(2) + ' M';
    if (a >= 1e3) return (n / 1e3).toFixed(1) + ' K';
    return String(n);
  }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  // show leading spaces of tokens visibly
  function tokText(t) { return esc(t).replace(/ /g, '␣').replace(/\n/g, '↵'); }

  // ------------------------------------------------------------------ registry
  var sections = [];
  var Course = window.Course = {
    TYPES: TYPES, PARTS: PARTS, EX: EX, REF: REF, math: M,
    sections: sections, slides: [], errors: [], warnings: [],
    register: function (sec) {
      try { validateSection(sec); } catch (e) { Course.errors.push('register: ' + e.message); }
      sections.push(sec);
    }
  };

  function validateSection(sec) {
    var w = function (m) { Course.warnings.push('[' + (sec && sec.id) + '] ' + m); };
    if (!sec || !sec.id) throw new Error('section without id');
    if (!PARTS[sec.part]) w('unknown part "' + sec.part + '" (use I, II or A)');
    if (typeof sec.order !== 'number') w('order must be a number');
    (sec.slides || []).forEach(function (s) {
      if (!s.id || !s.title) w('slide missing id/title');
      if (!s.critical || !(s.critical.understand || []).length || !(s.critical.details || []).length) w(s.id + ': missing critical.understand/details');
      if (!(s.dataTypes || []).length) w(s.id + ': missing dataTypes');
      (s.dataTypes || []).forEach(function (d) { if (!TYPES[d.type]) w(s.id + ': unknown type "' + d.type + '"'); });
      var n = (s.levels || []).length;
      if (n < 2 || n > 4) w(s.id + ': has ' + n + ' levels (need 2–4)');
    });
  }

  // ------------------------------------------------------------------ DOM helper
  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v == null || v === false) return;
        if (k === 'class') el.className = v;
        else if (k === 'html') el.innerHTML = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : v);
      });
    }
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function append(el, c) {
    if (c == null || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }

  // ------------------------------------------------------------------ state
  var state = { idx: 0, level: 0 };
  var timers = [], cleanups = [];
  var dom = {};
  var visited = loadVisited();
  var uid = 0;

  function loadVisited() {
    try { return JSON.parse(localStorage.getItem('llmtraining.visited') || '{}') || {}; } catch (e) { return {}; }
  }
  function saveVisited() {
    try { localStorage.setItem('llmtraining.visited', JSON.stringify(visited)); } catch (e) { /* file:// may block */ }
  }
  function clearTimers() {
    timers.forEach(function (t) { t.kind === 'i' ? clearInterval(t.id) : clearTimeout(t.id); });
    timers = [];
    cleanups.forEach(function (f) { try { f(); } catch (e) { /* ignore */ } });
    cleanups = [];
  }

  // ------------------------------------------------------------------ navigation targets
  function resolve(target) {
    // 'section' | 'section/slide' | 'section/slide/level' | numeric index
    if (typeof target === 'number') return { idx: target, level: 0 };
    var p = String(target).replace(/^#?\/?/, '').split('/');
    var secId = p[0], slideId = p[1], lvl = p[2] ? parseInt(p[2], 10) || 0 : 0;
    for (var i = 0; i < Course.slides.length; i++) {
      var s = Course.slides[i];
      if (s.section.id === secId && (!slideId || s.slide.id === slideId)) {
        return { idx: i, level: Math.max(0, Math.min(lvl, s.slide.levels.length - 1)) };
      }
    }
    return null;
  }
  Course.resolve = resolve;

  function go(target, level) {
    var r = typeof target === 'object' && target ? target : resolve(target);
    if (!r) { Course.errors.push('Unresolved link: ' + target); flash('Link not found: ' + target); return false; }
    if (level != null) r.level = level;
    show(r.idx, r.level);
    return true;
  }
  Course.go = go;

  function flash(msg) {
    var t = h('div', { class: 'toast', text: msg });
    document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 2200);
  }

  // ------------------------------------------------------------------ rendering
  function show(idx, level, opts) {
    opts = opts || {};
    var n = Course.slides.length;
    if (!n) return;
    idx = Math.max(0, Math.min(n - 1, idx | 0));
    var entry = Course.slides[idx];
    level = Math.max(0, Math.min(entry.slide.levels.length - 1, level | 0));
    var slideChanged = idx !== state.idx || !dom.rendered;
    var dir = level > state.level && !slideChanged ? 'in' : (level < state.level && !slideChanged ? 'out' : 'next');
    state.idx = idx; state.level = level;
    clearTimers();

    visited[entry.key] = 1; saveVisited();
    var hash = '#/' + entry.key + '/' + level;
    if (location.hash !== hash) { suppressHash = true; try { history.replaceState(null, '', hash); } catch (e) { try { location.hash = hash; } catch (e2) { /* sandboxed viewer */ } } suppressHash = false; }

    renderHead(entry, level);
    renderBody(entry, level, dir);
    renderPanel(entry);
    renderToc();
    renderFooter();
    dom.rendered = true;
    if (slideChanged && !opts.keepScroll) dom.stage.scrollTop = 0;
    document.title = entry.slide.title + ' · LLM Training';
  }
  Course.show = show;

  function renderHead(entry, level) {
    var lv = entry.slide.levels;
    dom.head.innerHTML = '';
    var crumbs = h('div', { class: 'crumbs' });
    lv.forEach(function (L, i) {
      if (i) crumbs.appendChild(h('span', { class: 'crumb-sep', text: '›' }));
      crumbs.appendChild(h('button', {
        class: 'crumb' + (i === level ? ' on' : ''), 'data-level': i,
        title: 'Zoom level ' + (i + 1) + ' of ' + lv.length,
        onclick: function () { show(state.idx, i); }
      }, h('span', { class: 'crumb-n', text: String(i + 1) }), ' ' + L.name));
    });
    dom.head.appendChild(h('div', { class: 'head-top' },
      h('div', { class: 'eyebrow' },
        h('span', { class: 'part-tag', text: entry.section.part === 'A' ? 'Appendix' : 'Part ' + entry.section.part }),
        ' ', entry.section.title, h('span', { class: 'slide-num', text: '  ·  slide ' + (entry.sIdx + 1) + '/' + entry.section.slides.length })),
      h('div', { class: 'zoom-btns' },
        h('button', { class: 'btn zoom-out', disabled: level === 0, title: 'Zoom out (↑)', onclick: zoomOut }, '▲ Zoom out'),
        h('button', { class: 'btn zoom-in primary', disabled: level >= lv.length - 1, title: 'Dive in (↓)', onclick: zoomIn }, '▼ Dive in'))));
    dom.head.appendChild(h('h1', { class: 'slide-title', text: entry.slide.title }));
    dom.head.appendChild(crumbs);
  }

  function renderBody(entry, level, dir) {
    dom.content.innerHTML = '';
    var L = entry.slide.levels[level];
    var wrap = h('div', { class: 'level level-' + dir, 'data-level-name': L.name });
    dom.content.appendChild(wrap);
    try {
      L.render(wrap, makeApi(entry, level, wrap));
    } catch (e) {
      var msg = '[' + entry.key + ' L' + level + '] ' + (e && e.stack || e);
      Course.errors.push(msg);
      if (window.console) console.error(msg);
      wrap.appendChild(h('div', { class: 'render-error' }, h('b', { text: 'Render error: ' }), String(e && e.message || e)));
    }
    if (level < entry.slide.levels.length - 1) {
      var nx = entry.slide.levels[level + 1];
      dom.content.appendChild(h('button', { class: 'dive-hint', onclick: zoomIn },
        '▼ Dive in: ', h('b', { text: nx.name })));
    } else if (state.idx < Course.slides.length - 1) {
      var nxt = Course.slides[state.idx + 1];
      dom.content.appendChild(h('button', { class: 'dive-hint next-hint', onclick: next },
        'Deepest level reached — next slide → ', h('b', { text: nxt.slide.title })));
    }
  }

  function renderPanel(entry) {
    var s = entry.slide, c = s.critical || {};
    dom.panel.innerHTML = '';
    var list = function (arr) { var ul = h('ul'); (arr || []).forEach(function (x) { ul.appendChild(h('li', { html: x })); }); return ul; };
    dom.panel.appendChild(h('div', { class: 'kp kp-understand' }, h('h3', { text: '★ Critical to understand' }), list(c.understand)));
    dom.panel.appendChild(h('div', { class: 'kp kp-details' }, h('h3', { text: '⚙ Critical details' }), list(c.details)));
    var dt = h('div', { class: 'kp kp-data' }, h('h3', { text: 'Data on this slide' }));
    (s.dataTypes || []).forEach(function (d) {
      dt.appendChild(h('div', { class: 'dt-row' },
        h('div', { class: 'dt-row-top' }, h('span', { class: 'dt-name', text: d.name }), h('span', { html: typeBadge(d.type) })),
        d.shape ? h('div', { class: 'dt-shape', html: 'shape <code>' + esc(d.shape) + '</code>' }) : null,
        d.example != null ? h('div', { class: 'dt-ex', html: 'e.g. <code>' + esc(d.example) + '</code>' }) : null));
    });
    dom.panel.appendChild(dt);
  }

  function renderToc() {
    var cur = Course.slides[state.idx];
    if (dom.tocBuilt) {
      dom.toc.querySelectorAll('.toc-slide').forEach(function (b) {
        var k = b.getAttribute('data-key');
        b.classList.toggle('on', k === cur.key);
        b.classList.toggle('visited', !!visited[k]);
      });
      dom.toc.querySelectorAll('.toc-sec').forEach(function (s) {
        s.classList.toggle('open', s.getAttribute('data-sec') === cur.section.id || s.classList.contains('pinned'));
      });
      var on = dom.toc.querySelector('.toc-slide.on');
      if (on && on.scrollIntoView && !isVisibleIn(on, dom.toc)) on.scrollIntoView({ block: 'nearest' });
      var nv = Object.keys(visited).filter(function (k) { return Course.slides.some(function (s) { return s.key === k; }); }).length;
      dom.tocCount.textContent = nv + ' / ' + Course.slides.length + ' slides visited';
      return;
    }
    dom.toc.innerHTML = '';
    dom.tocCount = h('div', { class: 'toc-count' });
    dom.toc.appendChild(dom.tocCount);
    var lastPart = null;
    sections.forEach(function (sec) {
      if (sec.part !== lastPart) { dom.toc.appendChild(h('div', { class: 'toc-part', text: PARTS[sec.part] || sec.part })); lastPart = sec.part; }
      var box = h('div', { class: 'toc-sec', 'data-sec': sec.id });
      box.appendChild(h('button', { class: 'toc-sec-title', onclick: function () {
        if (box.classList.contains('open')) { box.classList.remove('open'); box.classList.remove('pinned'); }
        else { box.classList.add('open'); box.classList.add('pinned'); }
      } }, h('span', { class: 'toc-caret', text: '▸' }), ' ', sec.title));
      var ul = h('div', { class: 'toc-slides' });
      sec.slides.forEach(function (sl) {
        var e = Course.slides.filter(function (x) { return x.section === sec && x.slide === sl; })[0];
        var dots = ''; for (var i = 0; i < sl.levels.length; i++) dots += '●';
        ul.appendChild(h('button', { class: 'toc-slide', 'data-key': e.key, title: sl.levels.length + ' zoom levels',
          onclick: function () { show(e.i, 0); closeTocMobile(); } },
          h('span', { class: 'toc-check', text: '✓' }), h('span', { class: 'toc-name', text: sl.title }),
          h('span', { class: 'toc-zoom', text: dots })));
      });
      box.appendChild(ul);
      dom.toc.appendChild(box);
    });
    dom.tocBuilt = true;
    renderToc();
  }
  function isVisibleIn(el, box) {
    var a = el.getBoundingClientRect(), b = box.getBoundingClientRect();
    return a.top >= b.top && a.bottom <= b.bottom;
  }
  function closeTocMobile() { document.body.classList.remove('toc-open'); }

  function renderFooter() {
    var n = Course.slides.length;
    dom.progress.style.width = ((state.idx + 1) / n * 100).toFixed(2) + '%';
    dom.counter.textContent = (state.idx + 1) + ' / ' + n;
    dom.back.disabled = state.idx === 0;
    dom.next.disabled = state.idx === n - 1;
    var p = Course.slides[state.idx - 1], q = Course.slides[state.idx + 1];
    dom.backLabel.textContent = p ? p.slide.title : '';
    dom.nextLabel.textContent = q ? q.slide.title : '';
  }

  function next() { if (state.idx < Course.slides.length - 1) show(state.idx + 1, 0); }
  function prev() { if (state.idx > 0) show(state.idx - 1, 0); }
  function zoomIn() { var e = Course.slides[state.idx]; if (state.level < e.slide.levels.length - 1) show(state.idx, state.level + 1, { keepScroll: true }); }
  function zoomOut() { if (state.level > 0) show(state.idx, state.level - 1, { keepScroll: true }); }
  Course.next = next; Course.prev = prev; Course.zoomIn = zoomIn; Course.zoomOut = zoomOut;
  Course.state = state;

  // ------------------------------------------------------------------ helper API (passed to render)
  function typeBadge(t, text) {
    var ok = !!TYPES[t];
    if (!ok) Course.errors.push('Unknown data type badge: ' + t);
    return '<span class="dt dt-' + esc(ok ? t : 'unknown') + '" title="' + esc(ok ? TYPES[t].desc : 'unknown type') + '">' + esc(text || t) + '</span>';
  }
  function shapeHtml(s) { return s ? '<span class="shape">' + esc(s) + '</span>' : ''; }

  function makeApi(entry, level, root) {
    var api = {
      section: entry.section, slide: entry.slide, level: level, levels: entry.slide.levels.length,
      ex: EX, ref: REF, math: M, types: TYPES,
      fmt: fmt, fmtInt: fmtInt, fmtBytes: fmtBytes, fmtBig: fmtBig, esc: esc, tokText: tokText,
      h: h,
      uid: function (p) { uid++; return (p || 'u') + uid; },

      // --- timers (auto-cleaned when the slide/level changes)
      every: function (ms, fn) { var id = setInterval(function () { if (!root.isConnected) return clearInterval(id); try { fn(); } catch (e) { report(e); clearInterval(id); } }, ms); timers.push({ kind: 'i', id: id }); return id; },
      after: function (ms, fn) { var id = setTimeout(function () { if (!root.isConnected) return; try { fn(); } catch (e) { report(e); } }, ms); timers.push({ kind: 't', id: id }); return id; },
      onCleanup: function (fn) { cleanups.push(fn); },

      // --- navigation
      zoomTo: function (n) { show(state.idx, n, { keepScroll: true }); },
      zoomIn: zoomIn, zoomOut: zoomOut,
      goto: function (target) { return go(target); },
      next: next, prev: prev,

      // --- inline HTML helpers (return strings)
      type: function (t, text) { return typeBadge(t, text); },
      shape: shapeHtml,
      badge: function (t, shape) { return typeBadge(t) + (shape ? ' ' + shapeHtml(shape) : ''); },
      link: function (target, text) { return '<a href="#" class="xlink" data-goto="' + esc(target) + '">' + esc(text) + '</a>'; },
      tok: function (t, id) { return '<span class="tok">' + tokText(t) + (id != null ? '<sub>' + id + '</sub>' : '') + '</span>'; },

      // --- element helpers (append to el, return element or controller)
      html: function (el, s) { var d = h('div', { html: s }); el.appendChild(d); return d; },
      callout: callout,
      illustrative: function (el, extra) { return callout(el, '<b>Illustrative values.</b> ' + (extra || 'Numbers are made up to show the mechanics; real models learn them.'), 'illustrative'); },
      docsNote: function (el, extra) { return callout(el, '<b>Check current docs.</b> Framework APIs change quickly; signatures here reflect the time of writing. ' + (extra || ''), 'docs'); },
      blocks: blocks, stepper: stepper, matrix: matrix,
      slider: slider, select: select, button: button, toggle: toggle, input: input,
      chart: chart, bars: bars, code: code, table: table, tokens: tokens, grid: grid, card: card
    };
    function report(e) { Course.errors.push('[' + entry.key + ' L' + level + ' async] ' + (e && e.stack || e)); }
    return api;

    function callout(el, html, kind) {
      kind = kind || 'info';
      var icons = { info: 'ℹ', tip: '💡', warn: '⚠', key: '★', illustrative: '≈', docs: '📘', hedge: '?' };
      var d = h('div', { class: 'callout callout-' + kind }, h('span', { class: 'callout-icon', text: icons[kind] || 'ℹ' }), h('div', { class: 'callout-body', html: html }));
      el.appendChild(d); return d;
    }

    // Diagram of blocks with animated flow arrows.
    // spec: { blocks:[...] } or { rows:[[...],[...]] }, direction:'row'|'column', animate:true, caption, rowArrows:true
    // block: { id, label, sub, type, shape, zoom:(number|true), goto:'sec/slide', note, color, edge:'arrow label', cls }
    function blocks(el, spec) {
      var rows = spec.rows || [spec.blocks || []];
      var col = spec.direction === 'column';
      var wrap = h('div', { class: 'blocks' + (col ? ' blocks-col' : '') + (spec.animate === false ? '' : ' animate') + (spec.cls ? ' ' + spec.cls : '') });
      var byId = {};
      rows.forEach(function (row, ri) {
        if (ri && spec.rowArrows !== false) wrap.appendChild(h('div', { class: 'blk-rowarrow' }, h('span', { class: 'flow-line v' }), h('span', { class: 'ah', text: '▼' })));
        var r = h('div', { class: 'blk-row' });
        row.forEach(function (b, i) {
          if (b === '|' || b == null) return;
          if (i && b.arrow !== false && spec.arrows !== false) {
            r.appendChild(h('div', { class: 'blk-arrow' }, b.edge ? h('span', { class: 'edge-label', html: b.edge }) : null,
              h('span', { class: 'flow-line' }), h('span', { class: 'ah', text: col ? '▼' : '▶' })));
          }
          var clickable = b.zoom != null || b.goto || b.onClick;
          var node = h('div', {
            class: 'blk' + (clickable ? ' clickable' : '') + (b.cls ? ' ' + b.cls : '') + (b.highlight ? ' hl' : ''),
            'data-id': b.id || null, 'data-link': b.goto || null, tabindex: clickable ? 0 : null,
            title: b.note || (b.zoom != null ? 'Click to dive in' : b.goto ? 'Click to jump to ' + b.goto : null),
            style: b.color ? { borderColor: b.color, boxShadow: 'inset 0 3px 0 ' + b.color } : null
          },
            h('div', { class: 'blk-label', html: b.label || '' }),
            b.sub ? h('div', { class: 'blk-sub', html: b.sub }) : null,
            (b.type || b.shape) ? h('div', { class: 'blk-meta', html: (b.type ? (Array.isArray(b.type) ? b.type.map(function (t) { return typeBadge(t); }).join(' ') : typeBadge(b.type)) : '') + (b.shape ? ' ' + shapeHtml(b.shape) : '') }) : null,
            clickable ? h('div', { class: 'blk-go', text: b.zoom != null ? '⤵ dive in' : b.goto ? '↗ open' : '▸' }) : null);
          if (clickable) {
            var act = function () {
              if (b.onClick) return b.onClick(b);
              if (b.zoom != null) api.zoomTo(b.zoom === true ? level + 1 : b.zoom);
              else if (b.goto) go(b.goto);
            };
            node.addEventListener('click', act);
            node.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
          }
          if (b.id) byId[b.id] = node;
          r.appendChild(node);
        });
        wrap.appendChild(r);
      });
      if (spec.caption) wrap.appendChild(h('div', { class: 'blk-caption', html: spec.caption }));
      el.appendChild(wrap);
      return {
        el: wrap,
        highlight: function (ids) {
          ids = ids == null ? [] : [].concat(ids);
          Object.keys(byId).forEach(function (k) { byId[k].classList.toggle('hl', ids.indexOf(k) >= 0); });
        },
        node: function (id) { return byId[id]; }
      };
    }

    // Step-by-step walkthrough with Prev / Next / Play.
    // spec: { steps:[{title, html | render(el, api)}], interval:ms, onStep(i), start:0 }
    function stepper(el, spec) {
      var steps = spec.steps || [], i = spec.start || 0, playing = null;
      var box = h('div', { class: 'stepper' });
      var title = h('div', { class: 'step-title' });
      var body = h('div', { class: 'step-body' });
      var count = h('span', { class: 'step-count' });
      var dots = h('div', { class: 'step-dots' });
      var bPrev = h('button', { class: 'btn step-prev', onclick: function () { stop(); goStep(i - 1); } }, '◀ Prev');
      var bNext = h('button', { class: 'btn step-next primary', onclick: function () { stop(); goStep(i + 1); } }, 'Next ▶');
      var bPlay = h('button', { class: 'btn step-play', onclick: function () { playing ? stop() : play(); } }, '▶ Play');
      steps.forEach(function (_, k) { dots.appendChild(h('button', { class: 'step-dot', title: 'Step ' + (k + 1), onclick: function () { stop(); goStep(k); } })); });
      box.appendChild(h('div', { class: 'step-ctrl' }, bPrev, bNext, bPlay, count, dots));
      box.appendChild(title); box.appendChild(body);
      el.appendChild(box);
      function goStep(k) {
        i = Math.max(0, Math.min(steps.length - 1, k));
        var s = steps[i] || {};
        title.innerHTML = '<span class="step-n">' + (i + 1) + '</span> ' + (s.title || '');
        body.innerHTML = '';
        try {
          if (s.render) s.render(body, api); else if (s.html) body.innerHTML = s.html;
          if (spec.onStep) spec.onStep(i, body);
        } catch (e) { report(e); body.appendChild(h('div', { class: 'render-error', text: 'Step error: ' + e.message })); }
        count.textContent = 'Step ' + (i + 1) + ' / ' + steps.length;
        bPrev.disabled = i === 0; bNext.disabled = i === steps.length - 1;
        [].forEach.call(dots.children, function (d, j) { d.classList.toggle('on', j === i); d.classList.toggle('done', j < i); });
      }
      function play() {
        if (i >= steps.length - 1) goStep(0);
        bPlay.textContent = '❚❚ Pause';
        playing = api.every(spec.interval || 1800, function () { if (i >= steps.length - 1) stop(); else goStep(i + 1); });
      }
      function stop() { if (playing) { clearInterval(playing); playing = null; } bPlay.textContent = '▶ Play'; }
      goStep(i);
      return { el: box, go: goStep, get index() { return i; }, count: steps.length };
    }

    // Heat-mapped matrix. spec: { data, rowLabels, colLabels, decimals, heat:true, causal:false, mask:[[bool]],
    //   maskText:'−∞', title, type, shape, highlight:{row, col, cell:[r,c]}, onCell(r,c), scale:'diverging'|'seq', small }
    function matrix(el, spec) {
      var box = h('div', { class: 'mx-wrap' + (spec.small ? ' mx-small' : '') });
      el.appendChild(box);
      function draw(sp) {
        spec = Object.assign({}, spec, sp || {});
        var D = spec.data || [], dec = spec.decimals == null ? 2 : spec.decimals;
        box.innerHTML = '';
        if (spec.title || spec.type || spec.shape) {
          box.appendChild(h('div', { class: 'mx-title', html: (spec.title ? '<b>' + spec.title + '</b> ' : '') + (spec.type ? typeBadge(spec.type) : '') + (spec.shape ? ' ' + shapeHtml(spec.shape) : '') }));
        }
        var masked = function (r, c) { return (spec.causal && c > r) || (spec.mask && spec.mask[r] && spec.mask[r][c]); };
        var vals = [];
        D.forEach(function (row, r) { row.forEach(function (v, c) { if (typeof v === 'number' && isFinite(v) && !masked(r, c)) vals.push(v); }); });
        var mx = vals.length ? Math.max.apply(null, vals.map(Math.abs)) : 1, mn = vals.length ? Math.min.apply(null, vals) : 0, mxv = vals.length ? Math.max.apply(null, vals) : 1;
        var seq = spec.scale === 'seq' || (spec.scale == null && mn >= 0);
        var t = h('table', { class: 'mx' });
        if (spec.colLabels) {
          var hr = h('tr', {}, h('th', {}));
          spec.colLabels.forEach(function (c, j) { hr.appendChild(h('th', { class: 'mx-col' + (spec.highlight && spec.highlight.col === j ? ' hlc' : ''), html: c })); });
          t.appendChild(hr);
        }
        D.forEach(function (row, r) {
          var tr = h('tr');
          if (spec.rowLabels) tr.appendChild(h('th', { class: 'mx-row' + (spec.highlight && spec.highlight.row === r ? ' hlr' : ''), html: spec.rowLabels[r] }));
          else if (spec.colLabels) tr.appendChild(h('th'));
          row.forEach(function (v, c) {
            var m = masked(r, c), bg = '';
            if (!m && spec.heat !== false && typeof v === 'number' && isFinite(v)) {
              if (seq) { var a = mxv > mn ? (v - mn) / (mxv - mn) : 0.5; bg = 'rgba(78,168,255,' + (0.08 + 0.62 * a).toFixed(3) + ')'; }
              else { var k = mx ? Math.abs(v) / mx : 0; bg = v >= 0 ? 'rgba(255,150,70,' + (0.08 + 0.6 * k).toFixed(3) + ')' : 'rgba(78,168,255,' + (0.08 + 0.6 * k).toFixed(3) + ')'; }
            }
            var hl = spec.highlight && ((spec.highlight.row === r && spec.highlight.col == null) || (spec.highlight.col === c && spec.highlight.row == null) ||
              (spec.highlight.cell && spec.highlight.cell[0] === r && spec.highlight.cell[1] === c) || (spec.highlight.row === r && spec.highlight.col === c));
            var td = h('td', { class: (m ? 'masked' : '') + (hl ? ' hl' : '') + (spec.onCell ? ' clickable' : ''), style: bg ? { background: bg } : null,
              text: m ? (spec.maskText != null ? spec.maskText : '−∞') : (typeof v === 'number' ? fmt(v, dec) : String(v)) });
            if (spec.onCell) td.addEventListener('click', function () { spec.onCell(r, c); });
            tr.appendChild(td);
          });
          t.appendChild(tr);
        });
        box.appendChild(h('div', { class: 'mx-scroll' }, t));
        if (spec.caption) box.appendChild(h('div', { class: 'mx-caption', html: spec.caption }));
      }
      draw();
      return { el: box, update: draw };
    }

    // Labelled range slider. spec: { label, min, max, step, value, format(v), onInput(v), init:true }
    function slider(el, spec) {
      var out = h('span', { class: 'ctl-val' });
      var inp = h('input', { type: 'range', min: spec.min, max: spec.max, step: spec.step || 1, value: spec.value });
      var f = spec.format || function (v) { return String(v); };
      var row = h('label', { class: 'ctl ctl-slider' }, h('span', { class: 'ctl-label', html: spec.label || '' }), inp, out);
      var fire = function () { var v = parseFloat(inp.value); out.innerHTML = f(v); if (spec.onInput) { try { spec.onInput(v); } catch (e) { report(e); } } };
      inp.addEventListener('input', fire);
      el.appendChild(row);
      if (spec.init !== false) fire(); else out.innerHTML = f(parseFloat(inp.value));
      return inp;
    }
    // spec: { label, options:[string | {value,label}], value, onChange(v), init:false }
    function select(el, spec) {
      var s = h('select');
      (spec.options || []).forEach(function (o) {
        var v = typeof o === 'object' ? o.value : o, l = typeof o === 'object' ? o.label : o;
        s.appendChild(h('option', { value: v, text: l, selected: String(v) === String(spec.value) }));
      });
      s.addEventListener('change', function () { try { spec.onChange && spec.onChange(s.value); } catch (e) { report(e); } });
      el.appendChild(h('label', { class: 'ctl ctl-select' }, spec.label ? h('span', { class: 'ctl-label', html: spec.label }) : null, s));
      if (spec.init) spec.onChange && spec.onChange(s.value);
      return s;
    }
    function button(el, label, onClick, cls) {
      var b = h('button', { class: 'btn ' + (cls || ''), html: label, onclick: function (e) { try { onClick(e); } catch (er) { report(er); } } });
      el.appendChild(b); return b;
    }
    function toggle(el, spec) {
      var c = h('input', { type: 'checkbox', checked: !!spec.value });
      c.addEventListener('change', function () { try { spec.onChange && spec.onChange(c.checked); } catch (e) { report(e); } });
      el.appendChild(h('label', { class: 'ctl ctl-toggle' }, c, h('span', { class: 'ctl-label', html: spec.label || '' })));
      if (spec.init && spec.onChange) spec.onChange(c.checked);
      return c;
    }
    // spec: { label, value, placeholder, onInput(v), multiline }
    function input(el, spec) {
      var i = spec.multiline ? h('textarea', { rows: spec.rows || 3, placeholder: spec.placeholder || '' }) : h('input', { type: 'text', placeholder: spec.placeholder || '' });
      i.value = spec.value || '';
      i.addEventListener('input', function () { try { spec.onInput && spec.onInput(i.value); } catch (e) { report(e); } });
      el.appendChild(h('label', { class: 'ctl ctl-input' }, spec.label ? h('span', { class: 'ctl-label', html: spec.label }) : null, i));
      if (spec.init !== false && spec.onInput) spec.onInput(i.value);
      return i;
    }

    // SVG line chart. spec: { series:[{name, points:[[x,y]], color, dashed}], xLabel, yLabel, width, height,
    //   xRange:[a,b], yRange:[a,b], logY, markers:[{x, label}], hlines:[{y,label}], points:true }
    function chart(el, spec) {
      var box = h('div', { class: 'chart' });
      el.appendChild(box);
      var palette = ['#4ea8ff', '#ff9a4d', '#5ad18a', '#b98cff', '#ff5d73', '#2fd0c9', '#e6e65c'];
      function draw(sp) {
        spec = Object.assign({}, spec, sp || {});
        var W = spec.width || 560, H = spec.height || 260, pl = 52, pr = 16, pt = 14, pb = 40;
        var all = []; (spec.series || []).forEach(function (s) { s.points.forEach(function (p) { if (isFinite(p[0]) && isFinite(p[1])) all.push(p); }); });
        var ty = function (y) { return spec.logY ? Math.log10(Math.max(y, 1e-12)) : y; };
        var xr = spec.xRange || [Math.min.apply(null, all.map(function (p) { return p[0]; })), Math.max.apply(null, all.map(function (p) { return p[0]; }))];
        var yr = spec.yRange ? spec.yRange.map(ty) : [Math.min.apply(null, all.map(function (p) { return ty(p[1]); })), Math.max.apply(null, all.map(function (p) { return ty(p[1]); }))];
        if (!all.length) { xr = [0, 1]; yr = [0, 1]; }
        if (xr[0] === xr[1]) xr = [xr[0] - 1, xr[1] + 1];
        if (yr[0] === yr[1]) yr = [yr[0] - 1, yr[1] + 1];
        var X = function (x) { return pl + (x - xr[0]) / (xr[1] - xr[0]) * (W - pl - pr); };
        var Y = function (y) { return pt + (1 - (ty(y) - yr[0]) / (yr[1] - yr[0])) * (H - pt - pb); };
        var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="chart-svg" preserveAspectRatio="xMidYMid meet">';
        for (var k = 0; k <= 4; k++) {
          var yv = yr[0] + (yr[1] - yr[0]) * k / 4, yy = pt + (1 - k / 4) * (H - pt - pb);
          var lab = spec.logY ? fmtTick(Math.pow(10, yv)) : fmtTick(yv);
          s += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + yy + '" y2="' + yy + '" class="grid"/>';
          s += '<text x="' + (pl - 6) + '" y="' + (yy + 4) + '" class="tick" text-anchor="end">' + lab + '</text>';
          var xv = xr[0] + (xr[1] - xr[0]) * k / 4, xx = X(xv);
          s += '<text x="' + xx + '" y="' + (H - pb + 16) + '" class="tick" text-anchor="middle">' + fmtTick(xv) + '</text>';
        }
        s += '<line x1="' + pl + '" x2="' + pl + '" y1="' + pt + '" y2="' + (H - pb) + '" class="axis"/>';
        s += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + (H - pb) + '" y2="' + (H - pb) + '" class="axis"/>';
        (spec.hlines || []).forEach(function (l) { var y = Y(l.y); s += '<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y + '" y2="' + y + '" class="mark"/><text x="' + (W - pr - 4) + '" y="' + (y - 4) + '" class="tick" text-anchor="end">' + esc(l.label || '') + '</text>'; });
        (spec.markers || []).forEach(function (m) { var x = X(m.x); s += '<line x1="' + x + '" x2="' + x + '" y1="' + pt + '" y2="' + (H - pb) + '" class="mark"/><text x="' + (x + 4) + '" y="' + (pt + 12) + '" class="tick">' + esc(m.label || '') + '</text>'; });
        (spec.series || []).forEach(function (se, i) {
          var c = se.color || palette[i % palette.length];
          var pts = se.points.filter(function (p) { return isFinite(p[0]) && isFinite(p[1]); });
          var d = pts.map(function (p, j) { return (j ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1); }).join(' ');
          s += '<path d="' + d + '" fill="none" stroke="' + c + '" stroke-width="2.2"' + (se.dashed ? ' stroke-dasharray="6 4"' : '') + '/>';
          if (spec.points || se.points.length <= 12) pts.forEach(function (p) { s += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="3" fill="' + c + '"/>'; });
        });
        if (spec.xLabel) s += '<text x="' + ((W + pl) / 2) + '" y="' + (H - 6) + '" class="axlabel" text-anchor="middle">' + esc(spec.xLabel) + '</text>';
        if (spec.yLabel) s += '<text x="12" y="' + ((H - pb + pt) / 2) + '" class="axlabel" text-anchor="middle" transform="rotate(-90 12 ' + ((H - pb + pt) / 2) + ')">' + esc(spec.yLabel) + '</text>';
        s += '</svg>';
        var leg = (spec.series || []).length > 1 || (spec.series && spec.series[0] && spec.series[0].name) ? '<div class="chart-legend">' + spec.series.map(function (se, i) {
          return '<span><i style="background:' + (se.color || palette[i % palette.length]) + '"></i>' + esc(se.name || '') + '</span>'; }).join('') + '</div>' : '';
        box.innerHTML = s + leg;
      }
      function fmtTick(v) { var a = Math.abs(v); return a >= 1e9 ? (v / 1e9).toFixed(1) + 'B' : a >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : a >= 1e3 ? (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'k' : a >= 10 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : a === 0 ? '0' : v.toFixed(a < 0.01 ? 4 : 2); }
      draw();
      return { el: box, update: draw };
    }

    // Horizontal bar chart. spec: { items:[{label, value, color, highlight, note}], max, format(v), title }
    function bars(el, spec) {
      var box = h('div', { class: 'bars' });
      el.appendChild(box);
      function draw(items) {
        if (items) spec.items = items;
        var its = spec.items || [];
        var mx = spec.max || Math.max.apply(null, its.map(function (x) { return Math.abs(x.value); }).concat([1e-12]));
        var f = spec.format || function (v) { return fmt(v, 3); };
        box.innerHTML = spec.title ? '<div class="bars-title">' + spec.title + '</div>' : '';
        its.forEach(function (it) {
          var w = Math.max(0, Math.min(100, Math.abs(it.value) / mx * 100));
          box.appendChild(h('div', { class: 'bar-row' + (it.highlight ? ' hl' : '') + (it.dim ? ' dim' : '') },
            h('span', { class: 'bar-label', html: it.label }),
            h('span', { class: 'bar-track' }, h('span', { class: 'bar-fill', style: { width: w.toFixed(2) + '%', background: it.color || '' } })),
            h('span', { class: 'bar-val', html: f(it.value) + (it.note ? ' <small>' + it.note + '</small>' : '') })));
        });
      }
      draw();
      return { el: box, update: draw };
    }

    function code(el, src, lang) {
      var pre = h('pre', { class: 'code' + (lang ? ' lang-' + lang : '') });
      if (lang) pre.appendChild(h('span', { class: 'code-lang', text: lang }));
      pre.appendChild(h('code', { html: highlight(esc(src), lang) }));
      el.appendChild(pre); return pre;
    }
    function table(el, spec) {
      var t = h('table', { class: 'tbl' + (spec.cls ? ' ' + spec.cls : '') });
      if (spec.headers) { var tr = h('tr'); spec.headers.forEach(function (x) { tr.appendChild(h('th', { html: x })); }); t.appendChild(tr); }
      (spec.rows || []).forEach(function (r) { var tr = h('tr'); r.forEach(function (x) { tr.appendChild(h('td', { html: x == null ? '' : String(x) })); }); t.appendChild(tr); });
      var w = h('div', { class: 'tbl-wrap' }, t); el.appendChild(w); return t;
    }
    // tokens: [{text, id, cls}] -> chips
    function tokens(el, toks, opts) {
      opts = opts || {};
      var d = h('div', { class: 'tokens' });
      toks.forEach(function (t, i) {
        d.appendChild(h('span', { class: 'tok' + (t.cls ? ' ' + t.cls : '') + (opts.highlight === i ? ' hl' : ''), style: t.color ? { borderColor: t.color, background: t.color + '22' } : null,
          html: tokText(t.text) + (t.id != null && opts.ids !== false ? '<sub>' + t.id + '</sub>' : '') }));
      });
      el.appendChild(d); return d;
    }
    // responsive grid container: cols = min column width in px
    function grid(el, minPx) { var g = h('div', { class: 'grid', style: { gridTemplateColumns: 'repeat(auto-fit, minmax(min(' + Math.min(minPx || 240, 280) + 'px, 100%), 1fr))' } }); el.appendChild(g); return g; }
    function card(el, title, html, cls) { var c = h('div', { class: 'card' + (cls ? ' ' + cls : '') }, title ? h('div', { class: 'card-title', html: title }) : null, h('div', { class: 'card-body', html: html || '' })); el.appendChild(c); return c; }
  }

  function highlight(s, lang) {
    if (lang === 'json') {
      return s.replace(/(&quot;[^&]*?&quot;)(\s*:)?/g, function (m, str, colon) { return colon ? '<span class="c-key">' + str + '</span>' + colon : '<span class="c-str">' + str + '</span>'; })
              .replace(/\b(true|false|null)\b/g, '<span class="c-kw">$1</span>')
              .replace(/(:\s*)(-?\d+(\.\d+)?)/g, '$1<span class="c-num">$2</span>');
    }
    if (lang === 'python' || lang === 'py' || lang === 'js' || lang === 'javascript' || lang === 'bash') {
      return s.split('\n').map(function (line) {
        var m = line.match(/^(.*?)(\s*(#|\/\/).*)$/);
        var body = m && !/(&quot;|&#39;|')[^'"]*(#|\/\/)/.test(m[1] + m[2].slice(0, 3)) ? m[1] : line, com = body === line ? '' : m[2];
        body = body.replace(/\b(def|class|return|import|from|as|if|else|elif|for|while|in|not|and|or|lambda|with|async|await|const|let|var|function|new|True|False|None|yield|try|except|raise)\b/g, '<span class="c-kw">$1</span>');
        return body + (com ? '<span class="c-com">' + com + '</span>' : '');
      }).join('\n');
    }
    return s;
  }

  // ------------------------------------------------------------------ boot
  var suppressHash = false;
  function boot() {
    sections.sort(function (a, b) { return a.order - b.order; });
    Course.slides = [];
    sections.forEach(function (sec) {
      (sec.slides || []).forEach(function (sl, j) {
        if (!sl.levels || !sl.levels.length) sl.levels = [{ name: 'Overview', render: function (el) { el.textContent = '(empty)'; } }];
        Course.slides.push({ section: sec, slide: sl, sIdx: j, i: Course.slides.length, key: sec.id + '/' + sl.id });
      });
    });
    buildShell();
    if (!Course.slides.length) { dom.content.innerHTML = '<p>No sections loaded.</p>'; return; }
    var r = location.hash ? resolve(location.hash) : null;
    show(r ? r.idx : 0, r ? r.level : 0);
    window.addEventListener('hashchange', function () { if (suppressHash) return; var q = resolve(location.hash); if (q) show(q.idx, q.level); });
  }

  function buildShell() {
    var app = document.getElementById('app') || document.body.appendChild(h('div', { id: 'app' }));
    app.innerHTML = '';
    dom.progress = h('div', { class: 'progress-fill' });
    dom.counter = h('span', { class: 'counter' });
    var header = h('header', { class: 'topbar' },
      h('button', { class: 'btn toc-toggle', title: 'Table of contents', onclick: function () { document.body.classList.toggle('toc-open'); document.body.classList.toggle('toc-hidden'); } }, '☰'),
      h('a', { class: 'brand', href: '#', onclick: function (e) { e.preventDefault(); show(0, 0); } }, h('span', { class: 'brand-mark', text: '◆' }), ' LLM Training'),
      h('div', { class: 'progress' }, dom.progress), dom.counter);
    dom.toc = h('nav', { class: 'toc', 'aria-label': 'Table of contents' });
    dom.head = h('div', { class: 'slide-head' });
    dom.content = h('div', { class: 'slide-content' });
    dom.panel = h('aside', { class: 'keypanel' });
    dom.stage = h('main', { class: 'stage' }, dom.head, h('div', { class: 'slide-body' }, dom.content, dom.panel));
    dom.backLabel = h('span', { class: 'nav-label' });
    dom.nextLabel = h('span', { class: 'nav-label' });
    dom.back = h('button', { class: 'btn nav-back', onclick: prev }, '← Back', dom.backLabel);
    dom.next = h('button', { class: 'btn nav-next primary', onclick: next }, dom.nextLabel, 'Next →');
    var footer = h('footer', { class: 'navbar' }, dom.back,
      h('span', { class: 'keys-hint', html: '<kbd>←</kbd><kbd>→</kbd> slides · <kbd>↓</kbd><kbd>↑</kbd> zoom' }), dom.next);
    app.appendChild(header);
    app.appendChild(h('div', { class: 'layout' }, dom.toc, h('div', { class: 'main-col' }, dom.stage, footer)));
    app.appendChild(h('div', { class: 'toc-scrim', onclick: closeTocMobile }));

    document.addEventListener('keydown', function (e) {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (t && t.isContentEditable)) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); zoomIn(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); zoomOut(); }
    });
    // delegated cross-links: <a data-goto="section/slide">, <x data-zoom="2">
    app.addEventListener('click', function (e) {
      var a = e.target.closest && e.target.closest('[data-goto],[data-zoom]');
      if (!a || !app.contains(a)) return;
      e.preventDefault();
      if (a.hasAttribute('data-goto')) go(a.getAttribute('data-goto'));
      else show(state.idx, parseInt(a.getAttribute('data-zoom'), 10) || 0, { keepScroll: true });
    });
  }

  window.addEventListener('error', function (e) { Course.errors.push('window.onerror: ' + (e.message || e)); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else setTimeout(boot, 0);
})();
