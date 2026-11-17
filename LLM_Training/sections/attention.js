/* Section: attention — THE core mechanism. Every number in the worked example is computed at runtime. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- toy weights (illustrative)
  // d_model = 4, 2 heads, d_head = 2. Head 1 uses columns 0–1 of Q/K/V, head 2 uses columns 2–3.
  var WQ = [[1.0, 0.5, -0.5, 1.0], [-0.5, 1.0, 1.0, 0.0], [1.5, -0.5, 0.5, 1.5], [0.5, 1.0, -1.0, 0.5]];
  var WK = [[1.5, 0.5, 0.0, 1.0], [0.0, 1.0, 1.5, -0.5], [0.5, -1.0, -0.5, 1.0], [1.0, 0.5, 1.0, 0.5]];
  var WV = [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [0.5, 0.5, 1.0, 0.0], [-0.5, 1.0, 0.0, 1.0]];
  var WO = [[0.5, 1.0, 0.0, -0.5], [1.0, 0.0, 0.5, 0.5], [0.0, 0.5, 1.0, 0.0], [-0.5, 0.5, 0.0, 1.0]];
  var DH = 2;

  function cols(A, a, b) { return A.map(function (r) { return r.slice(a, b); }); }

  // Full toy forward pass of one attention layer.
  function compute(api) {
    var M = api.math, X = api.ex.toyX, n = X.length;
    var Q = M.matmul(X, WQ), K = M.matmul(X, WK), V = M.matmul(X, WV);
    var heads = [0, 1].map(function (h) {
      var q = cols(Q, h * DH, h * DH + DH), k = cols(K, h * DH, h * DH + DH), v = cols(V, h * DH, h * DH + DH);
      var S = M.matmul(q, M.transpose(k));
      var Ss = M.scale(S, 1 / Math.sqrt(DH));
      var mask = S.map(function (r, i) { return r.map(function (_, j) { return j > i ? -Infinity : 0; }); });
      var Sm = Ss.map(function (r, i) { return r.map(function (x, j) { return j > i ? -Infinity : x; }); });
      var A = Sm.map(function (r) { return M.softmax(r); });
      var O = M.matmul(A, v);
      return { q: q, k: k, v: v, S: S, Ss: Ss, mask: mask, Sm: Sm, A: A, O: O };
    });
    var concat = heads[0].O.map(function (r, i) { return r.concat(heads[1].O[i]); });
    var out = M.matmul(concat, WO);
    return { X: X, n: n, Q: Q, K: K, V: V, heads: heads, concat: concat, out: out };
  }

  function tokLabels(api) { return api.ex.tokens.map(function (t) { return api.tokText(t.text); }); }
  function dimLabels(n, p) { var r = []; for (var i = 0; i < n; i++) r.push((p || 'd') + i); return r; }
  function headCols() { return ['h1·0', 'h1·1', 'h2·0', 'h2·1']; }
  function f(api, x, d) { return api.fmt(x, d == null ? 2 : d); }
  function paren(api, x) { var s = f(api, x); return x < 0 ? '(' + s + ')' : s; }

  // Small helper: a row of elements that wraps.
  function row(api, el) { var r = api.h('div', { class: 'row attention-mxrow' }); el.appendChild(r); return r; }
  function formula(api, el, html) { return api.html(el, '<div class="formula">' + html + '</div>'); }

  // Illustrative attention patterns for the interactive view (raw scores → causal softmax).
  function patternFrom(api, raw) {
    return raw.map(function (r, i) { return api.math.softmax(r.map(function (x, j) { return j > i ? -Infinity : x; })); });
  }
  var PATTERNS = {
    noun: { name: 'Head A: "which noun / subject?"', raw: [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0.5, 3, 0, 0, 0], [0, 1.5, 2.5, 0.5, 0], [0.3, 2.6, 1.2, 1.6, 0.4]],
      note: 'Content-based: " the" looks back for the thing being talked about (" cat") and the relation (" on").' },
    prev: { name: 'Head B: previous token', raw: [[0, 0, 0, 0, 0], [3, 0.5, 0, 0, 0], [0, 3, 0.5, 0, 0], [0, 0, 3, 0.5, 0], [0, 0, 0, 3, 0.5]],
      note: 'Position-based: each token mostly copies information from the token right before it.' },
    sink: { name: 'Head C: first-token "sink"', raw: [[0, 0, 0, 0, 0], [2.5, 0.5, 0, 0, 0], [2.5, 0.3, 0.3, 0, 0], [2.5, 0.2, 0.2, 0.4, 0], [2.5, 0.4, 0.2, 0.3, 0.6]],
      note: 'Many trained heads park unused attention on the first token when they have "nothing to do" (often called an attention sink).' }
  };

  // ---------------------------------------------------------------- SVG arcs view
  function arcView(api, el, opts) {
    var toks = api.ex.tokens, n = toks.length, sel = opts.start == null ? n - 1 : opts.start;
    var wrap = api.h('div', { class: 'attention-arcs' });
    el.appendChild(wrap);
    var chips = api.h('div', { class: 'attention-arcchips' });
    var svgBox = api.h('div', { class: 'attention-arcsvg' });
    var barsBox = api.h('div', { class: 'attention-arcbars' });
    wrap.appendChild(chips); wrap.appendChild(svgBox); wrap.appendChild(barsBox);
    var barCtl = null;
    function draw() {
      var A = opts.getA();
      chips.innerHTML = '';
      toks.forEach(function (t, i) {
        chips.appendChild(api.h('button', { class: 'btn attention-tokbtn' + (i === sel ? ' on' : ''), html: api.tokText(t.text),
          title: 'Show where "' + t.text + '" attends', onclick: function () { sel = i; draw(); } }));
      });
      var W = 520, H = 170, x0 = 52, dx = (W - 2 * x0) / (n - 1), base = 128;
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="attention-svg" role="img" aria-label="attention arcs">';
      for (var j = 0; j < n; j++) {
        var w = j <= sel ? A[sel][j] : 0;
        if (j <= sel && j !== sel) {
          var xa = x0 + sel * dx, xb = x0 + j * dx, mid = (xa + xb) / 2, hgt = 2 * (12 + Math.abs(xa - xb) / ((n - 1) * dx) * 80);
          s += '<path d="M' + xa + ' ' + (base - 14) + ' Q' + mid + ' ' + (base - 14 - hgt) + ' ' + xb + ' ' + (base - 14) + '" fill="none" stroke="#ff9a4d" stroke-linecap="round" stroke-width="' + (1 + 13 * w).toFixed(1) + '" opacity="' + (0.25 + 0.75 * w).toFixed(2) + '"/>';
        }
        if (j === sel && w > 0) {
          var cx = x0 + j * dx;
          s += '<circle cx="' + cx + '" cy="' + (base - 30) + '" r="' + (4 + 12 * w).toFixed(1) + '" fill="none" stroke="#ff9a4d" stroke-width="2.5" opacity="' + (0.3 + 0.7 * w).toFixed(2) + '"/>';
        }
      }
      for (var k = 0; k < n; k++) {
        var cxk = x0 + k * dx, on = k === sel, fut = k > sel;
        s += '<rect x="' + (cxk - 40) + '" y="' + (base - 12) + '" width="80" height="26" rx="6" class="attention-svgtok' + (on ? ' on' : '') + (fut ? ' fut' : '') + '"/>';
        s += '<text x="' + cxk + '" y="' + (base + 6) + '" text-anchor="middle" class="attention-svgtxt' + (fut ? ' fut' : '') + '">' + api.tokText(toks[k].text) + '</text>';
        s += '<text x="' + cxk + '" y="' + (base + 32) + '" text-anchor="middle" class="attention-svgw">' + (k <= sel ? (A[sel][k] * 100).toFixed(0) + '%' : 'masked') + '</text>';
      }
      s += '</svg>';
      svgBox.innerHTML = s;
      var items = toks.map(function (t, j) {
        return { label: api.tokText(t.text), value: j <= sel ? A[sel][j] : 0, highlight: j <= sel && A[sel][j] === Math.max.apply(null, A[sel].slice(0, sel + 1)), dim: j > sel, note: j > sel ? 'future' : '' };
      });
      if (!barCtl) barCtl = api.bars(barsBox, { items: items, max: 1, title: 'Attention weights of ' + api.tok(toks[sel].text) + ' ' + api.badge('probability', '[5]'), format: function (v) { return api.fmt(v, 2); } });
      else { barsBox.querySelector('.bars-title').innerHTML = 'Attention weights of ' + api.tok(toks[sel].text) + ' ' + api.badge('probability', '[5]'); barCtl.update(items); }
      if (opts.onDraw) opts.onDraw(sel);
    }
    draw();
    return { redraw: draw, select: function (i) { sel = i; draw(); } };
  }

  // ---------------------------------------------------------------- the worked-example steps (per head)
  function headSteps(api, C, h) {
    var H = C.heads[h], L = tokLabels(api), hl = { row: 4 }, c0 = h * DH;
    var hc = ['h' + (h + 1) + '·0', 'h' + (h + 1) + '·1'];
    return [
      { title: 'Input X — one row per token', render: function (b) {
        formula(api, b, 'X ' + api.badge('matrix', '[seq_len × d_model] = [5 × 4]'));
        api.matrix(b, { data: C.X, rowLabels: L, colLabels: dimLabels(4), title: 'X', type: 'matrix', shape: '[5 × 4]', highlight: hl });
        api.html(b, '<p class="muted">Each row is one token\'s embedding + position (' + api.link('embeddings/input-matrix', 'the input matrix') + '). The highlighted row is ' + api.tok(' the') + ' — the position whose output must predict ' + api.tok(' mat') + '.</p>');
      } },
      { title: 'Project: Q = X·W_q, K = X·W_k, V = X·W_v', render: function (b) {
        formula(api, b, '[5 × 4] · [4 × 4] → [5 × 4]   (three times, three different learned matrices)');
        var r = row(api, b);
        api.matrix(r, { data: C.Q, rowLabels: L, colLabels: headCols(), title: 'Q', type: 'matrix', shape: '[5 × 4]', small: true, highlight: hl });
        api.matrix(r, { data: C.K, rowLabels: L, colLabels: headCols(), title: 'K', type: 'matrix', shape: '[5 × 4]', small: true });
        api.matrix(r, { data: C.V, rowLabels: L, colLabels: headCols(), title: 'V', type: 'matrix', shape: '[5 × 4]', small: true });
      } },
      { title: 'Split into heads → take head ' + (h + 1) + ' (columns ' + c0 + '–' + (c0 + 1) + ')', render: function (b) {
        formula(api, b, 'Q, K, V ' + api.shape('[5 × 4]') + ' → reshape → ' + api.badge('tensor', '[2 heads × 5 × 2]') + ' ; head ' + (h + 1) + ' slice: ' + api.shape('[5 × 2]'));
        var r = row(api, b);
        api.matrix(r, { data: H.q, rowLabels: L, colLabels: hc, title: 'Q<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true, highlight: hl });
        api.matrix(r, { data: H.k, rowLabels: L, colLabels: hc, title: 'K<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true });
        api.matrix(r, { data: H.v, rowLabels: L, colLabels: hc, title: 'V<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true });
      } },
      { title: 'Scores S = Q·Kᵀ — every query dotted with every key', render: function (b) {
        formula(api, b, 'S = Q<sub>' + (h + 1) + '</sub> · K<sub>' + (h + 1) + '</sub>ᵀ : [5 × 2] · [2 × 5] → ' + api.badge('matrix', '[5 × 5]'));
        api.matrix(b, { data: H.S, rowLabels: L.map(function (x) { return 'q ' + x; }), colLabels: L.map(function (x) { return 'k ' + x; }), title: 'S (raw scores)', type: 'float', shape: '[5 × 5]', highlight: hl, scale: 'diverging' });
        api.html(b, '<p class="muted">Row = the asking token (query), column = the token being looked at (key). Bigger = "more relevant". Example: S[␣the, ␣cat] = ' +
          paren(api, H.q[4][0]) + '×' + paren(api, H.k[1][0]) + ' + ' + paren(api, H.q[4][1]) + '×' + paren(api, H.k[1][1]) + ' = <b>' + f(api, H.S[4][1]) + '</b>.</p>');
      } },
      { title: 'Scale: divide by √d_head = √2 ≈ ' + f(api, Math.sqrt(DH), 3), render: function (b) {
        formula(api, b, 'S / √d_head → ' + api.badge('matrix', '[5 × 5]') + '   keeps the scores\' spread ≈ 1 regardless of d_head (' + api.link('attention/softmax-scaling', 'why?') + ')');
        api.matrix(b, { data: H.Ss, rowLabels: L, colLabels: L, title: 'S / √2', type: 'float', shape: '[5 × 5]', highlight: hl, scale: 'diverging' });
      } },
      { title: 'Causal mask — no peeking at future tokens', render: function (b) {
        formula(api, b, 'S\' = S/√2 + M,  M[i][j] = 0 if j ≤ i else −∞   ' + api.badge('mask', '[5 × 5]'));
        var r = row(api, b);
        api.matrix(r, { data: H.mask.map(function (rr) { return rr.map(function () { return 0; }); }), causal: true, rowLabels: L, colLabels: L, title: 'M', type: 'mask', shape: '[5 × 5]', small: true, heat: false, decimals: 0 });
        api.matrix(r, { data: H.Sm, causal: true, rowLabels: L, colLabels: L, title: 'S\'', type: 'float', shape: '[5 × 5]', small: true, highlight: hl, scale: 'diverging' });
        api.html(b, '<p class="muted">Row ' + api.tok(' sat') + ' may only see The, ␣cat, ␣sat. See ' + api.link('attention/causal-mask', 'the causal mask') + '.</p>');
      } },
      { title: 'Softmax each row → attention weights A', render: function (b) {
        formula(api, b, 'A[i] = softmax(S\'[i])   ' + api.badge('probability', '[5 × 5]') + '   e<sup>−∞</sup> = 0, each row sums to 1');
        var withSum = H.A.map(function (r) { return r.concat([api.math.sum(r)]); });
        api.matrix(b, { data: withSum, causal: false, mask: H.A.map(function (r, i) { return r.map(function (_, j) { return j > i && j < 5; }).concat([false]); }), maskText: '0', rowLabels: L, colLabels: L.concat(['Σ']), title: 'A', type: 'probability', shape: '[5 × 5]', highlight: hl, scale: 'seq' });
        var top = api.math.argmax(H.A[4]);
        api.html(b, '<p class="muted">' + api.tok(' the') + ' puts its largest weight (' + f(api, H.A[4][top]) + ') on ' + api.tok(api.ex.tokens[top].text) + ' in this head.</p>');
      } },
      { title: 'Mix the values: O = A · V', render: function (b) {
        formula(api, b, 'O<sub>' + (h + 1) + '</sub> = A · V<sub>' + (h + 1) + '</sub> : [5 × 5] · [5 × 2] → ' + api.badge('matrix', '[5 × 2]'));
        var r = row(api, b);
        api.matrix(r, { data: H.A, causal: true, maskText: '0', rowLabels: L, colLabels: L, title: 'A', type: 'probability', shape: '[5 × 5]', small: true, highlight: hl, scale: 'seq' });
        api.matrix(r, { data: H.v, rowLabels: L, colLabels: hc, title: 'V<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true });
        api.matrix(r, { data: H.O, rowLabels: L, colLabels: hc, title: 'O<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true, highlight: hl });
        api.html(b, '<p class="muted">Row ' + api.tok(' the') + ' of O = ' + H.A[4].map(function (a, j) { return f(api, a) + '·v<sub>' + api.tokText(api.ex.tokens[j].text) + '</sub>'; }).join(' + ') +
          ' = [' + H.O[4].map(function (x) { return f(api, x); }).join(', ') + ']. ' + api.link('attention/multi-head', 'Next: concat heads → W_o') + '.</p>');
      } }
    ];
  }

  // ---------------------------------------------------------------- registration
  Course.register({
    id: 'attention',
    part: 'I',
    order: 50,
    title: 'Attention',
    slides: [
      // ============================================================ 1. Q/K/V intuition
      {
        id: 'qkv-intuition',
        title: 'Query, Key, Value: how tokens look at each other',
        critical: {
          understand: [
            'Attention is the <b>only</b> place where tokens exchange information; everything else works on one token at a time.',
            'Each token asks a <b>query</b> ("what am I looking for?"), advertises a <b>key</b> ("what do I contain?") and offers a <b>value</b> ("what I hand over if picked").',
            'Its output is a <b>weighted average of earlier tokens\' values</b>, weighted by how well its query matches their keys.'
          ],
          details: [
            'Query–key match = dot product q·k; weights = softmax over all visible keys, so they sum to 1.',
            'The patterns shown are illustrative; real heads are learned and many are hard to interpret.',
            'Each layer has many heads (32 in the reference model), each with its own pattern.'
          ]
        },
        dataTypes: [
          { name: 'prompt tokens', type: 'token', shape: '[5]', example: 'The ␣cat ␣sat ␣on ␣the' },
          { name: 'q, k, v (per token, per head)', type: 'vector', shape: '[d_head]', example: '[2] toy, [128] reference' },
          { name: 'score q·k', type: 'float', shape: 'scalar', example: '1.32' },
          { name: 'attention weights of one token', type: 'probability', shape: '[seq_len]', example: 'sums to 1' },
          { name: 'attention output (per token)', type: 'vector', shape: '[d_head]', example: 'weighted mix of v' }
        ],
        levels: [
          { name: 'Analogy', render: function (el, api) {
            api.html(el, '<p>Think of each token as a person at a meeting holding three cards. To update its own understanding, a token compares its ' +
              '<b class="attention-q">query</b> card with every earlier token\'s <b class="attention-k">key</b> card and collects a blend of their <b class="attention-v">value</b> cards — ' +
              'more from the ones that match best.</p>');
            api.blocks(el, { rows: [[
              { id: 'x', label: 'token vector x', sub: 'e.g. ' + api.tok(' the'), type: 'vector', shape: '[d_model]' },
              { id: 'q', label: '<span class="attention-q">Query q</span>', sub: '"what am I looking for?"', type: 'vector', shape: '[d_head]', edge: '·W_q', zoom: 1 },
              { id: 'k', label: '<span class="attention-k">Key k</span>', sub: '"what do I contain?"', type: 'vector', shape: '[d_head]', edge: '·W_k', zoom: 1 },
              { id: 'v', label: '<span class="attention-v">Value v</span>', sub: '"what I pass on"', type: 'vector', shape: '[d_head]', edge: '·W_v', zoom: 1 }
            ], [
              { id: 'm', label: 'match q·k with every earlier token', sub: '→ softmax → weights', type: ['float', 'probability'], shape: '[seq_len]', zoom: 2 },
              { id: 'o', label: 'output = Σ weight · v', sub: 'information pulled in from context', type: 'vector', shape: '[d_head]', goto: 'attention/worked-example' }
            ]], caption: 'Q, K and V are three different linear views of the same token vector (' + api.link('attention/projections', 'projections') + ').' });
            var g = api.grid(el, 200);
            api.card(g, '<span class="attention-q">Query</span>', api.tok(' the') + ' asks: "which noun am I the article for, and where?"');
            api.card(g, '<span class="attention-k">Key</span>', api.tok(' cat') + ' advertises: "I\'m a noun, the subject."');
            api.card(g, '<span class="attention-v">Value</span>', api.tok(' cat') + ' hands over: "animal, singular, sits on things…"');
            api.callout(el, 'Analogy only: real queries/keys are vectors of numbers, and what they "mean" is learned, not designed.', 'hedge');
          } },
          { name: 'Interactive view', render: function (el, api) {
            api.html(el, '<p>Click a token to see <b>where it looks</b>. Arc thickness = attention weight. Tokens to the right are invisible to it (' + api.link('attention/causal-mask', 'causal mask') + ').</p>');
            var key = 'noun', note;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            api.select(ctl, { label: 'Head', value: key, options: [
              { value: 'noun', label: PATTERNS.noun.name }, { value: 'prev', label: PATTERNS.prev.name }, { value: 'sink', label: PATTERNS.sink.name },
              { value: 'toy1', label: 'Toy head 1 (computed, worked example)' }, { value: 'toy2', label: 'Toy head 2 (computed, worked example)' }],
              onChange: function (v) { key = v; view.redraw(); } });
            var C = compute(api);
            var view = arcView(api, el, {
              getA: function () { return key === 'toy1' ? C.heads[0].A : key === 'toy2' ? C.heads[1].A : patternFrom(api, PATTERNS[key].raw); },
              onDraw: function () { if (note) note.innerHTML = key.indexOf('toy') === 0 ? 'Computed from toy X and toy weights — see the ' + api.link('attention/worked-example', 'full worked example') + '.' : PATTERNS[key].note; }
            });
            note = api.html(el, '').appendChild(api.h('div', { class: 'callout callout-info attention-note' }));
            view.redraw();
            api.illustrative(el, 'Heads A–C are hand-made patterns that resemble kinds of heads reported in interpretability research (previous-token heads, attention sinks, content lookups).');
          } },
          { name: 'One score', render: function (el, api) {
            var C = compute(api), H = C.heads[0], L = tokLabels(api);
            api.html(el, '<p>Zoom into toy head 1 and the query of ' + api.tok(' the') + '. Its score against each key is a plain dot product of two ' + api.badge('vector', '[d_head = 2]') + 's:</p>');
            var rows = [];
            for (var j = 0; j < 5; j++) {
              rows.push([L[j], '[' + H.k[j].map(function (x) { return f(api, x); }).join(', ') + ']',
                paren(api, H.q[4][0]) + '×' + paren(api, H.k[j][0]) + ' + ' + paren(api, H.q[4][1]) + '×' + paren(api, H.k[j][1]),
                '<b>' + f(api, H.S[4][j]) + '</b>', f(api, H.Ss[4][j]), '<b>' + f(api, H.A[4][j]) + '</b>']);
            }
            formula(api, el, 'q<sub>␣the</sub> = [' + H.q[4].map(function (x) { return f(api, x); }).join(', ') + ']  ' + api.badge('vector', '[2]'));
            api.table(el, { headers: ['key token', 'k ' + api.shape('[2]'), 'q·k worked', 'score ' + api.type('float'), '÷√2', 'softmax ' + api.type('probability')], rows: rows });
            api.html(el, '<p class="muted">The softmax column sums to ' + f(api, api.math.sum(H.A[4])) + '. Every number comes from ' + api.link('attention/worked-example', 'the full worked example') + ', where ' + api.tok(' the') + ' can see all 5 tokens because it is the last one.</p>');
          } }
        ]
      },

      // ============================================================ 2. projections
      {
        id: 'projections',
        title: 'Projections: X → Q, K, V and the split into heads',
        critical: {
          understand: [
            'Q, K and V are produced by multiplying the same input X by three learned weight matrices.',
            'The d_model columns are then split into heads: each head gets its own d_head-sized slice.',
            'It is one big matrix multiply per projection — all tokens at once, on the GPU.'
          ],
          details: [
            'Toy: X[5×4]·W_q[4×4] = Q[5×4] → 2 heads × [5×2]. Reference: X[n×4096]·W_q[4096×4096] → 32 heads × [n×128].',
            'With GQA, W_k and W_v are smaller: [4096 × 1024] for 8 KV heads (' + '<a href="#" class="xlink" data-goto="attention/gqa-cost">GQA</a>' + ').',
            'In practice the input is RMSNorm(x), not x itself (pre-norm; see ' + '<a href="#" class="xlink" data-goto="transformer-block/decoder-only">the block</a>' + ').'
          ]
        },
        dataTypes: [
          { name: 'X', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'W_q, W_k, W_v', type: 'matrix', shape: '[d_model × d_model]', example: '[4 × 4] (toy)' },
          { name: 'Q, K, V', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'Q per head', type: 'tensor', shape: '[n_heads × seq_len × d_head]', example: '[2 × 5 × 2]' },
          { name: 'one cell Q[i][j]', type: 'float', shape: 'scalar', example: 'dot(row i of X, column j of W_q)' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [[
              { id: 'x', label: 'X', sub: 'normed token vectors', type: 'matrix', shape: '[5 × 4]', goto: 'embeddings/input-matrix' },
              { id: 'q', label: 'Q = X·W_q', type: 'matrix', shape: '[5 × 4]', edge: 'W_q [4 × 4]', zoom: 1 },
              { id: 'k', label: 'K = X·W_k', type: 'matrix', shape: '[5 × 4]', edge: 'W_k', zoom: 1 },
              { id: 'v', label: 'V = X·W_v', type: 'matrix', shape: '[5 × 4]', edge: 'W_v', zoom: 1 }
            ], [
              { id: 'split', label: 'split columns into heads', sub: 'head 1: cols 0–1 · head 2: cols 2–3', type: 'tensor', shape: '[2 × 5 × 2]', zoom: 2 },
              { id: 'att', label: 'attention per head', type: 'matrix', shape: '[5 × 2] each', goto: 'attention/worked-example' }
            ]] });
            api.table(el, { headers: ['', 'toy', 'reference (Llama-3-8B-class)'], rows: [
              ['X', '[5 × 4]', '[n × 4,096]'],
              ['W_q', '[4 × 4]', '[4,096 × 4,096] = ' + api.fmtInt(api.ref.d_model * api.ref.n_heads * api.ref.d_head) + ' params'],
              ['W_k, W_v', '[4 × 4]', '[4,096 × 1,024] each (8 KV heads × 128)'],
              ['heads × d_head', '2 × 2', api.ref.n_heads + ' × ' + api.ref.d_head + ' = ' + api.fmtInt(api.ref.n_heads * api.ref.d_head)]
            ] });
          } },
          { name: 'Matrix multiply', render: function (el, api) {
            var C = compute(api), L = tokLabels(api), which = 'Q';
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            function draw() {
              box.innerHTML = '';
              var W = which === 'Q' ? WQ : which === 'K' ? WK : WV, R = C[which];
              formula(api, box, which + ' = X · W_' + which.toLowerCase() + '  :  ' + api.shape('[5 × 4]') + ' · ' + api.shape('[4 × 4]') + ' → ' + api.shape('[5 × 4]'));
              var r = row(api, box);
              api.matrix(r, { data: C.X, rowLabels: L, colLabels: dimLabels(4), title: 'X', type: 'matrix', shape: '[5 × 4]', small: true });
              r.appendChild(api.h('div', { class: 'attention-op', text: '·' }));
              api.matrix(r, { data: W, rowLabels: dimLabels(4), colLabels: headCols(), title: 'W_' + which.toLowerCase(), type: 'matrix', shape: '[4 × 4]', small: true });
              r.appendChild(api.h('div', { class: 'attention-op', text: '=' }));
              api.matrix(r, { data: R, rowLabels: L, colLabels: headCols(), title: which, type: 'matrix', shape: '[5 × 4]', small: true });
            }
            api.select(ctl, { label: 'Projection', value: 'Q', options: [{ value: 'Q', label: 'Q (queries)' }, { value: 'K', label: 'K (keys)' }, { value: 'V', label: 'V (values)' }], onChange: function (v) { which = v; draw(); } });
            draw();
            api.illustrative(el, 'X is the shared toy input (api.ex.toyX); W_q, W_k, W_v are made-up toy weights used throughout this section.');
          } },
          { name: 'Split heads', render: function (el, api) {
            var C = compute(api), L = tokLabels(api);
            api.html(el, '<p>No new math here — just bookkeeping. The 4 columns of Q are cut into 2 groups of d_head = 2. Each head runs attention independently on its slice.</p>');
            formula(api, el, 'Q ' + api.shape('[5 × 4]') + ' → reshape ' + api.shape('[5 × 2 × 2]') + ' → transpose → ' + api.badge('tensor', '[2 heads × 5 × 2]'));
            var r = row(api, el);
            api.matrix(r, { data: C.Q, rowLabels: L, colLabels: headCols(), title: 'Q', type: 'matrix', shape: '[5 × 4]', small: true });
            r.appendChild(api.h('div', { class: 'attention-op', text: '→' }));
            var hb = api.h('div', { class: 'attention-headstack' }); r.appendChild(hb);
            api.matrix(hb, { data: cols(C.Q, 0, 2), rowLabels: L, colLabels: ['0', '1'], title: 'head 1', type: 'matrix', shape: '[5 × 2]', small: true });
            api.matrix(hb, { data: cols(C.Q, 2, 4), rowLabels: L, colLabels: ['0', '1'], title: 'head 2', type: 'matrix', shape: '[5 × 2]', small: true });
            api.callout(el, 'Reference model: Q is [n × 4,096] → <b>32 heads × [n × 128]</b>; K and V are [n × 1,024] → 8 KV heads × [n × 128], each shared by 4 query heads (' + api.link('attention/gqa-cost', 'GQA') + ').', 'info');
          } },
          { name: 'One cell', render: function (el, api) {
            var C = compute(api), L = tokLabels(api), cell = [4, 0];
            api.html(el, '<p>Click any cell of Q. Each cell is a dot product: a row of X (one token) with a column of W_q (one learned "question direction").</p>');
            var r = row(api, el), info = api.h('div', { class: 'attention-cellcalc' });
            var mxQ = api.matrix(r, { data: C.Q, rowLabels: L, colLabels: headCols(), title: 'Q — click a cell', type: 'matrix', shape: '[5 × 4]', small: true, highlight: { cell: cell }, onCell: pick });
            el.appendChild(info);
            function pick(i, j) {
              cell = [i, j]; mxQ.update({ highlight: { cell: cell } });
              var xr = C.X[i], wc = WQ.map(function (rr) { return rr[j]; });
              info.innerHTML = '';
              formula(api, info, 'Q[' + L[i] + '][' + headCols()[j] + '] = ' + xr.map(function (x, k) { return paren(api, x) + '×' + paren(api, wc[k]); }).join(' + ') +
                ' = <b>' + f(api, C.Q[i][j]) + '</b> ' + api.type('float'));
              api.html(info, '<p class="muted">row X[' + L[i] + '] ' + api.badge('vector', '[4]') + ' · column ' + j + ' of W_q ' + api.badge('vector', '[4]') + '. Reference model: the same, with 4,096 terms per cell.</p>');
            }
            pick(4, 0);
          } }
        ]
      },

      // ============================================================ 3. worked example
      {
        id: 'worked-example',
        title: 'Full worked example: X → softmax(QKᵀ/√d)·V',
        critical: {
          understand: [
            'Attention(Q,K,V) = softmax(QKᵀ/√d_head + M)·V, computed independently for every head.',
            'Each token\'s output row is a <b>weighted average of the value rows</b> of itself and earlier tokens.',
            'The last row (' + '<span class="tok">␣the</span>' + ') is the one whose final hidden state is used to predict <span class="tok">␣mat</span>.'
          ],
          details: [
            'Scores are divided by √d_head before softmax to keep their variance ≈ 1.',
            'The causal mask sets future positions to −∞ so softmax gives them exactly 0 weight.',
            'Heads are concatenated back to [seq_len × d_model] and mixed by W_o; then the result is added to the residual stream.',
            'All values are computed live from api.ex.toyX and illustrative toy weights.'
          ]
        },
        dataTypes: [
          { name: 'X', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'W_q, W_k, W_v, W_o', type: 'matrix', shape: '[d_model × d_model]', example: '[4 × 4]' },
          { name: 'Q, K, V per head', type: 'tensor', shape: '[n_heads × seq_len × d_head]', example: '[2 × 5 × 2]' },
          { name: 'scores S', type: 'matrix', shape: '[seq_len × seq_len]', example: '[5 × 5] per head' },
          { name: 'causal mask M', type: 'mask', shape: '[seq_len × seq_len]', example: '0 / −∞' },
          { name: 'attention weights A', type: 'probability', shape: '[seq_len × seq_len]', example: 'row sums = 1' },
          { name: 'head output O_h', type: 'matrix', shape: '[seq_len × d_head]', example: '[5 × 2]' },
          { name: 'attention output', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' }
        ],
        levels: [
          { name: 'Pipeline', render: function (el, api) {
            var diag = api.blocks(el, { rows: [[
              { id: 'x', label: 'X', sub: '5 tokens', type: 'matrix', shape: '[5 × 4]', goto: 'embeddings/input-matrix' },
              { id: 'qkv', label: 'Q, K, V', sub: 'X·W_q, X·W_k, X·W_v', type: 'matrix', shape: '[5 × 4] ×3', edge: 'project', goto: 'attention/projections' },
              { id: 'split', label: 'split heads', type: 'tensor', shape: '[2 × 5 × 2]', zoom: 1 }
            ], [
              { id: 's', label: 'QKᵀ', sub: 'scores', type: 'float', shape: '[5 × 5]', zoom: 1 },
              { id: 'sc', label: '÷ √d_head', sub: '√2', type: 'float', shape: '[5 × 5]', goto: 'attention/softmax-scaling' },
              { id: 'm', label: '+ causal mask', type: 'mask', shape: '[5 × 5]', goto: 'attention/causal-mask' },
              { id: 'sm', label: 'softmax', sub: 'per row', type: 'probability', shape: '[5 × 5]', zoom: 1 }
            ], [
              { id: 'av', label: 'A · V', type: 'matrix', shape: '[5 × 2] per head', zoom: 1 },
              { id: 'cat', label: 'concat heads', type: 'matrix', shape: '[5 × 4]', zoom: 2 },
              { id: 'wo', label: '· W_o', type: 'matrix', shape: '[5 × 4]', zoom: 2, edge: 'mix heads' },
              { id: 'res', label: '+ residual', sub: 'back onto the stream', type: 'matrix', shape: '[5 × 4]', goto: 'transformer-block/residual-stream' }
            ]], caption: 'Click a block to dive in. Level 2 runs every step with real numbers; level 4 does the ␣the row by hand.' });
            var names = ['x', 'qkv', 'split', 's', 'sc', 'm', 'sm', 'av', 'cat', 'wo', 'res'], k = 0;
            api.every(900, function () { diag.highlight(names[k % names.length]); k++; });
          } },
          { name: 'Per head, step by step', render: function (el, api) {
            var C = compute(api), h = 0;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            var cur = 0;
            function build() {
              box.innerHTML = '';
              api.stepper(box, { steps: headSteps(api, C, h), interval: 2600, start: cur, onStep: function (i) { cur = i; } });
            }
            api.select(ctl, { label: 'Head', value: '0', options: [{ value: '0', label: 'Head 1 (Q/K/V cols 0–1)' }, { value: '1', label: 'Head 2 (Q/K/V cols 2–3)' }], onChange: function (v) { h = +v; build(); } });
            ctl.appendChild(api.h('span', { class: 'muted', html: 'Highlighted row = ' + api.tok(' the') + ' (predicts ' + api.tok(' mat') + ')' }));
            build();
            api.illustrative(el, 'X = api.ex.toyX; toy weights are made up. Every matrix is computed in your browser from them.');
          } },
          { name: 'Concat & W_o', render: function (el, api) {
            var C = compute(api), L = tokLabels(api), hl = { row: 4 };
            api.stepper(el, { interval: 2600, steps: [
              { title: 'Two head outputs', render: function (b) {
                formula(api, b, 'O<sub>1</sub>, O<sub>2</sub> : ' + api.badge('matrix', '[5 × 2]') + ' each');
                var r = row(api, b);
                api.matrix(r, { data: C.heads[0].O, rowLabels: L, colLabels: ['h1·0', 'h1·1'], title: 'O<sub>1</sub>', type: 'matrix', shape: '[5 × 2]', small: true, highlight: hl });
                api.matrix(r, { data: C.heads[1].O, rowLabels: L, colLabels: ['h2·0', 'h2·1'], title: 'O<sub>2</sub>', type: 'matrix', shape: '[5 × 2]', small: true, highlight: hl });
              } },
              { title: 'Concatenate → [5 × 4]', render: function (b) {
                formula(api, b, 'concat(O<sub>1</sub>, O<sub>2</sub>) → ' + api.badge('matrix', '[5 × 4]') + '  (back to d_model columns)');
                api.matrix(b, { data: C.concat, rowLabels: L, colLabels: headCols(), title: 'concat', type: 'matrix', shape: '[5 × 4]', small: true, highlight: hl });
              } },
              { title: 'Mix with W_o → attention output', render: function (b) {
                formula(api, b, 'out = concat · W_o : [5 × 4] · [4 × 4] → ' + api.badge('matrix', '[5 × 4]'));
                var r = row(api, b);
                api.matrix(r, { data: C.concat, rowLabels: L, colLabels: headCols(), title: 'concat', type: 'matrix', shape: '[5 × 4]', small: true, highlight: hl });
                r.appendChild(api.h('div', { class: 'attention-op', text: '·' }));
                api.matrix(r, { data: WO, colLabels: dimLabels(4), title: 'W_o', type: 'matrix', shape: '[4 × 4]', small: true });
                r.appendChild(api.h('div', { class: 'attention-op', text: '=' }));
                api.matrix(r, { data: C.out, rowLabels: L, colLabels: dimLabels(4), title: 'out', type: 'matrix', shape: '[5 × 4]', small: true, highlight: hl });
              } },
              { title: 'The ␣the row — what predicts ␣mat', render: function (b) {
                var g = api.grid(b, 230);
                [0, 1].forEach(function (h) {
                  var c = api.card(g, 'Head ' + (h + 1) + ': where ' + api.tok(' the') + ' looks', '');
                  api.bars(c.querySelector('.card-body'), { items: api.ex.tokens.map(function (t, j) { return { label: api.tokText(t.text), value: C.heads[h].A[4][j] }; }), max: 1, format: function (v) { return api.fmt(v, 2); } });
                });
                formula(api, b, 'out[␣the] = [' + C.out[4].map(function (x) { return f(api, x); }).join(', ') + ']  ' + api.badge('vector', '[d_model = 4]'));
                api.html(b, '<p>This vector is <b>added</b> to ' + api.tok(' the') + '\'s residual stream (' + api.link('transformer-block/residual-stream', 'residual stream') + '), then the ' + api.link('ffn-moe/ffn-role', 'FFN') +
                  ' runs, and so on for all 32 layers. After the last layer, this position\'s vector goes through ' + api.link('output-sampling/unembedding', 'the unembedding') + ' to produce logits — where ' + api.tok(' mat') + ' should come out on top in a trained model.</p>');
              } }
            ] });
            api.illustrative(el);
          } },
          { name: '␣the by hand', render: function (el, api) {
            var C = compute(api), L = tokLabels(api), M = api.math, h = 0;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            function draw() {
              box.innerHTML = '';
              var H = C.heads[h], q = H.q[4], s2 = Math.sqrt(DH);
              var ex = H.Ss[4].map(function (x) { return Math.exp(x); }), Z = M.sum(ex);
              api.stepper(box, { interval: 2600, steps: [
                { title: 'Query of ␣the and the 5 keys (head ' + (h + 1) + ')', render: function (b) {
                  formula(api, b, 'q = [' + q.map(function (x) { return f(api, x); }).join(', ') + '] ' + api.badge('vector', '[2]'));
                  api.matrix(b, { data: H.k, rowLabels: L, colLabels: ['0', '1'], title: 'K<sub>' + (h + 1) + '</sub>', type: 'matrix', shape: '[5 × 2]', small: true });
                } },
                { title: 'Dot products q·k_j and ÷√2', render: function (b) {
                  api.table(b, { headers: ['j', 'q·k_j', '= score', '÷ ' + f(api, s2, 3)], rows: H.k.map(function (k, j) {
                    return [L[j], paren(api, q[0]) + '×' + paren(api, k[0]) + ' + ' + paren(api, q[1]) + '×' + paren(api, k[1]), f(api, H.S[4][j], 3), '<b>' + f(api, H.Ss[4][j], 3) + '</b>'];
                  }) });
                  api.html(b, '<p class="muted">␣the is the last position, so the causal mask hides nothing in this row.</p>');
                } },
                { title: 'Softmax: exponentiate, sum, divide', render: function (b) {
                  api.table(b, { headers: ['j', 'scaled score', 'e<sup>x</sup>', 'e<sup>x</sup> / Σ = weight'], rows: ex.map(function (e, j) {
                    return [L[j], f(api, H.Ss[4][j], 3), f(api, e, 3), '<b>' + f(api, e / Z, 3) + '</b>'];
                  }).concat([['Σ', '', f(api, Z, 3), f(api, 1, 3)]]) });
                } },
                { title: 'Weighted sum of value vectors', render: function (b) {
                  api.table(b, { headers: ['j', 'weight a_j', 'v_j', 'a_j · v_j'], rows: H.v.map(function (v, j) {
                    return [L[j], f(api, H.A[4][j], 3), '[' + v.map(function (x) { return f(api, x); }).join(', ') + ']', '[' + v.map(function (x) { return f(api, x * H.A[4][j], 3); }).join(', ') + ']'];
                  }).concat([['Σ', '1.000', '', '<b>[' + H.O[4].map(function (x) { return f(api, x, 3); }).join(', ') + ']</b>']]) });
                  formula(api, b, 'o<sub>␣the</sub> (head ' + (h + 1) + ') = [' + H.O[4].map(function (x) { return f(api, x, 3); }).join(', ') + ']  ' + api.badge('vector', '[d_head = 2]'));
                  api.html(b, '<p class="muted">Matches row ␣the of O<sub>' + (h + 1) + '</sub> in ' + '<a href="#" class="xlink" data-zoom="1">step-by-step</a>' + '. Next: concat with the other head and multiply by W_o.</p>');
                } }
              ] });
            }
            api.select(ctl, { label: 'Head', value: '0', options: [{ value: '0', label: 'Head 1' }, { value: '1', label: 'Head 2' }], onChange: function (v) { h = +v; draw(); } });
            draw();
          } }
        ]
      },

      // ============================================================ 4. softmax & scaling
      {
        id: 'softmax-scaling',
        title: 'Softmax and the √d scaling playground',
        critical: {
          understand: [
            'Softmax turns any list of scores into positive weights that sum to 1: e<sup>s</sup> / Σ e<sup>s</sup>.',
            'Only <b>differences</b> between scores matter; large differences → almost all weight on one token.',
            'Dot products of random d-dim vectors grow like √d, so without ÷√d big heads would produce near one-hot weights.'
          ],
          details: [
            'If q, k have unit-variance entries, q·k has variance d; dividing by √d brings it back to ≈ 1.',
            'Near one-hot softmax has tiny gradients, which tends to make training slow or unstable.',
            'Dividing by a number is the same knob as a temperature (see ' + '<a href="#" class="xlink" data-goto="output-sampling/sampling-playground">sampling</a>' + ').'
          ]
        },
        dataTypes: [
          { name: 'scores', type: 'float', shape: '[seq_len]', example: '[2.0, 1.0, 0.5, −1.0]' },
          { name: 'scale / temperature', type: 'float', shape: 'scalar', example: '√d_head' },
          { name: 'd_head', type: 'integer', shape: 'scalar', example: '2 … 512' },
          { name: 'softmax weights', type: 'probability', shape: '[seq_len]', example: 'sums to 1' },
          { name: 'softmax', type: 'function', shape: '[n] → [n]', example: 'e^x / Σ e^x' }
        ],
        levels: [
          { name: 'Softmax', render: function (el, api) {
            var toks = api.ex.tokens.slice(0, 4), s = [0.6, 2.2, 0.4, 1.0];
            api.html(el, '<p>Drag the raw scores of ' + api.tok(' the') + ' against four earlier tokens. Watch the weights ' + api.type('probability') + ' always sum to 1.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            var b = api.bars(out, { items: [], max: 1, title: 'softmax(scores) ' + api.badge('probability', '[4]') });
            var fx = api.h('div'); out.appendChild(fx);
            function upd() {
              var p = api.math.softmax(s), ex = s.map(Math.exp), Z = api.math.sum(ex);
              b.update(toks.map(function (t, i) { return { label: api.tokText(t.text), value: p[i], highlight: i === api.math.argmax(p) }; }));
              fx.innerHTML = '<div class="formula">Σ e<sup>s</sup> = ' + ex.map(function (e) { return f(api, e); }).join(' + ') + ' = ' + f(api, Z) + ' ; weights sum = ' + f(api, api.math.sum(p)) + '</div>';
            }
            toks.forEach(function (t, i) {
              api.slider(ctl, { label: 'score ' + api.tokText(t.text), min: -3, max: 5, step: 0.1, value: s[i], format: function (v) { return api.fmt(v, 1); }, onInput: function (v) { s[i] = v; upd(); } });
            });
            api.callout(el, 'Try adding the same amount to every score: nothing changes. Softmax only cares about differences.', 'tip');
          } },
          { name: 'Scale knob', render: function (el, api) {
            var s = [1.2, 3.0, 0.4, 2.0, 0.8], toks = api.ex.tokens, div = 1;
            api.html(el, '<p>Same scores, divided by a scale factor before softmax. Small divisor → peaky (sharp); large divisor → flat (blurry).</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var g = api.grid(el, 240);
            var c1 = api.card(g, 'scores / scale ' + api.badge('float', '[5]'), ''), c2 = api.card(g, 'softmax ' + api.badge('probability', '[5]'), '');
            var b1 = api.bars(c1.querySelector('.card-body'), { items: [], max: 6, format: function (v) { return api.fmt(v, 2); } });
            var b2 = api.bars(c2.querySelector('.card-body'), { items: [], max: 1, format: function (v) { return api.fmt(v, 2); } });
            var info = api.h('div'); el.appendChild(info);
            api.slider(ctl, { label: 'divide scores by', min: 0.25, max: 4, step: 0.05, value: 1, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) {
              div = v;
              var sc = s.map(function (x) { return x / div; }), p = api.math.softmax(sc), mx = api.math.max(p);
              b1.update(toks.map(function (t, i) { return { label: api.tokText(t.text), value: sc[i] }; }));
              b2.update(toks.map(function (t, i) { return { label: api.tokText(t.text), value: p[i], highlight: p[i] === mx }; }));
              info.innerHTML = '<div class="formula">max weight = ' + f(api, mx) + (mx > 0.9 ? '  → nearly one-hot: the head ignores everything else' : mx < 0.3 ? '  → nearly uniform: the head averages everything' : '') + '</div>';
            } });
          } },
          { name: 'Why ÷√d', render: function (el, api) {
            var M = api.math, NK = 8, TR = 40;
            function gauss(r) { var u = Math.max(r(), 1e-9), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
            function trial(d, seed) {
              var r = M.rng(seed), q = [], ks = [], i, j;
              for (i = 0; i < d; i++) q.push(gauss(r));
              for (j = 0; j < NK; j++) { var k = []; for (i = 0; i < d; i++) k.push(gauss(r)); ks.push(k); }
              return ks.map(function (k) { return M.dot(q, k); });
            }
            function std(a) { var m = M.sum(a) / a.length; return Math.sqrt(M.sum(a.map(function (x) { return (x - m) * (x - m); })) / a.length); }
            var ds = [2, 4, 8, 16, 32, 64, 128, 256, 512], raw = [], sc = [];
            ds.forEach(function (d) {
              var a = 0, b = 0;
              for (var t = 0; t < TR; t++) { var s = trial(d, 1000 + t * 7 + d); a += M.max(M.softmax(s)); b += M.max(M.softmax(M.scale(s, 1 / Math.sqrt(d)))); }
              raw.push([Math.log2(d), a / TR]); sc.push([Math.log2(d), b / TR]);
            });
            api.html(el, '<p>Draw a random query and ' + NK + ' random keys with unit-variance entries in d dimensions. Their dot products spread out like √d. Move d and compare softmax with and without ÷√d.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var g = api.grid(el, 240);
            var c1 = api.card(g, 'softmax(q·k) — unscaled', ''), c2 = api.card(g, 'softmax(q·k / √d) — scaled', '');
            var b1 = api.bars(c1.querySelector('.card-body'), { items: [], max: 1, format: function (v) { return api.fmt(v, 2); } });
            var b2 = api.bars(c2.querySelector('.card-body'), { items: [], max: 1, format: function (v) { return api.fmt(v, 2); } });
            var info = api.h('div'); el.appendChild(info);
            var ch = api.h('div'); el.appendChild(ch);
            var chart = api.chart(ch, { series: [{ name: 'unscaled: avg max weight', points: raw, color: '#ff5d73' }, { name: 'scaled by 1/√d', points: sc, color: '#5ad18a' }],
              xLabel: 'log₂(d_head)', yLabel: 'avg max softmax weight', yRange: [0, 1], height: 220 });
            api.slider(ctl, { label: 'd_head = 2^', min: 1, max: 9, step: 1, value: 7, format: function (v) { return api.fmtInt(Math.pow(2, v)); }, onInput: function (e) {
              var d = Math.pow(2, e), s = trial(d, 42 + d), p1 = M.softmax(s), p2 = M.softmax(M.scale(s, 1 / Math.sqrt(d)));
              b1.update(p1.map(function (p, j) { return { label: 'k' + j, value: p, highlight: p === M.max(p1) }; }));
              b2.update(p2.map(function (p, j) { return { label: 'k' + j, value: p, highlight: p === M.max(p2) }; }));
              info.innerHTML = '<div class="formula">d = ' + d + ' · std of raw scores ≈ ' + f(api, std(s)) + ' (√d = ' + f(api, Math.sqrt(d)) + ') · std after ÷√d ≈ ' + f(api, std(s) / Math.sqrt(d)) + '</div>';
              chart.update({ markers: [{ x: e, label: 'd=' + d }] });
            } });
            api.callout(el, 'Reference model: d_head = 128, so scores are divided by √128 ≈ 11.3. Random vectors are only a model of the start of training; learned q·k are not random, but the scaling keeps them in a healthy range.', 'hedge');
          } }
        ]
      },

      // ============================================================ 5. causal mask
      {
        id: 'causal-mask',
        title: 'The causal mask: no peeking at the future',
        critical: {
          understand: [
            'A decoder-only LLM may only use tokens <b>at or before</b> the current position.',
            'The mask adds −∞ to every score above the diagonal, so softmax gives those positions weight 0.',
            'This lets training check <b>all positions in parallel</b> while each one still only sees its past.'
          ],
          details: [
            'Without the mask, position i could just copy token i+1 — the very answer it is trained to predict.',
            'At inference only the newest token\'s row is computed; earlier K,V are reused from the ' + '<a href="#" class="xlink" data-goto="inference/kv-cache">KV cache</a>' + '.',
            'Implementations often add a large negative number instead of true −∞; the effect is the same.'
          ]
        },
        dataTypes: [
          { name: 'causal mask M', type: 'mask', shape: '[seq_len × seq_len]', example: '0 on/below diagonal, −∞ above' },
          { name: 'scaled scores', type: 'float', shape: '[seq_len × seq_len]', example: '[5 × 5]' },
          { name: 'attention weights A', type: 'probability', shape: '[seq_len × seq_len]', example: 'lower-triangular' },
          { name: 'apply mask?', type: 'boolean', shape: 'scalar', example: 'true' },
          { name: 'training targets', type: 'token', shape: '[seq_len]', example: '␣cat ␣sat ␣on ␣the ␣mat' }
        ],
        levels: [
          { name: 'Picture', render: function (el, api) {
            var toks = api.ex.tokens, n = toks.length;
            api.html(el, '<p>Row = the token doing the looking. ✓ = allowed to see. Each token sees itself and everything to its left — a triangle.</p>');
            var t = api.h('table', { class: 'attention-vis' });
            var hr = api.h('tr', {}, api.h('th', { text: 'looks at →' }));
            toks.forEach(function (tk) { hr.appendChild(api.h('th', { html: api.tokText(tk.text) })); });
            t.appendChild(hr);
            toks.forEach(function (tk, i) {
              var tr = api.h('tr', {}, api.h('th', { html: api.tokText(tk.text) }));
              for (var j = 0; j < n; j++) tr.appendChild(api.h('td', { class: j <= i ? 'attention-ok' : 'attention-no', text: j <= i ? '✓' : '✗' }));
              t.appendChild(tr);
            });
            el.appendChild(api.h('div', { class: 'tbl-wrap' }, t));
            api.blocks(el, { blocks: [
              { label: 'scores', type: 'float', shape: '[5 × 5]' },
              { label: '+ mask', sub: '0 / −∞', type: 'mask', shape: '[5 × 5]', zoom: 1 },
              { label: 'softmax', sub: 'e<sup>−∞</sup> = 0', type: 'probability', shape: '[5 × 5]', zoom: 1 }
            ] });
          } },
          { name: 'Mask on/off', render: function (el, api) {
            var C = compute(api), H = C.heads[0], L = tokLabels(api), on = true;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            function draw() {
              box.innerHTML = '';
              var A = on ? H.A : H.Ss.map(function (r) { return api.math.softmax(r); });
              var r = row(api, box);
              api.matrix(r, { data: H.Ss, rowLabels: L, colLabels: L, title: 'S / √2 (head 1)', type: 'float', shape: '[5 × 5]', small: true, scale: 'diverging' });
              r.appendChild(api.h('div', { class: 'attention-op', text: '+' }));
              api.matrix(r, { data: H.Ss.map(function (rr) { return rr.map(function () { return 0; }); }), causal: on, rowLabels: L, colLabels: L, title: 'M', type: 'mask', shape: '[5 × 5]', small: true, heat: false, decimals: 0 });
              r.appendChild(api.h('div', { class: 'attention-op', text: '→' }));
              api.matrix(r, { data: A, causal: on, maskText: '0', rowLabels: L, colLabels: L, title: 'softmax → A', type: 'probability', shape: '[5 × 5]', small: true, scale: 'seq', highlight: on ? null : { cell: [2, 3] } });
              if (on) api.callout(box, 'Every row only puts weight on its past. Row ' + api.tok(' sat') + ' (which is trained to predict ' + api.tok(' on') + ') cannot see ' + api.tok(' on') + '.', 'tip');
              else api.callout(box, '<b>Leak!</b> Row ' + api.tok(' sat') + ' puts ' + f(api, A[2][3]) + ' weight on ' + api.tok(' on') + ' — the very token it is supposed to predict. A model trained like this learns to copy the answer and fails at generation time, when the future does not exist yet.', 'warn');
            }
            api.toggle(ctl, { label: 'apply causal mask ' + api.type('boolean'), value: true, onChange: function (v) { on = v; draw(); } });
            draw();
          } },
          { name: 'Training vs inference', render: function (el, api) {
            var toks = api.ex.tokens, tgt = toks.slice(1).map(function (t) { return t.text; }).concat([api.ex.nextToken.text]);
            api.html(el, '<p><b>Training</b> (' + api.link('training/cross-entropy', 'teacher forcing') + '): one forward pass over the whole sequence gives a prediction at <b>every</b> position, and the mask makes each one honest.</p>');
            api.table(el, { headers: ['position', 'sees ' + api.type('token'), 'trained to predict ' + api.type('token')], rows: toks.map(function (t, i) {
              return [String(i), toks.slice(0, i + 1).map(function (x) { return api.tok(x.text); }).join(' '), api.tok(tgt[i])];
            }) });
            api.html(el, '<p><b>Inference</b> (' + api.link('inference/prefill-decode', 'prefill & decode') + '): the prompt is processed once (masked, like training), then each new token only needs <b>its own row</b> of the score matrix — a [1 × n] slice. Earlier keys and values never change (they cannot see the future), so they are stored in the ' + api.link('inference/kv-cache', 'KV cache') + '.</p>');
            api.blocks(el, { blocks: [
              { label: 'new token', sub: api.tok(' mat'), type: 'token', shape: '[1]' },
              { label: 'q of new token', type: 'vector', shape: '[d_head]' },
              { label: '· cached Kᵀ', sub: 'all previous keys', type: 'matrix', shape: '[n × d_head]', goto: 'inference/kv-cache' },
              { label: 'scores', sub: 'no mask needed', type: 'float', shape: '[1 × n]' }
            ] });
          } }
        ]
      },

      // ============================================================ 6. multi-head
      {
        id: 'multi-head',
        title: 'Multi-head attention and W_o',
        critical: {
          understand: [
            'Instead of one big attention, the model runs <b>many small ones in parallel</b> (heads), each with its own pattern.',
            'Head outputs are concatenated and mixed by one more learned matrix, W_o.',
            'Different heads can track different relations at once (e.g. previous token, subject, syntax).'
          ],
          details: [
            'Reference: 32 heads × d_head 128 = 4,096 = d_model, so the total cost is similar to one big head.',
            'W_o is [4,096 × 4,096] ≈ 16.8 M params per layer.',
            'Heads are not cleanly specialised in general; interpretability work finds some clear ones and many mixed ones.'
          ]
        },
        dataTypes: [
          { name: 'head outputs', type: 'tensor', shape: '[n_heads × seq_len × d_head]', example: '[2 × 5 × 2]' },
          { name: 'concat', type: 'matrix', shape: '[seq_len × n_heads·d_head]', example: '[5 × 4]' },
          { name: 'W_o', type: 'matrix', shape: '[n_heads·d_head × d_model]', example: '[4 × 4]' },
          { name: 'attention output', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'per-head weights', type: 'probability', shape: '[seq_len × seq_len]', example: 'one per head' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [[
              { label: 'X', type: 'matrix', shape: '[n × 4096]' }
            ], [
              { label: 'head 1', sub: 'own W_q,W_k,W_v slice', type: 'matrix', shape: '[n × 128]', zoom: 1 },
              { label: 'head 2', type: 'matrix', shape: '[n × 128]', zoom: 1, arrow: false },
              { label: '…', sub: '28 more', arrow: false },
              { label: 'head 32', type: 'matrix', shape: '[n × 128]', zoom: 1, arrow: false }
            ], [
              { label: 'concat', type: 'matrix', shape: '[n × 4096]', zoom: 2 },
              { label: '· W_o', type: 'matrix', shape: '[4096 × 4096]', zoom: 2 },
              { label: 'output', sub: 'added to residual', type: 'matrix', shape: '[n × 4096]', goto: 'transformer-block/residual-stream' }
            ]], caption: 'Heads run in parallel — on a GPU it is one batched matrix multiply.' });
            api.callout(el, 'Why many heads? One softmax produces one weighted average per token. Several heads let a token gather several different things at once — "who is the subject?" <i>and</i> "what was the previous word?".', 'key');
          } },
          { name: 'Two toy heads', render: function (el, api) {
            var C = compute(api), L = tokLabels(api);
            api.html(el, '<p>The two heads of the ' + api.link('attention/worked-example', 'worked example') + ' see the same tokens but produce different attention patterns, because they use different slices of W_q and W_k.</p>');
            var r = row(api, el);
            [0, 1].forEach(function (h) {
              api.matrix(r, { data: C.heads[h].A, causal: true, maskText: '0', rowLabels: L, colLabels: L, title: 'A, head ' + (h + 1), type: 'probability', shape: '[5 × 5]', small: true, scale: 'seq', highlight: { row: 4 } });
            });
            var t0 = api.math.argmax(C.heads[0].A[4]), t1 = api.math.argmax(C.heads[1].A[4]);
            api.html(el, '<p class="muted">For ' + api.tok(' the') + ': head 1 looks most at ' + api.tok(api.ex.tokens[t0].text) + ' (' + f(api, C.heads[0].A[4][t0]) + '), head 2 at ' + api.tok(api.ex.tokens[t1].text) + ' (' + f(api, C.heads[1].A[4][t1]) + ').</p>');
            api.illustrative(el);
          } },
          { name: 'Concat · W_o', render: function (el, api) {
            var C = compute(api), L = tokLabels(api);
            formula(api, el, 'MultiHead(X) = concat(head<sub>1</sub>, …, head<sub>h</sub>) · W_o');
            var r = row(api, el);
            api.matrix(r, { data: C.concat, rowLabels: L, colLabels: headCols(), title: 'concat', type: 'matrix', shape: '[5 × 4]', small: true });
            r.appendChild(api.h('div', { class: 'attention-op', text: '·' }));
            api.matrix(r, { data: WO, colLabels: dimLabels(4), rowLabels: headCols(), title: 'W_o', type: 'matrix', shape: '[4 × 4]', small: true });
            r.appendChild(api.h('div', { class: 'attention-op', text: '=' }));
            api.matrix(r, { data: C.out, rowLabels: L, colLabels: dimLabels(4), title: 'output', type: 'matrix', shape: '[5 × 4]', small: true, highlight: { row: 4 } });
            api.html(el, '<p>Rows of W_o labelled h1·* read head 1\'s output, h2·* read head 2\'s. So W_o is where heads\' findings get <b>combined</b> and written back in the residual stream\'s coordinates.</p>');
            var pb = api.ref.paramBreakdown();
            api.table(el, { headers: ['reference matrix', 'shape', 'params / layer'], rows: [
              ['W_q', '[4,096 × 4,096]', api.fmtInt(pb.wq)], ['W_k', '[4,096 × 1,024]', api.fmtInt(pb.wk)], ['W_v', '[4,096 × 1,024]', api.fmtInt(pb.wv)], ['W_o', '[4,096 × 4,096]', api.fmtInt(pb.wo)],
              ['<b>attention total</b>', '', '<b>' + api.fmtInt(pb.attn) + '</b>']] });
            api.html(el, '<p class="muted">Full per-layer count incl. FFN: ' + api.link('transformer-block/param-count', 'parameter count') + '.</p>');
          } }
        ]
      },

      // ============================================================ 7. GQA and cost
      {
        id: 'gqa-cost',
        title: 'GQA / MQA and the O(n²) cost',
        critical: {
          understand: [
            'Every token\'s keys and values must be kept around during generation (the KV cache) — that memory grows with context.',
            '<b>GQA</b> lets several query heads share one K/V head: the reference model has 32 query heads but only 8 KV heads → 4× smaller KV cache.',
            'The score matrix is n × n per head: doubling the context ≈ 4× the attention compute.'
          ],
          details: [
            'KV cache per token = 2 (K,V) × 32 layers × 8 KV heads × 128 × 2 bytes = 128 KiB.',
            'MQA (1 KV head) is the extreme; it saves the most memory but tends to cost some quality.',
            'FlashAttention-style kernels avoid storing the n × n matrix but still do O(n²·d) work.'
          ]
        },
        dataTypes: [
          { name: 'n_heads, n_kv_heads', type: 'integer', shape: 'scalar', example: '32, 8' },
          { name: 'K, V cache', type: 'tensor', shape: '[layers × 2 × kv_heads × seq_len × d_head]', example: '[32 × 2 × 8 × n × 128]' },
          { name: 'KV bytes per token', type: 'integer', shape: 'scalar', example: '131,072 (128 KiB)' },
          { name: 'context length n', type: 'integer', shape: 'scalar', example: '8,192' },
          { name: 'score matrix', type: 'matrix', shape: '[n × n]', example: 'per head, per layer' }
        ],
        levels: [
          { name: 'Sharing K/V', render: function (el, api) {
            var R = api.ref, kv = R.n_kv_heads;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            function draw() {
              var nq = R.n_heads, grp = nq / kv, W = 640, H = 170, qw = (W - 20) / nq;
              var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="attention-svg">';
              s += '<text x="10" y="16" class="attention-svglbl">32 query heads</text>';
              var kvw = (W - 20) / kv;
              for (var i = 0; i < nq; i++) {
                var g = Math.floor(i / grp), qx = 10 + i * qw + qw / 2, kx = 10 + g * kvw + kvw / 2;
                s += '<line x1="' + qx + '" y1="44" x2="' + kx + '" y2="118" stroke="' + (g % 2 ? '#b98cff' : '#4ea8ff') + '" stroke-width="1" opacity=".55"/>';
                s += '<rect x="' + (10 + i * qw + 1) + '" y="24" width="' + (qw - 2) + '" height="20" rx="3" fill="' + (g % 2 ? '#b98cff' : '#4ea8ff') + '" opacity=".85"/>';
              }
              for (var k = 0; k < kv; k++) {
                s += '<rect x="' + (10 + k * kvw + 2) + '" y="118" width="' + (kvw - 4) + '" height="24" rx="4" fill="' + (k % 2 ? '#b98cff' : '#4ea8ff') + '" opacity=".35" stroke="#ff9a4d"/>';
              }
              s += '<text x="10" y="162" class="attention-svglbl">' + kv + ' K/V head' + (kv > 1 ? 's' : '') + ' (each = one K and one V)</text></svg>';
              box.innerHTML = s;
              var b = 2 * R.n_layers * kv * R.d_head * 2;
              api.html(box, '<div class="formula">KV cache per token = 2 × ' + R.n_layers + ' layers × ' + kv + ' KV heads × ' + R.d_head + ' × 2 bytes = <b>' + api.fmtBytes(b) + '</b> ' + api.type('integer') + (kv === 8 ? ' ← reference model' : '') + '</div>');
            }
            api.select(ctl, { label: 'Variant', value: '8', options: [{ value: '32', label: 'MHA — 32 KV heads' }, { value: '8', label: 'GQA — 8 KV heads (reference)' }, { value: '1', label: 'MQA — 1 KV head' }], onChange: function (v) { kv = +v; draw(); } });
            draw();
            api.html(el, '<p class="muted">Queries stay per-head (each head still asks its own question); only the keys/values it searches are shared within a group of ' + (R.n_heads / R.n_kv_heads) + '.</p>');
          } },
          { name: 'KV memory', render: function (el, api) {
            var R = api.ref, ctx = 8192, batch = 1;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            var b = api.bars(out, { items: [], format: function (v) { return api.fmtBytes(v); }, title: 'KV cache size ' + api.badge('integer', 'bytes') });
            var info = api.h('div'); out.appendChild(info);
            function upd() {
              var mk = function (kv) { return 2 * R.n_layers * kv * R.d_head * 2 * ctx * batch; };
              var w = api.ref.paramBreakdown().bytesBF16;
              b.update([
                { label: 'MHA (32 KV)', value: mk(32), color: '#ff5d73' },
                { label: 'GQA (8 KV)', value: mk(8), color: '#5ad18a', highlight: true },
                { label: 'MQA (1 KV)', value: mk(1), color: '#4ea8ff' },
                { label: 'weights (BF16)', value: w, color: '#8a94a6' }
              ]);
              info.innerHTML = '<div class="formula">' + api.fmtInt(ctx) + ' tokens × ' + batch + ' sequence' + (batch > 1 ? 's' : '') + ' × ' + api.fmtBytes(api.ref.kvBytesPerToken) + '/token (GQA) = <b>' + api.fmtBytes(mk(8)) + '</b></div>';
            }
            api.slider(ctl, { label: 'context tokens', min: 512, max: 32768, step: 512, value: ctx, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { ctx = v; upd(); } });
            api.slider(ctl, { label: 'batch (sequences)', min: 1, max: 32, step: 1, value: batch, onInput: function (v) { batch = v; upd(); } });
            api.html(el, '<p>The KV cache, not the weights, is often what limits how many users or how much context fit on one GPU. More in ' + api.link('inference/kv-cache', 'Inference → KV cache') + '.</p>');
          } },
          { name: 'O(n²)', render: function (el, api) {
            var R = api.ref, n = 8192;
            api.html(el, '<p>For every head in every layer, the score matrix QKᵀ has n × n entries. Generating is linear per new token (one row), but processing a long prompt — or training — is quadratic.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var info = api.h('div'); el.appendChild(info);
            var pts = [], lin = [];
            for (var k = 1; k <= 32; k++) { var x = k * 1024; pts.push([x / 1024, x * x / 1e6]); lin.push([x / 1024, x * 1024 / 1e6]); }
            var ch = api.chart(el, { series: [{ name: 'score entries n² (millions)', points: pts, color: '#ff9a4d' }, { name: 'linear n·1024 (for comparison)', points: lin, color: '#4ea8ff', dashed: true }], xLabel: 'context n (K tokens)', yLabel: 'entries (M)', height: 230 });
            api.slider(ctl, { label: 'context n', min: 1024, max: 32768, step: 1024, value: n, format: function (v) { return api.fmtInt(v); }, onInput: function (v) {
              n = v;
              var per = n * n, all = per * R.n_heads * R.n_layers, flops = 2 * 2 * n * n * R.d_head * R.n_heads * R.n_layers;
              info.innerHTML = '<div class="formula">one head: ' + api.fmtInt(n) + '² = ' + api.fmtBig(per) + ' scores (' + api.fmtBytes(per * 2) + ' in BF16)<br>× ' + R.n_heads + ' heads × ' + R.n_layers + ' layers = ' + api.fmtBig(all) + ' scores<br>QKᵀ and A·V ≈ ' + api.fmtBig(flops) + ' FLOPs for the whole prompt</div>';
              ch.update({ markers: [{ x: n / 1024, label: api.fmtInt(n) }] });
            } });
            api.callout(el, 'Double n → 4× attention work. This is why long-context models rely on tricks such as FlashAttention (tiles, never materialises the full n×n matrix), sliding-window or sparse attention — trade-offs differ by model.', 'info');
          } }
        ]
      }
    ]
  });
})();
