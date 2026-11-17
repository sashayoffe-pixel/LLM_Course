(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';

  // illustrative 2-D "meaning" coordinates (x ≈ royalty/size, y ≈ femaleness/liveliness) — made up
  var WORDS = {
    man: [0.2, 0.1], woman: [0.1, 2.0], king: [3.1, 0.3], queen: [3.2, 2.3], prince: [2.4, 0.2], princess: [2.5, 2.1],
    boy: [-0.9, -0.2], girl: [-1.0, 1.6],
    cat: [-3.2, -2.0], dog: [-2.6, -2.5], kitten: [-3.6, -1.2], puppy: [-3.0, -1.7],
    car: [-0.5, -3.6], truck: [0.6, -3.5], bus: [0.2, -4.0],
    paris: [4.0, -2.4], france: [4.2, -1.4], rome: [3.1, -2.9], italy: [3.3, -1.9]
  };
  var NAMES = Object.keys(WORDS);

  function nearest(api, v, exclude, k) {
    var M = api.math;
    return NAMES.filter(function (n) { return exclude.indexOf(n) < 0; }).map(function (n) { return { n: n, d: Math.sqrt(M.dot(M.sub(WORDS[n], v), M.sub(WORDS[n], v))), c: M.cosine(WORDS[n], v) }; })
      .sort(function (a, b) { return a.d - b.d; }).slice(0, k || 3);
  }

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  }
  // 2-D map: x in [-5,5], y in [-5,3.5]
  function mapSvg(api, opts) {
    var W = 560, H = 380, sx = function (x) { return W / 2 + x * 48; }, sy = function (y) { return 300 - y * 44; };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'embeddings-svg', role: 'img', 'aria-label': 'Illustrative 2-D word map' });
    svgEl('line', { x1: 0, y1: sy(0), x2: W, y2: sy(0), class: 'embeddings-axis' }, svg);
    svgEl('line', { x1: sx(0), y1: 0, x2: sx(0), y2: H, class: 'embeddings-axis' }, svg);
    var pts = {};
    NAMES.forEach(function (n) {
      var g = svgEl('g', { class: 'embeddings-pt' }, svg);
      var c = svgEl('circle', { cx: sx(WORDS[n][0]), cy: sy(WORDS[n][1]), r: 5, fill: opts.colors && opts.colors[n] || 'var(--t-vector)', stroke: 'var(--bg)' }, g);
      var t = svgEl('text', { x: sx(WORDS[n][0]) + 8, y: sy(WORDS[n][1]) + 4 }, g); t.textContent = n;
      g.addEventListener('click', function () { if (opts.onPick) opts.onPick(n); });
      pts[n] = { g: g, c: c };
    });
    return { svg: svg, sx: sx, sy: sy, pts: pts };
  }

  function toyE(api) {
    var M = api.math, r = M.rng(21), E = [];
    for (var i = 0; i < 8; i++) { var row = []; for (var j = 0; j < 4; j++) row.push(M.round((r() * 2 - 1) * 0.8, 2)); E.push(row); }
    return E;
  }
  var TOYVOC = ['The', ' cat', ' sat', ' on', ' the', ' mat', ' dog', '.'];

  Course.register({
    id: 'embeddings', part: 'I', order: 30, title: 'Embeddings & Position',
    slides: [
      // ---------------------------------------------------------------- lookup
      {
        id: 'lookup', title: 'Token ID → a row of the embedding table E',
        critical: {
          understand: ['The model cannot do arithmetic on the ID 3797. So each ID selects a <b>row of a learned table E</b>: a vector of d_model numbers.',
            'That vector is the token\'s starting representation; the layers will refine it using context.'],
          details: ['E has shape [vocab × d_model]: toy 8 × 4 here; 128,256 × 4,096 ≈ 525 M parameters in the 8B reference.',
            'A lookup is mathematically a one-hot vector times E, but implementations just index the row.']
        },
        dataTypes: [
          { name: 'token ID', type: 'integer', shape: 'scalar', example: '3797' },
          { name: 'embedding table E', type: 'matrix', shape: '[vocab × d_model]', example: '[8 × 4] toy; [128,256 × 4,096]' },
          { name: 'one-hot vector', type: 'vector', shape: '[vocab]', example: 'a single 1' },
          { name: 'embedding', type: 'vector', shape: '[d_model]', example: '[0.43, −0.12, …]' }
        ],
        levels: [
          { name: 'Lookup', render: function (el, api) {
            api.blocks(el, { blocks: [
              { label: 'Token', sub: '" cat"', type: 'token', shape: 'scalar' },
              { label: 'ID', sub: '3797', type: 'integer', shape: 'scalar', edge: 'tokenizer', goto: 'tokenization/text-to-ids' },
              { label: 'Row of E', sub: 'E[3797]', type: 'vector', shape: '[4,096]', edge: 'index', zoom: 1 },
              { label: 'Stack rows', sub: 'one per token', type: 'matrix', shape: '[5 × 4,096]', goto: 'embeddings/input-matrix' }
            ], caption: 'E is learned during training like any other weight matrix.' });
          } },
          { name: 'One-hot × E', render: function (el, api) {
            var E = toyE(api), M = api.math;
            var out = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.select(c, { label: 'Token', options: TOYVOC.map(function (w, i) { return { value: String(i), label: w.replace(/ /g, '␣') }; }), value: '1', onChange: draw });
            el.appendChild(out);
            function draw(v) {
              var i = +v, oh = TOYVOC.map(function (_, k) { return k === i ? 1 : 0; });
              out.innerHTML = '';
              api.html(out, '<div>one-hot ' + api.badge('vector', '[8]') + ':</div><div class="embeddings-onehot">' + oh.map(function (x) { return '<span class="' + (x ? 'on' : '') + '">' + x + '</span>'; }).join('') + '</div>');
              api.matrix(out, { data: E, rowLabels: TOYVOC.map(function (w) { return api.tokText(w); }), title: 'E', type: 'matrix', shape: '[8 × 4]', highlight: { row: i }, decimals: 2 });
              api.html(out, '<div class="embeddings-calc">one-hot · E = 1×E[' + i + '] + 0×(all other rows) = <b>[' + M.vecmat(oh, E).map(function (x) { return api.fmt(x, 2); }).join(', ') + ']</b> ' + api.badge('vector', '[4]') + '</div>');
            }
            api.illustrative(el, 'E values are made up (seeded random); real ones are learned.');
          } },
          { name: 'Sizes', render: function (el, api) {
            var R = api.ref, n = R.vocab * R.d_model;
            api.table(el, { headers: ['', 'Toy', '8B reference'], rows: [
              ['vocab', '8', api.fmtInt(R.vocab)], ['d_model', '4', api.fmtInt(R.d_model)],
              ['E shape ' + api.type('matrix'), '[8 × 4]', '[' + api.fmtInt(R.vocab) + ' × ' + api.fmtInt(R.d_model) + ']'],
              ['numbers in E', '32', api.fmtInt(n) + ' (≈ ' + api.fmtBig(n) + ')'],
              ['bytes in BF16', '64 B', api.fmtBytes(n * 2)]
            ] });
            api.html(el, '<div>Embeddings alone are ≈ ' + (100 * n / api.ref.paramBreakdown().total).toFixed(1) + ' % of the model; the output matrix (unembedding) is the same size unless tied. See ' + api.link('big-picture/scale', 'scale') + '.</div>');
          } }
        ]
      },
      // ---------------------------------------------------------------- geometry
      {
        id: 'geometry', title: 'Meaning as geometry: a map of words',
        critical: {
          understand: ['Training arranges embeddings so that words used in similar ways end up <b>close together</b>, and directions can carry meaning.',
            'Classic demo: <b>king − man + woman ≈ queen</b>, because "male → female" is roughly a consistent direction.'],
          details: ['The map here is 2-D and hand-made for illustration. Real embeddings have thousands of dimensions and are not human-readable.',
            'Analogy arithmetic works only approximately and in some models; do not expect it to be exact.']
        },
        dataTypes: [
          { name: 'word vector', type: 'vector', shape: '[2] here; [d_model] real', example: 'king = [3.1, 0.3]' },
          { name: 'analogy result', type: 'vector', shape: '[2]', example: 'king − man + woman' },
          { name: 'distance', type: 'float', shape: 'scalar', example: '0.2' }
        ],
        levels: [
          { name: 'Map', render: function (el, api) {
            var info = api.h('div', { class: 'embeddings-calc', text: 'Click a word.' });
            var m = mapSvg(api, { onPick: function (n) {
              var nn = nearest(api, WORDS[n], [n], 3);
              info.innerHTML = '<b>' + n + '</b> = [' + WORDS[n].map(function (x) { return api.fmt(x, 1); }).join(', ') + '] · nearest: ' + nn.map(function (x) { return x.n + ' (' + api.fmt(x.d, 2) + ')'; }).join(', ');
              Object.keys(m.pts).forEach(function (k) { m.pts[k].c.setAttribute('r', k === n ? 8 : 5); });
            } });
            el.appendChild(m.svg); el.appendChild(info);
            api.illustrative(el, 'Coordinates are invented to show clusters (royalty/people, animals, vehicles, places).');
          } },
          { name: 'Analogy', render: function (el, api) {
            var M = api.math, st = { a: 'king', b: 'man', c: 'woman' }, out = api.h('div'), holder = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var opts = NAMES.map(function (n) { return { value: n, label: n }; });
            api.select(c, { label: 'A', options: opts, value: 'king', onChange: function (v) { st.a = v; draw(); }, init: false });
            api.select(c, { label: '− B', options: opts, value: 'man', onChange: function (v) { st.b = v; draw(); }, init: false });
            api.select(c, { label: '+ C', options: opts, value: 'woman', onChange: function (v) { st.c = v; draw(); }, init: false });
            el.appendChild(holder); el.appendChild(out);
            function draw() {
              var a = WORDS[st.a], b = WORDS[st.b], cc = WORDS[st.c], r = M.add(M.sub(a, b), cc);
              var nn = nearest(api, r, [st.a, st.b, st.c], 3);
              holder.innerHTML = '';
              var m = mapSvg(api, {});
              var star = svgEl('path', { d: 'M0,-9 L2.6,-3 9,-2.8 4,1.5 5.6,8 0,4.3 -5.6,8 -4,1.5 -9,-2.8 -2.6,-3Z', fill: 'var(--warn)', transform: 'translate(' + m.sx(r[0]) + ',' + m.sy(r[1]) + ')' }, m.svg);
              svgEl('line', { x1: m.sx(b[0]), y1: m.sy(b[1]), x2: m.sx(a[0]), y2: m.sy(a[1]), stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '5 3' }, m.svg);
              svgEl('line', { x1: m.sx(cc[0]), y1: m.sy(cc[1]), x2: m.sx(r[0]), y2: m.sy(r[1]), stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '5 3' }, m.svg);
              holder.appendChild(m.svg);
              out.innerHTML = '<div class="embeddings-calc">' + st.a + ' − ' + st.b + ' + ' + st.c + '<br>= [' + a.map(function (x) { return api.fmt(x, 1); }).join(', ') + '] − [' + b.map(function (x) { return api.fmt(x, 1); }).join(', ') + '] + [' + cc.map(function (x) { return api.fmt(x, 1); }).join(', ') + ']<br>= <b>[' + r.map(function (x) { return api.fmt(x, 2); }).join(', ') + ']</b> ' + api.badge('vector', '[2]') + '<br>nearest words (excluding inputs): <b>' + nn.map(function (x) { return x.n + ' (d=' + api.fmt(x.d, 2) + ')'; }).join(', ') + '</b></div>';
            }
            draw();
            api.illustrative(el, 'With these made-up 2-D vectors the answer is queen by construction; in real models it holds only approximately.');
          } },
          { name: 'In reality', render: function (el, api) {
            var R = api.ref;
            var g = api.h('div', { class: 'embeddings-cols' }); el.appendChild(g);
            api.card(g, 'Toy map', '2 dimensions you can see. ' + api.badge('vector', '[2]'));
            api.card(g, '8B reference', api.fmtInt(R.d_model) + ' dimensions per token. ' + api.badge('vector', '[' + api.fmtInt(R.d_model) + ']') + ' Individual dimensions rarely have a clean human meaning; directions and neighbourhoods do.');
            api.card(g, 'Contextual', 'The table E gives a <b>context-free</b> start. After attention, the vector for "bank" differs in "river bank" vs "bank loan" — that is the point of the layers.');
            api.callout(el, 'Research suggests many concepts are encoded as directions (sometimes spread over several dimensions), but it is an active research area; treat the geometric picture as an intuition, not a precise law.', 'hedge');
          } }
        ]
      },
      // ---------------------------------------------------------------- cosine
      {
        id: 'cosine', title: 'Cosine similarity, step by step',
        critical: {
          understand: ['Similarity of two vectors = the <b>angle</b> between them: cos θ = (a·b) / (‖a‖ ‖b‖).',
            '1 = same direction, 0 = unrelated (perpendicular), −1 = opposite.'],
          details: ['Dividing by the lengths makes the score ignore magnitude, so a "louder" vector is not automatically more similar.',
            'Cosine ranking is the core of retrieval in RAG (Part II) and similar to the dot-product scores inside attention.']
        },
        dataTypes: [
          { name: 'vectors a, b', type: 'vector', shape: '[2] here', example: 'king, queen' },
          { name: 'dot product a·b', type: 'float', shape: 'scalar', example: '10.2' },
          { name: 'norms ‖a‖, ‖b‖', type: 'float', shape: 'scalar', example: '3.1' },
          { name: 'cosine', type: 'float', shape: 'scalar in [−1, 1]', example: '0.97' }
        ],
        levels: [
          { name: 'Idea', render: function (el, api) {
            var M = api.math, out = api.h('div'), st = { a: 'king', b: 'queen' };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var opts = NAMES.map(function (n) { return { value: n, label: n }; });
            api.select(c, { label: 'Word 1', options: opts, value: 'king', onChange: function (v) { st.a = v; draw(); }, init: false });
            api.select(c, { label: 'Word 2', options: opts, value: 'queen', onChange: function (v) { st.b = v; draw(); }, init: false });
            el.appendChild(out);
            function draw() {
              var cs = M.cosine(WORDS[st.a], WORDS[st.b]);
              out.innerHTML = '';
              var m = mapSvg(api, {});
              [st.a, st.b].forEach(function (n) { svgEl('line', { x1: m.sx(0), y1: m.sy(0), x2: m.sx(WORDS[n][0]), y2: m.sy(WORDS[n][1]), stroke: 'var(--warn)', 'stroke-width': 2 }, m.svg); });
              out.appendChild(m.svg);
              api.bars(out, { title: 'cosine(' + st.a + ', ' + st.b + ') ' + api.badge('float', 'scalar'), items: [{ label: 'cos θ', value: cs, highlight: true }], max: 1, format: function (v) { return api.fmt(v, 3); } });
            }
            draw();
            api.illustrative(el, 'Vectors from the made-up 2-D map; the lines are arrows from the origin.');
          } },
          { name: 'Step by step', render: function (el, api) {
            var M = api.math, st = { a: 'cat', b: 'kitten' }, holder = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var opts = NAMES.map(function (n) { return { value: n, label: n }; });
            var stp;
            api.select(c, { label: 'Word 1', options: opts, value: 'cat', onChange: function (v) { st.a = v; build(); }, init: false });
            api.select(c, { label: 'Word 2', options: opts, value: 'kitten', onChange: function (v) { st.b = v; build(); }, init: false });
            el.appendChild(holder);
            function f(v) { return '[' + v.map(function (x) { return api.fmt(x, 2); }).join(', ') + ']'; }
            function build() {
              holder.innerHTML = '';
              var a = WORDS[st.a], b = WORDS[st.b], d = M.dot(a, b), na = M.norm(a), nb = M.norm(b), cs = d / (na * nb);
              stp = api.stepper(holder, { interval: 2400, steps: [
                { title: 'The two vectors', html: '<div class="embeddings-calc">a = ' + st.a + ' = ' + f(a) + '<br>b = ' + st.b + ' = ' + f(b) + '</div>' },
                { title: 'Dot product a·b', html: '<div class="embeddings-calc">a·b = ' + a.map(function (x, i) { return '(' + api.fmt(x, 2) + ' × ' + api.fmt(b[i], 2) + ')'; }).join(' + ') + '<br>= <b>' + api.fmt(d, 3) + '</b></div>' },
                { title: 'Lengths ‖a‖ and ‖b‖', html: '<div class="embeddings-calc">‖a‖ = √(' + a.map(function (x) { return api.fmt(x * x, 2); }).join(' + ') + ') = <b>' + api.fmt(na, 3) + '</b><br>‖b‖ = √(' + b.map(function (x) { return api.fmt(x * x, 2); }).join(' + ') + ') = <b>' + api.fmt(nb, 3) + '</b></div>' },
                { title: 'Divide', html: '<div class="embeddings-calc">cos θ = ' + api.fmt(d, 3) + ' / (' + api.fmt(na, 3) + ' × ' + api.fmt(nb, 3) + ') = <b>' + api.fmt(cs, 3) + '</b></div><div>angle ≈ ' + api.fmt(Math.acos(Math.max(-1, Math.min(1, cs))) * 180 / Math.PI, 1) + '°</div>' },
                { title: 'Interpret', html: '<div>' + (cs > 0.8 ? 'Very similar direction.' : cs > 0.3 ? 'Somewhat related.' : cs > -0.3 ? 'Roughly unrelated (near perpendicular).' : 'Pointing in opposite directions.') + ' ' + api.badge('float', 'scalar') + '</div>' }
              ] });
            }
            build();
          } },
          { name: 'Why divide?', render: function (el, api) {
            var M = api.math, a = WORDS.cat, b = WORDS.dog, out = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Scale vector b by', min: 0.2, max: 5, step: 0.1, value: 1, format: function (v) { return '× ' + v.toFixed(1); }, onInput: function (k) {
              var bb = M.scale(b, k);
              out.innerHTML = '';
              api.table(out, { headers: ['Measure', 'Value'], rows: [['dot product a·(k·b)', api.fmt(M.dot(a, bb), 3) + ' (grows with k)'], ['cosine', '<b>' + api.fmt(M.cosine(a, bb), 3) + '</b> (unchanged)']] });
            } });
            el.appendChild(out);
            api.html(el, '<p>a = cat, b = dog. The raw dot product changes with vector length, cosine does not. Attention scores use raw dot products (then divide by √d_head); retrieval usually uses cosine or normalized vectors. See ' + api.link('attention/worked-example', 'attention') + '.</p>');
          } }
        ]
      },
      // ---------------------------------------------------------------- why position
      {
        id: 'why-position', title: 'Why position must be injected',
        critical: {
          understand: ['Attention treats its input as a <b>set</b>: shuffle the tokens and each token\'s output is the same, just moved. Without extra information, "dog bites man" = "man bites dog".',
            'So the model must be told <b>where</b> each token is: by adding position vectors, or by rotating Q and K (RoPE).'],
          details: ['Demo uses plain self-attention without a causal mask. With a causal mask a model can infer a little order implicitly, but it is weak and unreliable.',
            'Modern LLMs such as the 8B reference use RoPE (next slide).']
        },
        dataTypes: [
          { name: 'token embeddings', type: 'matrix', shape: '[3 × 4]', example: 'dog, bites, man' },
          { name: 'position vectors', type: 'matrix', shape: '[3 × 4]', example: 'pos 0, 1, 2' },
          { name: 'attention output', type: 'matrix', shape: '[3 × 4]', example: 'same rows, permuted' }
        ],
        levels: [
          { name: 'Problem', render: function (el, api) {
            var g = api.grid(el, 240);
            api.card(g, 'Sentence A', api.tok('dog') + ' ' + api.tok(' bites') + ' ' + api.tok(' man') + '<div class="muted">a dog bites a man (ordinary)</div>');
            api.card(g, 'Sentence B', api.tok('man') + ' ' + api.tok(' bites') + ' ' + api.tok(' dog') + '<div class="muted">a man bites a dog (news!)</div>');
            api.callout(el, 'Same three tokens, very different meaning. The embedding table E alone gives the same three vectors in both sentences; only their <b>order</b> differs.', 'key');
          } },
          { name: 'Demo', render: function (el, api) {
            var M = api.math, E = toyE(api), r = M.rng(33);
            var ids = { dog: 6, bites: 2, man: 3 };
            var A = [E[ids.dog], E[ids.bites], E[ids.man]], B = [E[ids.man], E[ids.bites], E[ids.dog]];
            function W() { var m = []; for (var i = 0; i < 4; i++) { var row = []; for (var j = 0; j < 4; j++) row.push((r() * 2 - 1) * 1.2); m.push(row); } return m; }
            var Wq = W(), Wk = W(), Wv = W();
            function attn(X) { var Q = M.matmul(X, Wq), K = M.matmul(X, Wk), V = M.matmul(X, Wv); var Sc = M.matmul(Q, M.transpose(K)).map(function (row) { return M.softmax(row.map(function (x) { return x / Math.sqrt(4); })); }); return M.matmul(Sc, V); }
            var oA = attn(A), oB = attn(B);
            var withPos = function (X) { return X.map(function (row, i) { return row.map(function (x, j) { return x + 0.5 * Math.sin((i + 1) * (j + 1)); }); }); };
            var pA = attn(withPos(A)), pB = attn(withPos(B));
            api.html(el, '<p>Output row for the token <b>"man"</b> after self-attention (no mask), in each sentence:</p>');
            api.table(el, { headers: ['', 'Sentence A: dog bites <b>man</b>', 'Sentence B: <b>man</b> bites dog'], rows: [
              ['Without position ' + api.type('vector'), '<code>' + oA[2].map(function (x) { return api.fmt(x, 3); }).join(', ') + '</code>', '<code>' + oB[0].map(function (x) { return api.fmt(x, 3); }).join(', ') + '</code>'],
              ['With position added ' + api.type('vector'), '<code>' + pA[2].map(function (x) { return api.fmt(x, 3); }).join(', ') + '</code>', '<code>' + pB[0].map(function (x) { return api.fmt(x, 3); }).join(', ') + '</code>']
            ] });
            var same = oA[2].every(function (x, i) { return Math.abs(x - oB[0][i]) < 1e-9; });
            api.callout(el, 'Without position the two vectors are ' + (same ? '<b>identical</b>' : 'different') + ': "man" cannot tell whether it is the biter or the bitten. With position vectors added they differ, so order information survives.', 'info');
            api.illustrative(el, 'Seeded random toy weights and simple sinusoidal position offsets.');
          } },
          { name: 'Fixes', render: function (el, api) {
            api.table(el, { headers: ['Method', 'How', 'Used by (examples)'], rows: [
              ['Learned absolute', 'a trainable vector per position, added to the embedding', 'GPT-2'],
              ['Sinusoidal', 'fixed sin/cos pattern added to the embedding', 'original Transformer'],
              ['<b>RoPE</b>', 'rotate Q and K pairs by an angle ∝ position', 'Llama family, many open models'],
              ['ALiBi', 'penalty on attention scores growing with distance', 'some models']
            ] });
            api.html(el, '<p>Next: ' + api.link('embeddings/rope', 'RoPE as rotation') + '.</p>');
          } }
        ]
      },
      // ---------------------------------------------------------------- rope
      {
        id: 'rope', title: 'RoPE: position as rotation',
        critical: {
          understand: ['RoPE rotates each (even, odd) pair of numbers in Q and K by an angle <b>proportional to the token\'s position</b>.',
            'Because rotations compose, the dot product q·k then depends only on the <b>relative offset</b> between the two tokens.'],
          details: ['Different pairs rotate at different speeds θᵢ = base^(−2i/d_head): fast pairs resolve nearby offsets, slow pairs long distances.',
            'RoPE is applied to Q and K inside each attention layer, not added to the embedding. The toy matrix X later is a simplification.']
        },
        dataTypes: [
          { name: 'position m', type: 'integer', shape: 'scalar', example: '3' },
          { name: 'angle m·θ', type: 'float', shape: 'scalar', example: '1.5 rad' },
          { name: 'rotation R(m·θ)', type: 'matrix', shape: '[2 × 2]', example: '[[cos, −sin],[sin, cos]]' },
          { name: 'rotated q, k', type: 'vector', shape: '[d_head]', example: '[2] for one pair' }
        ],
        levels: [
          { name: 'Rotation', render: function (el, api) {
            var q = [1.0, 0.3], theta = 0.5, out = api.h('div'), holder = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Position m', min: 0, max: 12, step: 1, value: 3, format: function (v) { return String(v); }, onInput: draw });
            el.appendChild(holder); el.appendChild(out);
            function draw(m) {
              var a = m * theta, ca = Math.cos(a), sa = Math.sin(a), r = [q[0] * ca - q[1] * sa, q[0] * sa + q[1] * ca];
              holder.innerHTML = '';
              var S = 90, cx = 150, cy = 110;
              var svg = svgEl('svg', { viewBox: '0 0 300 220', class: 'embeddings-svg', style: 'max-width:320px' }, holder);
              svgEl('circle', { cx: cx, cy: cy, r: S * Math.hypot(q[0], q[1]), fill: 'none', stroke: 'var(--border2)', 'stroke-dasharray': '4 3' }, svg);
              svgEl('line', { x1: cx, y1: cy, x2: cx + S * q[0], y2: cy - S * q[1], stroke: 'var(--muted)', 'stroke-width': 2 }, svg);
              svgEl('line', { x1: cx, y1: cy, x2: cx + S * r[0], y2: cy - S * r[1], stroke: 'var(--warn)', 'stroke-width': 3 }, svg);
              var t1 = svgEl('text', { x: cx + S * q[0] + 4, y: cy - S * q[1] }, svg); t1.textContent = 'q at m=0';
              var t2 = svgEl('text', { x: cx + S * r[0] + 4, y: cy - S * r[1] - 4 }, svg); t2.textContent = 'rotated, m=' + m;
              out.innerHTML = '<div class="embeddings-calc">angle = m·θ = ' + m + ' × ' + theta + ' = <b>' + api.fmt(a, 2) + ' rad</b> (' + api.fmt(a * 180 / Math.PI, 0) + '°)<br>q′ = [' + api.fmt(r[0], 3) + ', ' + api.fmt(r[1], 3) + '] ' + api.badge('vector', '[2]') + ' — same length (' + api.fmt(Math.hypot(r[0], r[1]), 3) + '), new direction</div>';
            }
            api.illustrative(el, 'One 2-D pair with θ = 0.5 rad per position (chosen for visibility).');
          } },
          { name: 'Relative offset', render: function (el, api) {
            var M = api.math, q = [1.0, 0.3], k = [0.8, -0.4], theta = 0.5, out = api.h('div');
            function rot(v, m) { var a = m * theta; return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)]; }
            var st = { m: 5, n: 2 };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Query position m', min: 0, max: 20, step: 1, value: 5, format: String, onInput: function (v) { st.m = v; draw(); } });
            api.slider(c, { label: 'Key position n', min: 0, max: 20, step: 1, value: 2, format: String, onInput: function (v) { st.n = v; draw(); } });
            el.appendChild(out);
            function draw() {
              var s = M.dot(rot(q, st.m), rot(k, st.n)), rel = st.m - st.n;
              var rows = [];
              [0, 4, 9].forEach(function (shift) { var m2 = st.m + shift, n2 = st.n + shift; rows.push(['m=' + m2 + ', n=' + n2, m2 - n2, api.fmt(M.dot(rot(q, m2), rot(k, n2)), 4)]); });
              out.innerHTML = '<div class="embeddings-calc">q′·k′ with (m=' + st.m + ', n=' + st.n + ') = <b>' + api.fmt(s, 4) + '</b> ' + api.badge('float', 'scalar') + '<br>offset m − n = ' + rel + '</div><div class="muted">Shift both positions by the same amount — the score does not change:</div>';
              api.table(out, { headers: ['Positions', 'm − n', 'q′·k′'], rows: rows });
            }
            api.callout(el, 'Math: R(mθ)q · R(nθ)k = q · R((n−m)θ)k, because rotation matrices satisfy R(a)ᵀR(b) = R(b−a). Only the offset remains.', 'key');
          } },
          { name: 'Many speeds', render: function (el, api) {
            var dh = 8, base = 10000, rows = [];
            for (var i = 0; i < dh / 2; i++) { var th = Math.pow(base, -2 * i / dh); rows.push(['pair ' + i, api.fmt(th, 4) + ' rad/position', api.fmt(2 * Math.PI / th, 1) + ' positions per full turn']); }
            api.html(el, '<p>With d_head = ' + dh + ' and base = ' + api.fmtInt(base) + ' (original RoPE setting), θᵢ = base<sup>−2i/d_head</sup>:</p>');
            api.table(el, { headers: ['Pair ' + api.type('vector', '[2]'), 'θᵢ ' + api.type('float'), 'Period'], rows: rows });
            api.html(el, '<p>The 8B reference has d_head = ' + api.ref.d_head + ' → ' + api.ref.d_head / 2 + ' pairs per head, from very fast to very slow rotations. Llama-3-class models reportedly use a much larger base (≈ 500,000) to support longer contexts; treat this as approximate.</p>');
            api.html(el, '<p>Where it is used: ' + api.link('attention/projections', 'Q and K projections') + '.</p>');
          } }
        ]
      },
      // ---------------------------------------------------------------- input matrix
      {
        id: 'input-matrix', title: 'The input matrix X',
        critical: {
          understand: ['Stack the token vectors (with position information) as rows: that is <b>X [seq_len × d_model]</b>, the input to the first transformer block.',
            'For "The cat sat on the" the toy X is [5 × 4]; the 8B model has [5 × 4,096].'],
          details: ['In RoPE models position is applied inside attention; the course\'s toy X combines embedding + position into one matrix for simplicity.',
            'Row i is the working representation of token i; the layers will update every row in parallel.']
        },
        dataTypes: [
          { name: 'X', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'row of X', type: 'vector', shape: '[d_model]', example: '[0.9, 0.3, −0.2, 0.5]' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[464, 3797, 3332, 319, 262]' }
        ],
        levels: [
          { name: 'X', render: function (el, api) {
            var T = api.ex.tokens;
            api.matrix(el, { data: api.ex.toyX, rowLabels: T.map(function (t) { return api.tokText(t.text); }), colLabels: ['d0', 'd1', 'd2', 'd3'], title: 'X', type: 'matrix', shape: '[5 × 4]', decimals: 1 });
            api.html(el, '<p>Each row = one token. Next stop: ' + api.link('attention/worked-example', 'the attention worked example') + ' uses exactly this matrix.</p>');
            api.illustrative(el, 'Toy values chosen for hand calculation.');
          } },
          { name: 'How built', render: function (el, api) {
            var T = api.ex.tokens;
            api.blocks(el, { blocks: [
              { label: 'Token IDs', sub: '[' + T.map(function (t) { return t.id; }).join(', ') + ']', type: 'integer', shape: '[5]' },
              { label: 'Rows of E', type: 'matrix', shape: '[5 × 4]', edge: 'lookup', goto: 'embeddings/lookup' },
              { label: '+ position info', type: 'matrix', shape: '[5 × 4]', goto: 'embeddings/rope', edge: 'add or RoPE' },
              { label: 'X', type: 'matrix', shape: '[5 × 4]', goto: 'attention/worked-example' }
            ], caption: 'Absolute-position models add position vectors here; RoPE models rotate Q and K later.' });
            api.html(el, '<div class="embeddings-calc">row 1 (" cat") = [' + api.ex.toyX[1].map(function (x) { return api.fmt(x, 1); }).join(', ') + ']</div>');
          } },
          { name: 'Sizes', render: function (el, api) {
            var R = api.ref;
            api.table(el, { headers: ['', 'Toy', '8B reference'], rows: [
              ['X shape ' + api.type('matrix'), '[5 × 4]', '[5 × ' + api.fmtInt(R.d_model) + ']'],
              ['numbers in X', '20', api.fmtInt(5 * R.d_model)],
              ['max context', '—', api.fmtInt(R.context) + ' tokens → [' + api.fmtInt(R.context) + ' × ' + api.fmtInt(R.d_model) + ']']
            ] });
            api.html(el, '<p>The same shape [seq_len × d_model] flows through every block — see ' + api.link('transformer-block/residual-stream', 'the residual stream') + '.</p>');
          } }
        ]
      }
    ]
  });
})();
