(function () {
  'use strict';

  // Fixed illustrative toy weights (d_model = 4, d_ff = 8).
  var W_GATE = [
    [ 0.6, -0.4,  0.3,  0.8, -0.5,  0.2, -0.7,  0.4],
    [-0.3,  0.7,  0.5, -0.2,  0.6, -0.8,  0.1,  0.3],
    [ 0.5,  0.2, -0.6,  0.4,  0.3,  0.7, -0.2, -0.5],
    [-0.4,  0.1,  0.8, -0.6, -0.2,  0.5,  0.9, -0.3]
  ];
  var W_UP = [
    [ 0.4,  0.5, -0.3,  0.6,  0.2, -0.6,  0.7, -0.1],
    [ 0.7, -0.2,  0.4, -0.5, -0.3,  0.3,  0.2,  0.6],
    [-0.5,  0.6,  0.2,  0.1,  0.8,  0.4, -0.4,  0.3],
    [ 0.2, -0.7,  0.5,  0.3, -0.6,  0.1,  0.5, -0.8]
  ];
  var W_DOWN = [
    [ 0.5, -0.3,  0.2,  0.4],
    [-0.2,  0.6,  0.1, -0.5],
    [ 0.3,  0.4, -0.6,  0.2],
    [ 0.7, -0.1,  0.3, -0.2],
    [-0.4,  0.5,  0.6,  0.1],
    [ 0.1, -0.6, -0.3,  0.7],
    [ 0.6,  0.2, -0.5, -0.4],
    [-0.3,  0.3,  0.4,  0.5]
  ];
  var TOKS = ['The', ' cat', ' sat', ' on', ' the'];
  var EXPERT_NAMES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'];

  function swiglu(x, M) {
    var gate = M.vecmat(x, W_GATE), sg = gate.map(M.silu), up = M.vecmat(x, W_UP);
    var hid = sg.map(function (v, i) { return v * up[i]; });
    var out = M.vecmat(hid, W_DOWN);
    return { gate: gate, sg: sg, up: up, hid: hid, out: out };
  }
  function gauss(r) {
    var u = Math.max(r(), 1e-9), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function topK(arr, k) {
    return arr.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return b[0] - a[0]; }).slice(0, k).map(function (p) { return p[1]; });
  }
  function vecRow(api, el, label, v, type, shape, dec) {
    api.matrix(el, { data: [v], title: label, type: type, shape: shape, decimals: dec == null ? 2 : dec, small: true, scale: 'diverging' });
  }

  Course.register({
    id: 'ffn-moe', part: 'I', order: 60, title: 'Feed-Forward & MoE',
    slides: [
      // ------------------------------------------------------------------ 1
      {
        id: 'ffn-role', title: 'The feed-forward network: per-token compute',
        critical: {
          understand: [
            'Attention <b>mixes information between tokens</b>; the FFN then transforms <b>each token\'s vector on its own</b>, with the same weights at every position.',
            'The FFN expands 4096 → 14336 → 4096 dimensions: a wide "thinking space" in the middle.',
            'In the reference model the FFN holds about two thirds of every block\'s parameters.'
          ],
          details: [
            'Position-wise means: output row <i>t</i> depends only on input row <i>t</i>. There is no communication between rows inside the FFN.',
            'Llama-style models use a gated variant (SwiGLU) with three matrices (gate, up, down) instead of two.',
            'Because rows are independent, the FFN is trivially parallel across all tokens during prefill (' + '<a href="#" class="xlink" data-goto="inference/prefill-decode">prefill</a>).'
          ]
        },
        dataTypes: [
          { name: 'x (one token)', type: 'vector', shape: '[d_model]', example: '[4096]' },
          { name: 'hidden activation', type: 'vector', shape: '[d_ff]', example: '[14336]' },
          { name: 'W_gate, W_up', type: 'matrix', shape: '[d_model × d_ff]', example: '[4096 × 14336]' },
          { name: 'W_down', type: 'matrix', shape: '[d_ff × d_model]', example: '[14336 × 4096]' },
          { name: 'FFN output', type: 'vector', shape: '[d_model]', example: '[4096]' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'in', label: 'x', sub: 'residual stream after attention', type: 'matrix', shape: '[seq_len × d_model]' },
              { id: 'norm', label: 'RMSNorm', sub: 'per row', type: 'function', goto: 'transformer-block/rmsnorm' },
              { id: 'ffn', label: 'FFN', sub: '4096 → 14336 → 4096', type: 'function', zoom: true, color: 'var(--t-function)' },
              { id: 'add', label: '+ residual', sub: 'add back to x', type: 'matrix', shape: '[seq_len × d_model]', goto: 'transformer-block/residual-stream' }
            ], caption: 'The second half of every transformer block. Attention came first (' + api.link('attention', 'attention') + ').' });
            api.callout(el, '<b>Attention = communication, FFN = computation.</b> After attention has gathered context into each token\'s vector, the FFN processes every vector independently — a bit like every token going to its own private "thinking" step with the same brain.', 'key');
            var b = api.ref.paramBreakdown();
            api.html(el, '<div class="ffn-moe-note">In the reference model, per layer: attention ≈ <b>' + api.fmtBig(b.attn) + '</b> params, FFN ≈ <b>' + api.fmtBig(b.ffn) + '</b> params (' + api.fmt(100 * b.ffn / b.perLayer, 0) + '% of the layer).</div>');
          } },
          { name: 'Per token', render: function (el, api) {
            var X = api.ex.toyX, M = api.math;
            var sel = 4, bumped = false;
            var box = api.h('div');
            function draw() {
              box.innerHTML = '';
              var X2 = X.map(function (r) { return r.slice(); });
              if (bumped) X2[1] = X2[1].map(function (v) { return v + 0.5; });
              var row = api.h('div', { class: 'ffn-moe-tokrow' });
              X2.forEach(function (x, t) {
                var o = swiglu(x, M).out;
                var changed = bumped && t === 1;
                row.appendChild(api.h('div', { class: 'ffn-moe-col' + (changed ? ' sel' : '') },
                  api.h('div', { html: api.tok(TOKS[t]) }),
                  api.h('div', { class: 'v', text: 'x = [' + x.map(function (v) { return api.fmt(v, 1); }).join(', ') + ']' }),
                  api.h('div', { class: 'ffn-box', text: '↓ same FFN ↓' }),
                  api.h('div', { class: 'v', text: 'out = [' + o.map(function (v) { return api.fmt(v, 2); }).join(', ') + ']' })));
              });
              box.appendChild(row);
              box.appendChild(api.h('div', { class: 'ffn-moe-note', html: bumped ? 'Only the <b>' + api.esc(TOKS[1]) + '</b> column changed: its neighbours\' outputs are identical to before. Contrast with attention, where changing one token would shift every later row.' : 'Five rows in, five rows out. Click the button to perturb one token and see what moves.' }));
            }
            var c = api.h('div', { class: 'controls' });
            api.button(c, 'Perturb the " cat" vector', function () { bumped = !bumped; draw(); }, 'primary');
            el.appendChild(c); el.appendChild(box);
            api.html(el, 'Toy FFN from the next slide (' + api.link('ffn-moe/swiglu', 'SwiGLU') + '), d_model = 4. ' + api.badge('vector', '[d_model]') + ' per token.');
            api.illustrative(el, 'The input rows are ' + api.type('matrix', 'api.ex.toyX') + '; FFN weights are fixed toy numbers.');
          } },
          { name: 'Sizes', render: function (el, api) {
            var r = api.ref, b = api.ref.paramBreakdown();
            var per = r.d_model * r.d_ff;
            api.table(el, { headers: ['Matrix', 'Shape', 'Parameters'], rows: [
              ['W_gate', api.shape('[' + r.d_model + ' × ' + r.d_ff + ']'), api.fmtInt(per)],
              ['W_up', api.shape('[' + r.d_model + ' × ' + r.d_ff + ']'), api.fmtInt(per)],
              ['W_down', api.shape('[' + r.d_ff + ' × ' + r.d_model + ']'), api.fmtInt(per)],
              ['<b>FFN per layer</b>', '', '<b>' + api.fmtInt(b.ffn) + '</b>'],
              ['× ' + r.n_layers + ' layers', '', '<b>' + api.fmtBig(b.ffn * r.n_layers) + '</b> (' + api.fmt(100 * b.ffn * r.n_layers / b.total, 0) + '% of ' + api.fmtBig(b.total) + ')']
            ] });
            api.html(el, '<div class="formula ffn-moe-f">FLOPs per token per layer ≈ 2 × 3 × d_model × d_ff = 2 × 3 × ' + api.fmtInt(r.d_model) + ' × ' + api.fmtInt(r.d_ff) + ' ≈ ' + api.fmtBig(6 * r.d_model * r.d_ff) + '</div>');
            api.html(el, '<div class="ffn-moe-note">The expansion ratio is d_ff / d_model = ' + api.fmt(r.d_ff / r.d_model, 2) + ' (classic transformers used 4×; SwiGLU models shrink it to keep parameter count comparable with three matrices instead of two). Total parameters: see ' + api.link('transformer-block/param-count', 'param count') + '.</div>');
          } }
        ]
      },
      // ------------------------------------------------------------------ 2
      {
        id: 'swiglu', title: 'SwiGLU worked example (d_model = 4, d_ff = 8)',
        critical: {
          understand: [
            'SwiGLU runs two parallel projections: a <b>gate</b> path (through SiLU) and an <b>up</b> path. Their element-wise product is projected back down.',
            'The gate decides, per hidden unit, how much of the "up" signal passes through.',
            'Formula: <code>out = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down</code>.'
          ],
          details: [
            'All matrices here are fixed illustrative numbers; real ones are learned. Shapes: W_gate, W_up are [4 × 8]; W_down is [8 × 4].',
            'The reference model uses exactly this structure with 4096 → 14336 → 4096.',
            'There are no biases in Llama-style FFNs.'
          ]
        },
        dataTypes: [
          { name: 'x', type: 'vector', shape: '[d_model]', example: '[4]' },
          { name: 'W_gate, W_up', type: 'matrix', shape: '[d_model × d_ff]', example: '[4 × 8]' },
          { name: 'gate, up', type: 'vector', shape: '[d_ff]', example: '[8]' },
          { name: 'SiLU', type: 'function', shape: 'scalar → scalar', example: 'x·σ(x)' },
          { name: 'W_down', type: 'matrix', shape: '[d_ff × d_model]', example: '[8 × 4]' },
          { name: 'out', type: 'vector', shape: '[d_model]', example: '[4]' }
        ],
        levels: [
          { name: 'Diagram', render: function (el, api) {
            api.blocks(el, { rows: [[
              { id: 'x', label: 'x', type: 'vector', shape: '[4]' },
              { id: 'g', label: 'x · W_gate', sub: 'then SiLU', type: 'vector', shape: '[8]', edge: 'gate path' },
              { id: 'mul', label: '⊙', sub: 'element-wise product', type: 'vector', shape: '[8]' },
              { id: 'd', label: '· W_down', type: 'vector', shape: '[4]', zoom: true }
            ], [
              { id: 'u', label: 'x · W_up', sub: 'no activation', type: 'vector', shape: '[8]', edge: 'up path' }
            ]], caption: 'Two branches from x; the gate branch is squashed by SiLU and multiplies the up branch.' });
            api.callout(el, 'Compared with a classic FFN (<code>W₂ · act(x·W₁)</code>), the extra <b>up</b> projection gives the network multiplicative interactions: the gate can switch a feature off while the up path carries its content.', 'tip');
            api.html(el, 'Activation functions: ' + api.link('ffn-moe/activations', 'next slide') + '.');
          } },
          { name: 'Numbers', render: function (el, api) {
            var M = api.math, wrap = api.h('div'), tokSel = 4, step = 0;
            var holder = api.h('div');
            var c = api.h('div', { class: 'controls' });
            api.select(c, { label: 'Input token', options: TOKS.map(function (t, i) { return { value: String(i), label: '"' + t + '" (row ' + i + ' of toyX)' }; }), value: '4', onChange: function (v) { tokSel = +v; draw(); } });
            wrap.appendChild(c); wrap.appendChild(holder); el.appendChild(wrap);
            function draw() {
              holder.innerHTML = '';
              var x = api.ex.toyX[tokSel], o = swiglu(x, M), cl = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8'].map(function (_, i) { return 'h' + (i + 1); });
              api.stepper(holder, { start: step, onStep: function (i) { step = i; }, steps: [
                { title: 'Input row x', render: function (b) { vecRow(api, b, 'x = toyX[' + tokSel + '] ("' + api.esc(TOKS[tokSel]) + '")', x, 'vector', '[4]'); } },
                { title: 'gate = x · W_gate', render: function (b) {
                  api.matrix(b, { data: W_GATE, title: 'W_gate', type: 'matrix', shape: '[4 × 8]', small: true, colLabels: cl, scale: 'diverging' });
                  vecRow(api, b, 'gate', o.gate, 'vector', '[8]');
                  api.html(b, '<div class="ffn-moe-note">Each gate value is a dot product of x with a column of W_gate, e.g. h1 = ' + x.map(function (v, k) { return api.fmt(v, 1) + '×' + api.fmt(W_GATE[k][0], 1); }).join(' + ') + ' = <b>' + api.fmt(o.gate[0], 3) + '</b>.</div>');
                } },
                { title: 'SiLU(gate) = gate · σ(gate)', render: function (b) {
                  vecRow(api, b, 'gate', o.gate, 'vector', '[8]'); vecRow(api, b, 'SiLU(gate)', o.sg, 'vector', '[8]', 3);
                  api.html(b, '<div class="ffn-moe-note">Large positive values pass almost unchanged; negative ones are squashed towards 0 (slightly negative dip near −1.3).</div>');
                } },
                { title: 'up = x · W_up', render: function (b) {
                  api.matrix(b, { data: W_UP, title: 'W_up', type: 'matrix', shape: '[4 × 8]', small: true, colLabels: cl, scale: 'diverging' });
                  vecRow(api, b, 'up', o.up, 'vector', '[8]');
                } },
                { title: 'hidden = SiLU(gate) ⊙ up', render: function (b) {
                  vecRow(api, b, 'SiLU(gate)', o.sg, 'vector', '[8]', 3); vecRow(api, b, 'up', o.up, 'vector', '[8]'); vecRow(api, b, 'hidden', o.hid, 'vector', '[8]', 3);
                } },
                { title: 'out = hidden · W_down', render: function (b) {
                  api.matrix(b, { data: W_DOWN, title: 'W_down', type: 'matrix', shape: '[8 × 4]', small: true, colLabels: ['d1', 'd2', 'd3', 'd4'], scale: 'diverging' });
                  vecRow(api, b, 'out', o.out, 'vector', '[4]', 3);
                  var res = x.map(function (v, i) { return v + o.out[i]; });
                  api.html(b, '<div class="ffn-moe-note">Back in the residual stream: x + out = [' + res.map(function (v) { return api.fmt(v, 3); }).join(', ') + '].</div>');
                } }
              ] });
            }
            draw();
            api.illustrative(el, 'W_gate, W_up and W_down are fixed toy matrices; all products are computed live.');
          } },
          { name: 'At scale', render: function (el, api) {
            var r = api.ref;
            api.table(el, { headers: ['', 'Toy', 'Reference model'], rows: [
              ['d_model', '4', api.fmtInt(r.d_model)],
              ['d_ff (hidden)', '8', api.fmtInt(r.d_ff)],
              ['Hidden units gated', '8', api.fmtInt(r.d_ff)],
              ['Params in 3 matrices', api.fmtInt(3 * 4 * 8), api.fmtInt(3 * r.d_model * r.d_ff)]
            ] });
            api.callout(el, 'Same computation, bigger matrices. For one token that is two matrix–vector products of [4096 × 14336], an element-wise multiply, and one of [14336 × 4096]. During <b>decode</b> the cost is dominated by <i>reading</i> these weights from memory (' + api.link('inference/prefill-decode', 'prefill vs decode') + ').', 'info');
          } }
        ]
      },
      // ------------------------------------------------------------------ 3
      {
        id: 'activations', title: 'Activation functions: ReLU, GELU, SiLU',
        critical: {
          understand: [
            'Without a non-linearity, stacking matrix multiplications collapses into one matrix. The activation is what makes depth useful.',
            'ReLU = max(0, x) is simple and sparse; GELU and SiLU are smooth versions that let small negative values through.',
            'Modern LLMs usually use SiLU (as "Swish") inside a gated FFN.'
          ],
          details: [
            'SiLU(x) = x · σ(x), where σ is the logistic sigmoid.',
            'GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³))) (the common tanh approximation).',
            'Which one is "best" is empirical; differences are small compared with architecture and data.'
          ]
        },
        dataTypes: [
          { name: 'x', type: 'float', shape: 'scalar', example: '0.8' },
          { name: 'ReLU / GELU / SiLU', type: 'function', shape: 'scalar → scalar', example: 'element-wise' },
          { name: 'activation output', type: 'float', shape: 'scalar', example: '0.63' }
        ],
        levels: [
          { name: 'Curves', render: function (el, api) {
            var M = api.math, xs = M.linspace(-4, 4, 81);
            api.chart(el, { series: [
              { name: 'ReLU', points: xs.map(function (x) { return [x, M.relu(x)]; }), color: '#ff9a4d' },
              { name: 'GELU', points: xs.map(function (x) { return [x, M.gelu(x)]; }), color: '#5ad18a' },
              { name: 'SiLU', points: xs.map(function (x) { return [x, M.silu(x)]; }), color: '#4ea8ff' }
            ], xLabel: 'input x', yLabel: 'activation(x)', xRange: [-4, 4], yRange: [-1, 4], height: 280 });
            api.callout(el, 'All three are ≈ identity for large positive x and ≈ 0 for large negative x. They differ near zero: ReLU has a sharp corner; GELU/SiLU are smooth and dip slightly below zero.', 'info');
          } },
          { name: 'Probe', render: function (el, api) {
            var M = api.math, out = api.h('div', { class: 'ffn-moe-probe' }), plot = null;
            var c = api.h('div', { class: 'controls' });
            var xs = M.linspace(-4, 4, 81);
            function mk(name, f, color) { return { name: name, points: xs.map(function (x) { return [x, f(x)]; }), color: color }; }
            var base = [mk('ReLU', M.relu, '#ff9a4d'), mk('GELU', M.gelu, '#5ad18a'), mk('SiLU', M.silu, '#4ea8ff')];
            var holder = api.h('div');
            el.appendChild(c); el.appendChild(holder); el.appendChild(out);
            plot = api.chart(holder, { series: base, xRange: [-4, 4], yRange: [-1, 4], height: 260, xLabel: 'x', markers: [{ x: 0, label: 'x = 0' }] });
            api.slider(c, { label: 'Probe x', min: -4, max: 4, step: 0.1, value: 0.8, format: function (v) { return api.fmt(v, 1); }, onInput: function (v) {
              out.innerHTML = '';
              [['ReLU', M.relu], ['GELU', M.gelu], ['SiLU', M.silu]].forEach(function (p) {
                api.card(out, p[0] + '(' + api.fmt(v, 1) + ')', '<div class="big-num">' + api.fmt(p[1](v), 3) + '</div>');
              });
              plot.update({ markers: [{ x: v, label: 'x = ' + api.fmt(v, 1) }] });
            } });
            api.html(el, '<div class="ffn-moe-note">Try x = −1: ReLU gives exactly 0, while GELU and SiLU return small negative values — gradient still flows during training.</div>');
          } }
        ]
      },
      // ------------------------------------------------------------------ 4
      {
        id: 'kv-memory', title: 'The FFN as key–value memory (an intuition)',
        critical: {
          understand: [
            'One way to read an FFN: each hidden unit is a <b>pattern detector</b> (a "key" = a column of W_up/W_gate) paired with an <b>output direction</b> (a "value" = a row of W_down).',
            'How strongly a pattern matches decides how much of its value is added to the residual stream.',
            'This is an <b>interpretive lens</b>, not a proven description of what every unit does.'
          ],
          details: [
            'Research (e.g. "Transformer Feed-Forward Layers Are Key-Value Memories", Geva et al. 2021) found some units that respond to recognisable patterns; many units are polysemantic and messy.',
            'Facts such as "Paris is in France" are <em>often hypothesised</em> to be stored largely in FFN weights; evidence is suggestive, not settled.',
            'Do not confuse with the attention KV cache (' + '<a href="#" class="xlink" data-goto="inference/kv-cache">KV cache</a>), which stores per-token keys/values of the <i>context</i>.'
          ]
        },
        dataTypes: [
          { name: 'keys (cols of W_up)', type: 'matrix', shape: '[d_model × d_ff]', example: '[4 × 8]' },
          { name: 'match scores', type: 'vector', shape: '[d_ff]', example: '[8]' },
          { name: 'values (rows of W_down)', type: 'matrix', shape: '[d_ff × d_model]', example: '[8 × 4]' },
          { name: 'memory read-out', type: 'vector', shape: '[d_model]', example: '[4]' }
        ],
        levels: [
          { name: 'Idea', render: function (el, api) {
            api.blocks(el, { blocks: [
              { label: 'x', sub: 'query-like input', type: 'vector', shape: '[d_model]' },
              { label: 'match keys', sub: 'x · W_up  (gated)', type: 'vector', shape: '[d_ff]', zoom: true },
              { label: 'weights', sub: 'how strongly each pattern fires', type: 'vector', shape: '[d_ff]' },
              { label: 'sum of values', sub: 'Σ weightᵢ · row i of W_down', type: 'vector', shape: '[d_model]' }
            ], caption: 'Like a soft lookup table with 14,336 slots per layer.' });
            api.callout(el, 'It resembles attention (compare a query against keys, read out values), but the keys and values are <b>fixed weights</b> learned in training — not computed from the context.', 'hedge');
          } },
          { name: 'Toy lookup', render: function (el, api) {
            var M = api.math;
            var keys = ['"animal" pattern', '"furniture" pattern', '"place" pattern', '"verb" pattern'];
            var K = [[1, 0.2, -0.3, 0], [-0.2, 1, 0.3, 0.1], [0.1, -0.3, 1, 0.2], [0, 0.1, -0.2, 1]];
            var V = [[0.9, 0.1, 0, 0], [0, 0.8, 0.3, 0], [0, 0.1, 0.9, 0.2], [0.1, 0, 0.2, 0.9]];
            var vlabels = ['feat A', 'feat B', 'feat C', 'feat D'];
            var out = api.h('div');
            var ex = [{ n: '" cat"', x: api.ex.toyX[1] }, { n: '" sat"', x: api.ex.toyX[2] }, { n: '" on"', x: api.ex.toyX[3] }];
            var sel = 0;
            var c = api.h('div', { class: 'controls' });
            api.select(c, { label: 'Input', options: ex.map(function (e, i) { return { value: String(i), label: e.n }; }), value: '0', onChange: function (v) { sel = +v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            function draw() {
              out.innerHTML = '';
              var x = ex[sel].x;
              var score = K.map(function (k) { return M.dot(k, x); });
              var w = score.map(M.silu);
              var res = M.vecmat(w, V);
              api.bars(out, { title: 'Pattern match strength (SiLU of x · key)', items: keys.map(function (k, i) { return { label: k, value: Math.max(0, w[i]), highlight: i === M.argmax(w) }; }), format: function (v) { return api.fmt(v, 3); }, max: 1 });
              api.matrix(out, { data: V, title: 'Value rows (row i of W_down)', rowLabels: keys, colLabels: vlabels, small: true, type: 'matrix', shape: '[4 × 4]' });
              vecRow(api, out, 'Read-out = Σ weightᵢ · valueᵢ', res, 'vector', '[4]', 3);
            }
            draw();
            api.illustrative(el, 'Keys, values and labels are invented to show the mechanism; real units are rarely this tidy.');
          } }
        ]
      },
      // ------------------------------------------------------------------ 5
      {
        id: 'moe', title: 'Mixture of Experts: router, top-k, load balancing',
        critical: {
          understand: [
            'An MoE layer replaces one big FFN with <b>E expert FFNs</b> plus a small <b>router</b>. Each token is sent to only its <b>top-k</b> experts (e.g. 2 of 8).',
            'Total parameters grow with E, but compute per token only grows with k: more knowledge capacity at similar FLOPs.',
            'Training adds a <b>load-balancing</b> loss so tokens spread across experts instead of collapsing onto a few.'
          ],
          details: [
            'Router: <code>logits = x · W_r</code> ([d_model × E]), softmax, keep top-k, renormalise those k weights; the output is the weighted sum of the chosen experts.',
            'All experts must still sit in memory, so MoE saves compute, not weight memory. Mixtral-style 8×7B has ≈ 47 B total but ≈ 13 B active parameters.',
            'The reference model is <b>dense</b>; the MoE numbers here are an illustrative variant of it.'
          ]
        },
        dataTypes: [
          { name: 'x', type: 'vector', shape: '[d_model]', example: '[4]' },
          { name: 'W_router', type: 'matrix', shape: '[d_model × E]', example: '[4 × 8]' },
          { name: 'router logits', type: 'vector', shape: '[E]', example: '[8]' },
          { name: 'gate weights', type: 'probability', shape: '[E]', example: 'top-2 sum to 1' },
          { name: 'chosen experts', type: 'integer', shape: 'list[k]', example: '[2, 5]' },
          { name: 'top-k mask', type: 'mask', shape: '[E]', example: '8 × bool' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var b = api.blocks(el, { rows: [[
              { id: 'x', label: 'token vector x', type: 'vector', shape: '[d_model]' },
              { id: 'r', label: 'Router', sub: 'x · W_r → softmax → top-k', type: 'function', zoom: true, color: 'var(--accent)' },
              { id: 'e', label: 'k chosen experts', sub: 'each is a SwiGLU FFN', type: 'function', zoom: 3 },
              { id: 'o', label: 'Σ gᵢ · expertᵢ(x)', type: 'vector', shape: '[d_model]' }
            ]], caption: 'Only k of E experts run for this token; the others stay idle.' });
            var row = api.h('div'); for (var i = 0; i < 8; i++) row.appendChild(api.h('span', { class: 'ffn-moe-expert ' + (i === 2 || i === 5 ? 'on' : 'off'), text: 'Expert ' + i }));
            el.appendChild(row);
            api.callout(el, 'Different tokens go to different experts. Experts are <em>not</em> hand-assigned topics; any specialisation emerges in training and is often subtler than "this one does maths".', 'hedge');
          } },
          { name: 'Router', render: function (el, api) {
            var M = api.math, E = 8;
            // fixed toy router weights [4 x 8]
            var WR = [[0.9, -0.5, 0.2, 0.1, -0.8, 0.6, -0.2, 0.3], [-0.4, 0.8, 0.5, -0.6, 0.2, -0.3, 0.7, 0.1], [0.3, 0.1, -0.7, 0.9, 0.4, -0.2, 0.5, -0.6], [-0.2, 0.6, 0.4, -0.3, 0.9, 0.5, -0.8, 0.2]];
            var sel = 1, k = 2, out = api.h('div');
            var c = api.h('div', { class: 'controls' });
            api.select(c, { label: 'Token', options: TOKS.map(function (t, i) { return { value: String(i), label: '"' + t + '"' }; }), value: '1', onChange: function (v) { sel = +v; draw(); } });
            api.slider(c, { label: 'top-k', min: 1, max: 4, step: 1, value: 2, init: false, onInput: function (v) { k = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            function draw() {
              out.innerHTML = '';
              var x = api.ex.toyX[sel], logits = M.vecmat(x, WR), p = M.softmax(logits), top = topK(p, k);
              var psum = M.sum(top.map(function (i) { return p[i]; }));
              var g = p.map(function (v, i) { return top.indexOf(i) >= 0 ? v / psum : 0; });
              vecRow(api, out, 'router logits = x · W_r', logits, 'vector', '[8]', 2);
              api.bars(out, { title: 'softmax(logits)  — top-' + k + ' highlighted', max: 1, items: p.map(function (v, i) { return { label: EXPERT_NAMES[i], value: v, highlight: top.indexOf(i) >= 0, dim: top.indexOf(i) < 0 }; }), format: function (v) { return api.fmt(v, 3); } });
              api.bars(out, { title: 'Renormalised gate weights over chosen experts (sum = 1)', max: 1, items: top.map(function (i) { return { label: EXPERT_NAMES[i], value: g[i], color: 'var(--good)' }; }), format: function (v) { return api.fmt(v, 3); } });
              api.html(out, '<div class="formula ffn-moe-f">y = ' + top.map(function (i) { return api.fmt(g[i], 2) + '·' + EXPERT_NAMES[i] + '(x)'; }).join(' + ') + '</div>');
              api.html(out, '<div class="ffn-moe-note">Experts used: ' + top.map(function (i) { return api.type('integer', String(i)); }).join(' ') + ' — computed for this token only; the other ' + (E - k) + ' experts do no work for it.</div>');
            }
            draw();
            api.illustrative(el, 'The router weights W_r are fixed toy numbers.');
          } },
          { name: 'Load balance', render: function (el, api) {
            var M = api.math, E = 8, N = 2000;
            var bias = [1.0, 0.6, 0.2, 0.0, -0.2, -0.5, -0.8, -1.0];
            var out = api.h('div'), c = api.h('div', { class: 'controls' });
            el.appendChild(c); el.appendChild(out);
            function draw(skew) {
              var r = M.rng(2024), cnt = new Array(E).fill(0), psum = new Array(E).fill(0);
              for (var t = 0; t < N; t++) {
                var lg = bias.map(function (b) { return skew * b + gauss(r); });
                var p = M.softmax(lg), tp = topK(lg, 2);
                tp.forEach(function (i) { cnt[i]++; });
                p.forEach(function (v, i) { psum[i] += v; });
              }
              var f = cnt.map(function (v) { return v / (2 * N); }), P = psum.map(function (v) { return v / N; });
              var aux = E * M.sum(f.map(function (fi, i) { return fi * P[i]; }));
              out.innerHTML = '';
              api.bars(out, { title: 'Share of routing slots per expert (' + api.fmtInt(N) + ' tokens × top-2; ideal = 12.5%)', max: 0.5, items: f.map(function (v, i) { return { label: EXPERT_NAMES[i], value: v, color: v > 0.2 ? 'var(--bad)' : v < 0.06 ? 'var(--warn)' : 'var(--good)' }; }), format: function (v) { return api.fmt(100 * v, 1) + '%'; } });
              api.html(out, '<div class="formula ffn-moe-f">aux loss = E · Σ fᵢ·Pᵢ = ' + api.fmt(aux, 3) + '   (1.0 when perfectly balanced; larger = more imbalanced)</div>');
              api.html(out, '<div class="ffn-moe-note">fᵢ = fraction of tokens routed to expert i, Pᵢ = mean router probability. Minimising this auxiliary loss pushes the router to spread load. Overloaded experts become bottlenecks and, if capacity-limited, drop tokens. Some newer designs use bias-based balancing instead of an aux loss.</div>');
            }
            api.slider(c, { label: 'Router skew', min: 0, max: 3, step: 0.1, value: 1.2, format: function (v) { return api.fmt(v, 1); }, onInput: draw });
            api.illustrative(el, 'Simulated routing with a seeded random generator: fixed popularity bias per expert × skew + Gaussian noise.');
          } },
          { name: 'Params', render: function (el, api) {
            var r = api.ref, b = api.ref.paramBreakdown(), out = api.h('div'), c = api.h('div', { class: 'controls' });
            var E = 8, k = 2;
            el.appendChild(c); el.appendChild(out);
            function draw() {
              var shared = b.embed + b.unembed + b.finalNorm + r.n_layers * (b.attn + b.norms + r.d_model * E);
              var total = shared + r.n_layers * E * b.ffn, active = shared + r.n_layers * k * b.ffn;
              out.innerHTML = '';
              api.bars(out, { title: 'Parameters (illustrative MoE built from reference-model dims)', items: [
                { label: 'Dense reference', value: b.total, color: 'var(--muted)', note: api.fmtBig(b.total) },
                { label: 'MoE total', value: total, color: 'var(--accent)', note: api.fmtBig(total) },
                { label: 'MoE active', value: active, color: 'var(--good)', note: api.fmtBig(active) }
              ], max: Math.max(total, b.total) * 1.02, format: function () { return ''; } });
              api.html(out, '<div class="formula ffn-moe-f">active = shared + layers × k × FFN = ' + api.fmtBig(active) + ' (' + api.fmt(100 * active / total, 0) + '% of total)</div>');
              api.html(out, '<div class="ffn-moe-note">Weights in memory (BF16): ≈ ' + api.fmtBytes(total * 2) + ' vs ' + api.fmtBytes(b.bytesBF16) + ' dense. Per-token compute scales with <i>active</i> parameters. See ' + api.link('inference/quantization', 'quantization') + ' for shrinking the memory side.</div>');
            }
            api.slider(c, { label: 'Experts E', min: 2, max: 64, step: 1, value: 8, onInput: function (v) { E = v; if (k > E) k = E; draw(); } });
            api.slider(c, { label: 'top-k', min: 1, max: 8, step: 1, value: 2, onInput: function (v) { k = Math.min(v, E); draw(); } });
            api.illustrative(el, 'Attention, embeddings and norms are copied from the dense reference; only the FFN is multiplied. Real MoEs (Mixtral, DeepSeek, …) differ in details such as shared experts and fine-grained experts.');
          } }
        ]
      }
    ]
  });
})();
