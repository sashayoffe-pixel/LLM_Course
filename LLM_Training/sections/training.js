/* Section: training — how an LLM is trained (pretraining → SFT → preference tuning), loss, optimisation, scaling, LoRA. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- local helpers
  function ctl(api, el) { return api.h('div', { class: 'controls' }); }
  function sci(x, d) { // 7.23e23 -> "7.23 × 10²³"
    if (!isFinite(x) || x === 0) return String(x);
    var e = Math.floor(Math.log10(Math.abs(x))), m = x / Math.pow(10, e);
    if (m >= 9.995) { m /= 10; e += 1; }
    var sup = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return m.toFixed(d == null ? 2 : d) + ' × 10' + String(e).split('').map(function (c) { return sup[c]; }).join('');
  }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  function row(api, el) { var r = api.h('div', { class: 'row' }); el.appendChild(r); return r; }

  // Illustrative next-token probabilities used for " The cat sat on the" → teacher forcing
  var TF = [ // input prefix ends with ... → target, illustrative p(target)
    { ctx: 'The', target: ' cat', p: 0.012 },
    { ctx: 'The cat', target: ' sat', p: 0.09 },
    { ctx: 'The cat sat', target: ' on', p: 0.41 },
    { ctx: 'The cat sat on', target: ' the', p: 0.62 },
    { ctx: 'The cat sat on the', target: ' mat', p: 0.31 }
  ];
  var TF_IDS = [3797, 3332, 319, 262, 2603];

  // ---------------------------------------------------------------- slide 1: lifecycle
  var lifecycle = {
    id: 'lifecycle',
    title: 'The training lifecycle: pretraining → SFT → preference tuning → eval & deploy',
    critical: {
      understand: [
        '<b>Pretraining</b> on a huge pile of text teaches next-token prediction — this is where almost all knowledge and compute go.',
        '<b>SFT</b> (supervised fine-tuning) on curated conversations teaches the <i>format</i> of being an assistant.',
        '<b>Preference tuning</b> (RLHF, DPO, RLAIF…) nudges which of several plausible answers the model prefers.',
        'Every stage uses the same network and the same forward pass — only the data and the loss change.'
      ],
      details: [
        'Pretraining corpora for 8B-class models are commonly reported in the <b>trillions</b> of tokens (Llama 3: ~15 T reported).',
        'SFT / preference data is tiny by comparison: often thousands to millions of examples.',
        'Evaluation (benchmarks, red-teaming, human review) gates each release; exact recipes vary by lab and are often unpublished.'
      ]
    },
    dataTypes: [
      { name: 'raw corpus', type: 'text', shape: 'list[str]', example: 'web pages, books, code' },
      { name: 'training tokens', type: 'integer', shape: '[n_tokens]', example: '≈ 15 T token IDs' },
      { name: 'model weights θ', type: 'tensor', shape: '[≈ 8.03 B params]', example: 'BF16, ≈ 16 GB' },
      { name: 'chat examples', type: 'json', shape: 'list[object]', example: '{messages:[…]}' },
      { name: 'preference pairs', type: 'json', shape: 'list[object]', example: '{prompt, chosen, rejected}' },
      { name: 'loss', type: 'float', shape: 'scalar', example: '≈ 1.8 nats' }
    ],
    levels: [
      {
        name: 'Overview',
        render: function (el, api) {
          api.html(el, '<p>A chat model is built in stages. Each stage starts from the previous weights and changes them a little. Click a block to jump to its slide.</p>');
          api.blocks(el, {
            rows: [[
              { id: 'data', label: 'Raw text', sub: 'web, books, code', type: 'text', shape: 'list[str]', goto: 'tokenization/why-tokens' },
              { id: 'pre', label: 'Pretraining', sub: 'predict next token', type: 'float', shape: 'loss: scalar', goto: 'training/cross-entropy', edge: 'tokens' },
              { id: 'base', label: 'Base model', sub: 'autocompletes text', type: 'tensor', shape: '[8.03 B]' }
            ], [
              { id: 'sft', label: 'SFT', sub: 'imitate good chats', type: 'json', shape: 'messages', goto: 'training/sft' },
              { id: 'pref', label: 'Preference tuning', sub: 'RLHF / DPO / RLAIF', type: 'json', shape: '{chosen, rejected}', goto: 'training/rlhf' },
              { id: 'eval', label: 'Eval & deploy', sub: 'benchmarks, red-team, serve', type: 'function', goto: 'inference/prefill-decode' }
            ]],
            caption: 'Same network throughout; only the data and the objective change.'
          });
          api.callout(el, 'After pretraining, “The cat sat on the” → “ mat” works fine, but asking a <i>question</i> may get you… more questions, because the base model just continues documents. SFT and preference tuning turn it into an assistant.', 'key');
        }
      },
      {
        name: 'Stages compared',
        render: function (el, api) {
          api.table(el, {
            headers: ['Stage', 'Data ' + api.type('text'), 'Typical size (hedged)', 'Objective', 'What changes'],
            rows: [
              ['Pretraining', 'Filtered web, books, code, math', 'trillions of tokens', 'next-token cross-entropy on every token', 'knowledge, grammar, skills'],
              ['SFT / instruction tuning', 'Curated prompt → answer chats', '≈ 10³–10⁶ examples', 'cross-entropy on <b>assistant</b> tokens only', 'format, following instructions'],
              ['Preference tuning', 'Pairs: chosen vs rejected answer', '≈ 10⁴–10⁶ comparisons', 'reward model + RL, or DPO', 'helpfulness, tone, refusals'],
              ['RL with verifiable rewards', 'Math/code problems with checkers', 'varies', 'reward = answer correct?', 'reasoning ' + api.link('reasoning/reasoning-models', '(see reasoning)')],
              ['Eval & deploy', 'Held-out benchmarks, red-team prompts', '—', 'no training: measure', 'go / no-go, quantize, serve']
            ]
          });
          api.callout(el, 'Sizes are rough orders of magnitude from public reports; labs rarely publish full recipes, and many mix stages (e.g. several SFT + preference rounds).', 'hedge');
        }
      },
      {
        name: 'Data budget',
        render: function (el, api) {
          var out = api.h('div');
          var c = ctl(api); el.appendChild(c); el.appendChild(out);
          api.slider(c, { label: 'pretraining tokens (T)', min: 1, max: 20, step: 1, value: 15, format: function (v) { return v + ' T'; },
            onInput: function (T) {
              var tokens = T * 1e12, words = tokens * 0.75, wpm = 250;
              var years = words / wpm / 60 / 24 / 365;
              var bytesIds = tokens * 4; // int32 ids
              var N = api.ref.paramBreakdown().total;
              out.innerHTML = '<div class="grid training-kpis">' +
                '<div class="card"><div class="muted">tokens ' + api.badge('integer', 'scalar') + '</div><div class="big-num">' + api.fmtBig(tokens) + '</div></div>' +
                '<div class="card"><div class="muted">≈ words (0.75 words/token)</div><div class="big-num">' + api.fmtBig(words) + '</div></div>' +
                '<div class="card"><div class="muted">reading non-stop at ' + wpm + ' wpm</div><div class="big-num">' + api.fmtInt(years) + ' yrs</div></div>' +
                '<div class="card"><div class="muted">tokens per parameter (8.03 B)</div><div class="big-num">' + api.fmtInt(tokens / N) + '</div></div>' +
                '</div><p class="muted">Token IDs alone stored as int32: ' + api.fmtBytes(bytesIds) + '. Compare: the trained weights are only ' + api.fmtBytes(api.ref.paramBreakdown().bytesBF16) + ' (BF16) — training <i>compresses</i> the corpus into patterns, not a copy.</p>';
            } });
          api.illustrative(el, '0.75 words/token and 250 words/minute are rough rules of thumb for English.');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 2: cross-entropy
  var crossEntropy = {
    id: 'cross-entropy',
    title: 'Cross-entropy loss and teacher forcing',
    critical: {
      understand: [
        'Loss for one position = <b>−log p(correct next token)</b>. Confident and right → near 0; confident and wrong → large.',
        '<b>Teacher forcing</b>: feed the <i>real</i> text, predict every next token at once. Targets are the inputs shifted left by one.',
        'One sequence of n tokens gives n−1 training signals in a single forward pass (thanks to the ' + '<a href="#" class="xlink" data-goto="attention/causal-mask">causal mask</a>).'
      ],
      details: [
        'Training loss = mean of −log p over all predicted positions (in nats when using ln).',
        'Perplexity = exp(mean loss): “as uncertain as choosing among k equally likely tokens”.',
        'Computed from logits via log-softmax for numerical stability; the full distribution is over the whole vocab (128,256 for the ref model).'
      ]
    },
    dataTypes: [
      { name: 'input tokens', type: 'token', shape: '[seq_len]', example: 'The ␣cat ␣sat ␣on ␣the' },
      { name: 'target IDs', type: 'integer', shape: '[seq_len]', example: '3797 3332 319 262 2603' },
      { name: 'logits', type: 'matrix', shape: '[seq_len × vocab]', example: '[5 × 128,256]' },
      { name: 'p(next token)', type: 'probability', shape: '[vocab]', example: 'p(␣mat) = 0.31' },
      { name: 'loss', type: 'float', shape: 'scalar', example: '−ln 0.31 = 1.17' }
    ],
    levels: [
      {
        name: 'One position',
        render: function (el, api) {
          api.html(el, '<p>The model sees ' + api.ex.tokens.map(function (t) { return api.tok(t.text); }).join('') + ' and outputs a probability for every token in the vocab. The training text says the next token is ' + api.tok(' mat', 2603) + '.</p>');
          var probs = [[' mat', 0.31], [' floor', 0.22], [' couch', 0.14], [' bed', 0.09], [' table', 0.06], [' roof', 0.03]];
          var other = 1 - probs.reduce(function (s, p) { return s + p[1]; }, 0);
          api.bars(el, { title: 'p(next token) ' + api.badge('probability', '[vocab]'),
            items: probs.map(function (p) { return { label: api.tok(p[0]), value: p[1], highlight: p[0] === ' mat' }; }).concat([{ label: 'all other 128,250', value: other, dim: true }]),
            max: 0.4 });
          var L = -Math.log(0.31);
          api.html(el, '<div class="formula">loss = −ln p(␣mat) = −ln 0.31 = <b>' + api.fmt(L, 3) + '</b> ' + api.badge('float', 'scalar') + '</div>');
          api.callout(el, 'Only the probability of the <b>correct</b> token enters the loss — but because of softmax, pushing it up automatically pushes the others down.', 'key');
          api.illustrative(el);
        }
      },
      {
        name: 'Teacher forcing',
        render: function (el, api) {
          api.html(el, '<p>Inputs and targets are the <b>same sequence shifted by one</b>. All positions are predicted in parallel in one forward pass; the causal mask stops each position from peeking at its own answer.</p>');
          var inp = api.ex.tokens, tgt = TF.map(function (r, i) { return { text: r.target, id: TF_IDS[i] }; });
          api.html(el, '<div class="training-shift"><div><span class="muted">input ' + api.badge('token', '[5]') + '</span></div></div>');
          var wrap = el.lastChild;
          api.tokens(wrap, inp);
          wrap.appendChild(api.h('div', { class: 'muted training-shift-arrow', text: '↓ predict each next token (shift left by 1)' }));
          wrap.appendChild(api.h('div', { html: '<span class="muted">target ' + api.badge('integer', '[5]') + '</span>' }));
          api.tokens(wrap, tgt);
          var total = 0;
          var rows = TF.map(function (r, i) {
            var l = -Math.log(r.p); total += l;
            return [i, api.esc(r.ctx).replace(/ /g, '␣'), api.tok(r.target, TF_IDS[i]), api.fmt(r.p, 3), api.fmt(l, 3)];
          });
          var mean = total / TF.length;
          rows.push(['', '<b>mean</b>', '', '', '<b>' + api.fmt(mean, 3) + '</b>']);
          api.table(el, { headers: ['pos', 'sees', 'target', 'p(target) ' + api.type('probability'), '−ln p ' + api.type('float')], rows: rows });
          api.html(el, '<div class="formula">sequence loss = (1/5)·Σ −ln p = ' + api.fmt(mean, 3) + ' nats · perplexity = e<sup>' + api.fmt(mean, 3) + '</sup> = ' + api.fmt(Math.exp(mean), 1) + '</div>');
          api.illustrative(el, 'Probabilities are made up. In practice “The” is also predicted from a beginning-of-sequence token.');
        }
      },
      {
        name: 'Loss curve',
        render: function (el, api) {
          api.html(el, '<p>Drag the probability the model gives the correct token and watch the loss.</p>');
          var c = ctl(api), out = api.h('div'), chartBox = api.h('div');
          el.appendChild(c); el.appendChild(out); el.appendChild(chartBox);
          var pts = []; for (var i = 1; i <= 100; i++) pts.push([i / 100, -Math.log(i / 100)]);
          var ch = api.chart(chartBox, { series: [{ name: '−ln p', points: pts }], xLabel: 'p(correct token)', yLabel: 'loss (nats)', yRange: [0, 4.7], height: 230 });
          api.slider(c, { label: 'p(␣mat)', min: 0.01, max: 0.99, step: 0.01, value: 0.31, format: function (v) { return v.toFixed(2); },
            onInput: function (p) {
              var L = -Math.log(p);
              out.innerHTML = '<div class="row"><div class="formula">loss = −ln ' + p.toFixed(2) + ' = <b>' + api.fmt(L, 3) + '</b></div>' +
                '<div class="formula">perplexity = 1/p = ' + api.fmt(1 / p, 2) + '</div>' +
                '<div class="formula">gradient ∂L/∂logit(␣mat) = p − 1 = ' + api.fmt(p - 1, 2) + '</div></div>';
              ch.update({ markers: [{ x: p, label: 'loss ' + api.fmt(L, 2) }] });
            } });
          api.callout(el, 'Uniform guessing over 128,256 tokens gives loss ln(128,256) = <b>' + api.fmt(Math.log(api.ref.vocab), 2) + '</b> — roughly where an untrained model starts. Well-trained 8B-class models tend to reach ≈ 1.5–2 nats on typical web text (hedged; depends on data).', 'info');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 3: backprop
  // tiny model: hidden h [3] → logits = W h (W [3 × 3]) over a 3-token vocab; target index 0 (" mat")
  var TINY = {
    vocab: [' mat', ' floor', ' roof'],
    h: [1.0, 0.5, -0.5],
    W: [[0.2, -0.1, 0.4], [0.3, 0.2, -0.2], [-0.1, 0.1, 0.3]],
    target: 0,
    lr: 0.5
  };
  function tinyStep(api) {
    var M = api.math, z = M.matvec(TINY.W, TINY.h), p = M.softmax(z), L = -Math.log(p[TINY.target]);
    var y = TINY.vocab.map(function (_, i) { return i === TINY.target ? 1 : 0; });
    var dz = M.sub(p, y);
    var dW = dz.map(function (g) { return TINY.h.map(function (hj) { return g * hj; }); });
    var W2 = M.sub(TINY.W, M.scale(dW, TINY.lr));
    var z2 = M.matvec(W2, TINY.h), p2 = M.softmax(z2), L2 = -Math.log(p2[TINY.target]);
    return { z: z, p: p, L: L, y: y, dz: dz, dW: dW, W2: W2, z2: z2, p2: p2, L2: L2 };
  }
  // gradient descent on a 1-D quadratic loss L(w) = a (w - w*)^2 + L0 (illustrative "loss valley")
  function gdCurve(lr, steps) {
    var a = 1, w = 3, wStar = 0.5, L0 = 0.2, pts = [];
    for (var t = 0; t <= steps; t++) {
      var L = a * (w - wStar) * (w - wStar) + L0;
      pts.push([t, Math.min(L, 50)]);
      w = w - lr * 2 * a * (w - wStar);
      if (!isFinite(w) || Math.abs(w) > 1e6) { w = 1e6; }
    }
    return pts;
  }

  var backprop = {
    id: 'backprop',
    title: 'Backprop and gradient descent: one worked update, learning rate, AdamW',
    critical: {
      understand: [
        'Training loop: forward pass → loss → <b>backward pass</b> (chain rule gives ∂loss/∂every weight) → small step downhill.',
        'For softmax + cross-entropy the gradient at the logits is beautifully simple: <b>p − y</b> (prediction minus one-hot target).',
        'The <b>learning rate</b> is the step size: too small crawls, too large overshoots and diverges.',
        '<b>AdamW</b> adapts the step per weight using running averages of gradients and squared gradients, plus decoupled weight decay.'
      ],
      details: [
        'Backward costs ≈ 2× the forward FLOPs, so a training step ≈ 3× a forward pass (hence ≈ 6·N FLOPs per token).',
        'Real runs use a warm-up then a decaying schedule (e.g. cosine) and gradient clipping; peak LRs for ~8B models are often around 10⁻⁴ (hedged).',
        'AdamW keeps 2 extra floats per parameter (m and v) — a big part of training memory.'
      ]
    },
    dataTypes: [
      { name: 'hidden state h', type: 'vector', shape: '[3]', example: '[1, 0.5, −0.5]' },
      { name: 'weights W', type: 'matrix', shape: '[3 × 3]', example: 'toy unembedding' },
      { name: 'logits z', type: 'vector', shape: '[3]', example: 'W·h' },
      { name: 'p', type: 'probability', shape: '[3]', example: 'softmax(z)' },
      { name: 'one-hot target y', type: 'vector', shape: '[3]', example: '[1, 0, 0]' },
      { name: 'gradient ∂L/∂W', type: 'matrix', shape: '[3 × 3]', example: '(p − y)·hᵀ' },
      { name: 'learning rate η', type: 'float', shape: 'scalar', example: '0.5' },
      { name: 'Adam moments m, v', type: 'float', shape: 'per weight', example: 'm̂, v̂' }
    ],
    levels: [
      {
        name: 'The loop',
        render: function (el, api) {
          var b = api.blocks(el, {
            blocks: [
              { id: 'f', label: 'Forward', sub: 'tokens → logits', type: 'matrix', shape: '[seq × vocab]', goto: 'big-picture/forward-pass' },
              { id: 'l', label: 'Loss', sub: 'mean −ln p(target)', type: 'float', shape: 'scalar', goto: 'training/cross-entropy', edge: 'p' },
              { id: 'b', label: 'Backward', sub: 'chain rule', type: 'tensor', shape: '∂L/∂θ [8.03 B]', zoom: 1, edge: '∂L' },
              { id: 'u', label: 'Update', sub: 'θ ← θ − η·step', type: 'function', zoom: 3, edge: 'grads' }
            ],
            caption: 'Repeat for ~10⁵–10⁶ steps, each on a batch of millions of tokens (hedged; varies by run).'
          });
          var ids = ['f', 'l', 'b', 'u'], k = 0;
          b.highlight('f');
          api.every(1100, function () { k = (k + 1) % ids.length; b.highlight(ids[k]); });
          api.callout(el, '“Backpropagation” is just the chain rule applied layer by layer from the loss back to the embeddings. Every one of the 8.03 B weights gets its own gradient number each step.', 'key');
        }
      },
      {
        name: 'Worked update',
        render: function (el, api) {
          var r = tinyStep(api), M = api.math;
          api.html(el, '<p>A tiny “unembedding” with a 3-token vocab. Target is ' + api.tok(' mat') + '. We compute the gradient and take <b>one</b> step with η = ' + TINY.lr + '.</p>');
          api.stepper(el, { steps: [
            { title: '1 · Forward: logits z = W·h', render: function (b) {
              var g = api.grid(b, 200);
              api.matrix(g, { data: TINY.W, title: 'W', type: 'matrix', shape: '[3 × 3]', rowLabels: TINY.vocab.map(api.tokText), small: true });
              api.matrix(g, { data: [TINY.h], title: 'h', type: 'vector', shape: '[3]', small: true });
              api.matrix(g, { data: [r.z], title: 'z = W·h', type: 'vector', shape: '[3]', colLabels: TINY.vocab.map(api.tokText), small: true, decimals: 3 });
            } },
            { title: '2 · Softmax and loss', render: function (b) {
              api.bars(b, { items: r.p.map(function (p, i) { return { label: api.tok(TINY.vocab[i]), value: p, highlight: i === 0 }; }), max: 1 });
              api.html(b, '<div class="formula">L = −ln p(␣mat) = −ln ' + api.fmt(r.p[0], 3) + ' = <b>' + api.fmt(r.L, 3) + '</b></div>');
            } },
            { title: '3 · Gradient at logits: p − y', render: function (b) {
              api.html(b, '<div class="formula">∂L/∂z = p − y = [' + r.p.map(function (x) { return api.fmt(x, 3); }).join(', ') + '] − [1, 0, 0] = [' + r.dz.map(function (x) { return api.fmt(x, 3); }).join(', ') + ']</div>' +
                '<p class="muted">Negative for the target (push its logit up), positive for the others (push down).</p>');
            } },
            { title: '4 · Chain rule to weights: ∂L/∂W = (p − y)·hᵀ', render: function (b) {
              api.matrix(b, { data: r.dW, title: '∂L/∂W', type: 'matrix', shape: '[3 × 3]', rowLabels: TINY.vocab.map(api.tokText), scale: 'diverging', decimals: 3, small: true });
            } },
            { title: '5 · Update: W ← W − η·∂L/∂W, then re-check', render: function (b) {
              api.matrix(b, { data: r.W2, title: 'W (new)', type: 'matrix', shape: '[3 × 3]', rowLabels: TINY.vocab.map(api.tokText), decimals: 3, small: true });
              api.html(b, '<div class="formula">p(␣mat): ' + api.fmt(r.p[0], 3) + ' → <b>' + api.fmt(r.p2[0], 3) + '</b> · loss: ' + api.fmt(r.L, 3) + ' → <b>' + api.fmt(r.L2, 3) + '</b> (−' + pct((r.L - r.L2) / r.L) + ')</div>');
            } }
          ] });
          api.illustrative(el, 'Toy numbers; a real model does exactly this for billions of weights, with gradients flowing back through every attention and FFN layer.');
        }
      },
      {
        name: 'Learning rate',
        render: function (el, api) {
          api.html(el, '<p>Gradient descent on an illustrative 1-D loss valley L(w) = (w − 0.5)² + 0.2, starting at w = 3. Drag the learning rate.</p>');
          var c = ctl(api), out = api.h('div'), box = api.h('div');
          el.appendChild(c); el.appendChild(out); el.appendChild(box);
          var steps = 25;
          var ref = [
            { name: 'too small (0.02)', points: gdCurve(0.02, steps), color: '#ff9a4d', dashed: true },
            { name: 'good (0.3)', points: gdCurve(0.3, steps), color: '#5ad18a', dashed: true },
            { name: 'divergent (1.05)', points: gdCurve(1.05, steps), color: '#ff5d73', dashed: true }
          ];
          var ch = api.chart(box, { series: ref, xLabel: 'step', yLabel: 'loss', yRange: [0.1, 50], logY: true, height: 250 });
          api.slider(c, { label: 'learning rate η', min: -2.5, max: 0.1, step: 0.05, value: -0.7,
            format: function (v) { return Math.pow(10, v).toFixed(3); },
            onInput: function (lv) {
              var lr = Math.pow(10, lv), pts = gdCurve(lr, steps), last = pts[pts.length - 1][1];
              var verdict = lr >= 1 ? '<b class="training-bad">diverges</b> (each step overshoots further: |1 − 2η| = ' + api.fmt(Math.abs(1 - 2 * lr), 2) + ' > 1)'
                : lr < 0.05 ? '<b class="training-warn">too slow</b>' : '<b class="training-good">converges</b>';
              out.innerHTML = '<p>η = ' + lr.toFixed(3) + ' → after ' + steps + ' steps loss = ' + (last >= 50 ? '≥ 50' : api.fmt(last, 3)) + ' (minimum 0.200): ' + verdict + '</p>';
              ch.update({ series: ref.concat([{ name: 'yours (' + lr.toFixed(3) + ')', points: pts, color: '#4ea8ff' }]) });
            } });
          api.illustrative(el, 'For this quadratic each step multiplies the distance to the minimum by (1 − 2η); real loss surfaces are far messier, which is why LR warm-up and decay schedules are used.');
        }
      },
      {
        name: 'AdamW',
        render: function (el, api) {
          var b1 = 0.9, b2 = 0.95, eps = 1e-8, lr = 0.01, wd = 0.1;
          var grads = [0.5, 0.3, -0.2, 0.4];
          var w = 1.0, m = 0, v = 0, rows = [];
          grads.forEach(function (g, i) {
            var t = i + 1;
            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * g * g;
            var mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t));
            var step = lr * mh / (Math.sqrt(vh) + eps), decay = lr * wd * w;
            var w2 = w - step - decay;
            rows.push([t, api.fmt(g, 2), api.fmt(m, 4), api.fmt(v, 5), api.fmt(mh, 4), api.fmt(vh, 4), api.fmt(step, 5), api.fmt(decay, 5), api.fmt(w2, 5)]);
            w = w2;
          });
          api.html(el, '<p>One weight, four steps. AdamW hyper-parameters: η = ' + lr + ', β₁ = ' + b1 + ', β₂ = ' + b2 + ', weight decay λ = ' + wd + '.</p>' +
            '<div class="formula">m ← β₁m + (1−β₁)g &nbsp;·&nbsp; v ← β₂v + (1−β₂)g² &nbsp;·&nbsp; m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)<br>w ← w − η·m̂/(√v̂ + ε) − η·λ·w</div>');
          api.table(el, { headers: ['t', 'g', 'm', 'v', 'm̂', 'v̂', 'Adam step', 'decay η·λ·w', 'w (new)'], rows: rows });
          api.callout(el, '<b>Bias correction</b> matters early: at t = 1, m = 0.1·g is tiny, but m̂ = m/(1 − 0.9) = g. <b>Normalising</b> by √v̂ makes the step ≈ η regardless of gradient scale. <b>Decoupled weight decay</b> (the “W” in AdamW) shrinks weights directly instead of via the gradient.', 'key');
          var N = api.ref.paramBreakdown().total;
          api.html(el, '<p class="muted">Memory: m and v in FP32 = 8 bytes × ' + api.fmtBig(N) + ' params = <b>' + api.fmtBytes(8 * N) + '</b> just for optimizer state (see ' + api.link('training/scaling/2', 'precision & memory') + ').</p>');
          api.illustrative(el, 'Gradients are made up; β₂ = 0.95 is a common choice for LLMs (PyTorch’s default is 0.999).');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 4: scaling
  var scaling = {
    id: 'scaling',
    title: 'Scale and scaling laws',
    critical: {
      understand: [
        'Empirically, loss falls as a smooth <b>power law</b> in compute, parameters and data — bigger runs are predictable.',
        'Training compute ≈ <b>6 · N · D</b> FLOPs (N params, D tokens): 2 for forward, 4 for backward per param per token.',
        '“Chinchilla” rule of thumb: for a fixed compute budget, ≈ <b>20 tokens per parameter</b> is roughly compute-optimal.',
        'Training uses low precision (<b>BF16</b>) for speed and memory, with some FP32 for stability.'
      ],
      details: [
        'Scaling-law fits are approximate, vary with data and architecture, and are an active research area.',
        'Modern small models are often trained far beyond 20 tok/param (Llama 3 8B: ~15 T tokens reported ≈ 1,900 tok/param) because inference is cheaper with a smaller model.',
        'BF16 keeps FP32’s 8-bit exponent (same range) but only 7 mantissa bits (≈ 2–3 decimal digits).'
      ]
    },
    dataTypes: [
      { name: 'N (parameters)', type: 'integer', shape: 'scalar', example: '8,030,261,248' },
      { name: 'D (training tokens)', type: 'integer', shape: 'scalar', example: '15 × 10¹²' },
      { name: 'C (compute)', type: 'float', shape: 'scalar', example: '≈ 7.2 × 10²³ FLOPs' },
      { name: 'loss L(C)', type: 'float', shape: 'scalar', example: 'power law' },
      { name: 'BF16 / FP32 value', type: 'float', shape: '16 / 32 bits', example: '1-8-7 / 1-8-23' }
    ],
    levels: [
      {
        name: 'Power law',
        render: function (el, api) {
          // illustrative L(C) = E + A / C^alpha  with C in FLOPs
          var E = 1.7, A = 1.1e3, alpha = 0.13;
          var Lc = function (c) { return E + A * Math.pow(c, -alpha); };
          var pts = []; for (var lc = 18; lc <= 26; lc += 0.25) pts.push([lc, Lc(Math.pow(10, lc))]);
          var c = ctl(api), out = api.h('div'), box = api.h('div');
          el.appendChild(c); el.appendChild(box); el.appendChild(out);
          var ch = api.chart(box, { series: [{ name: 'L(C) = E + A·C^−α (illustrative fit)', points: pts }, { name: 'irreducible E', points: [[18, E], [26, E]], dashed: true, color: '#888' }],
            xLabel: 'log₁₀ training compute (FLOPs)', yLabel: 'loss (nats)', height: 240 });
          api.slider(c, { label: 'compute', min: 18, max: 26, step: 0.25, value: 23.86, format: function (v) { return '10^' + v.toFixed(2); },
            onInput: function (v) {
              var L = Lc(Math.pow(10, v)), L10 = Lc(Math.pow(10, v + 1));
              ch.update({ markers: [{ x: v, label: 'L = ' + api.fmt(L, 2) }] });
              out.innerHTML = '<p>Loss at 10^' + v.toFixed(2) + ' FLOPs: <b>' + api.fmt(L, 3) + '</b>. 10× more compute → ' + api.fmt(L10, 3) + ' (−' + api.fmt(L - L10, 3) + '). Each 10× buys a <i>smaller</i> absolute gain: diminishing but predictable returns.</p>';
            } });
          api.callout(el, 'On log–log axes the reducible part (L − E) is a straight line. Labs use small runs to fit such curves and predict big-run loss before spending the compute. Real exponents and constants depend on the setup.', 'hedge');
          api.illustrative(el, 'E, A and α chosen for display only.');
        }
      },
      {
        name: 'FLOPs & Chinchilla',
        render: function (el, api) {
          var N = api.ref.paramBreakdown().total;
          var c = ctl(api), out = api.h('div');
          el.appendChild(c); el.appendChild(out);
          var state = { D: 15, gpu: 400 };
          function draw() {
            var D = state.D * 1e12, C = 6 * N * D, Dchin = 20 * N, Cchin = 6 * N * Dchin;
            var gpuSec = C / (state.gpu * 1e12), gpuHours = gpuSec / 3600;
            out.innerHTML = '<div class="formula">C ≈ 6 · N · D = 6 × ' + api.fmtInt(N) + ' × ' + sci(D, 1) + ' = <b>' + sci(C) + ' FLOPs</b></div>' +
              '<div class="grid training-kpis">' +
              '<div class="card"><div class="muted">tokens per param</div><div class="big-num">' + api.fmtInt(D / N) + '</div><div class="muted">Chinchilla ≈ 20 → ' + api.fmtBig(Dchin) + ' tokens</div></div>' +
              '<div class="card"><div class="muted">vs Chinchilla-optimal compute</div><div class="big-num">' + api.fmt(C / Cchin, 0) + '×</div><div class="muted">' + sci(Cchin) + ' FLOPs at 20 tok/param</div></div>' +
              '<div class="card"><div class="muted">GPU-hours at ' + state.gpu + ' TFLOP/s effective</div><div class="big-num">' + api.fmtBig(gpuHours) + '</div><div class="muted">≈ ' + api.fmtInt(gpuHours / 24 / 1000) + ' days on 1,000 GPUs</div></div>' +
              '</div>';
          }
          api.slider(c, { label: 'D (trillion tokens)', min: 0.1, max: 20, step: 0.1, value: 15, format: function (v) { return v.toFixed(1) + ' T'; }, onInput: function (v) { state.D = v; draw(); } });
          api.slider(c, { label: 'effective TFLOP/s per GPU', min: 100, max: 1000, step: 50, value: 400, onInput: function (v) { state.gpu = v; draw(); } });
          api.callout(el, '<b>Why 6?</b> Forward ≈ 2 FLOPs per param per token (a multiply and an add); backward ≈ twice that. Attention adds a seq-length-dependent term that this estimate ignores.', 'info');
          api.callout(el, 'The ~15 T token figure is what Meta reported for Llama 3; effective GPU throughput (utilisation) varies a lot, so treat GPU-hours as an order-of-magnitude estimate.', 'hedge');
        }
      },
      {
        name: 'Precision & memory',
        render: function (el, api) {
          var fmts = [
            { n: 'FP32', s: 1, e: 8, m: 23 },
            { n: 'BF16', s: 1, e: 8, m: 7 },
            { n: 'FP16', s: 1, e: 5, m: 10 }
          ];
          var box = api.h('div', { class: 'training-bits' }); el.appendChild(box);
          fmts.forEach(function (f) {
            var cells = '';
            for (var i = 0; i < f.s; i++) cells += '<i class="training-bit-s"></i>';
            for (i = 0; i < f.e; i++) cells += '<i class="training-bit-e"></i>';
            for (i = 0; i < f.m; i++) cells += '<i class="training-bit-m"></i>';
            var maxv = (2 - Math.pow(2, -f.m)) * Math.pow(2, Math.pow(2, f.e - 1) - 1);
            box.appendChild(api.h('div', { class: 'training-bitrow', html: '<b>' + f.n + '</b> <span class="muted">(' + (f.s + f.e + f.m) + ' bits: sign ' + f.s + ' · exponent ' + f.e + ' · mantissa ' + f.m + ')</span><div class="training-bitcells">' + cells + '</div>' +
              '<span class="muted">max ≈ ' + sci(maxv, 1) + ' · precision ≈ ' + Math.floor(f.m * Math.log10(2) + 1) + ' significant digits</span>' }));
          });
          var N = api.ref.paramBreakdown().total;
          api.table(el, { headers: ['Training memory item (mixed precision + AdamW)', 'bytes / param', 'for 8.03 B params'],
            rows: [
              ['weights (BF16)', 2, api.fmtBytes(2 * N)],
              ['gradients (BF16)', 2, api.fmtBytes(2 * N)],
              ['FP32 master weights', 4, api.fmtBytes(4 * N)],
              ['Adam m + v (FP32)', 8, api.fmtBytes(8 * N)],
              ['<b>total (before activations)</b>', '<b>16</b>', '<b>' + api.fmtBytes(16 * N) + '</b>']
            ] });
          api.callout(el, 'Inference needs only the 2-byte weights (≈ 16 GB, see ' + api.link('inference/quantization', 'quantization') + '); full training needs ~8× that plus activations, so it is sharded across many GPUs. Exact layouts vary by framework.', 'info');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 5: SFT
  var SFT_TOKS = [
    { t: '<|user|>', role: 'tmpl' }, { t: 'What', role: 'user' }, { t: ' is', role: 'user' }, { t: ' 2', role: 'user' }, { t: '+', role: 'user' }, { t: '2', role: 'user' }, { t: '?', role: 'user' },
    { t: '<|assistant|>', role: 'tmpl' }, { t: '2', role: 'asst', p: 0.62 }, { t: '+', role: 'asst', p: 0.81 }, { t: '2', role: 'asst', p: 0.93 }, { t: ' =', role: 'asst', p: 0.88 },
    { t: ' 4', role: 'asst', p: 0.97 }, { t: '.', role: 'asst', p: 0.71 }, { t: '<|end|>', role: 'asst', p: 0.55 }
  ];
  var sft = {
    id: 'sft',
    title: 'Supervised fine-tuning (SFT) with a masked loss',
    critical: {
      understand: [
        'SFT = the same next-token training, but on example <b>conversations</b> written or curated to show ideal assistant behaviour.',
        'Conversations are flattened with a ' + '<a href="#" class="xlink" data-goto="chat-api/chat-template">chat template</a> into one token sequence.',
        'A <b>loss mask</b> keeps only assistant tokens in the loss: the model learns to <i>answer</i>, not to imitate users.'
      ],
      details: [
        'Masked positions are often given label −100, which PyTorch’s cross-entropy ignores by default (ignore_index).',
        'Including the end-of-turn token in the loss teaches the model when to stop.',
        'Whether to also train on prompts is a design choice; masking them is common practice (hedged).'
      ]
    },
    dataTypes: [
      { name: 'chat example', type: 'json', shape: '{messages: list[object]}', example: '[{role:"user"…},{role:"assistant"…}]' },
      { name: 'templated tokens', type: 'token', shape: '[seq_len]', example: '15 tokens' },
      { name: 'loss mask', type: 'mask', shape: '[seq_len]', example: '0 for user, 1 for assistant' },
      { name: 'labels', type: 'integer', shape: '[seq_len]', example: '−100 where masked' },
      { name: 'per-token loss', type: 'float', shape: '[seq_len]', example: '−ln p' },
      { name: 'SFT loss', type: 'float', shape: 'scalar', example: 'masked mean' }
    ],
    levels: [
      {
        name: 'Chat example',
        render: function (el, api) {
          api.code(el, JSON.stringify({ messages: [{ role: 'user', content: 'What is 2+2?' }, { role: 'assistant', content: '2+2 = 4.' }] }, null, 2), 'json');
          api.html(el, '<p>↓ chat template ' + api.badge('function') + ' → one token sequence ' + api.badge('token', '[15]') + '</p>');
          var box = api.h('div', { class: 'training-sft-toks' }); el.appendChild(box);
          SFT_TOKS.forEach(function (x) { box.appendChild(api.h('span', { class: 'tok training-role-' + x.role, html: api.tokText(x.t) })); });
          api.html(el, '<div class="row training-legend"><span class="tok training-role-tmpl">template</span><span class="tok training-role-user">user</span><span class="tok training-role-asst">assistant (trained)</span></div>');
          api.callout(el, 'Real templates use model-specific special tokens; the ones here are simplified. See ' + api.link('tokenization/special-tokens', 'special tokens') + '.', 'info');
        }
      },
      {
        name: 'Loss mask',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div');
          el.appendChild(c); el.appendChild(out);
          api.toggle(c, { label: 'apply loss mask (train only on assistant tokens)', value: true, onChange: draw });
          draw(true);
          function draw(masked) {
            if (masked == null) masked = true;
            var rows = [], sum = 0, n = 0;
            SFT_TOKS.forEach(function (x, i) {
              // illustrative p for user/template tokens (harder to predict)
              var p = x.p != null ? x.p : [0.02, 0.05, 0.3, 0.1, 0.15, 0.4, 0.6, 0.9][i % 8];
              var on = !masked || x.role === 'asst';
              if (i === 0) on = false; // first token has no prediction
              var l = -Math.log(p);
              if (on) { sum += l; n++; }
              rows.push('<div class="training-mrow' + (on ? '' : ' off') + '"><span class="tok training-role-' + x.role + '">' + api.tokText(x.t) + '</span><span class="training-mbit">' + (on ? 1 : 0) + '</span><span>' + (i === 0 ? '—' : api.fmt(l, 2)) + '</span></div>');
            });
            out.innerHTML = '<div class="training-mhead"><span>token ' + api.type('token') + '</span><span>mask ' + api.type('mask') + '</span><span>−ln p ' + api.type('float') + '</span></div><div class="training-mgrid">' + rows.join('') + '</div>' +
              '<div class="formula">loss = Σ(mask · −ln p) / Σ mask = ' + api.fmt(sum, 2) + ' / ' + n + ' = <b>' + api.fmt(sum / n, 3) + '</b></div>' +
              (masked ? '' : '<p class="training-warn">Without the mask, hard-to-predict user tokens dominate the loss and the model is also trained to write user turns.</p>');
          }
          api.illustrative(el);
        }
      },
      {
        name: 'Code',
        render: function (el, api) {
          api.code(el, [
            '# Pseudo-code in PyTorch style: masked SFT loss',
            'ids    = chat_template(messages)            # [seq_len] token IDs',
            'labels = ids.clone()',
            'labels[~is_assistant_token] = -100           # loss mask: ignored positions',
            'logits = model(ids[:-1])                     # [seq_len-1, vocab]  teacher forcing',
            'loss = F.cross_entropy(logits, labels[1:],   # targets shifted by one',
            '                       ignore_index=-100)    # mean over unmasked tokens',
            'loss.backward(); optimizer.step(); optimizer.zero_grad()'
          ].join('\n'), 'python');
          api.docsNote(el, 'Fine-tuning libraries often build this mask for you from the chat template (e.g. an “assistant-only loss” option).');
          api.html(el, '<p>Same optimizer as pretraining (' + api.link('training/backprop/3', 'AdamW') + '), usually with a smaller learning rate and far fewer steps. Parameter-efficient variants train only small adapters: ' + api.link('training/lora', 'LoRA') + '.</p>');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 6: RLHF / DPO
  var rlhf = {
    id: 'rlhf',
    title: 'Preference tuning: reward models, PPO, DPO, RLAIF',
    critical: {
      understand: [
        'Humans (or an AI judge) compare two answers to the same prompt: <b>chosen</b> vs <b>rejected</b>.',
        '<b>RLHF</b>: train a <b>reward model</b> on those comparisons, then optimise the policy with RL (classically PPO) to get high reward without drifting too far (KL penalty).',
        '<b>DPO</b> skips the reward model and RL loop: a simple classification-style loss directly on the preference pairs.',
        '<b>RLAIF</b> / constitutional approaches use an AI model, guided by written principles, to produce the preference labels.'
      ],
      details: [
        'Reward model loss (Bradley–Terry): −ln σ(r_chosen − r_rejected).',
        'DPO loss: −ln σ(β·[(log π/π_ref)(chosen) − (log π/π_ref)(rejected)]).',
        'Failure mode: <b>reward hacking</b> — exploiting reward-model quirks (e.g. verbosity). The KL term and data quality mitigate it.'
      ]
    },
    dataTypes: [
      { name: 'preference pair', type: 'json', shape: '{prompt, chosen, rejected}', example: 'one comparison' },
      { name: 'reward r', type: 'float', shape: 'scalar', example: 'r_chosen = 1.3' },
      { name: 'P(chosen ≻ rejected)', type: 'probability', shape: 'scalar', example: 'σ(Δr)' },
      { name: 'log-probs log π(y|x)', type: 'float', shape: 'scalar', example: 'sum over answer tokens' },
      { name: 'β', type: 'float', shape: 'scalar', example: '0.1' },
      { name: 'reward model', type: 'function', shape: '(prompt, answer) → scalar', example: 'LLM + scalar head' }
    ],
    levels: [
      {
        name: 'Pipelines',
        render: function (el, api) {
          api.html(el, '<p><b>RLHF (classic)</b></p>');
          api.blocks(el, { blocks: [
            { label: 'SFT model', sub: 'samples 2+ answers', type: 'text' },
            { label: 'Humans rank', sub: 'chosen ≻ rejected', type: 'json', shape: 'pair', edge: 'answers' },
            { label: 'Reward model', sub: 'answer → score', type: 'function', zoom: 1, edge: 'pairs' },
            { label: 'PPO', sub: 'max reward − β·KL', type: 'float', shape: 'r: scalar', edge: 'r' }
          ] });
          api.html(el, '<p><b>DPO</b></p>');
          api.blocks(el, { blocks: [
            { label: 'Preference pairs', type: 'json', shape: '{chosen, rejected}' },
            { label: 'DPO loss', sub: 'policy vs frozen ref', type: 'float', shape: 'scalar', zoom: 2 },
            { label: 'Tuned policy', type: 'tensor', shape: '[8.03 B]' }
          ] });
          api.html(el, '<p><b>RLAIF / constitutional</b></p>');
          api.blocks(el, { blocks: [
            { label: 'Principles', sub: 'written rules', type: 'text' },
            { label: 'AI judge', sub: 'compares answers', type: 'function', edge: 'prompt' },
            { label: 'AI preference labels', type: 'json', edge: 'pairs' },
            { label: 'RM + RL, or DPO', type: 'float', zoom: 3 }
          ] });
        }
      },
      {
        name: 'Reward model',
        render: function (el, api) {
          api.html(el, '<p>Prompt: <i>“Explain why the sky is blue.”</i> The reward model ' + api.badge('function') + ' scores each answer ' + api.badge('float', 'scalar') + '. Bradley–Terry turns the score gap into a preference probability.</p>');
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          var st = { a: 1.3, b: 0.4 };
          function draw() {
            var d = st.a - st.b, P = api.math.sigmoid(d), L = -Math.log(P);
            out.innerHTML = '<div class="formula">P(chosen ≻ rejected) = σ(r_c − r_r) = σ(' + api.fmt(d, 2) + ') = <b>' + api.fmt(P, 3) + '</b> &nbsp; loss = −ln P = <b>' + api.fmt(L, 3) + '</b></div>' +
              '<p class="muted">' + (d > 0 ? 'Agrees with the human label; training still pushes the gap wider (gradient ∝ 1 − P = ' + api.fmt(1 - P, 3) + ').' : 'Disagrees with the human — large loss, strong correction.') + '</p>';
          }
          api.slider(c, { label: 'r(chosen)', min: -3, max: 3, step: 0.1, value: 1.3, onInput: function (v) { st.a = v; draw(); } });
          api.slider(c, { label: 'r(rejected)', min: -3, max: 3, step: 0.1, value: 0.4, onInput: function (v) { st.b = v; draw(); } });
          api.html(el, '<div class="formula">PPO objective (sketch): maximise E[ r(x, y) ] − β · KL(π ‖ π_SFT)</div>');
          api.callout(el, 'The KL penalty keeps the policy close to the SFT model so it can’t drift into weird text that merely fools the reward model.', 'info');
          api.illustrative(el);
        }
      },
      {
        name: 'DPO numbers',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          var st = { beta: 0.1, shift: 0 };
          // log-probs (sum over answer tokens), illustrative
          var ref = { c: -42.0, r: -40.5 };
          function draw() {
            var pol = { c: -42.0 + st.shift, r: -40.5 - st.shift };
            var rc = pol.c - ref.c, rr = pol.r - ref.r;
            var z = st.beta * (rc - rr), L = -Math.log(api.math.sigmoid(z));
            out.innerHTML = '';
            api.table(out, { headers: ['', 'log π_θ ' + api.type('float'), 'log π_ref', 'log-ratio'], rows: [
              ['chosen', api.fmt(pol.c, 1), api.fmt(ref.c, 1), api.fmt(rc, 1)],
              ['rejected', api.fmt(pol.r, 1), api.fmt(ref.r, 1), api.fmt(rr, 1)]
            ] });
            out.appendChild(api.h('div', { class: 'formula', html: 'z = β·(' + api.fmt(rc, 1) + ' − ' + api.fmt(rr, 1) + ') = ' + api.fmt(z, 3) + ' &nbsp; L_DPO = −ln σ(z) = <b>' + api.fmt(L, 3) + '</b>' }));
            out.appendChild(api.h('p', { class: 'muted', html: st.shift === 0 ? 'At the start policy = reference, so z = 0 and L = ln 2 = ' + api.fmt(Math.log(2), 3) + '.' : 'Training raises the chosen answer’s log-prob relative to the reference and lowers the rejected one’s; the loss falls.' }));
          }
          api.slider(c, { label: 'training progress (log-prob shift)', min: 0, max: 20, step: 1, value: 0, onInput: function (v) { st.shift = v; draw(); } });
          api.slider(c, { label: 'β', min: 0.02, max: 0.5, step: 0.01, value: 0.1, onInput: function (v) { st.beta = v; draw(); } });
          api.callout(el, 'Note the reference already slightly preferred the <i>rejected</i> answer (−40.5 &gt; −42.0). DPO cares about the change relative to the reference, not raw probabilities. β plays the same role as PPO’s KL strength.', 'key');
          api.illustrative(el);
        }
      },
      {
        name: 'RLAIF & caveats',
        render: function (el, api) {
          var g = api.grid(el, 240);
          api.card(g, 'RLAIF / Constitutional AI', 'An AI model critiques or compares answers using a written list of principles; its judgements replace (some) human labels. Cheaper and more consistent at scale; quality depends on the judge and the principles.');
          api.card(g, 'Reward hacking', 'Policies tend to find what the reward model over-rewards — e.g. longer answers, confident tone, flattery. Mitigations: KL penalty, better RM data, length control, held-out human eval.');
          api.card(g, 'PPO vs DPO in practice', 'PPO needs 4 models in memory (policy, reference, reward, value) and is finicky; DPO needs 2 and is stable. Many recipes combine methods; the trade-offs are still debated.');
          api.callout(el, 'These methods change <i>which</i> plausible answer is preferred much more than they add knowledge. Research suggests most capabilities come from pretraining.', 'hedge');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 7: LoRA
  var lora = {
    id: 'lora',
    title: 'LoRA: fine-tuning with tiny low-rank adapters',
    critical: {
      understand: [
        'Freeze the big weight W. Learn a small update <b>ΔW = B·A</b> with rank r ≪ d. Effective weight: <b>W + (α/r)·B·A</b>.',
        'For a 4096 × 4096 projection, r = 16 trains 2·16·4096 = 131,072 numbers instead of 16.8 M.',
        'Adapters are a few MB to a few hundred MB — swap them per task on one shared base model.'
      ],
      details: [
        'B is initialised to zero, so training starts exactly at the base model.',
        'After training, B·A can be merged into W: zero extra inference latency.',
        'QLoRA keeps the frozen base in 4-bit to save even more memory (see ' + '<a href="#" class="xlink" data-goto="clearml/lora-tracking">LoRA tracking</a>).'
      ]
    },
    dataTypes: [
      { name: 'W (frozen)', type: 'matrix', shape: '[d_out × d_in]', example: '[4096 × 4096]' },
      { name: 'A (trainable)', type: 'matrix', shape: '[r × d_in]', example: '[16 × 4096]' },
      { name: 'B (trainable)', type: 'matrix', shape: '[d_out × r]', example: '[4096 × 16]' },
      { name: 'rank r', type: 'integer', shape: 'scalar', example: '16' },
      { name: 'scale α/r', type: 'float', shape: 'scalar', example: '2' },
      { name: 'x', type: 'vector', shape: '[d_in]', example: 'token hidden state' }
    ],
    levels: [
      {
        name: 'Idea',
        render: function (el, api) {
          api.blocks(el, { rows: [[
            { label: 'x', type: 'vector', shape: '[4096]' },
            { label: 'W (frozen)', sub: '16.8 M params', type: 'matrix', shape: '[4096 × 4096]', color: '#556' },
            { label: 'W·x', type: 'vector', shape: '[4096]' }
          ], [
            { label: 'x', type: 'vector', shape: '[4096]' },
            { label: 'A', sub: 'down-project', type: 'matrix', shape: '[r × 4096]', color: '#5ad18a' },
            { label: 'B', sub: 'up-project (init 0)', type: 'matrix', shape: '[4096 × r]', color: '#5ad18a', edge: '[r]' },
            { label: '+ (α/r)·B·A·x', type: 'vector', shape: '[4096]', zoom: 1 }
          ]], caption: 'h = W·x + (α/r)·B·(A·x) — only A and B get gradients.' });
          api.callout(el, 'Intuition: the change needed to adapt a model to a task is (empirically) low-rank — a few directions matter. LoRA bets on that.', 'key');
        }
      },
      {
        name: 'Rank slider',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          var st = { r: 16, shape: 'q' };
          var d = api.ref.d_model, kv = api.ref.n_kv_heads * api.ref.d_head;
          var shapes = { q: [d, d], k: [kv, d], gate: [api.ref.d_ff, d] };
          function draw() {
            var s = shapes[st.shape], full = s[0] * s[1], lr = st.r * (s[0] + s[1]);
            out.innerHTML = '<div class="formula">full ΔW: ' + api.fmtInt(s[0]) + ' × ' + api.fmtInt(s[1]) + ' = <b>' + api.fmtInt(full) + '</b> &nbsp;·&nbsp; LoRA: r·(d_out + d_in) = ' + st.r + ' × (' + api.fmtInt(s[0]) + ' + ' + api.fmtInt(s[1]) + ') = <b>' + api.fmtInt(lr) + '</b></div>';
            api.bars(out, { items: [
              { label: 'full fine-tune', value: full, color: '#ff9a4d' },
              { label: 'LoRA r = ' + st.r, value: lr, color: '#5ad18a', note: pct(lr / full, 2) + ' · ' + api.fmt(full / lr, 0) + '× fewer' }
            ], format: function (v) { return api.fmtBig(v); } });
          }
          api.select(c, { label: 'projection', value: 'q', options: [
            { value: 'q', label: 'W_q [4096 × 4096]' }, { value: 'k', label: 'W_k [1024 × 4096] (GQA)' }, { value: 'gate', label: 'W_gate [14336 × 4096]' }
          ], onChange: function (v) { st.shape = v; draw(); } });
          api.slider(c, { label: 'rank r', min: 1, max: 256, step: 1, value: 16, onInput: function (v) { st.r = v; draw(); } });
        }
      },
      {
        name: 'Whole model',
        render: function (el, api) {
          var R = api.ref, P = R.paramBreakdown(), d = R.d_model, kv = R.n_kv_heads * R.d_head;
          var mats = [
            { k: 'q', label: 'W_q', s: [d, d], on: true }, { k: 'k', label: 'W_k', s: [kv, d], on: false },
            { k: 'v', label: 'W_v', s: [kv, d], on: true }, { k: 'o', label: 'W_o', s: [d, d], on: false },
            { k: 'gate', label: 'W_gate', s: [R.d_ff, d], on: false }, { k: 'up', label: 'W_up', s: [R.d_ff, d], on: false },
            { k: 'down', label: 'W_down', s: [d, R.d_ff], on: false }
          ];
          var c = ctl(api), c2 = ctl(api), out = api.h('div');
          el.appendChild(c); el.appendChild(c2); el.appendChild(out);
          var r = 16;
          function draw() {
            var perLayer = 0;
            mats.forEach(function (m) { if (m.on) perLayer += r * (m.s[0] + m.s[1]); });
            var total = perLayer * R.n_layers;
            var fullFT = 16 * P.total, loraMem = 2 * P.total + 16 * total;
            out.innerHTML = '<div class="grid training-kpis">' +
              '<div class="card"><div class="muted">trainable params ' + api.badge('integer', 'scalar') + '</div><div class="big-num">' + api.fmtBig(total) + '</div><div class="muted">' + api.fmtInt(perLayer) + ' per layer × ' + R.n_layers + '</div></div>' +
              '<div class="card"><div class="muted">share of 8.03 B</div><div class="big-num">' + pct(total / P.total, 3) + '</div></div>' +
              '<div class="card"><div class="muted">adapter file (BF16)</div><div class="big-num">' + api.fmtBytes(total * 2) + '</div></div>' +
              '<div class="card"><div class="muted">train memory (weights + optimizer, no activations)</div><div class="big-num">' + api.fmtBytes(loraMem) + '</div><div class="muted">vs full FT ≈ ' + api.fmtBytes(fullFT) + '</div></div>' +
              '</div>';
          }
          mats.forEach(function (m) { api.toggle(c, { label: m.label, value: m.on, onChange: function (v) { m.on = v; draw(); } }); });
          api.slider(c2, { label: 'rank r', min: 1, max: 128, step: 1, value: 16, onInput: function (v) { r = v; draw(); } });
          api.callout(el, 'Memory model: frozen BF16 base (2 B/param) + 16 B/param for trainable adapter weights, grads and AdamW state (as on the ' + api.link('training/scaling/2', 'precision slide') + '). Activations come on top.', 'info');
        }
      },
      {
        name: 'Tiny numbers',
        render: function (el, api) {
          var M = api.math, alpha = 2, r = 1;
          var W = [[0.5, -0.2, 0.1, 0.0], [0.1, 0.4, -0.3, 0.2], [0.0, 0.3, 0.6, -0.1], [-0.2, 0.1, 0.0, 0.5]];
          var A = [[0.3, -0.1, 0.2, 0.4]];       // [1 × 4]
          var B = [[0.5], [-0.2], [0.1], [0.3]]; // [4 × 1]
          var BA = M.matmul(B, A), dW = M.scale(BA, alpha / r), We = M.add(W, dW);
          var x = api.ex.toyX[1];
          var g = api.grid(el, 200);
          api.matrix(g, { data: B, title: 'B', type: 'matrix', shape: '[4 × 1]', small: true });
          api.matrix(g, { data: A, title: 'A', type: 'matrix', shape: '[1 × 4]', small: true });
          api.matrix(g, { data: dW, title: '(α/r)·B·A, α=2, r=1', type: 'matrix', shape: '[4 × 4]', small: true, scale: 'diverging', decimals: 3 });
          api.matrix(g, { data: We, title: 'W + ΔW', type: 'matrix', shape: '[4 × 4]', small: true, decimals: 3 });
          var y0 = M.matvec(W, x), y1 = M.matvec(We, x);
          api.html(el, '<div class="formula">x = toyX[␣cat] = [' + x.map(function (v) { return api.fmt(v, 1); }).join(', ') + ']<br>W·x = [' + y0.map(function (v) { return api.fmt(v, 3); }).join(', ') + ']<br>(W + ΔW)·x = [' + y1.map(function (v) { return api.fmt(v, 3); }).join(', ') + ']</div>');
          api.html(el, '<p class="muted">ΔW is 16 numbers but has rank 1: it was made from only 4 + 4 = 8 trainable numbers.</p>');
          api.illustrative(el);
        }
      }
    ]
  };

  Course.register({
    id: 'training',
    part: 'I',
    order: 80,
    title: 'Training',
    slides: [lifecycle, crossEntropy, backprop, scaling, sft, rlhf, lora]
  });
})();
