/* Section: transformer-block — the decoder-only stack, residual stream, RMSNorm, layer roles, parameter count. */
(function () {
  'use strict';

  var GAIN = [1.0, 0.8, 1.2, 1.0]; // illustrative learned RMSNorm gain for the toy d_model = 4

  function formula(api, el, html) { return api.html(el, '<div class="formula">' + html + '</div>'); }
  function vec(api, v, d) { return '[' + v.map(function (x) { return api.fmt(x, d == null ? 2 : d); }).join(', ') + ']'; }
  function tokLabels(api) { return api.ex.tokens.map(function (t) { return api.tokText(t.text); }); }

  // parameter count for an arbitrary config (same formula as api.ref.paramBreakdown)
  function countParams(c) {
    var kvDim = c.kv * c.dh, qDim = c.heads * c.dh;
    var wq = c.d * qDim, wk = c.d * kvDim, wv = c.d * kvDim, wo = qDim * c.d;
    var ffn = 3 * c.d * c.ff, norms = 2 * c.d, per = wq + wk + wv + wo + ffn + norms;
    var embed = c.vocab * c.d;
    return { per: per, attn: wq + wk + wv + wo, ffn: ffn, embed: embed, unembed: embed, total: embed * 2 + per * c.layers + c.d };
  }

  Course.register({
    id: 'transformer-block',
    part: 'I',
    order: 40,
    title: 'The Transformer Block',
    slides: [
      // ============================================================ 1. decoder-only
      {
        id: 'decoder-only',
        title: 'Decoder-only architecture: a stack of 32 identical blocks',
        critical: {
          understand: [
            'An LLM is: embed tokens → <b>the same block repeated N times</b> → final norm → unembed to next-token scores.',
            'Each block has two sub-layers: <b>attention</b> (tokens exchange information) and an <b>FFN</b> (each token processed on its own).',
            'Shape stays <b>[seq_len × d_model]</b> all the way through the stack; only the content changes.'
          ],
          details: [
            'Pre-norm: each sub-layer reads RMSNorm(x) and its output is <b>added</b> back to x (residual connection).',
            'Reference model: 32 blocks, d_model 4,096; blocks share structure but each has its own weights.',
            '"Decoder-only" = no separate encoder; a causal mask lets every position see only its past.'
          ]
        },
        dataTypes: [
          { name: 'input tokens', type: 'token', shape: '[seq_len]', example: 'The ␣cat ␣sat ␣on ␣the' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[464, 3797, 3332, 319, 262]' },
          { name: 'hidden states / residual stream', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4096]' },
          { name: 'logits', type: 'float', shape: '[seq_len × vocab]', example: '[5 × 128,256]' },
          { name: 'next-token distribution', type: 'probability', shape: '[vocab]', example: 'P(␣mat) highest' },
          { name: 'block', type: 'function', shape: '[n × d] → [n × d]', example: 'one of 32' }
        ],
        levels: [
          { name: 'The stack', render: function (el, api) {
            var R = api.ref;
            var d = api.blocks(el, { direction: 'column', rows: [
              [{ id: 't', label: 'tokens', sub: api.ex.tokens.map(function (t) { return api.tok(t.text); }).join(' '), type: ['token', 'integer'], shape: '[5]', goto: 'tokenization/text-to-ids' }],
              [{ id: 'e', label: 'embedding lookup', sub: 'ID → row of E', type: 'matrix', shape: '[5 × ' + api.fmtInt(R.d_model) + ']', goto: 'embeddings/lookup' }],
              [{ id: 'b1', label: 'block 1', sub: 'RMSNorm → Attention → + · RMSNorm → FFN → +', type: 'matrix', shape: '[5 × 4,096]', zoom: 1 }],
              [{ id: 'b2', label: 'block 2 … block 31', sub: 'same structure, different weights', type: 'matrix', shape: '[5 × 4,096]', zoom: 1 }],
              [{ id: 'b32', label: 'block 32', type: 'matrix', shape: '[5 × 4,096]', zoom: 1 }],
              [{ id: 'n', label: 'final RMSNorm', type: 'matrix', shape: '[5 × 4,096]', goto: 'transformer-block/rmsnorm' }],
              [{ id: 'u', label: 'unembedding', sub: '· W_U → logits', type: 'float', shape: '[5 × ' + api.fmtInt(R.vocab) + ']', goto: 'output-sampling/unembedding' }],
              [{ id: 'p', label: 'softmax of last row', sub: '→ ' + api.tok(' mat') + ' most likely', type: 'probability', shape: '[' + api.fmtInt(R.vocab) + ']', goto: 'output-sampling/sampling-playground' }]
            ] });
            var ids = ['t', 'e', 'b1', 'b2', 'b32', 'n', 'u', 'p'], k = 0;
            api.every(800, function () { d.highlight(ids[k % ids.length]); k++; });
            api.callout(el, 'The whole model is this one picture. Everything else in Part I zooms into one of these boxes. Click a block.', 'key');
          } },
          { name: 'Inside a block', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ label: 'x', sub: 'residual stream in', type: 'matrix', shape: '[n × d_model]', goto: 'transformer-block/residual-stream' },
               { label: 'RMSNorm', type: 'function', shape: '[n × d]', goto: 'transformer-block/rmsnorm' },
               { label: 'Multi-head attention', sub: 'mixes across tokens', type: 'matrix', shape: '[n × d]', goto: 'attention/qkv-intuition' }],
              [{ label: '⊕ add', sub: 'h = x + attn(norm(x))', type: 'matrix', shape: '[n × d]', cls: 'transformer-block-add' },
               { label: 'RMSNorm', type: 'function', shape: '[n × d]', goto: 'transformer-block/rmsnorm' },
               { label: 'FFN (SwiGLU)', sub: 'per token, d → 14,336 → d', type: 'matrix', shape: '[n × d]', goto: 'ffn-moe/ffn-role' }],
              [{ label: '⊕ add', sub: 'x\' = h + ffn(norm(h))', type: 'matrix', shape: '[n × d]', cls: 'transformer-block-add' },
               { label: 'x\'', sub: 'to next block', type: 'matrix', shape: '[n × d_model]' }]
            ], caption: 'Two sub-layers, each wrapped as: <b>x ← x + Sublayer(RMSNorm(x))</b>' });
            var g = api.grid(el, 240);
            api.card(g, 'Attention = communication', 'The only step where a token can read other tokens (its past). ' + api.link('attention/worked-example', 'Worked example') + '.');
            api.card(g, 'FFN = computation', 'Applied to each token independently; holds most of the parameters. ' + api.link('ffn-moe/ffn-role', 'FFN section') + '.');
            api.card(g, 'Norm + add = plumbing', 'Keeps numbers well-scaled and lets each sub-layer make a small edit rather than rewrite everything.');
          } },
          { name: 'Pseudo-code', render: function (el, api) {
            api.code(el, [
              '# Pseudo-code (shapes for the reference model; n = seq_len)',
              'def llm_forward(token_ids):                 # [n]            integer',
              '    x = E[token_ids]                        # [n × 4096]     matrix',
              '    for block in blocks:                    # 32 times',
              '        x = x + block.attn(rmsnorm(x, block.g1))   # [n × 4096]',
              '        x = x + block.ffn(rmsnorm(x, block.g2))    # [n × 4096]',
              '    x = rmsnorm(x, g_final)                 # [n × 4096]',
              '    logits = x @ W_U                        # [n × 128256]   float',
              '    return softmax(logits[-1])              # [128256]       probability'
            ].join('\n'), 'python');
            api.table(el, { headers: ['step', 'type', 'shape (toy)', 'shape (reference)'], rows: [
              ['token IDs', api.type('integer'), '[5]', '[n]'],
              ['x after embedding', api.type('matrix'), '[5 × 4]', '[n × 4,096]'],
              ['x after each block', api.type('matrix'), '[5 × 4]', '[n × 4,096]'],
              ['logits', api.type('float'), '—', '[n × 128,256]'],
              ['next-token probs', api.type('probability'), '—', '[128,256]']
            ] });
            api.html(el, '<p class="muted">Note the loop: 32 times the same two lines. Depth (not width) is what lets later blocks build on what earlier blocks computed. Positions are injected inside attention via ' + api.link('embeddings/rope', 'RoPE') + '.</p>');
          } }
        ]
      },

      // ============================================================ 2. residual stream
      {
        id: 'residual-stream',
        title: 'The residual stream: a shared whiteboard',
        critical: {
          understand: [
            'Each token has one vector [d_model] that flows through all 32 blocks: the <b>residual stream</b>.',
            'Every sub-layer <b>reads</b> the stream (through a norm) and <b>adds</b> its result back — it never overwrites it.',
            'So the final vector is the input embedding <b>plus a sum of 64 edits</b> (32 attention + 32 FFN).'
          ],
          details: [
            'The add (skip connection) gives gradients a direct path back to early layers, which makes deep stacks trainable.',
            'Because all layers write in the same space, an intermediate stream can be decoded with the unembedding ("logit lens").',
            'The "whiteboard" picture is a helpful mental model popularised by interpretability research, not a formal property.'
          ]
        },
        dataTypes: [
          { name: 'stream x (one token)', type: 'vector', shape: '[d_model]', example: '[4] toy, [4,096] ref.' },
          { name: 'sub-layer edit Δ', type: 'vector', shape: '[d_model]', example: 'attn or FFN output' },
          { name: 'stream for all tokens', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
          { name: 'sub-layer', type: 'function', shape: '[d] → [d]', example: 'Attention, FFN' }
        ],
        levels: [
          { name: 'Whiteboard', render: function (el, api) {
            api.html(el, '<p>Picture each token\'s vector as a <b>whiteboard</b> that travels up the stack. Each sub-layer walks up, reads it, and writes a few notes on it with a "+" — nobody erases.</p>');
            var W = 640, H = 160, y = 110, s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="transformer-block-svg" role="img" aria-label="residual stream">';
            s += '<defs><marker id="tb-ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#4ea8ff"/></marker></defs>';
            s += '<line x1="20" y1="' + y + '" x2="' + (W - 30) + '" y2="' + y + '" stroke="#4ea8ff" stroke-width="10" opacity=".35"/>';
            s += '<line x1="20" y1="' + y + '" x2="' + (W - 30) + '" y2="' + y + '" class="transformer-block-flow" stroke="#4ea8ff" stroke-width="3" marker-end="url(#tb-ah)"/>';
            s += '<text x="20" y="' + (y + 34) + '" class="transformer-block-svglbl">embedding</text><text x="' + (W - 30) + '" y="' + (y + 34) + '" text-anchor="end" class="transformer-block-svglbl">→ unembedding</text>';
            var subs = ['Attn 1', 'FFN 1', 'Attn 2', 'FFN 2', '…', 'FFN 32'];
            subs.forEach(function (lb, i) {
              var x = 90 + i * 92;
              if (lb === '…') { s += '<text x="' + x + '" y="' + (y - 50) + '" text-anchor="middle" class="transformer-block-svglbl">…</text>'; return; }
              var col = lb.indexOf('Attn') === 0 ? '#ff9a4d' : '#5ad18a';
              s += '<rect x="' + (x - 34) + '" y="18" width="68" height="30" rx="6" fill="' + col + '" opacity=".22" stroke="' + col + '"/>';
              s += '<text x="' + x + '" y="38" text-anchor="middle" class="transformer-block-svgtxt">' + lb + '</text>';
              s += '<path d="M' + (x - 14) + ' ' + (y - 6) + ' C' + (x - 14) + ' 80 ' + (x - 14) + ' 70 ' + (x - 14) + ' 50" stroke="#9aa6ba" stroke-width="1.5" fill="none" stroke-dasharray="3 3"/>';
              s += '<path d="M' + (x + 14) + ' 50 L' + (x + 14) + ' ' + (y - 8) + '" stroke="' + col + '" stroke-width="2" fill="none"/>';
              s += '<circle cx="' + (x + 14) + '" cy="' + y + '" r="8" fill="#0f141d" stroke="' + col + '" stroke-width="2"/><text x="' + (x + 14) + '" y="' + (y + 4) + '" text-anchor="middle" class="transformer-block-svgtxt">+</text>';
              s += '<text x="' + (x - 18) + '" y="72" text-anchor="end" class="transformer-block-svgsm">read</text><text x="' + (x + 18) + '" y="72" class="transformer-block-svgsm">write</text>';
            });
            s += '</svg>';
            api.html(el, s);
            var g = api.grid(el, 220);
            api.card(g, 'Read', 'Sub-layer input = RMSNorm(stream) ' + api.badge('vector', '[d_model]'));
            api.card(g, 'Write', 'Stream ← stream + output ' + api.badge('vector', '[d_model]'));
            api.card(g, 'Shared', 'Later layers can use anything earlier layers wrote — the stream is their only channel.');
          } },
          { name: 'Adding up', render: function (el, api) {
            var x0 = api.ex.toyX[4];
            var edits = [
              { who: 'Attention 1', d: [0.10, 0.25, -0.05, 0.15], note: 'pulls in info from ␣cat / ␣on' },
              { who: 'FFN 1', d: [-0.05, 0.10, 0.20, 0.05], note: 'per-token processing' },
              { who: 'Attention 2', d: [0.20, -0.10, 0.05, 0.30], note: 'more context' },
              { who: 'FFN 2', d: [0.05, 0.15, -0.10, 0.20], note: 'more features' }
            ];
            var states = [x0];
            edits.forEach(function (e) { states.push(api.math.add(states[states.length - 1], e.d)); });
            api.html(el, '<p>Follow the stream of ' + api.tok(' the') + ' (toy d_model = 4) through two blocks. Each step: <b>new = old + Δ</b>.</p>');
            var steps = [{ title: 'Start: embedding of ␣the', render: function (b) {
              formula(api, b, 'x<sub>0</sub> = ' + vec(api, x0) + ' ' + api.badge('vector', '[4]'));
              api.matrix(b, { data: [x0], rowLabels: ['x₀'], colLabels: ['d0', 'd1', 'd2', 'd3'], title: 'stream', type: 'vector', shape: '[4]', scale: 'diverging' });
            } }];
            edits.forEach(function (e, i) {
              steps.push({ title: e.who + ' writes Δ (' + e.note + ')', render: function (b) {
                formula(api, b, 'x<sub>' + (i + 1) + '</sub> = x<sub>' + i + '</sub> + Δ = ' + vec(api, states[i]) + ' + ' + vec(api, e.d) + ' = <b>' + vec(api, states[i + 1]) + '</b>');
                api.matrix(b, { data: [states[i], e.d, states[i + 1]], rowLabels: ['x' + i, '+ Δ ' + e.who, '= x' + (i + 1)], colLabels: ['d0', 'd1', 'd2', 'd3'], title: 'stream update', type: 'vector', shape: '[4] each', scale: 'diverging', highlight: { row: 2 } });
              } });
            });
            steps.push({ title: 'Result: embedding + sum of edits', render: function (b) {
              var tot = edits.reduce(function (a, e) { return api.math.add(a, e.d); }, [0, 0, 0, 0]);
              formula(api, b, 'x<sub>4</sub> = x<sub>0</sub> + ΣΔ = ' + vec(api, x0) + ' + ' + vec(api, tot) + ' = <b>' + vec(api, states[4]) + '</b>');
              api.html(b, '<p class="muted">In the reference model this sum has 64 terms (32 attention + 32 FFN) and lives in 4,096 dimensions.</p>');
            } });
            api.stepper(el, { steps: steps, interval: 2000 });
            api.illustrative(el, 'The Δ vectors are made up; real sub-layer outputs come from the computations in ' + api.link('attention/worked-example', 'attention') + ' and ' + api.link('ffn-moe/swiglu', 'the FFN') + '.');
          } },
          { name: 'Formula', render: function (el, api) {
            formula(api, el, 'h<sub>ℓ</sub> = x<sub>ℓ</sub> + Attn<sub>ℓ</sub>(RMSNorm(x<sub>ℓ</sub>))<br>x<sub>ℓ+1</sub> = h<sub>ℓ</sub> + FFN<sub>ℓ</sub>(RMSNorm(h<sub>ℓ</sub>))<br>⇒ x<sub>32</sub> = x<sub>0</sub> + Σ<sub>ℓ</sub> (attn edits + ffn edits)');
            var g = api.grid(el, 240);
            api.card(g, 'Why add instead of replace?', 'The derivative of x + f(x) w.r.t. x includes an identity term (1), so the training signal reaches early layers without fading through 32 multiplications (' + api.link('training/backprop', 'backprop') + ').');
            api.card(g, 'Why one fixed width?', 'Every sub-layer reads from and writes to the same d_model = 4,096 space, so all layers "speak the same language" and outputs can simply be summed.');
            api.card(g, 'Logit lens', 'Because the space is shared, you can apply the final norm + unembedding to the stream at any layer and see what it "would predict" so far (' + api.link('transformer-block/layer-roles', 'layer roles') + ').');
            api.callout(el, 'Pre-norm (norm inside the branch, as here) is what Llama-style models use; the original 2017 Transformer used post-norm (norm after the add).', 'info');
          } }
        ]
      },

      // ============================================================ 3. RMSNorm
      {
        id: 'rmsnorm',
        title: 'RMSNorm: keeping the numbers in range',
        critical: {
          understand: [
            'RMSNorm rescales a vector so its <b>root-mean-square</b> is 1, then multiplies each dimension by a learned gain.',
            'It keeps the input to each sub-layer at a predictable scale, however big the stream has grown.',
            'It works per token: one row at a time, no mixing between tokens.'
          ],
          details: [
            'RMSNorm(x) = x / √(mean(x²) + ε) ⊙ g, ε ≈ 1e-5, g ∈ ℝ<sup>d_model</sup> learned.',
            'Unlike LayerNorm it does not subtract the mean and has no bias — cheaper, and works about as well in practice.',
            'Reference model: 2 norms per block + 1 final = 65 gain vectors × 4,096 = 266,240 params.'
          ]
        },
        dataTypes: [
          { name: 'x (one token row)', type: 'vector', shape: '[d_model]', example: 'row of toyX, [4]' },
          { name: 'mean square, rms', type: 'float', shape: 'scalar', example: 'rms ≈ 0.62' },
          { name: 'ε', type: 'float', shape: 'scalar', example: '1e-5' },
          { name: 'gain g', type: 'vector', shape: '[d_model]', example: '[1.0, 0.8, 1.2, 1.0]' },
          { name: 'output', type: 'vector', shape: '[d_model]', example: 'rms ≈ 1 before gain' },
          { name: 'RMSNorm', type: 'function', shape: '[d] → [d]', example: 'per token' }
        ],
        levels: [
          { name: 'What it does', render: function (el, api) {
            var x = api.ex.toyX[1], k = 1;
            api.html(el, '<p>Scale the input row of ' + api.tok(' cat') + ' up or down. The <b>normalized</b> output stays the same — only the direction of the vector survives.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var g = api.grid(el, 240);
            var c1 = api.card(g, 'input k·x ' + api.badge('vector', '[4]'), ''), c2 = api.card(g, 'RMSNorm(k·x) (gain = 1) ' + api.badge('vector', '[4]'), '');
            var b1 = api.bars(c1.querySelector('.card-body'), { items: [], max: 9, format: function (v) { return api.fmt(v, 2); } });
            var b2 = api.bars(c2.querySelector('.card-body'), { items: [], max: 2, format: function (v) { return api.fmt(v, 2); } });
            var info = api.h('div'); el.appendChild(info);
            api.slider(ctl, { label: 'scale k', min: 0.1, max: 10, step: 0.1, value: 1, format: function (v) { return api.fmt(v, 1) + '×'; }, onInput: function (v) {
              k = v;
              var xi = api.math.scale(x, k), y = api.math.rmsnorm(xi, null, 1e-5);
              var rms = Math.sqrt(api.math.sum(xi.map(function (a) { return a * a; })) / 4);
              b1.update(xi.map(function (a, i) { return { label: 'd' + i, value: a, color: a < 0 ? '#4ea8ff' : '#ff9a4d', note: a < 0 ? '(neg)' : '' }; }));
              b2.update(y.map(function (a, i) { return { label: 'd' + i, value: a, color: a < 0 ? '#4ea8ff' : '#5ad18a', note: a < 0 ? '(neg)' : '' }; }));
              info.innerHTML = '<div class="formula">rms(input) = ' + api.fmt(rms, 3) + ' → output = input / ' + api.fmt(rms, 3) + ' → rms(output) ≈ 1.000</div>';
            } });
            api.callout(el, 'Why care? The stream grows as 64 edits pile up. Without a norm, later sub-layers would see ever-larger inputs and training tends to become unstable.', 'tip');
          } },
          { name: 'Formula', render: function (el, api) {
            formula(api, el, 'RMSNorm(x)<sub>i</sub> = x<sub>i</sub> / √( (1/d) Σ<sub>j</sub> x<sub>j</sub>² + ε ) · g<sub>i</sub>');
            api.blocks(el, { blocks: [
              { label: 'x', type: 'vector', shape: '[d]' },
              { label: 'square & mean', sub: 'mean(x²)', type: 'float', shape: 'scalar', zoom: 2 },
              { label: '√( · + ε)', sub: 'rms', type: 'float', shape: 'scalar', zoom: 2 },
              { label: 'x / rms', type: 'vector', shape: '[d]', zoom: 2 },
              { label: '⊙ g', sub: 'learned gain', type: 'vector', shape: '[d]', zoom: 2 }
            ] });
            api.table(el, { headers: ['', 'LayerNorm (2016)', 'RMSNorm (Llama-style)'], rows: [
              ['subtract mean', 'yes', 'no'], ['divide by', 'standard deviation', 'root mean square'], ['learned params', 'gain + bias (2d)', 'gain only (d)'], ['used in', 'GPT-2, BERT', 'Llama, Mistral and many recent LLMs']
            ] });
          } },
          { name: 'Worked example', render: function (el, api) {
            var ri = 1, eps = 1e-5, L = tokLabels(api);
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var box = api.h('div'); el.appendChild(box);
            function draw() {
              box.innerHTML = '';
              var x = api.ex.toyX[ri], sq = x.map(function (a) { return a * a; }), ms = api.math.sum(sq) / x.length, rms = Math.sqrt(ms + eps);
              var nx = x.map(function (a) { return a / rms; }), y = api.math.rmsnorm(x, GAIN, eps);
              api.stepper(box, { interval: 2200, steps: [
                { title: 'Take the row of ' + L[ri] + ' from X', render: function (b) {
                  api.matrix(b, { data: [x], rowLabels: ['x'], colLabels: ['d0', 'd1', 'd2', 'd3'], title: 'x = X[' + L[ri] + ']', type: 'vector', shape: '[4]', scale: 'diverging' });
                } },
                { title: 'Square each element', render: function (b) {
                  formula(api, b, 'x² = ' + x.map(function (a) { return api.fmt(a, 2) + '²'; }).join(', ') + ' = ' + vec(api, sq, 4));
                } },
                { title: 'Mean of the squares', render: function (b) {
                  formula(api, b, 'mean(x²) = (' + sq.map(function (a) { return api.fmt(a, 4); }).join(' + ') + ') / 4 = <b>' + api.fmt(ms, 4) + '</b> ' + api.type('float'));
                } },
                { title: 'Root: rms = √(mean + ε)', render: function (b) {
                  formula(api, b, 'rms = √(' + api.fmt(ms, 4) + ' + 0.00001) = <b>' + api.fmt(rms, 4) + '</b> ' + api.type('float'));
                } },
                { title: 'Divide: x / rms', render: function (b) {
                  formula(api, b, 'x / ' + api.fmt(rms, 4) + ' = ' + vec(api, nx, 3) + ' ' + api.badge('vector', '[4]'));
                  var chk = Math.sqrt(api.math.sum(nx.map(function (a) { return a * a; })) / 4);
                  api.html(b, '<p class="muted">Check: rms of this vector = ' + api.fmt(chk, 3) + '.</p>');
                } },
                { title: 'Multiply by the learned gain g', render: function (b) {
                  formula(api, b, vec(api, nx, 3) + ' ⊙ ' + vec(api, GAIN, 1) + ' = <b>' + vec(api, y, 3) + '</b>');
                  api.matrix(b, { data: [x, nx, GAIN, y], rowLabels: ['x', 'x / rms', 'g', 'output'], colLabels: ['d0', 'd1', 'd2', 'd3'], title: 'RMSNorm(' + L[ri] + ')', type: 'vector', shape: '[4] each', decimals: 3, scale: 'diverging', highlight: { row: 3 } });
                  api.html(b, '<p class="muted">This output is what the attention sub-layer reads (' + api.link('attention/projections', 'projections') + '). The stream itself is left untouched.</p>');
                } }
              ] });
            }
            api.select(ctl, { label: 'Token row of X', value: '1', options: api.ex.tokens.map(function (t, i) { return { value: String(i), label: t.text.replace(/^ /, '␣') }; }), onChange: function (v) { ri = +v; draw(); } });
            draw();
            api.illustrative(el, 'X is the shared toy input; the gain g is made up (a trained g is learned and starts at all-ones).');
          } }
        ]
      },

      // ============================================================ 4. layer roles
      {
        id: 'layer-roles',
        title: 'What layers tend to do: early, middle, late',
        critical: {
          understand: [
            'All 32 blocks have the same structure, but training makes them <b>specialise</b> loosely by depth.',
            'A common picture: early layers handle <b>local/surface</b> features, middle layers build <b>meaning and relations</b>, late layers shape the <b>next-token prediction</b>.',
            'These are tendencies from interpretability research, not hard rules — roles overlap and vary by model.'
          ],
          details: [
            'Evidence comes from probing classifiers, the logit lens / tuned lens, ablations and sparse-autoencoder features.',
            'Factual recall is often associated with middle-layer FFNs; attention heads route information between positions.',
            'Removing a single middle layer often hurts surprisingly little — another sign of redundancy.'
          ]
        },
        dataTypes: [
          { name: 'layer index', type: 'integer', shape: 'scalar', example: '0 … 32' },
          { name: 'stream at layer ℓ', type: 'vector', shape: '[d_model]', example: 'for ␣the' },
          { name: 'logit-lens distribution', type: 'probability', shape: '[vocab]', example: 'top-6 shown' },
          { name: 'unembedding', type: 'function', shape: '[d_model] → [vocab]', example: 'final norm + W_U' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var g = api.grid(el, 220);
            api.card(g, '<span class="transformer-block-early">Early layers (≈ 1–8)</span>', 'Tend to work on the surface: token identity, nearby words, fixing up multi-token words, simple syntax. Many "previous-token" style heads.');
            api.card(g, '<span class="transformer-block-mid">Middle layers (≈ 9–24)</span>', 'Tend to build richer meaning: who-did-what, entities, facts recalled from FFN "memories" (' + api.link('ffn-moe/kv-memory', 'key-value memory') + '), long-range links.');
            api.card(g, '<span class="transformer-block-late">Late layers (≈ 25–32)</span>', 'Tend to turn that into a prediction: assemble the next-token candidates, sharpen or calibrate the distribution.');
            api.blocks(el, { blocks: [
              { label: 'embedding', type: 'vector', shape: '[4096]' },
              { label: 'early', sub: 'surface', cls: 'transformer-block-b-early', zoom: 1 },
              { label: 'middle', sub: 'meaning', cls: 'transformer-block-b-mid', zoom: 1 },
              { label: 'late', sub: 'prediction', cls: 'transformer-block-b-late', zoom: 1 },
              { label: 'logits', type: 'float', shape: '[128,256]', goto: 'output-sampling/unembedding' }
            ] });
            api.callout(el, 'Layer ranges are rough and illustrative. Findings come from specific models and methods; treat them as tendencies, not a blueprint.', 'hedge');
          } },
          { name: 'Logit lens', render: function (el, api) {
            var cands = [' the', ' a', ' floor', ' table', ' mat', ' couch'];
            // illustrative logits at anchor layers 0, 8, 16, 24, 32
            var anchors = [0, 8, 16, 24, 32];
            var logit = {
              ' the': [3.0, 2.0, 0.5, -0.5, -1.5], ' a': [1.0, 1.5, 0.5, 0.0, -1.0], ' floor': [-1.0, 0.0, 2.0, 2.2, 1.8],
              ' table': [-1.0, 0.0, 1.8, 2.0, 1.4], ' mat': [-2.0, -1.0, 1.0, 2.6, 3.6], ' couch': [-2.0, -0.5, 1.2, 1.5, 1.0]
            };
            function at(arr, l) {
              for (var i = 0; i < anchors.length - 1; i++) if (l <= anchors[i + 1]) { var t = (l - anchors[i]) / (anchors[i + 1] - anchors[i]); return arr[i] + t * (arr[i + 1] - arr[i]); }
              return arr[arr.length - 1];
            }
            api.html(el, '<p><b>Logit lens</b>: take ' + api.tok(' the') + '\'s stream after layer ℓ, apply the final norm + unembedding, and look at the predicted next token. Drag through the layers.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var b = api.bars(el, { items: [], max: 1, title: 'decoded next-token guess ' + api.badge('probability', '[top-6 of vocab]'), format: function (v) { return api.fmt(v, 2); } });
            var info = api.h('div'); el.appendChild(info);
            api.slider(ctl, { label: 'after layer ℓ', min: 0, max: 32, step: 1, value: 0, format: function (v) { return String(v); }, onInput: function (l) {
              var lg = cands.map(function (c) { return at(logit[c], l); }), p = api.math.softmax(lg), top = api.math.argmax(p);
              b.update(cands.map(function (c, i) { return { label: api.tokText(c), value: p[i], highlight: i === top }; }));
              var phase = l <= 8 ? 'early: mostly echoes the current token (' + api.tok(' the') + ' → another article)' : l <= 24 ? 'middle: plausible "things you sit on" emerge' : 'late: the prediction sharpens on ' + api.tok(' mat');
              info.innerHTML = '<div class="formula">layer ' + l + ' → top guess ' + api.tok(cands[top]) + ' (' + api.fmt(p[top], 2) + ') · ' + phase + '</div>';
            } });
            api.illustrative(el, 'The trajectory is invented to show the typical shape of logit-lens plots; real curves are noisier, and early layers often decode poorly.');
          } },
          { name: 'How we know', render: function (el, api) {
            api.table(el, { headers: ['method', 'idea', 'what it suggests'], rows: [
              ['Probing', 'train a small classifier on the stream of layer ℓ', 'syntax tends to be decodable earlier than semantics'],
              ['Logit / tuned lens', 'decode intermediate streams with the unembedding', 'predictions tend to form gradually in later layers'],
              ['Ablation / patching', 'remove or swap a layer\'s or head\'s output', 'some heads/layers are critical, many are redundant'],
              ['Sparse autoencoders', 'split the stream into many sparse "features"', 'interpretable features exist at many depths']
            ] });
            api.callout(el, 'Caveats: methods disagree at times, results vary across models, and "this layer does X" is always a simplification. Interpretability is an active research area.', 'hedge');
            api.html(el, '<p class="muted">Related: ' + api.link('attention/qkv-intuition', 'attention patterns') + ' · ' + api.link('ffn-moe/kv-memory', 'FFN as memory') + ' · ' + api.link('reasoning/limits', 'faithfulness limits') + '.</p>');
          } }
        ]
      },

      // ============================================================ 5. parameter count
      {
        id: 'param-count',
        title: 'Counting parameters: per layer × 32 = 8.03 B',
        critical: {
          understand: [
            'Almost all parameters are in the 32 blocks: each has ≈ 218 M (attention ≈ 42 M + FFN ≈ 176 M).',
            'The FFN holds about <b>80%</b> of each block\'s parameters; attention about 20%.',
            'Embedding and unembedding add ≈ 525 M each because the vocabulary is large (128,256).'
          ],
          details: [
            'Total = embed + 32 × per-layer + final norm + unembed = 8,030,261,248.',
            'In BF16 (2 bytes each) that is ≈ 16 GB just to hold the weights.',
            'GQA makes W_k and W_v 4× smaller than W_q (1,024 vs 4,096 output columns).'
          ]
        },
        dataTypes: [
          { name: 'weight matrices', type: 'matrix', shape: '[d_in × d_out]', example: 'W_q [4,096 × 4,096]' },
          { name: 'norm gains', type: 'vector', shape: '[d_model]', example: '[4,096]' },
          { name: 'parameter counts', type: 'integer', shape: 'scalar', example: '8,030,261,248' },
          { name: 'model config', type: 'json', shape: 'object', example: '{d_model, n_layers, d_ff, …}' }
        ],
        levels: [
          { name: 'Total', render: function (el, api) {
            var p = api.ref.paramBreakdown(), layers = p.perLayer * api.ref.n_layers;
            api.html(el, '<div class="transformer-block-total"><span class="big-num">' + api.fmtInt(p.total) + '</span> ' + api.badge('integer', 'scalar') + '<div class="muted">≈ ' + api.fmtBig(p.total) + ' parameters · ' + api.fmtBytes(p.bytesBF16) + ' in BF16</div></div>');
            api.bars(el, { title: 'Where the parameters live', items: [
              { label: 'embedding E', value: p.embed, color: '#2fd0c9' },
              { label: '32 × block', value: layers, color: '#ff9a4d', highlight: true },
              { label: 'final norm', value: p.finalNorm, color: '#8a94a6' },
              { label: 'unembedding W_U', value: p.unembed, color: '#b98cff' }
            ], format: function (v) { return api.fmtBig(v) + ' (' + api.fmt(v / p.total * 100, 1) + '%)'; } });
            formula(api, el, api.fmtInt(p.embed) + ' + 32 × ' + api.fmtInt(p.perLayer) + ' + ' + api.fmtInt(p.finalNorm) + ' + ' + api.fmtInt(p.unembed) + ' = <b>' + api.fmtInt(p.total) + '</b>');
            api.html(el, '<p class="muted">What 8 B parameters means in practice: ' + api.link('big-picture/scale', 'scale') + '. Shrinking them: ' + api.link('inference/quantization', 'quantization') + '.</p>');
          } },
          { name: 'One layer', render: function (el, api) {
            var R = api.ref, p = api.ref.paramBreakdown();
            var rows = [
              { k: 'W_q', shape: '[4,096 × 4,096]', n: p.wq, part: 'attn', why: '32 heads × 128' },
              { k: 'W_k', shape: '[4,096 × 1,024]', n: p.wk, part: 'attn', why: '8 KV heads × 128 (GQA)' },
              { k: 'W_v', shape: '[4,096 × 1,024]', n: p.wv, part: 'attn', why: '8 KV heads × 128 (GQA)' },
              { k: 'W_o', shape: '[4,096 × 4,096]', n: p.wo, part: 'attn', why: 'concat heads → d_model' },
              { k: 'W_gate', shape: '[4,096 × 14,336]', n: p.ffn / 3, part: 'ffn', why: 'SwiGLU gate' },
              { k: 'W_up', shape: '[4,096 × 14,336]', n: p.ffn / 3, part: 'ffn', why: 'SwiGLU up' },
              { k: 'W_down', shape: '[14,336 × 4,096]', n: p.ffn / 3, part: 'ffn', why: 'back to d_model' },
              { k: 'g₁, g₂ (RMSNorm)', shape: '2 × [4,096]', n: p.norms, part: 'norm', why: 'gain vectors' }
            ];
            var sel = 0;
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var bx = api.h('div'); el.appendChild(bx);
            var detail = api.h('div'); el.appendChild(detail);
            var colors = { attn: '#ff9a4d', ffn: '#5ad18a', norm: '#8a94a6' };
            var br = api.bars(bx, { items: [], title: 'Parameters per block ' + api.badge('integer', 'count'), format: function (v) { return api.fmtInt(v); } });
            function draw() {
              br.update(rows.map(function (r, i) { return { label: r.k, value: r.n, color: colors[r.part], highlight: i === sel }; }));
              var r = rows[sel];
              detail.innerHTML = '';
              formula(api, detail, '<b>' + r.k + '</b> ' + api.badge(r.part === 'norm' ? 'vector' : 'matrix', r.shape) + ' = ' + api.fmtInt(r.n) + ' params · ' + r.why + ' · ' + api.fmt(r.n / p.perLayer * 100, 1) + '% of the block');
            }
            api.select(ctl, { label: 'Inspect', value: '0', options: rows.map(function (r, i) { return { value: String(i), label: r.k }; }), onChange: function (v) { sel = +v; draw(); } });
            draw();
            api.table(el, { headers: ['group', 'params / block', 'share'], rows: [
              ['attention (' + api.link('attention/multi-head', 'W_q,k,v,o') + ')', api.fmtInt(p.attn), api.fmt(p.attn / p.perLayer * 100, 1) + '%'],
              ['FFN (' + api.link('ffn-moe/swiglu', 'SwiGLU') + ')', api.fmtInt(p.ffn), api.fmt(p.ffn / p.perLayer * 100, 1) + '%'],
              ['norms', api.fmtInt(p.norms), api.fmt(p.norms / p.perLayer * 100, 3) + '%'],
              ['<b>per block</b>', '<b>' + api.fmtInt(p.perLayer) + '</b>', '100%'],
              ['<b>× 32 blocks</b>', '<b>' + api.fmtInt(p.perLayer * R.n_layers) + '</b>', '']
            ] });
          } },
          { name: 'Calculator', render: function (el, api) {
            var R = api.ref, c = { d: R.d_model, layers: R.n_layers, heads: R.n_heads, kv: R.n_kv_heads, dh: R.d_head, ff: R.d_ff, vocab: R.vocab };
            api.html(el, '<p>Change the config and recount. Defaults = the reference model; the formula is the same one used by ' + '<code>api.ref.paramBreakdown()</code>.</p>');
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            function upd() {
              var r = countParams(c);
              out.innerHTML = '';
              api.html(out, '<span class="big-num">' + api.fmtBig(r.total) + '</span> <span class="muted">(' + api.fmtInt(r.total) + ' params · ' + api.fmtBytes(r.total * 2) + ' BF16)</span>');
              api.bars(out, { items: [
                { label: 'embed + unembed', value: r.embed * 2, color: '#b98cff' },
                { label: 'attention × layers', value: r.attn * c.layers, color: '#ff9a4d' },
                { label: 'FFN × layers', value: r.ffn * c.layers, color: '#5ad18a' }
              ], format: function (v) { return api.fmtBig(v); } });
              api.code(out, JSON.stringify({ d_model: c.d, n_layers: c.layers, n_heads: c.heads, n_kv_heads: c.kv, d_head: c.dh, d_ff: c.ff, vocab: c.vocab }, null, 0), 'json');
            }
            api.slider(ctl, { label: 'n_layers', min: 1, max: 80, step: 1, value: c.layers, init: false, onInput: function (v) { c.layers = v; upd(); } });
            api.slider(ctl, { label: 'd_ff', min: 1024, max: 28672, step: 1024, value: c.ff, init: false, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { c.ff = v; upd(); } });
            api.slider(ctl, { label: 'vocab', min: 32000, max: 256000, step: 1000, value: c.vocab, init: false, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { c.vocab = v; upd(); } });
            api.select(ctl, { label: 'n_kv_heads', value: String(c.kv), options: ['1', '4', '8', '16', '32'], onChange: function (v) { c.kv = +v; upd(); } });
            api.button(ctl, 'Reset to reference', function () { c.layers = R.n_layers; c.ff = R.d_ff; c.vocab = R.vocab; c.kv = R.n_kv_heads; api.zoomTo(api.level); });
            upd();
            api.callout(el, 'd_model and heads are fixed here (4,096 = 32 × 128). Rule of thumb: per layer ≈ 12·d² for a classic GPT block; Llama-style blocks differ because of GQA and the 3-matrix SwiGLU FFN.', 'info');
          } }
        ]
      }
    ]
  });
})();
