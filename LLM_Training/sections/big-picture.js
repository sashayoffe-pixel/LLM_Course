(function () {
  'use strict';

  // ---- toy forward pass (illustrative weights, deterministic), computed at runtime
  function toyForward() {
    var M = Course.math, X = Course.EX.toyX, d = 4, dh = 2;
    var rnd = M.rng(7);
    function W(r, c, s) { var m = []; for (var i = 0; i < r; i++) { var row = []; for (var j = 0; j < c; j++) row.push((rnd() * 2 - 1) * s); m.push(row); } return m; }
    var Wq = W(d, d, 0.9), Wk = W(d, d, 0.9), Wv = W(d, d, 0.9), Wo = W(d, d, 0.9);
    var W1 = W(d, 8, 0.8), W2 = W(8, d, 0.8);
    var g = [1, 1, 1, 1];
    var N1 = X.map(function (r) { return M.rmsnorm(r, g); });
    var Q = M.matmul(N1, Wq), K = M.matmul(N1, Wk), V = M.matmul(N1, Wv);
    var heads = [], concat = X.map(function () { return []; });
    for (var h = 0; h < 2; h++) {
      var sl = function (A) { return A.map(function (r) { return r.slice(h * dh, h * dh + dh); }); };
      var q = sl(Q), k = sl(K), v = sl(V);
      var S = M.matmul(q, M.transpose(k)).map(function (r, i) { return r.map(function (x, j) { return j > i ? -Infinity : x / Math.sqrt(dh); }); });
      var A = S.map(function (r) { return M.softmax(r); });
      var O = M.matmul(A, v);
      O.forEach(function (r, i) { concat[i] = concat[i].concat(r); });
      heads.push({ S: S, A: A, O: O });
    }
    var attn = M.matmul(concat, Wo);
    var R1 = M.add(X, attn);
    var N2 = R1.map(function (r) { return M.rmsnorm(r, g); });
    var F = M.matmul(N2, W1).map(function (r) { return r.map(M.silu); });
    var F2 = M.matmul(F, W2);
    var R2 = M.add(R1, F2);
    var Nf = R2.map(function (r) { return M.rmsnorm(r, g); });
    var last = Nf[4];
    var vocab = [' mat', ' floor', ' dog', ' roof', ' chair', ' moon'];
    // illustrative unembedding: column 0 (" mat") is deliberately aligned with the final hidden state
    var U = [[], [], [], []];
    var r2 = M.rng(11);
    for (var i = 0; i < d; i++) for (var j = 0; j < vocab.length; j++) U[i].push(j === 0 ? last[i] * 0.9 : (r2() * 2 - 1) * 0.9);
    var logits = M.vecmat(last, U);
    var probs = M.softmax(logits);
    return { X: X, N1: N1, Q: Q, K: K, V: V, heads: heads, concat: concat, attn: attn, R1: R1, F2: F2, R2: R2, Nf: Nf, last: last, vocab: vocab, U: U, logits: logits, probs: probs };
  }

  function vec(api, v, dec) { return '[' + v.map(function (x) { return api.fmt(x, dec == null ? 2 : dec); }).join(', ') + ']'; }

  Course.register({
    id: 'big-picture', part: 'I', order: 10, title: 'The Big Picture',
    slides: [
      // ------------------------------------------------------------------ welcome
      {
        id: 'welcome', title: 'Welcome: how to use this course',
        critical: {
          understand: ['Every slide has <b>zoom levels</b>: start with the simple picture, then dive in to the mechanism and the numbers.',
            'Every value on screen carries a <b>type</b> badge and a <b>shape</b> label, so you always know what kind of data flows where.'],
          details: ['Keys: ← → change slide, ↓ ↑ zoom in / out. The right panel summarises what is critical on each slide.',
            'Numbers are computed in your browser; made-up values are labelled "illustrative".']
        },
        dataTypes: [
          { name: 'all 12 data types', type: 'text', shape: 'legend', example: 'text, token, integer, float, …' },
          { name: 'shape label', type: 'text', shape: 'string', example: '[seq_len × d_model]' }
        ],
        levels: [
          { name: 'Navigate', render: function (el, api) {
            api.html(el, '<p>This course explains <b>how large language models work</b> (Part I) and how they become <b>agents</b> (Part II). No ML background is assumed.</p>');
            var keys = api.h('div', { class: 'big-picture-keys' });
            keys.innerHTML = '<div><kbd>←</kbd> / <kbd>→</kbd> or <b>Back / Next</b>: previous / next slide</div>' +
              '<div><kbd>↓</kbd> or <b>▼ Dive in</b>: one zoom level deeper</div>' +
              '<div><kbd>↑</kbd> or <b>▲ Zoom out</b>: back to the simpler view</div>' +
              '<div><b>TOC</b>: jump anywhere; ✓ = visited, ●●● = number of zoom levels</div>' +
              '<div><b>Right panel</b>: the critical points + every datapoint on the slide</div>' +
              '<div>Click a block in a diagram: it dives in or jumps to the right section</div>';
            el.appendChild(keys);
            api.blocks(el, { blocks: [
              { label: 'Overview', sub: 'simple picture', zoom: 1 },
              { label: 'Mechanism', sub: 'how it works', zoom: 2 },
              { label: 'Numbers', sub: 'worked calculation', zoom: 2 }
            ], caption: 'Typical zoom path: Overview → Mechanism → Numbers (clicking a block here dives in).' });
            api.callout(el, 'Try it now: press <kbd>↓</kbd> to see the legend of the 12 data types used on every slide.', 'tip');
          } },
          { name: 'Legend', render: function (el, api) {
            api.html(el, '<p>Everything in an LLM is one of these <b>12 data types</b>. Badges use the same colours everywhere in the course.</p>');
            var ex = {
              text: '"The cat sat on the"', token: '" cat"', integer: '3797', float: 'logit 2.31',
              probability: '0.42', vector: '[0.9, 0.3, −0.2, 0.5]', matrix: 'X: [5 × 4]', tensor: 'Q: [2 × 5 × 2]',
              mask: 'causal mask (▲ = blocked)', boolean: 'stop? = true', json: '{"tool":"get_weather"}', function: 'softmax(·), a tool'
            };
            var sh = { text: 'string', token: 'string', integer: 'scalar', float: 'scalar', probability: 'scalar or [vocab]', vector: '[d_model]', matrix: '[seq_len × d_model]', tensor: '[n_heads × seq_len × d_head]', mask: '[seq_len × seq_len]', boolean: 'scalar', json: 'object', function: 'callable' };
            var g = api.h('div', { class: 'big-picture-legend' });
            Object.keys(Course.TYPES).forEach(function (t) {
              var c = api.h('div', { class: 'card' });
              c.innerHTML = '<div>' + api.badge(t, sh[t]) + '</div><div class="muted" style="margin:6px 0">' + api.esc(Course.TYPES[t].desc) + '</div><div><b>e.g.</b> <code>' + api.esc(ex[t]) + '</code></div>';
              g.appendChild(c);
            });
            el.appendChild(g);
          } },
          { name: 'Shapes', render: function (el, api) {
            api.html(el, '<p>A <b>shape</b> says how many numbers there are along each axis. We write it with <b>×</b>.</p>');
            api.table(el, { headers: ['Shape', 'Meaning', 'Example'], rows: [
              ['scalar', 'one number', api.badge('float', 'scalar') + ' a loss of 2.1'],
              ['[d_model]', 'a vector of d_model numbers', 'one token\'s embedding: [4] in the toy, [4,096] in the 8B model'],
              ['[seq_len × d_model]', 'one row per token', 'X for "The cat sat on the": [5 × 4] (toy) or [5 × 4,096]'],
              ['[n_heads × seq_len × d_head]', 'a 3-D tensor', 'Q split into heads: [2 × 5 × 2] (toy)'],
              ['[vocab]', 'one number per vocabulary token', 'logits / probabilities: 128,256 entries in the 8B reference']
            ] });
            api.callout(el, 'Running examples: the prompt <b>"The cat sat on the"</b> (5 tokens, next token " mat"), a <b>toy model</b> with d_model = 4, and the <b>8B reference model</b> (d_model 4,096, 32 layers). Part II follows an agent that books a restaurant and a coding agent fixing a failing test.', 'info');
          } }
        ]
      },
      // ------------------------------------------------------------------ next token
      {
        id: 'next-token', title: 'An LLM predicts the next token — in a loop',
        critical: {
          understand: ['The model does exactly one thing: given the tokens so far, output a <b>probability for every possible next token</b>.',
            'Text generation = pick a token, append it, run the model again. Repeat until a stop token.'],
          details: ['Choosing the token from the distribution is <b>sampling</b> (see ' + '<span class="xlink" data-goto="output-sampling">Output &amp; sampling</span>).',
            'Agents (Part II) are the same loop with tools wrapped around it.']
        },
        dataTypes: [
          { name: 'prompt', type: 'text', shape: 'string', example: '"The cat sat on the"' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[464, 3797, 3332, 319, 262]' },
          { name: 'next-token probabilities', type: 'probability', shape: '[vocab]', example: '128,256 values, sum = 1' },
          { name: 'chosen token', type: 'token', shape: 'scalar', example: '" mat" (id 2603)' }
        ],
        levels: [
          { name: 'The loop', render: function (el, api) {
            var b = api.blocks(el, { blocks: [
              { id: 'p', label: 'Prompt', sub: '"The cat sat on the"', type: 'text', shape: 'string' },
              { id: 'm', label: 'Model', sub: 'transformer', type: 'function', shape: '[seq_len] → [vocab]', goto: 'transformer-block' },
              { id: 'pr', label: 'Probabilities', type: 'probability', shape: '[vocab]' },
              { id: 'pick', label: 'Pick a token', sub: '" mat"', type: 'token', shape: 'scalar', goto: 'output-sampling' },
              { id: 'app', label: 'Append', sub: '… the mat', type: 'text', shape: 'string' }
            ], caption: 'The appended text becomes the next prompt. This loop is how every chatbot answer is produced.' });
            var order = ['p', 'm', 'pr', 'pick', 'app'], i = 0;
            api.every(900, function () { b.highlight(order[i++ % order.length]); });
            api.callout(el, 'Foreshadowing Part II: an <b>agent</b> is this same loop, plus ordinary code that reads certain outputs (tool calls), runs them, and appends the results.', 'key');
          } },
          { name: 'Distribution', render: function (el, api) {
            var M = api.math;
            var words = [' mat', ' floor', ' couch', ' bed', ' roof', ' moon'];
            var logits = [4.0, 2.9, 2.4, 1.6, 0.9, -1.2];
            var st = { T: 1 };
            var out = api.h('div'); var bar;
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Temperature', min: 0.2, max: 2, step: 0.1, value: 1, format: function (v) { return v.toFixed(1); }, onInput: function (v) { st.T = v; draw(); } });
            el.appendChild(out);
            function draw() {
              var p = M.softmax(logits, st.T); out.innerHTML = '';
              api.bars(out, { title: 'P(next token | "The cat sat on the") — ' + api.badge('probability', '[vocab]') + ' (top 6 shown)', items: words.map(function (w, i) { return { label: api.tokText(w), value: p[i], highlight: i === 0 }; }), max: 1, format: function (v) { return api.fmt(v, 3); } });
              out.appendChild(api.h('div', { class: 'muted', html: 'The full vector has 128,256 entries (8B reference); the rest share the remaining probability mass.' }));
            }
            api.illustrative(el, 'Logits are made up; a real model produces a different distribution at every step.');
          } },
          { name: 'Generate', render: function (el, api) {
            var M = api.math;
            var steps = [
              { ctx: 'The cat sat on the', opts: [[' mat', 0.46], [' floor', 0.21], [' couch', 0.12], [' bed', 0.08]], pick: 0 },
              { ctx: 'The cat sat on the mat', opts: [['.', 0.52], [',', 0.2], [' and', 0.15], [' in', 0.05]], pick: 0 },
              { ctx: 'The cat sat on the mat.', opts: [[' It', 0.3], [' The', 0.25], ['<|eot|>', 0.2], [' Then', 0.1]], pick: 2 }
            ];
            api.stepper(el, { interval: 2200, steps: steps.map(function (s, k) {
              return { title: 'Step ' + (k + 1) + ': feed ' + api.badge('text', 'string') + ' → model → choose', render: function (body) {
                api.html(body, '<div>Context: <code>' + api.esc(s.ctx) + '</code></div>');
                api.bars(body, { items: s.opts.map(function (o, i) { return { label: api.tokText(o[0]), value: o[1], highlight: i === s.pick }; }), max: 1, format: function (v) { return api.fmt(v, 2); } });
                api.html(body, '<div>Chosen: ' + api.tok(s.opts[s.pick][0]) + (k === 2 ? ' → <b>stop</b>: generation ends.' : ' → appended; loop again.') + '</div>');
              } };
            }) });
            api.illustrative(el, 'Probabilities are invented to show the shape of the process.');
          } }
        ]
      },
      // ------------------------------------------------------------------ architecture map
      {
        id: 'architecture-map', title: 'The master map: text in, text out',
        critical: {
          understand: ['Read top to bottom: text → tokens → IDs → vectors → N × transformer blocks → logits → a sampled token → back to text.',
            'Around that core sit training, inference tricks, the chat API, and reasoning; Part II wraps it all in agent machinery.'],
          details: ['Every block shows its <b>type</b> and <b>shape</b>; click any block to jump to its section.',
            'Dive in for the shapes of the 8B reference model and the inside of one transformer block.']
        },
        dataTypes: [
          { name: 'text', type: 'text', shape: 'string', example: '"The cat sat on the"' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[5]' },
          { name: 'hidden states', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4,096]' },
          { name: 'logits', type: 'vector', shape: '[vocab]', example: '128,256 values' },
          { name: 'probabilities', type: 'probability', shape: '[vocab]', example: 'sum = 1' }
        ],
        levels: [
          { name: 'Whole system', render: function (el, api) {
            api.blocks(el, { caption: 'Click any block to jump to its section. Dashed flow = data moving forward.', rows: [
              [{ label: 'Text', type: 'text', shape: 'string', goto: 'tokenization' },
               { label: 'Tokenizer', type: 'function', shape: 'string → [seq_len]', goto: 'tokenization', edge: 'BPE' },
               { label: 'Token IDs', type: 'integer', shape: '[seq_len]', goto: 'tokenization/text-to-ids' },
               { label: 'Embedding + position', type: ['matrix'], shape: '[seq_len × d_model]', goto: 'embeddings' }],
              [{ label: 'Transformer block × N', sub: 'attention · FFN/MoE · residual · norm', type: 'matrix', shape: '[seq_len × d_model]', goto: 'transformer-block', zoom: 1 },
               { label: 'Attention', type: 'tensor', shape: '[n_heads × seq × seq]', goto: 'attention' },
               { label: 'FFN / MoE', type: 'matrix', shape: '[seq_len × d_ff]', goto: 'ffn-moe' }],
              [{ label: 'Unembedding', type: 'matrix', shape: '[d_model × vocab]', goto: 'output-sampling/unembedding' },
               { label: 'Logits', type: 'vector', shape: '[vocab]', goto: 'output-sampling/unembedding' },
               { label: 'Softmax → probabilities', type: 'probability', shape: '[vocab]', goto: 'output-sampling' },
               { label: 'Sampling', type: 'function', shape: '[vocab] → scalar', goto: 'output-sampling/sampling-playground' },
               { label: 'Next token', type: 'token', shape: 'scalar', goto: 'output-sampling/autoregressive-loop', edge: 'append, repeat' }]
            ] });
            api.html(el, '<h4 style="margin:12px 0 2px">Around the core</h4>');
            api.blocks(el, { arrows: false, animate: false, blocks: [
              { label: 'Training', sub: 'cross-entropy, backprop, SFT, RLHF, LoRA', type: 'function', shape: 'weights ← gradients', goto: 'training' },
              { label: 'Experiment tracking', sub: 'ClearML', type: 'json', shape: 'object', goto: 'clearml' },
              { label: 'Inference', sub: 'KV cache, quantization, batching', type: 'tensor', shape: '[layers × 2 × kv_heads × seq × d_head]', goto: 'inference' },
              { label: 'Chat API & template', sub: 'messages → one sequence', type: 'json', shape: 'object', goto: 'chat-api' },
              { label: 'Reasoning', sub: 'chain of thought, test-time compute', type: 'text', shape: 'string', goto: 'reasoning' }
            ] });
            api.html(el, '<h4 style="margin:12px 0 2px">Part II — agents on top</h4>');
            api.blocks(el, { caption: 'The agent loop wraps repeated next-token prediction with tools, memory and control.', rows: [
              [{ label: 'Agent loop', type: 'function', shape: 'goal → actions', goto: 'llm-to-agent' },
               { label: 'Tool calling', type: 'json', shape: 'object', goto: 'tool-calling' },
               { label: 'Memory / RAG', type: 'vector', shape: '[d_model]', goto: 'memory-rag' },
               { label: 'Planning & multi-agent', type: 'json', shape: 'list[object]', goto: 'planning-multiagent' }],
              [{ label: 'LangChain', type: 'function', shape: 'Runnable', goto: 'langchain' },
               { label: 'LangGraph', type: 'json', shape: 'graph state', goto: 'langgraph' },
               { label: 'Skills', type: 'text', shape: 'SKILL.md', goto: 'skills' },
               { label: 'ComfyUI', type: 'tensor', shape: '[4 × 64 × 64] latent', goto: 'comfyui' },
               { label: 'Safety & evals', type: 'boolean', shape: 'scalar', goto: 'safety-eval' }]
            ] });
          } },
          { name: 'Reference shapes', render: function (el, api) {
            var R = api.ref;
            api.html(el, '<p>The same path with the <b>8B reference model</b> numbers for the prompt "The cat sat on the" (seq_len = 5). Training sits beside it: the same forward pass, then a loss and weight updates.</p>');
            api.blocks(el, { rows: [
              [{ label: 'Prompt', type: 'text', shape: 'string' },
               { label: 'Tokenizer', sub: 'vocab 128,256', type: 'function', shape: 'string → [5]', goto: 'tokenization' },
               { label: 'IDs', type: 'integer', shape: '[5]', sub: '[464, 3797, …]' },
               { label: 'E lookup', sub: 'E: [128,256 × 4,096]', type: 'matrix', shape: '[5 × 4,096]', goto: 'embeddings/lookup' },
               { label: '+ RoPE', sub: 'rotates Q, K', type: 'function', shape: '[5 × 4,096]', goto: 'embeddings/rope' }],
              [{ label: '32 × block', sub: 'in: [5 × 4,096]', type: 'matrix', shape: '[5 × 4,096]', goto: 'transformer-block' },
               { label: 'Attention', sub: '32 Q heads, 8 KV heads, d_head 128', type: 'tensor', shape: '[32 × 5 × 128]', goto: 'attention/gqa-cost' },
               { label: 'FFN (SwiGLU)', sub: 'd_ff 14,336', type: 'matrix', shape: '[5 × 14,336]', goto: 'ffn-moe/swiglu' },
               { label: 'Final norm', type: 'matrix', shape: '[5 × 4,096]', goto: 'transformer-block/rmsnorm' }],
              [{ label: 'Last row', sub: 'only the final position predicts', type: 'vector', shape: '[4,096]' },
               { label: 'Unembed', sub: 'U: [4,096 × 128,256]', type: 'matrix', shape: '[128,256]', goto: 'output-sampling/unembedding' },
               { label: 'Softmax', type: 'probability', shape: '[128,256]', goto: 'output-sampling' },
               { label: 'Sample', type: 'token', shape: 'scalar', sub: '" mat" = 2603', goto: 'output-sampling/sampling-playground' }],
              [{ label: 'Training', sub: 'loss = −log p(true next token)', type: 'float', shape: 'scalar', goto: 'training/cross-entropy' },
               { label: 'KV cache', sub: R.kvBytesPerToken / 1024 + ' KiB per token', type: 'tensor', shape: '[2 × 32 × 8 × seq × 128]', goto: 'inference/kv-cache' },
               { label: 'Chat template', sub: 'roles → one sequence', type: 'text', shape: 'string', goto: 'chat-api/chat-template' },
               { label: 'Reasoning', sub: 'more tokens = more compute', type: 'text', shape: 'string', goto: 'reasoning' }]
            ] });
            api.callout(el, 'Weights total ' + api.fmtInt(R.paramBreakdown().total) + ' numbers ≈ ' + api.fmtBytes(R.paramBreakdown().bytesBF16) + ' in BF16 — see the <span class="xlink" data-goto="big-picture/scale">scale</span> slide.', 'info');
          } },
          { name: 'Inside a block', render: function (el, api) {
            api.html(el, '<p>One decoder block, repeated 32 times. The <b>residual stream</b> (the main matrix) passes through; each sub-layer reads it and <b>adds</b> its result.</p>');
            api.blocks(el, { direction: 'column', caption: 'Shapes for the 8B reference with seq_len = 5. Pre-norm layout (norm before each sub-layer).', rows: [
              [{ label: 'x (residual stream)', type: 'matrix', shape: '[5 × 4,096]', goto: 'transformer-block/residual-stream' }],
              [{ label: 'RMSNorm', type: 'function', shape: '[5 × 4,096]', goto: 'transformer-block/rmsnorm' },
               { label: 'Q K V projections', sub: 'Q: 4,096→4,096 · K,V: 4,096→1,024', type: 'matrix', shape: '[5 × 4,096], [5 × 1,024]', goto: 'attention/projections' },
               { label: 'Scores QKᵀ/√128 + causal mask', type: ['matrix', 'mask'], shape: '[32 × 5 × 5]', goto: 'attention/causal-mask' },
               { label: 'softmax · V, concat, W_o', type: 'matrix', shape: '[5 × 4,096]', goto: 'attention/multi-head' }],
              [{ label: 'x + attention', type: 'matrix', shape: '[5 × 4,096]', sub: 'residual add' }],
              [{ label: 'RMSNorm', type: 'function', shape: '[5 × 4,096]' },
               { label: 'gate, up (4,096→14,336)', type: 'matrix', shape: '[5 × 14,336]', goto: 'ffn-moe/swiglu' },
               { label: 'SiLU(gate) ⊙ up, down', type: 'matrix', shape: '[5 × 4,096]', goto: 'ffn-moe/ffn-role', note: 'In MoE models this is replaced by a router + several expert FFNs.' },
               { label: 'MoE variant', sub: 'router picks top-k experts', type: 'json', shape: '[5 × n_experts]', goto: 'ffn-moe/moe' }],
              [{ label: 'x + FFN → next block', type: 'matrix', shape: '[5 × 4,096]', goto: 'transformer-block/decoder-only' }]
            ] });
          } }
        ]
      },
      // ------------------------------------------------------------------ forward pass
      {
        id: 'forward-pass', title: 'One forward pass, with numbers',
        critical: {
          understand: ['A forward pass turns <b>5 token IDs</b> into <b>one probability distribution</b> over the whole vocabulary.',
            'Same stages in the toy model and in the 8B model — only the sizes differ.'],
          details: ['Toy weights here are illustrative (seeded random, one block, d_model = 4); the pipeline and shapes are real.',
            'Only the <b>last row</b> of the final hidden matrix is used to predict the next token.']
        },
        dataTypes: [
          { name: 'X (embeddings + position)', type: 'matrix', shape: '[5 × 4]', example: 'toy; [5 × 4,096] in 8B' },
          { name: 'Q, K, V', type: 'matrix', shape: '[5 × 4]', example: 'toy' },
          { name: 'logits', type: 'vector', shape: '[6]', example: 'toy vocab of 6; [128,256] in 8B' },
          { name: 'probabilities', type: 'probability', shape: '[6]', example: 'sum = 1' }
        ],
        levels: [
          { name: 'Stages', render: function (el, api) {
            var R = api.ref;
            api.table(el, { headers: ['#', 'Stage', 'Toy model', '8B reference'], rows: [
              [1, 'Token IDs ' + api.type('integer'), '[5]', '[5]'],
              [2, 'Embed + position ' + api.type('matrix'), '[5 × 4]', '[5 × 4,096]'],
              [3, 'Attention ' + api.type('tensor'), '[2 × 5 × 5] weights', '[32 × 5 × 5] weights'],
              [4, 'FFN ' + api.type('matrix'), '[5 × 8] hidden', '[5 × 14,336] hidden'],
              [5, 'Blocks', '1 (illustrative)', R.n_layers],
              [6, 'Final hidden → logits ' + api.type('vector'), '[6]', '[128,256]'],
              [7, 'Softmax → ' + api.type('probability'), '[6]', '[128,256]']
            ] });
            api.callout(el, 'Dive in to step through the toy model with real arithmetic.', 'tip');
          } },
          { name: 'Toy numbers', render: function (el, api) {
            var T = toyForward(), M = api.math, labels = ['The', '␣cat', '␣sat', '␣on', '␣the'];
            var mx = function (title, data, type, shape, extra) { return function (body) { api.matrix(body, Object.assign({ data: data, rowLabels: labels, title: title, type: type, shape: shape, decimals: 2 }, extra || {})); }; };
            api.stepper(el, { interval: 2600, steps: [
              { title: 'Input X: embeddings + position', render: mx('X', T.X, 'matrix', '[5 × 4]') },
              { title: 'Attention (head 1 weights, causal)', render: function (b) {
                api.matrix(b, { data: T.heads[0].A, rowLabels: labels, colLabels: labels, title: 'A (head 1)', type: 'probability', shape: '[5 × 5]', causal: true, decimals: 2 });
                api.html(b, '<div class="muted">Each row sums to 1; "the" (last row) mixes values of earlier tokens. Computed from X via Q, K, V — details in ' + api.link('attention/worked-example', 'attention') + '.</div>'); } },
              { title: 'Residual add: X + attention output', render: mx('X + attn', T.R1, 'matrix', '[5 × 4]') },
              { title: 'FFN + residual add', render: mx('after FFN', T.R2, 'matrix', '[5 × 4]') },
              { title: 'Take the last row (position of "the")', render: function (b) {
                api.html(b, '<div>Final-normed hidden vector ' + api.badge('vector', '[4]') + ': <code>' + vec(api, T.last) + '</code></div>'); } },
              { title: 'Unembed → logits', render: function (b) {
                api.html(b, '<div>last · U ' + api.badge('vector', '[6]') + ' = <code>' + vec(api, T.logits) + '</code></div>');
                api.html(b, '<div class="muted">vocab: ' + T.vocab.map(function (w) { return api.tok(w); }).join(' ') + '</div>'); } },
              { title: 'Softmax → probabilities', render: function (b) {
                api.bars(b, { items: T.vocab.map(function (w, i) { return { label: api.tokText(w), value: T.probs[i], highlight: i === M.argmax(T.probs) }; }), max: 1 });
                api.html(b, '<div>Σ = ' + api.fmt(M.sum(T.probs), 3) + ' · most likely: <span class="big-picture-pick">' + api.tokText(T.vocab[M.argmax(T.probs)]) + '</span></div>'); } }
            ] });
            api.illustrative(el, 'One block, seeded random weights; the unembedding is arranged so " mat" wins. A real 32-block model learns all of this.');
          } },
          { name: '8B shapes', render: function (el, api) {
            var R = api.ref;
            api.stepper(el, { interval: 2600, steps: [
              { title: 'IDs → E rows', html: api.badge('integer', '[5]') + ' → E ' + api.shape('[128,256 × 4,096]') + ' → ' + api.badge('matrix', '[5 × 4,096]') },
              { title: 'Per block: attention', html: 'Q ' + api.shape('[5 × 4,096]') + ' → 32 heads × 128; K, V ' + api.shape('[5 × 1,024]') + ' → 8 KV heads × 128; scores ' + api.shape('[32 × 5 × 5]') + '; output ' + api.shape('[5 × 4,096]') },
              { title: 'Per block: FFN', html: 'gate/up ' + api.shape('[5 × 14,336]') + ' → SiLU(gate) ⊙ up → down ' + api.shape('[5 × 4,096]') },
              { title: '× 32 blocks', html: '32 repeats of the same shapes: ' + api.fmtInt(R.paramBreakdown().perLayer) + ' weights each.' },
              { title: 'Last row → logits', html: 'row ' + api.shape('[4,096]') + ' · U ' + api.shape('[4,096 × 128,256]') + ' = logits ' + api.badge('vector', '[128,256]') },
              { title: 'Softmax → sample', html: api.badge('probability', '[128,256]') + ' → pick id 2603 → ' + api.tok(' mat', 2603) }
            ] });
          } }
        ]
      },
      // ------------------------------------------------------------------ scale
      {
        id: 'scale', title: 'What "8 billion parameters" means',
        critical: {
          understand: ['A parameter is one learned number (a weight). The 8B reference model has ≈ 8.03 billion of them.',
            'Stored as 2-byte BF16 numbers that is ≈ 16 GB — just to hold the weights.'],
          details: ['Generating one token costs roughly <b>2 × N</b> floating-point operations (one multiply + one add per weight).',
            'Most weights sit in the FFN layers; attention is smaller; embeddings are ~13 % of the total.']
        },
        dataTypes: [
          { name: 'parameter count N', type: 'integer', shape: 'scalar', example: '8,030,261,248' },
          { name: 'weight matrices', type: 'matrix', shape: '[d_model × d_ff] etc.', example: 'W_gate [4,096 × 14,336]' },
          { name: 'bytes', type: 'integer', shape: 'scalar', example: '≈ 16 GB in BF16' },
          { name: 'FLOPs per token', type: 'float', shape: 'scalar', example: '≈ 2·N ≈ 16 GFLOPs' }
        ],
        levels: [
          { name: 'Headline', render: function (el, api) {
            var B = api.ref.paramBreakdown();
            var s = api.h('div', { class: 'big-picture-stat' });
            s.innerHTML = '<div><span class="big-num">' + api.fmtBig(B.total) + '</span>parameters</div>' +
              '<div><span class="big-num">' + api.fmtBytes(B.bytesBF16) + '</span>in BF16 (2 bytes each)</div>' +
              '<div><span class="big-num">' + api.fmtBig(2 * B.total).replace(' B', ' G') + 'FLOPs</span>per generated token (≈ 2·N)</div>';
            el.appendChild(s);
            api.html(el, '<p>Every parameter is a number inside a weight matrix. "Training" means adjusting all ' + api.fmtBig(B.total) + ' of them; "inference" means reading them all (roughly once per token).</p>');
            api.table(el, { headers: ['Comparison', 'Rough figure (illustrative)'], rows: [
              ['Weights in BF16', api.fmtBytes(B.bytesBF16) + ' ≈ several HD movies'],
              ['Fits on', 'one 24 GB GPU only if quantized (e.g. 8-bit ≈ 8 GB) or with little headroom; comfortably on 40–80 GB'],
              ['A frontier model', 'often rumoured/reported to be 10–100× larger; exact sizes are usually not published']
            ] });
          } },
          { name: 'Breakdown', render: function (el, api) {
            var B = api.ref.paramBreakdown(), L = api.ref.n_layers;
            var items = [
              ['Embedding E', B.embed], ['FFN (×32)', B.ffn * L], ['Attention (×32)', B.attn * L], ['Unembedding U', B.unembed], ['Norms', B.norms * L + B.finalNorm]
            ];
            api.bars(el, { title: 'Parameters by group (' + api.badge('integer', 'scalar') + ')', items: items.map(function (x) { return { label: x[0], value: x[1], note: (100 * x[1] / B.total).toFixed(1) + ' %' }; }), format: function (v) { return api.fmtBig(v); } });
            api.table(el, { headers: ['Per layer', 'Formula', 'Count'], rows: [
              ['W_q', '4,096 × 4,096', api.fmtInt(B.wq)], ['W_k', '4,096 × 1,024', api.fmtInt(B.wk)], ['W_v', '4,096 × 1,024', api.fmtInt(B.wv)], ['W_o', '4,096 × 4,096', api.fmtInt(B.wo)],
              ['FFN (gate, up, down)', '3 × 4,096 × 14,336', api.fmtInt(B.ffn)], ['2 norms', '2 × 4,096', api.fmtInt(B.norms)], ['<b>One layer</b>', '', '<b>' + api.fmtInt(B.perLayer) + '</b>']
            ] });
            api.html(el, '<div class="formula">total = E + 32 × layer + final norm + U = ' + api.fmtInt(B.embed) + ' + 32 × ' + api.fmtInt(B.perLayer) + ' + ' + api.fmtInt(B.finalNorm) + ' + ' + api.fmtInt(B.unembed) + ' = ' + api.fmtInt(B.total) + '</div>');
            api.html(el, '<div>More in ' + api.link('transformer-block/param-count', 'parameter count per layer') + '. (Some models tie E and U, which would save ' + api.fmtBig(B.unembed) + ' parameters.)</div>');
          } },
          { name: 'Compute', render: function (el, api) {
            var B = api.ref.paramBreakdown(), st = { gpu: 100 };
            api.html(el, '<div class="formula">FLOPs per token ≈ 2 × N = 2 × ' + api.fmtInt(B.total) + ' ≈ ' + (2 * B.total / 1e9).toFixed(1) + ' GFLOPs</div>');
            api.html(el, '<p>Each weight is used in one multiply and one add. (Attention over long contexts adds more, but at short context this estimate is the usual rule of thumb.)</p>');
            var out = api.h('div'); var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Hardware throughput (illustrative)', min: 10, max: 1000, step: 10, value: 100, format: function (v) { return v + ' TFLOP/s'; }, onInput: function (v) { st.gpu = v; draw(); } });
            el.appendChild(out);
            function draw() {
              var perTok = 2 * B.total, ideal = st.gpu * 1e12 / perTok;
              out.innerHTML = '<div>Compute-bound ceiling: ' + api.fmtInt(ideal) + ' tokens/s (' + api.badge('float', 'scalar') + ') — but single-stream decoding is usually limited by <b>memory bandwidth</b> (reading ' + api.fmtBytes(B.bytesBF16) + ' of weights per token), so real speeds are far lower. See ' + api.link('inference/prefill-decode', 'prefill vs decode') + '.</div>';
            }
            api.illustrative(el, 'Throughput numbers are hypothetical, not a benchmark.');
          } }
        ]
      },
      // ------------------------------------------------------------------ llm vs software
      {
        id: 'llm-vs-software', title: 'LLMs vs software, search engines and databases',
        critical: {
          understand: ['Ordinary software is <b>deterministic</b>: same input, same output, by rules someone wrote. An LLM is <b>probabilistic</b>: it samples from learned patterns.',
            'Use each for what it is good at — and often combine them: the LLM decides, ordinary code executes.'],
          details: ['A database returns exactly what was stored; an LLM generates plausible text and may "hallucinate".',
            'Search finds existing documents; an LLM composes new text. RAG (Part II) joins the two.']
        },
        dataTypes: [
          { name: 'query / prompt', type: 'text', shape: 'string', example: '"capital of France?"' },
          { name: 'function output', type: 'boolean', shape: 'scalar', example: 'is_even(7) = false' },
          { name: 'next-token distribution', type: 'probability', shape: '[vocab]', example: 'sampled, may vary' },
          { name: 'tool', type: 'function', shape: 'callable', example: 'sql_query(…)' }
        ],
        levels: [
          { name: 'Comparison', render: function (el, api) {
            api.html(el, '<div class="big-picture-cmp"></div>');
            api.table(el, { cls: 'big-picture-cmp', headers: ['', 'Ordinary software', 'Search engine', 'Database', 'LLM'], rows: [
              ['Logic comes from', 'rules a human wrote', 'indexing + ranking', 'schema + query', 'patterns learned from data'],
              ['Same input → same output?', 'yes', 'mostly', 'yes', 'not necessarily (sampling)'],
              ['Returns', 'computed result', 'links to documents', 'stored rows', 'newly generated text'],
              ['Can be wrong how?', 'bugs', 'irrelevant results', 'stale/wrong data', 'plausible but false statements'],
              ['Strength', 'exactness, speed, auditability', 'finding existing sources', 'transactions, precise lookups', 'language, flexibility, fuzzy tasks'],
              ['Cost per call', 'tiny', 'small', 'small', 'large (≈ 2·N FLOPs per token)']
            ] });
          } },
          { name: 'Deterministic?', render: function (el, api) {
            var M = api.math, st = { T: 0.8, n: 0 };
            api.html(el, '<p>Left: a normal function. Right: next-token sampling from a fixed distribution. Press the button several times.</p>');
            var g = api.grid(el, 260);
            var a = api.h('div', { class: 'card' }), b = api.h('div', { class: 'card' }); g.appendChild(a); g.appendChild(b);
            a.innerHTML = '<b>' + api.badge('function', 'callable') + ' is_even(7)</b><div id="bp-det" style="margin-top:6px"></div>';
            b.innerHTML = '<b>' + api.badge('probability', '[vocab]') + ' sample " The cat sat on the …"</b>';
            var out = api.h('div', { class: 'big-picture-seq' }); b.appendChild(out);
            var det = a.querySelector('#bp-det');
            var words = [' mat', ' floor', ' couch', ' roof'], logits = [3, 2.2, 1.5, 0.5], rnd = M.rng(5), hist = [];
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Temperature', min: 0.1, max: 2, step: 0.1, value: 0.8, format: function (v) { return v.toFixed(1); }, onInput: function (v) { st.T = v; } });
            api.button(c, 'Run again', function () {
              st.n++;
              det.innerHTML = 'run ' + st.n + ': <b>false</b> ' + api.type('boolean');
              var p = M.softmax(logits, st.T), i = M.sampleIndex(p, rnd());
              hist.push(words[i]); if (hist.length > 8) hist.shift();
              out.innerHTML = hist.map(function (w) { return api.tok(w); }).join(' ');
            }, 'primary');
            det.innerHTML = 'press "Run again"';
            api.callout(el, 'At temperature → 0 the LLM becomes (almost) deterministic, but it is still not <i>rule-based</i>: correctness is not guaranteed by construction.', 'hedge');
            api.illustrative(el, 'The distribution is made up.');
          } },
          { name: 'When to use', render: function (el, api) {
            var cases = [
              ['Compute a tax total exactly', 'Ordinary code', 'Needs exactness and auditability.'],
              ['Find the policy document that mentions X', 'Search (or RAG)', 'The answer exists in a document; retrieve, then optionally summarise.'],
              ['Look up order #4821', 'Database', 'Precise key lookup; an LLM could invent a plausible answer.'],
              ['Summarise a messy email thread and draft a reply', 'LLM', 'Fuzzy language task with no exact rule.'],
              ['Decide which tool to call for "book a table"', 'LLM + code (agent)', 'LLM chooses; code executes and validates (Part II).']
            ];
            var out = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.select(c, { label: 'Task', options: cases.map(function (x, i) { return { value: String(i), label: x[0] }; }), value: '0', onChange: show });
            el.appendChild(out);
            function show(v) { var x = cases[+v]; out.innerHTML = ''; api.callout(out, '<b>Best fit: ' + x[1] + '.</b> ' + x[2], 'key'); }
            show('0');
            api.html(el, '<p>Rule of thumb: <b>let the LLM handle ambiguity, let code handle guarantees.</b> See ' + api.link('llm-to-agent', 'LLM → agent') + ' and ' + api.link('tool-calling', 'tool calling') + '.</p>');
          } }
        ]
      }
    ]
  });
})();
