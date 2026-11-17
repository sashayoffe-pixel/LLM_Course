(function () {
  'use strict';

  var HW = {
    a100: { label: 'A100-class (illustrative: 2.0 TB/s, 300 TFLOP/s)', bw: 2.0e12, flops: 3.0e14 },
    h100: { label: 'H100-class (illustrative: 3.3 TB/s, 800 TFLOP/s)', bw: 3.3e12, flops: 8.0e14 }
  };
  var GB = 1e9;
  function kvBytes(ref, ctx, batch, kvHeads, bytes) {
    return 2 * ref.n_layers * kvHeads * ref.d_head * bytes * ctx * batch;
  }
  function pct(api, v) { return api.fmt(100 * v, 0) + '%'; }
  function hwSelect(api, c, cb) {
    var cur = 'a100';
    api.select(c, { label: 'GPU class', options: Object.keys(HW).map(function (k) { return { value: k, label: HW[k].label }; }), value: cur, onChange: function (v) { cur = v; cb(HW[v]); } });
    return HW[cur];
  }

  Course.register({
    id: 'inference', part: 'I', order: 90, title: 'Inference & Serving',
    slides: [
      // ------------------------------------------------------------------ 1
      {
        id: 'prefill-decode', title: 'Prefill vs decode',
        critical: {
          understand: [
            '<b>Prefill</b> processes the whole prompt in one parallel pass (many tokens at once) and fills the KV cache. It is typically <b>compute-bound</b>.',
            '<b>Decode</b> generates one token per pass; each pass must stream all the weights from GPU memory, so it is typically <b>memory-bandwidth-bound</b>.',
            'That is why "time to first token" grows with prompt length while "tokens per second" is mostly about memory bandwidth.'
          ],
          details: [
            'Rule of thumb: decode time per token ≳ weight bytes ÷ memory bandwidth (16 GB ÷ 2 TB/s ≈ 8 ms at batch 1).',
            'Prefill FLOPs ≈ 2 × parameters × prompt tokens. Decode does the same per token, but only for one token — too little work to keep the GPU busy.',
            'Hardware figures on these slides are illustrative round numbers, not benchmarks.'
          ]
        },
        dataTypes: [
          { name: 'prompt tokens', type: 'integer', shape: '[prompt_len]', example: '[1000]' },
          { name: 'KV cache', type: 'tensor', shape: '[layers × 2 × kv_heads × seq_len × d_head]', example: '128 KiB per token' },
          { name: 'time to first token (TTFT)', type: 'float', shape: 'scalar', example: 'ms' },
          { name: 'time per output token', type: 'float', shape: 'scalar', example: 'ms' }
        ],
        levels: [
          { name: 'Two phases', render: function (el, api) {
            api.blocks(el, { rows: [[
              { id: 'p', label: 'Prefill', sub: 'all prompt tokens at once', type: 'function', color: 'var(--t-matrix)', zoom: true },
              { id: 'kv', label: 'KV cache filled', type: 'tensor', goto: 'inference/kv-cache' },
              { id: 'd', label: 'Decode', sub: 'one new token per pass', type: 'function', color: 'var(--t-probability)', zoom: true },
              { id: 't', label: 'tokens streamed', type: 'token' }
            ]], caption: 'Prefill = matrix × matrix (parallel). Decode = matrix × vector, repeated (sequential).' });
            api.callout(el, '<b>Why decode cannot be parallelised across time:</b> token <i>t+1</i> depends on the sampled token <i>t</i> (' + api.link('output-sampling/autoregressive-loop', 'autoregressive loop') + '). The only parallelism left is across <i>requests</i> (' + api.link('inference/batching-speculative', 'batching') + ') or via tricks like speculative decoding.', 'key');
          } },
          { name: 'Latency', render: function (el, api) {
            var P = api.ref.paramBreakdown().total, wBytes = api.ref.paramBreakdown().bytesBF16;
            var hw = HW.a100, L = 2000, N = 200;
            var out = api.h('div'), chartBox = api.h('div'), ch = null;
            var c = api.h('div', { class: 'controls' });
            function draw() {
              var tPre = 2 * P * L / hw.flops;
              var tTok = Math.max(wBytes / hw.bw, 2 * P / hw.flops) + api.ref.kvBytesPerToken * L / hw.bw;
              var tot = tPre + N * tTok;
              out.innerHTML = '';
              api.html(out, '<div class="inference-two"><div class="card"><div class="card-title">Prefill (' + api.fmtInt(L) + ' tokens)</div><div class="big-num">' + api.fmt(tPre * 1000, 0) + ' ms</div><div class="inference-note">2 × ' + api.fmtBig(P) + ' × ' + api.fmtInt(L) + ' ≈ ' + api.fmt(2 * P * L / 1e12, 1) + ' TFLOP ÷ ' + api.fmt(hw.flops / 1e12, 0) + ' TFLOP/s</div></div>' +
                '<div class="card"><div class="card-title">Decode (per token)</div><div class="big-num">' + api.fmt(tTok * 1000, 1) + ' ms</div><div class="inference-note">read ' + api.fmtBytes(wBytes) + ' weights + ' + api.fmtBytes(api.ref.kvBytesPerToken * L) + ' KV cache ÷ ' + api.fmt(hw.bw / 1e12, 1) + ' TB/s → ≈ ' + api.fmt(1 / tTok, 0) + ' tok/s</div></div></div>');
              api.html(out, '<div class="formula inference-f">total ≈ ' + api.fmt(tPre * 1000, 0) + ' ms + ' + N + ' × ' + api.fmt(tTok * 1000, 1) + ' ms = ' + api.fmt(tot, 2) + ' s</div>');
              var pts = [], x;
              for (x = 0; x <= N; x += Math.max(1, Math.round(N / 20))) pts.push([x, (tPre + x * tTok) * 1000]);
              pts.push([N, tot * 1000]);
              var ser = [{ name: 'time since request (ms)', points: pts, color: '#4ea8ff' }];
              if (ch) ch.update({ series: ser, xRange: [0, N] });
              else ch = api.chart(chartBox, { series: ser, xLabel: 'output tokens generated', yLabel: 'ms', height: 240, xRange: [0, N] });
            }
            hwSelect(api, c, function (h) { hw = h; draw(); });
            api.slider(c, { label: 'Prompt tokens', min: 100, max: 32000, step: 100, value: 2000, format: api.fmtInt, onInput: function (v) { L = v; draw(); } });
            api.slider(c, { label: 'Output tokens', min: 10, max: 1000, step: 10, value: 200, format: api.fmtInt, onInput: function (v) { N = v; draw(); } });
            el.appendChild(c); el.appendChild(out); el.appendChild(chartBox);
            api.html(el, '<div class="inference-note">The jump at x = 0 is time-to-first-token (prefill); the slope is per-token decode time. Longer prompts raise the jump and, via a bigger KV cache, slightly the slope.</div>');
            api.illustrative(el, 'Round hardware numbers, ideal utilisation, batch 1, no overheads. Real systems are slower and vary by kernel and software.');
          } },
          { name: 'Why', render: function (el, api) {
            var P = api.ref.paramBreakdown().total, wBytes = api.ref.paramBreakdown().bytesBF16, hw = HW.a100;
            var ridge = hw.flops / hw.bw;
            var out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw(n) {
              var flop = 2 * P * n, bytes = wBytes, ai = flop / bytes;
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">tokens processed together n = ' + api.fmtInt(n) + '\nFLOPs = 2 × params × n = ' + api.fmt(flop / 1e12, 2) + ' TFLOP\nbytes moved ≈ weights = ' + api.fmtBytes(bytes) + '\narithmetic intensity = FLOPs ÷ bytes = ' + api.fmt(ai, 0) + ' FLOP/byte\nGPU ridge point = ' + api.fmt(hw.flops / 1e12, 0) + ' TFLOP/s ÷ ' + api.fmt(hw.bw / 1e12, 1) + ' TB/s = ' + api.fmt(ridge, 0) + ' FLOP/byte</div>');
              api.callout(out, ai < ridge ? '<b>Memory-bound:</b> intensity ' + api.fmt(ai, 0) + ' is below the ridge (' + api.fmt(ridge, 0) + '): the GPU waits for weights to arrive; compute units are mostly idle.' : '<b>Compute-bound:</b> intensity ' + api.fmt(ai, 0) + ' exceeds the ridge: arithmetic is the bottleneck.', ai < ridge ? 'warn' : 'tip');
              api.bars(out, { title: 'Arithmetic intensity vs GPU ridge (FLOP/byte)', items: [{ label: 'this workload', value: ai, color: ai < ridge ? 'var(--warn)' : 'var(--good)' }, { label: 'GPU ridge', value: ridge, color: 'var(--muted)' }], max: Math.max(ai, ridge) * 1.05, format: function (v) { return api.fmt(v, 0); } });
            }
            api.slider(c, { label: 'Tokens per weight pass', min: 1, max: 2048, step: 1, value: 1, format: api.fmtInt, onInput: draw });
            el.appendChild(c); el.appendChild(out);
            api.html(el, '<div class="inference-note">Decode with batch 1 → n = 1 (intensity ≈ 1 FLOP/byte). Prefill of a 2,000-token prompt → n = 2,000. Batching many users' + ' decode steps raises n without waiting — see ' + api.link('inference/batching-speculative', 'batching') + '. Dense-attention FLOPs are ignored for simplicity.</div>');
            api.illustrative(el, 'Illustrative GPU numbers.');
          } }
        ]
      },
      // ------------------------------------------------------------------ 2
      {
        id: 'kv-cache', title: 'KV cache deep-dive and memory calculator',
        critical: {
          understand: [
            'Every past token\'s <b>keys and values</b> are saved so decode never recomputes them: each new token only computes its own K, V and attends to the cache.',
            'Cost: <b>128 KiB per token</b> in the reference model = 2 (K,V) × 32 layers × 8 KV heads × 128 × 2 bytes. It grows linearly with context × batch size.',
            'At long contexts and large batches the cache can exceed the 16 GB of weights — GQA, quantised KV and paging exist to tame this.'
          ],
          details: [
            'GQA shares each K/V head among 4 query heads: 8 KV heads instead of 32 gives a 4× smaller cache (' + '<a href="#" class="xlink" data-goto="attention/gqa-cost">GQA cost</a>).',
            'Prompt caching in chat APIs reuses a stored prefix cache across requests (' + '<a href="#" class="xlink" data-goto="chat-api/cost-caching">cost &amp; caching</a>).',
            'Without a cache, recomputing everything every step makes total work grow ~quadratically with output length.'
          ]
        },
        dataTypes: [
          { name: 'K, V cache', type: 'tensor', shape: '[layers × 2 × kv_heads × seq_len × d_head]', example: '[32 × 2 × 8 × n × 128]' },
          { name: 'bytes per token', type: 'integer', shape: 'scalar', example: '131,072' },
          { name: 'context length n', type: 'integer', shape: 'scalar', example: '8,192' },
          { name: 'batch size', type: 'integer', shape: 'scalar', example: '1–64' },
          { name: 'total cache bytes', type: 'float', shape: 'scalar', example: 'GB' }
        ],
        levels: [
          { name: 'Idea', render: function (el, api) {
            api.blocks(el, { rows: [[
              { label: 'new token', type: 'token' },
              { label: 'compute q, k, v', sub: 'only for this token', type: 'function' },
              { label: 'append k, v to cache', type: 'tensor', zoom: 2 },
              { label: 'attend over cached K,V', sub: 'q · Kᵀ → weights → V', type: 'function', goto: 'attention/qkv-intuition' }
            ]], caption: 'Past tokens\' K and V never change (causal mask), so they can be stored.' });
            var toks = api.ex.tokens.concat([api.ex.nextToken]);
            api.html(el, '<div class="inference-note">Example: after "' + api.esc(api.ex.prompt) + '" the cache holds K,V for these ' + api.ex.tokens.length + ' tokens in every layer; generating ' + api.tok(api.ex.nextToken.text) + ' adds one more entry, giving ' + toks.length + '.</div>');
            api.callout(el, 'Why only K and V — not Q? Queries are used once (for the current token) and then never needed again. Keys and values are needed by <i>every future token</i>.', 'tip');
          } },
          { name: 'Per token', render: function (el, api) {
            var r = api.ref;
            api.stepper(el, { steps: [
              { title: 'One K vector and one V vector per KV head', html: 'A single head stores K [d_head] and V [d_head] = ' + api.fmtInt(2 * r.d_head) + ' numbers per token.' },
              { title: '× KV heads', html: r.n_kv_heads + ' KV heads (GQA): 2 × ' + r.d_head + ' × ' + r.n_kv_heads + ' = <b>' + api.fmtInt(2 * r.d_head * r.n_kv_heads) + '</b> numbers per token per layer.' },
              { title: '× layers', html: r.n_layers + ' layers: ' + api.fmtInt(2 * r.d_head * r.n_kv_heads) + ' × ' + r.n_layers + ' = <b>' + api.fmtInt(2 * r.d_head * r.n_kv_heads * r.n_layers) + '</b> numbers per token.' },
              { title: '× bytes per number', html: 'BF16 = 2 bytes: ' + api.fmtInt(2 * r.d_head * r.n_kv_heads * r.n_layers) + ' × 2 = <b>' + api.fmtInt(r.kvBytesPerToken) + ' bytes = ' + api.fmtBytes(r.kvBytesPerToken) + '</b> per token.' },
              { title: '× context length', html: 'An ' + api.fmtInt(r.context) + '-token context: ' + api.fmtBytes(r.kvBytesPerToken * r.context) + ' per sequence. The weights are ' + api.fmtBytes(r.paramBreakdown().bytesBF16) + '.' }
            ] });
            api.html(el, '<div class="formula inference-f">kv_bytes_per_token = 2 × n_layers × n_kv_heads × d_head × bytes\n                   = 2 × ' + r.n_layers + ' × ' + r.n_kv_heads + ' × ' + r.d_head + ' × 2 = ' + api.fmtInt(r.kvBytesPerToken) + '</div>');
          } },
          { name: 'Calculator', render: function (el, api) {
            var r = api.ref, ctx = 8192, batch = 1, kvh = 8, bytes = 2;
            var out = api.h('div'), c = api.h('div', { class: 'controls' });
            var wGB = r.paramBreakdown().bytesBF16;
            function draw() {
              var kv = kvBytes(r, ctx, batch, kvh, bytes), per = kvBytes(r, 1, 1, kvh, bytes);
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">per token = 2 × ' + r.n_layers + ' × ' + kvh + ' × ' + r.d_head + ' × ' + bytes + ' B = ' + api.fmtBytes(per) + '\ntotal = ' + api.fmtBytes(per) + ' × ' + api.fmtInt(ctx) + ' tokens × ' + batch + ' sequences = <b>' + api.fmtBytes(kv) + '</b></div>');
              api.bars(out, { title: 'GPU memory: weights vs KV cache (GB)', items: [
                { label: 'Weights (BF16)', value: wGB / GB, color: 'var(--t-matrix)', note: 'fixed' },
                { label: 'KV cache', value: kv / GB, color: kv > wGB ? 'var(--bad)' : 'var(--t-tensor)', highlight: kv > wGB }
              ], max: Math.max(wGB, kv) / GB * 1.02, format: function (v) { return api.fmt(v, 1) + ' GB'; } });
              api.html(out, '<div class="inference-note">KV cache = <b>' + api.fmt(100 * kv / wGB, 0) + '%</b> of the weight size. ' + (kv > wGB ? 'The cache is now larger than the model itself.' : '') + ' MHA (32 KV heads) would need ' + api.fmtBytes(kvBytes(r, ctx, batch, 32, bytes)) + ' for the same settings.</div>');
            }
            api.slider(c, { label: 'Context length', min: 1024, max: 131072, step: 1024, value: 8192, format: api.fmtInt, onInput: function (v) { ctx = v; draw(); } });
            api.slider(c, { label: 'Batch size', min: 1, max: 64, step: 1, value: 1, onInput: function (v) { batch = v; draw(); } });
            api.select(c, { label: 'KV heads', options: [{ value: '1', label: '1 (MQA)' }, { value: '2', label: '2' }, { value: '4', label: '4' }, { value: '8', label: '8 (reference, GQA)' }, { value: '16', label: '16' }, { value: '32', label: '32 (MHA)' }], value: '8', onChange: function (v) { kvh = +v; draw(); } });
            api.select(c, { label: 'Precision', options: [{ value: '2', label: 'BF16/FP16 (2 B)' }, { value: '1', label: 'FP8/INT8 (1 B)' }, { value: '0.5', label: 'INT4 (0.5 B)' }], value: '2', onChange: function (v) { bytes = +v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            api.html(el, '<div class="inference-note">Try context 128k × batch 8: the cache is far larger than the weights. Compare the effect of GQA in ' + api.link('attention/gqa-cost', 'GQA cost') + '. (The reference model\'s own window is 8,192 — longer contexts here are "what if".)</div>');
          } },
          { name: 'No cache?', render: function (el, api) {
            var P = api.ref.paramBreakdown().total, L = 1000, N = 500, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              var withC = L + N, without = 0, i;
              for (i = 0; i < N; i++) without += L + i + 1;
              without += 0; // each step reprocesses the full sequence
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">with cache:    prefill ' + api.fmtInt(L) + ' + ' + api.fmtInt(N) + ' decode steps × 1 token = ' + api.fmtInt(withC) + ' tokens processed\nwithout cache: Σ over steps of (current length) = ' + api.fmtInt(without) + ' tokens processed\nratio = ' + api.fmt(without / withC, 1) + '×   (≈ ' + api.fmt(2 * P * without / 1e15, 1) + ' vs ' + api.fmt(2 * P * withC / 1e15, 2) + ' PFLOP, ignoring attention cost)</div>');
              api.bars(out, { title: 'Tokens pushed through the network', items: [{ label: 'With KV cache', value: withC, color: 'var(--good)' }, { label: 'Without cache', value: without, color: 'var(--bad)' }], max: without, format: api.fmtInt });
              var pts = [], n;
              for (n = 0; n <= N; n += Math.max(1, Math.round(N / 25))) { var s = 0, j; for (j = 0; j < n; j++) s += L + j + 1; pts.push([n, s]); }
              api.chart(out, { series: [{ name: 'without cache', points: pts, color: '#ff5d73' }, { name: 'with cache', points: pts.map(function (p) { return [p[0], L + p[0]]; }), color: '#5ad18a' }], xLabel: 'output tokens generated', yLabel: 'tokens processed (cumulative)', height: 230 });
            }
            api.slider(c, { label: 'Prompt', min: 100, max: 8000, step: 100, value: 1000, format: api.fmtInt, onInput: function (v) { L = v; draw(); } });
            api.slider(c, { label: 'Output', min: 50, max: 2000, step: 50, value: 500, format: api.fmtInt, onInput: function (v) { N = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            api.html(el, '<div class="inference-note">The cache trades <b>memory</b> for <b>compute</b>: work per new token stays constant instead of growing with the text so far. (Attention itself still reads the whole cache each step.)</div>');
          } }
        ]
      },
      // ------------------------------------------------------------------ 3
      {
        id: 'context-window', title: 'Context window and "lost in the middle"',
        critical: {
          understand: [
            'The context window is the maximum number of tokens (prompt + generated reply) the model can attend to at once; the reference model\'s is 8,192.',
            'Everything the model "knows" about your conversation must fit in the window; older text has to be truncated or summarised.',
            'Research suggests models often use information near the <b>start and end</b> of a long prompt better than information in the <b>middle</b> ("lost in the middle").'
          ],
          details: [
            'Longer windows cost more: KV cache grows linearly, attention compute quadratically (' + '<a href="#" class="xlink" data-goto="attention/gqa-cost">attention cost</a>).',
            'The U-shape is a hedged empirical finding (e.g. Liu et al. 2023) that varies by model; newer models and training recipes reduce it.',
            'Practical rule: put critical instructions at the start or end and retrieve only relevant chunks (' + '<a href="#" class="xlink" data-goto="memory-rag/rag-pipeline">RAG</a>).'
          ]
        },
        dataTypes: [
          { name: 'context length', type: 'integer', shape: 'scalar', example: '8,192' },
          { name: 'tokens used', type: 'integer', shape: 'scalar', example: 'prompt + reply' },
          { name: 'position of key fact', type: 'float', shape: 'scalar', example: '0.0 = start, 1.0 = end' },
          { name: 'retrieval accuracy', type: 'probability', shape: 'scalar', example: '0.72' }
        ],
        levels: [
          { name: 'Window', render: function (el, api) {
            var W = api.ref.context, sys = 300, hist = 3000, user = 600, reply = 1000;
            var bar = api.h('div'), info = api.h('div', { class: 'inference-note' }), c = api.h('div', { class: 'controls' });
            function draw() {
              var used = sys + hist + user + reply, over = used > W;
              bar.innerHTML = '';
              var seg = [['system', sys, 'var(--t-text)'], ['history', hist, 'var(--t-matrix)'], ['user', user, 'var(--t-token)'], ['reply (max)', reply, 'var(--t-probability)']];
              var tot = Math.max(used, W), b = api.h('div', { class: 'inference-ctxbar' });
              seg.forEach(function (s) { b.appendChild(api.h('span', { style: { width: (100 * s[1] / tot) + '%', background: s[2], color: '#000' }, title: s[0], text: s[1] / tot > 0.08 ? s[0] : '' })); });
              if (used < W) b.appendChild(api.h('span', { style: { width: (100 * (W - used) / tot) + '%' }, text: '' }));
              bar.appendChild(b);
              info.innerHTML = 'Used <b>' + api.fmtInt(used) + '</b> of ' + api.fmtInt(W) + ' tokens (' + pct(api, used / W) + ')' + (over ? ' — <span style="color:var(--bad)"><b>over by ' + api.fmtInt(used - W) + '</b>: something must be truncated, summarised or dropped (or the API returns an error)</span>.' : '. Remaining: ' + api.fmtInt(W - used) + '.');
            }
            api.slider(c, { label: 'System prompt', min: 0, max: 2000, step: 50, value: sys, format: api.fmtInt, onInput: function (v) { sys = v; draw(); } });
            api.slider(c, { label: 'Chat history', min: 0, max: 8000, step: 100, value: hist, format: api.fmtInt, onInput: function (v) { hist = v; draw(); } });
            api.slider(c, { label: 'New user msg', min: 0, max: 4000, step: 100, value: user, format: api.fmtInt, onInput: function (v) { user = v; draw(); } });
            api.slider(c, { label: 'Reserved reply', min: 0, max: 4000, step: 100, value: reply, format: api.fmtInt, onInput: function (v) { reply = v; draw(); } });
            el.appendChild(c); el.appendChild(bar); el.appendChild(info);
            api.html(el, '<div class="inference-note">Window = input + output. Chat APIs resend the whole history each turn (' + api.link('chat-api/cost-caching', 'cost grows per turn') + '); agents manage this with compaction (' + api.link('memory-rag/context-management', 'context management') + ').</div>');
          } },
          { name: 'Lost in middle', render: function (el, api) {
            var len = 8000, depth = 1;
            var holder = api.h('div'), ch = null, c = api.h('div', { class: 'controls' });
            var xs = []; for (var i = 0; i <= 20; i++) xs.push(i / 20);
            function acc(x, L, d) { var u = 1 - Math.pow(2 * (x - 0.5), 2); return 0.95 - d * 0.35 * u * Math.min(1, L / 16000 + 0.15); }
            function draw() {
              var ser = [{ name: 'illustrative accuracy, context ' + api.fmtInt(len) + ' tokens', points: xs.map(function (x) { return [x, acc(x, len, depth)]; }), color: '#4ea8ff' }, { name: 'short context (reference)', points: xs.map(function (x) { return [x, acc(x, 1000, depth)]; }), color: '#5ad18a', dashed: true }];
              if (ch) ch.update({ series: ser }); else ch = api.chart(holder, { series: ser, xLabel: 'where the answer sits in the prompt (0 = start, 1 = end)', yLabel: 'accuracy', xRange: [0, 1], yRange: [0, 1], height: 260 });
            }
            api.slider(c, { label: 'Context length', min: 1000, max: 32000, step: 1000, value: 8000, format: api.fmtInt, onInput: function (v) { len = v; draw(); } });
            api.slider(c, { label: 'Effect strength', min: 0, max: 1.5, step: 0.1, value: 1, format: function (v) { return api.fmt(v, 1); }, onInput: function (v) { depth = v; draw(); } });
            el.appendChild(c); el.appendChild(holder);
            api.callout(el, 'Research (e.g. Liu et al., "Lost in the Middle", 2023) <b>reported</b> a U-shaped curve in multi-document question answering: answers placed at the start or end were found more reliably than in the middle. The effect size depends heavily on the model, task and prompt, and recent long-context models often show much smaller dips.', 'hedge');
            api.illustrative(el, 'The curve is a schematic made-up function, not measured data.');
          } },
          { name: 'Practice', render: function (el, api) {
            api.table(el, { headers: ['Situation', 'What tends to help'], rows: [
              ['Long document + question', 'Put the question and key instructions at the end (and optionally at the start too)'],
              ['Many retrieved chunks', 'Retrieve fewer, rerank, put the best chunks first or last (' + api.link('memory-rag/advanced-rag', 'advanced RAG') + ')'],
              ['Long agent run', 'Summarise old tool results; keep the goal pinned near the top (' + api.link('memory-rag/context-management', 'compaction') + ')'],
              ['Huge window available', 'More tokens cost more KV memory and latency — a long window is not free (' + api.link('inference/kv-cache', 'KV cache') + ')']
            ] });
            api.callout(el, 'These are heuristics, not guarantees. If something matters, test your own model on your own task with the fact placed at different positions.', 'hedge');
          } }
        ]
      },
      // ------------------------------------------------------------------ 4
      {
        id: 'quantization', title: 'Quantization: smaller weights, less memory',
        critical: {
          understand: [
            'Quantization stores weights with fewer bits: BF16 (16 bit) → INT8 (8 bit) → INT4 (4 bit) cuts weight memory by 2× or 4×.',
            'Since decode is memory-bandwidth-bound, fewer bytes per weight usually also means faster decoding.',
            'The price is rounding error; careful schemes (per-group scales, outlier handling) keep quality loss small, but it is not zero.'
          ],
          details: [
            'Symmetric INT8: <code>scale = max|w| / 127</code>, <code>q = round(w / scale)</code>, dequantised weight = <code>q × scale</code>.',
            'One outlier weight inflates the scale and hurts all the others — hence per-group scales (e.g. 1 scale per 64–128 weights).',
            'Quality impact depends on model, method and task; INT8 is often near-lossless, INT4 shows some degradation (hedge: verify on your task).'
          ]
        },
        dataTypes: [
          { name: 'weights w', type: 'vector', shape: '[n]', example: '8 floats' },
          { name: 'scale', type: 'float', shape: 'scalar', example: 'max|w| / 127' },
          { name: 'quantized q', type: 'integer', shape: '[n]', example: 'range −127…127' },
          { name: 'dequantized ŵ', type: 'vector', shape: '[n]', example: 'q × scale' },
          { name: 'parameter count', type: 'integer', shape: 'scalar', example: '8,030,261,248' }
        ],
        levels: [
          { name: 'Memory', render: function (el, api) {
            var P = api.ref.paramBreakdown().total;
            var fmts = [['BF16 / FP16', 16], ['INT8 / FP8', 8], ['INT4', 4]];
            api.bars(el, { title: 'Weight memory for ' + api.fmtBig(P) + ' parameters', items: fmts.map(function (f) { return { label: f[0], value: P * f[1] / 8, note: f[1] + ' bit' }; }), max: P * 2, format: api.fmtBytes });
            api.callout(el, 'A 16 GB model becomes ≈ 8 GB (INT8) or ≈ 4 GB (INT4) — the difference between needing a data-centre GPU and fitting on a laptop or phone-class device. Details on the next level.', 'tip');
            api.html(el, '<div class="inference-note">Related: the KV cache can be quantised too (' + api.link('inference/kv-cache', 'KV cache calculator') + '), and training uses BF16 for different reasons (' + api.link('training/scaling', 'scaling &amp; precision') + ').</div>');
          } },
          { name: 'Calculator', render: function (el, api) {
            var P = api.ref.paramBreakdown().total, bits = 4, group = 64, gpu = 16, ctx = 8192, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              var overhead = group > 0 ? 16 / group : 0;      // one FP16 scale per group
              var eff = bits + (bits < 16 ? overhead : 0), wb = P * eff / 8, kv = api.ref.kvBytesPerToken * ctx, tot = wb + kv;
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">memory = params × (bits + scale overhead) / 8\n       = ' + api.fmtBig(P) + ' × (' + bits + ' + ' + api.fmt(bits < 16 ? overhead : 0, 3) + ') / 8 = <b>' + api.fmtBytes(wb) + '</b>\nplus KV cache for ' + api.fmtInt(ctx) + ' tokens (BF16): ' + api.fmtBytes(kv) + '   →   total ≈ ' + api.fmtBytes(tot) + '</div>');
              api.bars(out, { title: 'Total vs GPU memory (GB)', items: [{ label: 'weights', value: wb / GB, color: 'var(--t-matrix)' }, { label: 'KV cache', value: kv / GB, color: 'var(--t-tensor)' }, { label: 'GPU ' + gpu + ' GB', value: gpu, color: tot / GB <= gpu ? 'var(--good)' : 'var(--bad)' }], max: Math.max(gpu, tot / GB) * 1.05, format: function (v) { return api.fmt(v, 1) + ' GB'; } });
              api.callout(out, tot / GB <= gpu ? 'Fits in ' + gpu + ' GB (before activations/runtime overhead, which also need some memory).' : 'Does <b>not</b> fit in ' + gpu + ' GB.', tot / GB <= gpu ? 'tip' : 'warn');
            }
            api.select(c, { label: 'Format', options: [{ value: '16', label: 'BF16 (16 bit)' }, { value: '8', label: 'INT8 (8 bit)' }, { value: '4', label: 'INT4 (4 bit)' }], value: '4', onChange: function (v) { bits = +v; draw(); } });
            api.select(c, { label: 'Scale group', options: [{ value: '0', label: 'none' }, { value: '32', label: '32' }, { value: '64', label: '64' }, { value: '128', label: '128' }], value: '64', onChange: function (v) { group = +v; draw(); } });
            api.select(c, { label: 'GPU memory', options: ['8', '16', '24', '48', '80'].map(function (g) { return { value: g, label: g + ' GB' }; }), value: '16', onChange: function (v) { gpu = +v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            api.html(el, '<div class="inference-note">Group-wise quantization stores one FP16 scale (16 bits) per group of weights, adding 16/group bits per weight (0.25 bit for group 64).</div>');
          } },
          { name: 'INT8 example', render: function (el, api) {
            var base = [0.12, -0.5, 0.33, 0.9, -0.07, 0.61, -0.88, 0.24], bits = 8, outlier = 0.9, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              var w = base.slice(); w[3] = outlier;
              var qmax = Math.pow(2, bits - 1) - 1, mx = Math.max.apply(null, w.map(Math.abs)), scale = mx / qmax;
              var q = w.map(function (v) { return Math.round(v / scale); }), dq = q.map(function (v) { return v * scale; });
              var err = w.map(function (v, i) { return dq[i] - v; }), mae = api.math.sum(err.map(Math.abs)) / w.length;
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">scale = max|w| / ' + qmax + ' = ' + api.fmt(mx, 3) + ' / ' + qmax + ' = ' + api.fmt(scale, 5) + '\nq = round(w / scale)      ŵ = q × scale      error = ŵ − w</div>');
              api.table(out, { headers: ['i', 'w ' + api.type('float'), 'w / scale', 'q ' + api.type('integer'), 'ŵ = q·scale', 'error'], rows: w.map(function (v, i) { return [i, api.fmt(v, 3), api.fmt(v / scale, 2), q[i], api.fmt(dq[i], 4), api.fmt(err[i], 4)]; }) });
              api.html(out, '<div class="inference-note">Mean absolute error = <b>' + api.fmt(mae, 5) + '</b> (' + api.fmt(100 * mae / (api.math.sum(w.map(Math.abs)) / w.length), 2) + '% of mean |w|). Storage: ' + bits + ' bits per weight + one shared scale instead of 16 bits per weight.</div>');
            }
            api.select(c, { label: 'Bits', options: [{ value: '8', label: 'INT8 (±127)' }, { value: '4', label: 'INT4 (±7)' }], value: '8', onChange: function (v) { bits = +v; draw(); } });
            api.slider(c, { label: 'Largest weight', min: 0.9, max: 6, step: 0.1, value: 0.9, format: function (v) { return api.fmt(v, 1); }, onInput: function (v) { outlier = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            api.callout(el, 'Slide the "largest weight" up: one outlier stretches the scale, so all the small weights collapse onto few integer levels and their relative error grows. That is why practical schemes use per-group scales or special outlier handling.', 'tip');
            api.illustrative(el, 'Eight made-up weights; the table is computed live.');
          } }
        ]
      },
      // ------------------------------------------------------------------ 5
      {
        id: 'batching-speculative', title: 'Batching and speculative decoding',
        critical: {
          understand: [
            '<b>Batching</b>: serve many requests per weight pass. Weights are read once for all sequences, so throughput (tokens/s overall) rises almost linearly until the GPU becomes compute-bound or KV memory runs out.',
            '<b>Continuous batching</b> swaps finished requests out and new ones in at every step, keeping the GPU full.',
            '<b>Speculative decoding</b>: a small draft model guesses k tokens; the big model checks all k in <i>one</i> pass and keeps the correct prefix — the output distribution is unchanged.'
          ],
          details: [
            'Expected tokens per big-model pass = (1 − α^(k+1)) / (1 − α), where α is the per-token acceptance rate (assumed independent).',
            'Batching improves throughput but each user\'s tokens/s may fall as the batch gets heavier; the KV cache bounds the batch size (' + '<a href="#" class="xlink" data-goto="inference/kv-cache">KV cache</a>).',
            'Hardware numbers are illustrative; speedups in real systems depend on draft quality and overheads.'
          ]
        },
        dataTypes: [
          { name: 'batch size B', type: 'integer', shape: 'scalar', example: '1–128' },
          { name: 'throughput', type: 'float', shape: 'scalar', example: 'tokens / s' },
          { name: 'draft tokens', type: 'token', shape: '[k]', example: '[" mat", ".", …]' },
          { name: 'acceptance rate α', type: 'probability', shape: 'scalar', example: '0.8' },
          { name: 'accepted?', type: 'boolean', shape: '[k]', example: '[true, true, false, …]' }
        ],
        levels: [
          { name: 'Batching', render: function (el, api) {
            var P = api.ref.paramBreakdown().total, wBytes = api.ref.paramBreakdown().bytesBF16, hw = HW.a100, ctx = 2000, B = 8;
            var out = api.h('div'), chartBox = api.h('div'), ch = null, c = api.h('div', { class: 'controls' });
            function stepT(b) { return Math.max(wBytes / hw.bw + b * api.ref.kvBytesPerToken * ctx / hw.bw, b * 2 * P / hw.flops); }
            function draw() {
              var t = stepT(B), thr = B / t, per = 1 / t, memKV = B * api.ref.kvBytesPerToken * ctx;
              out.innerHTML = '';
              api.html(out, '<div class="inference-two"><div class="card"><div class="card-title">Total throughput</div><div class="big-num">' + api.fmt(thr, 0) + ' tok/s</div><div class="inference-note">B / step time = ' + B + ' / ' + api.fmt(t * 1000, 1) + ' ms</div></div><div class="card"><div class="card-title">Per user</div><div class="big-num">' + api.fmt(per, 0) + ' tok/s</div><div class="inference-note">KV cache for batch: ' + api.fmtBytes(memKV) + '</div></div></div>');
              var p1 = [], p2 = [], b;
              for (b = 1; b <= 128; b += (b < 16 ? 1 : 8)) { p1.push([b, b / stepT(b)]); }
              var ser = [{ name: 'total tokens/s', points: p1, color: '#4ea8ff' }];
              if (ch) ch.update({ series: ser, markers: [{ x: B, label: 'B = ' + B }] }); else ch = api.chart(chartBox, { series: ser, xLabel: 'batch size', yLabel: 'tokens / s', height: 230, markers: [{ x: B, label: 'B = ' + B }] });
            }
            api.slider(c, { label: 'Batch size', min: 1, max: 128, step: 1, value: 8, onInput: function (v) { B = v; draw(); } });
            api.slider(c, { label: 'Context per user', min: 500, max: 16000, step: 500, value: 2000, format: api.fmtInt, onInput: function (v) { ctx = v; draw(); } });
            el.appendChild(c); el.appendChild(out); el.appendChild(chartBox);
            api.html(el, '<div class="inference-note">Step time = max(memory time, compute time): weights are streamed once per step no matter how many sequences share it; each extra sequence adds its own KV reads. Throughput climbs steeply, then flattens as the KV cache dominates (and it must also fit in GPU memory — see ' + api.link('inference/kv-cache', 'KV cache') + ').</div>');
            api.illustrative(el, 'A100-class round numbers (2 TB/s, 300 TFLOP/s), ideal kernels, no scheduling overhead.');
          } },
          { name: 'Continuous', render: function (el, api) {
            var lens = [4, 9, 3, 6, 5, 2, 7, 4], S = 4;
            var colors = ['#4ea8ff', '#ff9a4d', '#5ad18a', '#b98cff', '#ff5d73', '#2fd0c9', '#e6e65c', '#c0c0c0'];
            function staticSim() { var rows = [], i; for (i = 0; i < lens.length; i += S) { var grp = lens.slice(i, i + S), m = Math.max.apply(null, grp); for (var s = 0; s < S; s++) { rows[s] = rows[s] || []; for (var t = 0; t < m; t++) rows[s].push(grp[s] != null && t < grp[s] ? i + s : -1); } } return rows; }
            function contSim() { var rows = [[], [], [], []], free = [0, 0, 0, 0], nextReq = 0, steps = 0, active = [-1, -1, -1, -1], left = [0, 0, 0, 0];
              while (true) { for (var s = 0; s < S; s++) if (left[s] === 0 && nextReq < lens.length) { active[s] = nextReq; left[s] = lens[nextReq]; nextReq++; }
                var any = false; for (s = 0; s < S; s++) { if (left[s] > 0) { rows[s].push(active[s]); left[s]--; any = true; } else rows[s].push(-1); }
                if (!any) { rows.forEach(function (r) { r.pop(); }); break; } steps++; }
              return rows; }
            function render(title, rows) {
              var T = rows[0].length, useful = 0; rows.forEach(function (r) { r.forEach(function (v) { if (v >= 0) useful++; }); });
              api.html(el, '<b>' + title + '</b> — ' + T + ' steps, GPU slot utilisation <b>' + pct(api, useful / (T * S)) + '</b>');
              var wrap = api.h('div', { class: 'inference-tlwrap' });
              var g = api.h('div', { class: 'inference-tl', style: { gridTemplateColumns: 'auto repeat(' + T + ', minmax(14px, 1fr))' } });
              rows.forEach(function (r, s) { g.appendChild(api.h('span', { class: 'lbl', text: 'slot ' + s })); r.forEach(function (v, t) { g.appendChild(api.h('span', { class: 'c', title: v >= 0 ? 'request ' + v + ', step ' + (t + 1) : 'idle', style: v >= 0 ? { background: colors[v % colors.length] } : null })); }); });
              wrap.appendChild(g); el.appendChild(wrap);
              return T;
            }
            api.html(el, '<div class="inference-note">8 requests needing ' + lens.join(', ') + ' output tokens, 4 GPU slots. Each coloured cell = one decode step of one request; grey = idle slot.</div>');
            var a = render('Static batching (wait for the longest request in each group)', staticSim());
            var b = render('Continuous batching (refill a slot the moment a request ends)', contSim());
            api.callout(el, 'Same work, ' + a + ' vs ' + b + ' steps (' + api.fmt(a / b, 2) + '× faster here). Real engines (vLLM, TGI, TensorRT-LLM, …) also page the KV cache so memory is not wasted on padding.', 'key');
            api.illustrative(el, 'Request lengths are made up.');
          } },
          { name: 'Speculative', render: function (el, api) {
            var M = api.math, alpha = 0.8, k = 4, cost = 0.1, out = api.h('div'), chartBox = api.h('div'), ch = null, c = api.h('div', { class: 'controls' });
            function exp(a, kk) { return a >= 0.9999 ? kk + 1 : (1 - Math.pow(a, kk + 1)) / (1 - a); }
            function speed(a, kk, cc) { return exp(a, kk) / (1 + kk * cc); }
            function draw() {
              var e = exp(alpha, k), sp = speed(alpha, k, cost);
              out.innerHTML = '';
              api.html(out, '<div class="formula inference-f">E[tokens per target pass] = (1 − α^(k+1)) / (1 − α) = (1 − ' + api.fmt(alpha, 2) + '^' + (k + 1) + ') / (1 − ' + api.fmt(alpha, 2) + ') = <b>' + api.fmt(e, 2) + '</b>\ncost per round ≈ 1 target pass + k × c draft passes = 1 + ' + k + ' × ' + api.fmt(cost, 2) + ' = ' + api.fmt(1 + k * cost, 2) + '  (c = draft cost relative to target)\nspeedup ≈ ' + api.fmt(e, 2) + ' / ' + api.fmt(1 + k * cost, 2) + ' = <b>' + api.fmt(sp, 2) + '×</b></div>');
              var pts = []; for (var kk = 1; kk <= 10; kk++) pts.push([kk, speed(alpha, kk, cost)]);
              var ser = [{ name: 'speedup vs k (α = ' + api.fmt(alpha, 2) + ')', points: pts, color: '#5ad18a' }];
              if (ch) ch.update({ series: ser, markers: [{ x: k, label: 'k = ' + k }] }); else ch = api.chart(chartBox, { series: ser, xLabel: 'draft tokens per round k', yLabel: 'speedup ×', height: 220, markers: [{ x: k, label: 'k = ' + k }] });
            }
            api.blocks(el, { blocks: [
              { label: 'Draft model', sub: 'small, fast: proposes k tokens', type: 'function' },
              { label: 'k draft tokens', type: 'token', shape: '[k]' },
              { label: 'Target model', sub: 'verifies all k in one pass', type: 'function', zoom: true },
              { label: 'accepted prefix + 1 token', type: 'token' }
            ], caption: 'Verification is a prefill-like parallel pass, so checking k tokens costs about the same as generating one.' });
            api.slider(c, { label: 'Acceptance α', min: 0.1, max: 0.98, step: 0.02, value: 0.8, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { alpha = v; draw(); } });
            api.slider(c, { label: 'Draft length k', min: 1, max: 10, step: 1, value: 4, onInput: function (v) { k = v; draw(); } });
            api.slider(c, { label: 'Draft cost c', min: 0.02, max: 0.5, step: 0.02, value: 0.1, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { cost = v; draw(); } });
            el.appendChild(c); el.appendChild(out); el.appendChild(chartBox);
            api.callout(el, 'Why the output is unchanged: a rejection-sampling rule accepts draft token x with probability min(1, p_target(x) / p_draft(x)) and resamples from the corrected distribution on a rejection, so the final text follows the target model\'s distribution exactly (in exact arithmetic).', 'info');
            api.illustrative(el, 'α and c are slider inputs, not measurements. A larger k only helps while the draft keeps being right.');
          } },
          { name: 'Verify', render: function (el, api) {
            var draft = [' mat', '.', ' It', ' slept'], fix = ' purred';
            var out = api.h('div'), c = api.h('div', { class: 'controls' }), accept = 2;
            var sim = api.h('div');
            function draw() {
              out.innerHTML = '';
              var d = api.h('div', { class: 'inference-draft' });
              api.ex.tokens.forEach(function (t) { d.appendChild(api.h('span', { class: 'tok', html: api.tokText(t.text) })); });
              draft.forEach(function (t, i) { d.appendChild(api.h('span', { class: 'tok ' + (i < accept ? 'ok' : i === accept ? 'bad' : 'skip'), html: api.tokText(t) })); });
              out.appendChild(d);
              api.html(out, 'Draft proposed <b>' + draft.length + '</b> tokens; the target accepted the first <b>' + accept + '</b>' + (accept < draft.length ? ', rejected ' + api.tok(draft[accept]) + ' and sampled its own replacement ' + api.tok(accept === 2 ? fix : ' (resampled)') + '; later draft tokens are discarded.' : ' and appends a bonus token from the same pass.') + '<br>Tokens gained in this single target pass: <b>' + (accept + 1) + '</b>.');
            }
            api.slider(c, { label: 'Tokens the target accepts', min: 0, max: 4, step: 1, value: 2, onInput: function (v) { accept = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            // Monte Carlo
            var alpha = 0.8, k = 4;
            var c2 = api.h('div', { class: 'controls' });
            api.slider(c2, { label: 'α for simulation', min: 0.3, max: 0.95, step: 0.05, value: 0.8, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { alpha = v; } });
            api.button(c2, 'Simulate 2,000 rounds (k = 4)', function () {
              var r = api.math.rng(99), tot = 0, n = 2000;
              for (var i = 0; i < n; i++) { var a = 0; while (a < k && r() < alpha) a++; tot += a + 1; }
              var th = alpha >= 0.9999 ? k + 1 : (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
              sim.innerHTML = 'Simulated mean tokens per target pass: <b>' + api.fmt(tot / n, 3) + '</b> · formula: <b>' + api.fmt(th, 3) + '</b>';
            }, 'primary');
            el.appendChild(c2); el.appendChild(sim);
            api.html(el, '<div class="inference-note">Each round: draft proposes up to k tokens, the target keeps the longest correct prefix and always contributes one extra token (a correction, or a bonus if all were accepted) — so progress is at least one token per target pass. Related: ' + api.link('output-sampling/autoregressive-loop', 'the autoregressive loop') + '.</div>');
            api.illustrative(el, 'Draft tokens are scripted for the demo; the simulation uses a seeded RNG.');
          } }
        ]
      }
    ]
  });
})();
