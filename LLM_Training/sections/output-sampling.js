(function () {
  'use strict';

  // Illustrative candidate tokens with toy unembedding columns (d_model = 4).
  var H = [1.2, -0.4, 0.9, 0.6];
  var CANDS = [
    { t: ' mat',   w: [ 2.0, -0.5,  1.5,  1.0] },
    { t: ' couch', w: [ 1.2, -0.8,  1.4,  0.9] },
    { t: ' floor', w: [ 1.8,  0.2,  1.0,  0.5] },
    { t: ' rug',   w: [ 1.6, -0.2,  0.6,  0.9] },
    { t: ' bed',   w: [ 0.9, -0.3,  1.2,  0.8] },
    { t: ' chair', w: [ 0.7,  0.1,  0.9,  0.3] },
    { t: ' table', w: [ 0.5,  0.4,  0.8,  0.2] },
    { t: ' roof',  w: [ 0.2,  0.5, -0.1,  0.1] },
    { t: ' moon',  w: [-0.3,  0.6, -0.2, -0.5] },
    { t: '.',      w: [-0.8,  0.1,  0.3, -0.2] },
    { t: ' dog',   w: [-0.6,  0.9,  0.1, -0.4] },
    { t: ' the',   w: [-1.0,  0.2, -0.5,  0.3] }
  ];
  function candLogits(api) { return CANDS.map(function (c) { return api.math.dot(H, c.w); }); }
  function order(arr) { return arr.map(function (v, i) { return i; }).sort(function (a, b) { return arr[b] - arr[a]; }); }

  // temperature -> softmax -> top-k -> top-p. Returns {p (after T), keep[], final[]}
  function sampleDist(M, logits, T, k, topP) {
    var p;
    if (T <= 0.001) { p = logits.map(function () { return 0; }); p[M.argmax(logits)] = 1; }
    else p = M.softmax(logits, T);
    var ord = order(p), keep = logits.map(function () { return false; });
    ord.forEach(function (i, r) { if (r < k) keep[i] = true; });
    var s = M.sum(ord.filter(function (i) { return keep[i]; }).map(function (i) { return p[i]; }));
    var q = p.map(function (v, i) { return keep[i] ? v / s : 0; });
    var cum = 0, keep2 = keep.slice();
    ord.forEach(function (i) {
      if (!keep[i]) return;
      if (cum >= topP - 1e-12) keep2[i] = false;
      cum += q[i];
    });
    var s2 = M.sum(p.map(function (v, i) { return keep2[i] ? v : 0; }));
    var fin = p.map(function (v, i) { return keep2[i] ? v / s2 : 0; });
    return { p: p, keep: keep2, final: fin, order: ord };
  }
  function pct(api, v) { return api.fmt(100 * v, 1) + '%'; }

  Course.register({
    id: 'output-sampling', part: 'I', order: 70, title: 'Output & Sampling',
    slides: [
      // ------------------------------------------------------------------ 1
      {
        id: 'unembedding', title: 'From hidden state to a probability distribution',
        critical: {
          understand: [
            'The last layer\'s vector for the final position (<code>h</code>, 4096 numbers) is multiplied by the <b>unembedding matrix</b> W_U to get one score (<b>logit</b>) for every token in the vocabulary.',
            'Softmax turns the 128,256 logits into probabilities that sum to 1. Only the <i>last</i> position is needed to choose the next token.',
            'A logit is a raw score; a probability is the normalised version. Only differences between logits matter.'
          ],
          details: [
            'Shapes: h [d_model] · W_U [d_model × vocab] → logits [vocab]. In the reference model that is 4096 × 128,256 ≈ 525 M parameters.',
            'Many models <b>tie</b> W_U to the embedding matrix E (W_U = Eᵀ); Llama-3-8B keeps them separate, which is why the reference breakdown counts both.',
            'Before the matmul, a final RMSNorm is applied to h (' + '<a href="#" class="xlink" data-goto="transformer-block/rmsnorm">RMSNorm</a>).'
          ]
        },
        dataTypes: [
          { name: 'h (last position)', type: 'vector', shape: '[d_model]', example: '[4096]' },
          { name: 'W_U', type: 'matrix', shape: '[d_model × vocab]', example: '[4096 × 128256]' },
          { name: 'logits', type: 'vector', shape: '[vocab]', example: '[128256]' },
          { name: 'probabilities', type: 'probability', shape: '[vocab]', example: 'sums to 1' },
          { name: 'next token', type: 'token', shape: 'scalar', example: '" mat" (id 2603)' }
        ],
        levels: [
          { name: 'Pipeline', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'h', label: 'h', sub: 'last position, final layer', type: 'vector', shape: '[4096]', goto: 'transformer-block/residual-stream' },
              { id: 'wu', label: '· W_U', sub: 'unembedding', type: 'matrix', shape: '[4096 × 128256]', zoom: true },
              { id: 'lg', label: 'logits', type: 'vector', shape: '[128256]' },
              { id: 'sm', label: 'softmax', type: 'function' },
              { id: 'p', label: 'probabilities', type: 'probability', shape: '[128256]', zoom: true }
            ], caption: 'Prompt: "' + api.esc(api.ex.prompt) + '" → ' + api.tok(api.ex.nextToken.text, api.ex.nextToken.id) + ' is the most likely next token.' });
            api.callout(el, 'The output is <b>not a word</b> but a probability for every token in the vocabulary. Picking one is a separate step — <i>sampling</i> (next slide), with the autoregressive loop around it (' + api.link('output-sampling/autoregressive-loop', 'loop') + ').', 'key');
          } },
          { name: 'Logits', render: function (el, api) {
            var M = api.math, lg = candLogits(api), p = M.softmax(lg), ord = order(lg);
            api.html(el, '<div class="output-sampling-note">A tiny model: h has 4 numbers (not 4096) and we show 12 candidate columns of W_U (not 128,256). Each logit is a dot product h · column.</div>');
            api.matrix(el, { data: [H], title: 'h', type: 'vector', shape: '[4]', small: true, scale: 'diverging' });
            api.matrix(el, { data: M.transpose(CANDS.map(function (c) { return c.w; })), title: 'W_U (12 candidate columns)', type: 'matrix', shape: '[4 × 12]', colLabels: CANDS.map(function (c) { return api.tokText(c.t); }), small: true, scale: 'diverging' });
            api.table(el, { headers: ['Token', 'logit = h · column', 'probability'], rows: ord.map(function (i) {
              return [api.tok(CANDS[i].t), api.fmt(lg[i], 2), pct(api, p[i])]; }) });
            api.illustrative(el, 'Hand-picked vectors so that ' + api.tok(' mat') + ' wins; real logits are computed over the whole vocabulary.');
          } },
          { name: 'Softmax', render: function (el, api) {
            var M = api.math, lg = candLogits(api), p = M.softmax(lg), mx = M.max(lg);
            var e = lg.map(function (v) { return Math.exp(v - mx); }), s = M.sum(e), ord = order(lg).slice(0, 6);
            api.html(el, '<div class="formula">pᵢ = exp(logitᵢ − max) / Σⱼ exp(logitⱼ − max)     (subtracting the max keeps numbers stable)</div>');
            api.table(el, { headers: ['Token', 'logit', 'logit − max', 'exp(·)', 'p = exp / ' + api.fmt(s, 3)], rows: ord.map(function (i) {
              return [api.tok(CANDS[i].t), api.fmt(lg[i], 2), api.fmt(lg[i] - mx, 2), api.fmt(e[i], 3), api.fmt(p[i], 3)]; }) });
            api.bars(el, { title: 'Probabilities (12 candidates, sum = ' + api.fmt(M.sum(p), 3) + ')', max: 1, items: order(p).map(function (i) { return { label: api.tokText(CANDS[i].t), value: p[i], highlight: i === 0 }; }), format: function (v) { return pct(api, v); } });
            api.html(el, '<div class="output-sampling-note">Adding the same constant to every logit changes nothing; multiplying them (temperature) changes the shape — see ' + api.link('output-sampling/sampling-playground', 'sampling playground') + '. Same softmax as in ' + api.link('attention/softmax-scaling', 'attention') + ', now over the vocabulary.</div>');
          } }
        ]
      },
      // ------------------------------------------------------------------ 2
      {
        id: 'sampling-playground', title: 'Sampling playground: temperature, top-k, top-p',
        critical: {
          understand: [
            '<b>Temperature</b> divides logits before softmax: low = sharper (more deterministic), high = flatter (more random). T → 0 is greedy decoding.',
            '<b>Top-k</b> keeps only the k most likely tokens; <b>top-p</b> (nucleus) keeps the smallest set whose probability reaches p.',
            'After filtering, the remaining probabilities are renormalised and one token is drawn at random.'
          ],
          details: [
            'Order used here: temperature → softmax → top-k → top-p → renormalise → sample. Implementations differ in details (some apply temperature after filtering).',
            'Sampling is the only random part of inference; with the same seed and settings the output is reproducible (in principle — GPU nondeterminism can still intervene).',
            'Typical defaults are hedged guidance: low T for extraction or code, higher T for brainstorming.'
          ]
        },
        dataTypes: [
          { name: 'logits', type: 'vector', shape: '[vocab]', example: '12 candidates shown' },
          { name: 'temperature T', type: 'float', shape: 'scalar', example: '0.8' },
          { name: 'top-k', type: 'integer', shape: 'scalar', example: '5' },
          { name: 'top-p', type: 'float', shape: 'scalar', example: '0.9' },
          { name: 'keep filter', type: 'mask', shape: '[vocab]', example: 'true = candidate stays' },
          { name: 'filtered distribution', type: 'probability', shape: '[vocab]', example: 'renormalised' }
        ],
        levels: [
          { name: 'Playground', render: function (el, api) {
            var M = api.math, lg = candLogits(api), T = 1, K = 12, P = 1;
            var rng = M.rng(7), counts = new Array(CANDS.length).fill(0), total = 0, last = null;
            var bars = api.h('div'), hist = api.h('div', { class: 'output-sampling-hist' }), info = api.h('div', { class: 'output-sampling-note' });
            var c = api.h('div', { class: 'controls' });
            api.slider(c, { label: 'Temperature', min: 0, max: 2, step: 0.05, value: 1, format: function (v) { return v === 0 ? '0 (greedy)' : api.fmt(v, 2); }, onInput: function (v) { T = v; draw(); } });
            api.slider(c, { label: 'top-k', min: 1, max: 12, step: 1, value: 12, onInput: function (v) { K = v; draw(); } });
            api.slider(c, { label: 'top-p', min: 0.05, max: 1, step: 0.05, value: 1, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { P = v; draw(); } });
            var c2 = api.h('div', { class: 'controls' });
            function cur() { return sampleDist(M, lg, T, K, P); }
            function draw() {
              var d = cur(), ord = d.order;
              var barsC = ord.map(function (i) { return { label: api.tokText(CANDS[i].t), value: d.keep[i] ? d.final[i] : d.p[i], dim: !d.keep[i], highlight: last === i, note: d.keep[i] ? '' : 'filtered' }; });
              bars.innerHTML = '';
              api.bars(bars, { title: 'Probability per candidate (dimmed = removed by top-k / top-p)', items: barsC, max: 1, format: function (v) { return pct(api, v); } });
              var nk = d.keep.filter(Boolean).length;
              info.innerHTML = 'Candidates kept: <b>' + nk + '</b> of ' + CANDS.length + (last != null ? ' · last sample: ' + api.tok(CANDS[last].t) : '') + '. Entropy of kept distribution: ' + api.fmt(-M.sum(d.final.map(function (v) { return v > 0 ? v * Math.log2(v) : 0; })), 2) + ' bits.';
              drawHist();
            }
            function drawHist() {
              hist.innerHTML = '';
              if (!total) { hist.innerHTML = '<div class="output-sampling-note">No samples yet. Draw some to compare the empirical histogram with the probabilities above.</div>'; return; }
              api.bars(hist, { title: 'Histogram of ' + api.fmtInt(total) + ' samples (fraction)', max: 1, items: order(counts).map(function (i) { return { label: api.tokText(CANDS[i].t), value: counts[i] / total, color: 'var(--t-token)' }; }), format: function (v) { return pct(api, v); } });
            }
            function draw1(n) {
              var d = cur();
              for (var i = 0; i < n; i++) { last = M.sampleIndex(d.final, rng()); counts[last]++; total++; }
              draw();
            }
            api.button(c2, 'Sample 1', function () { draw1(1); }, 'primary');
            api.button(c2, 'Sample 200', function () { draw1(200); });
            api.button(c2, 'Reset histogram', function () { counts = new Array(CANDS.length).fill(0); total = 0; last = null; draw(); });
            el.appendChild(c); el.appendChild(bars); el.appendChild(info); el.appendChild(c2); el.appendChild(hist);
            draw();
            api.illustrative(el, 'Twelve candidate tokens for "' + api.esc(api.ex.prompt) + '" with logits from the previous slide; sampling uses a seeded RNG, so results are reproducible.');
          } },
          { name: 'Pipeline', render: function (el, api) {
            var M = api.math, lg = candLogits(api), T = 0.7, K = 5, P = 0.9;
            var d = sampleDist(M, lg, T, K, P), p1 = M.softmax(lg), ord = d.order;
            var kk = ord.slice(0, K);
            var s = M.sum(kk.map(function (i) { return d.p[i]; }));
            api.html(el, '<div class="output-sampling-note">Worked pipeline with T = ' + T + ', top-k = ' + K + ', top-p = ' + P + ' (fixed). Showing the 8 most likely tokens.</div>');
            api.stepper(el, { steps: [
              { title: 'Raw logits', render: function (b) { api.bars(b, { items: ord.slice(0, 8).map(function (i) { return { label: api.tokText(CANDS[i].t), value: lg[i] }; }), format: function (v) { return api.fmt(v, 2); } }); } },
              { title: 'Divide by temperature, softmax', render: function (b) {
                api.html(b, '<div class="formula">pᵢ = softmax(logitᵢ / T), T = ' + T + '</div>');
                api.bars(b, { max: 1, items: ord.slice(0, 8).map(function (i) { return { label: api.tokText(CANDS[i].t), value: d.p[i], note: 'T=1: ' + pct(api, p1[i]) }; }), format: function (v) { return pct(api, v); } });
              } },
              { title: 'Top-k filter (keep ' + K + ')', render: function (b) {
                api.bars(b, { max: 1, items: ord.slice(0, 8).map(function (i, r) { return { label: api.tokText(CANDS[i].t), value: d.p[i], dim: r >= K, note: r >= K ? 'removed' : '' }; }), format: function (v) { return pct(api, v); } });
              } },
              { title: 'Renormalise the survivors', render: function (b) {
                api.html(b, '<div class="formula">p′ᵢ = pᵢ / ' + api.fmt(s, 3) + '  (sum of kept)</div>');
                api.bars(b, { max: 1, items: kk.map(function (i) { return { label: api.tokText(CANDS[i].t), value: d.p[i] / s }; }), format: function (v) { return pct(api, v); } });
              } },
              { title: 'Top-p filter (cumulative ≥ ' + P + ')', render: function (b) {
                var cum = 0, items = kk.map(function (i) { var v = d.p[i] / s, before = cum; cum += v; return { label: api.tokText(CANDS[i].t), value: v, dim: before >= P - 1e-12, note: 'cum ' + api.fmt(cum, 2) }; });
                api.bars(b, { max: 1, items: items, format: function (v) { return pct(api, v); } });
              } },
              { title: 'Final distribution → draw one token', render: function (b) {
                api.bars(b, { max: 1, items: ord.filter(function (i) { return d.keep[i]; }).map(function (i) { return { label: api.tokText(CANDS[i].t), value: d.final[i], color: 'var(--good)' }; }), format: function (v) { return pct(api, v); } });
                api.html(b, '<div class="output-sampling-note">A uniform random r ∈ [0,1) is compared against the cumulative sum to pick the token (<code>sampleIndex</code>).</div>');
              } }
            ], interval: 2200 });
          } }
        ]
      },
      // ------------------------------------------------------------------ 3
      {
        id: 'autoregressive-loop', title: 'The autoregressive loop',
        critical: {
          understand: [
            'Generation is a loop: run the model, sample one token, <b>append it to the input</b>, run again.',
            'It stops when the model emits an end-of-sequence token (EOS) or a <code>max_tokens</code> limit is hit.',
            'Every token of a reply costs one full forward pass — this is why long answers are slow, and the same loop underlies agents (' + '<a href="#" class="xlink" data-goto="llm-to-agent/agent-loop">agent loop</a>).'
          ],
          details: [
            'The KV cache (' + '<a href="#" class="xlink" data-goto="inference/kv-cache">KV cache</a>) means each step only processes the new token, not the whole text again.',
            'Stop conditions in APIs: EOS token, <code>max_tokens</code>, or custom stop strings.',
            'The model never "plans" the whole sentence; each token is chosen given everything so far.'
          ]
        },
        dataTypes: [
          { name: 'context tokens', type: 'integer', shape: '[seq_len]', example: '[464, 3797, 3332, 319, 262]' },
          { name: 'next-token distribution', type: 'probability', shape: '[vocab]', example: 'one per step' },
          { name: 'sampled token', type: 'token', shape: 'scalar', example: '" mat"' },
          { name: 'stop?', type: 'boolean', shape: 'scalar', example: 'EOS or max_tokens' }
        ],
        levels: [
          { name: 'Loop', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'ctx', label: 'tokens so far', type: 'integer', shape: '[seq_len]' },
              { id: 'fwd', label: 'Transformer', sub: 'forward pass', type: 'function', goto: 'big-picture/forward-pass' },
              { id: 'dist', label: 'probabilities', type: 'probability', shape: '[vocab]' },
              { id: 'smp', label: 'sample', sub: 'temperature, top-k/p', type: 'function', goto: 'output-sampling/sampling-playground' },
              { id: 'tok', label: 'new token', type: 'token', zoom: true },
              { id: 'stop', label: 'EOS or max?', type: 'boolean' }
            ], caption: 'The new token is appended to "tokens so far" and the loop repeats until the stop check passes.' });
            api.callout(el, 'Connection to agents: a tool call is just tokens produced by this same loop; the harness reads them, runs the tool, appends the result, and the loop continues (' + api.link('llm-to-agent', 'Part II') + ').', 'key');
          } },
          { name: 'Step by step', render: function (el, api) {
            var M = api.math;
            var SCRIPT = [
              { pick: ' mat', alts: [[' mat', 4.4], [' couch', 3.5], [' floor', 3.1], [' rug', 2.9]] },
              { pick: '.', alts: [['.', 4.0], [' and', 2.6], [',', 2.4], [' in', 1.0]] },
              { pick: ' It', alts: [[' It', 3.2], [' The', 3.0], [' Then', 2.1], [' She', 1.2]] },
              { pick: ' purred', alts: [[' purred', 3.4], [' was', 3.0], [' slept', 2.6], [' sat', 2.1]] },
              { pick: '.', alts: [['.', 4.2], [' softly', 2.9], [',', 2.2], [' loudly', 0.8]] },
              { pick: '<EOS>', alts: [['<EOS>', 3.8], [' The', 2.5], ['\n', 2.4], [' It', 1.4]] }
            ];
            var steps = SCRIPT.map(function (s, n) {
              return { title: 'Forward pass #' + (n + 1) + ' → ' + api.tok(s.pick === '<EOS>' ? '<EOS>' : s.pick), render: function (b) {
                var p = M.softmax(s.alts.map(function (a) { return a[1]; }));
                var toks = api.ex.tokens.map(function (t) { return { text: t.text, id: t.id }; });
                SCRIPT.slice(0, n).forEach(function (q) { toks.push({ text: q.pick, color: '#5ad18a' }); });
                api.tokens(b, toks, { ids: false });
                api.bars(b, { max: 1, title: 'Next-token distribution (top 4, greedy pick highlighted; illustrative)', items: s.alts.map(function (a, i) { return { label: api.tokText(a[0]), value: p[i], highlight: i === 0 }; }), format: function (v) { return pct(api, v); } });
                api.html(b, '<div class="output-sampling-note">Context length now ' + (api.ex.tokens.length + n) + ' → after appending ' + api.tok(s.pick) + ': ' + (api.ex.tokens.length + n + 1) + '. ' + (s.pick === '<EOS>' ? '<b>EOS reached — stop.</b>' : 'Not EOS — loop again.') + '</div>');
              } };
            });
            api.stepper(el, { steps: steps, interval: 1500 });
            api.illustrative(el, 'The candidate lists and logits are invented; the second reply sentence is made up for the demo.');
          } },
          { name: 'Stop rules', render: function (el, api) {
            var words = [' mat', '.', ' It', ' purred', ' and', ' slept', ' all', ' day', '.', '<EOS>'];
            var maxT = 6, out = api.h('div', { class: 'output-sampling-gen' }), info = api.h('div', { class: 'output-sampling-note' });
            var c = api.h('div', { class: 'controls' });
            var timer = null;
            api.slider(c, { label: 'max_tokens', min: 1, max: 12, step: 1, value: 6, onInput: function (v) { maxT = v; } });
            function run() {
              if (timer) clearInterval(timer);
              var n = 0; out.innerHTML = ''; info.innerHTML = '';
              api.tokens(out, api.ex.tokens.map(function (t) { return { text: t.text }; }), { ids: false });
              timer = api.every(600, function () {
                if (!el.isConnected) { clearInterval(timer); return; }
                var w = words[n]; n++;
                out.appendChild(api.h('span', { class: 'tok new' + (w === '<EOS>' ? ' eos' : ''), html: api.tokText(w) }));
                if (w === '<EOS>') { clearInterval(timer); info.innerHTML = 'Stopped: <b>EOS</b> after ' + n + ' generated tokens (stop_reason = end_turn / stop).'; }
                else if (n >= maxT) { clearInterval(timer); info.innerHTML = 'Stopped: <b>max_tokens = ' + maxT + '</b> reached — the reply is cut off mid-thought (stop_reason = max_tokens / length).'; }
              });
            }
            api.button(c, '▶ Generate', run, 'primary');
            el.appendChild(c); el.appendChild(out); el.appendChild(info);
            api.html(el, '<div class="output-sampling-note">Each green chip is one forward pass. The scripted reply needs 10 tokens including EOS: set max_tokens below 10 to see truncation. See also ' + api.link('chat-api/messages-api', 'stop_reason in chat APIs') + '.</div>');
            api.illustrative(el, 'Scripted tokens, not a real model.');
          } }
        ]
      },
      // ------------------------------------------------------------------ 4
      {
        id: 'logprobs-constraints', title: 'Logprobs, penalties and constrained decoding',
        critical: {
          understand: [
            '<b>Logprobs</b> are natural logs of token probabilities. Summing them over a reply gives how likely the model found that whole text; APIs can return them.',
            '<b>Penalties</b> (repetition, frequency, presence) lower the logits of tokens already seen, discouraging loops.',
            '<b>Constrained decoding</b> masks logits of tokens that would break a grammar or JSON schema to −∞, so output is guaranteed to parse.'
          ],
          details: [
            'logprob = ln p. p = 0.5 → −0.69; p = 0.01 → −4.6. Perplexity = exp(−mean logprob).',
            'Constrained decoding works by editing logits before softmax; the model itself is unchanged. See ' + '<a href="#" class="xlink" data-goto="tool-calling/valid-json">valid JSON</a> for how tool calls use it.',
            'Penalty formulas vary between providers; the ones shown are common textbook forms.'
          ]
        },
        dataTypes: [
          { name: 'logprob', type: 'float', shape: 'scalar', example: '−0.47' },
          { name: 'token counts', type: 'integer', shape: '[vocab]', example: 'counts in context' },
          { name: 'penalised logits', type: 'vector', shape: '[vocab]', example: 'logit − penalties' },
          { name: 'grammar mask', type: 'mask', shape: '[vocab]', example: 'allowed? true/false' },
          { name: 'JSON schema / grammar', type: 'json', shape: 'object', example: '{"type":"number"}' }
        ],
        levels: [
          { name: 'Logprobs', render: function (el, api) {
            var M = api.math;
            var seq = [[' mat', 0.62], ['.', 0.55], [' It', 0.31], [' purred', 0.12], ['.', 0.71]];
            var tot = 0;
            var rows = seq.map(function (s) { var lp = Math.log(s[1]); tot += lp; return [api.tok(s[0]), api.fmt(s[1], 2), api.fmt(lp, 3), api.fmt(-lp / Math.LN2, 2) + ' bits']; });
            api.table(el, { headers: ['Chosen token', 'probability', 'logprob = ln p', 'surprise'], rows: rows });
            var mean = tot / seq.length;
            api.html(el, '<div class="formula">sequence logprob = Σ ln pᵢ = ' + api.fmt(tot, 3) + '   →   joint probability = exp(' + api.fmt(tot, 3) + ') = ' + api.fmt(Math.exp(tot), 4) + '\nmean logprob = ' + api.fmt(mean, 3) + '   →   perplexity = exp(−mean) = ' + api.fmt(Math.exp(-mean), 2) + '</div>');
            api.callout(el, 'Low-probability tokens (large negative logprob) mark spots where the model was unsure — a rough but imperfect hint for hallucination detection or classification confidence. Logprobs measure <i>token</i> likelihood, not factual truth.', 'hedge');
            api.illustrative(el, 'Probabilities of the chosen tokens are invented. Cross-entropy training loss is the same idea (' + api.link('training/cross-entropy', 'cross-entropy') + ').');
          } },
          { name: 'Penalties', render: function (el, api) {
            var M = api.math;
            var toks = [' the', ' cat', ' sat', ' mat', ' on', ' rug', ' floor', ' dog'];
            var base = [2.6, 2.4, 2.0, 2.2, 1.6, 1.4, 1.2, 0.2];
            var cnt = [3, 2, 2, 1, 2, 0, 0, 0];
            var rep = 1, fq = 0, pr = 0, out = api.h('div');
            var c = api.h('div', { class: 'controls' });
            function draw() {
              var adj = base.map(function (l, i) {
                var v = l;
                if (cnt[i] > 0) v = v > 0 ? v / rep : v * rep;
                return v - fq * cnt[i] - pr * (cnt[i] > 0 ? 1 : 0);
              });
              var p0 = M.softmax(base), p1 = M.softmax(adj);
              out.innerHTML = '';
              var g = api.h('div', { class: 'output-sampling-row2' }); out.appendChild(g);
              var l = api.h('div'), r = api.h('div'); g.appendChild(l); g.appendChild(r);
              api.bars(l, { title: 'Before penalties', max: 0.5, items: toks.map(function (t, i) { return { label: api.tokText(t), value: p0[i], note: 'seen ×' + cnt[i] }; }), format: function (v) { return pct(api, v); } });
              api.bars(r, { title: 'After penalties', max: 0.5, items: toks.map(function (t, i) { return { label: api.tokText(t), value: p1[i], color: cnt[i] ? 'var(--warn)' : 'var(--good)' }; }), format: function (v) { return pct(api, v); } });
              api.html(out, '<div class="formula">logit′ = (logit / rep  if logit>0 else logit × rep) − frequency × count − presence × [count>0]\n e.g. " the": ' + api.fmt(base[0], 2) + ' → ' + api.fmt(adj[0], 2) + '</div>');
            }
            api.slider(c, { label: 'Repetition (×)', min: 1, max: 2, step: 0.05, value: 1, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { rep = v; draw(); } });
            api.slider(c, { label: 'Frequency', min: 0, max: 1.5, step: 0.05, value: 0, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { fq = v; draw(); } });
            api.slider(c, { label: 'Presence', min: 0, max: 1.5, step: 0.05, value: 0, format: function (v) { return api.fmt(v, 2); }, onInput: function (v) { pr = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            api.html(el, '<div class="output-sampling-note">Context so far (illustrative): "the cat sat on the mat . the cat sat on the" — counts: the 3, cat 2, sat 2, on 2, mat 1. <b>Frequency</b> penalty grows with each repeat; <b>presence</b> is a one-time penalty for having appeared at all; <b>repetition</b> is multiplicative.</div>');
            api.illustrative(el, 'Logits and counts are invented. Penalties help against loops but can also suppress legitimately repeated words (names, code).');
          } },
          { name: 'Constrained', render: function (el, api) {
            var M = api.math;
            var toks = ['Sure', '{', '}', ',', '"temp"', '"', '2', '1', '-', ':'];
            var raw = [3.2, 2.0, 0.5, 0.8, 1.8, 1.9, 1.2, 1.0, 0.4, 0.6];
            var states = {
              'after "{"  (expect a key)': { prefix: '{', allow: ['"temp"', '"'] },
              'after  "temp":  (expect a number)': { prefix: '{"temp": ', allow: ['2', '1', '-'] },
              'after  "temp": 2  (digit, comma or close)': { prefix: '{"temp": 2', allow: ['2', '1', ',', '}'] }
            };
            var keys = Object.keys(states), sel = keys[1], on = true, out = api.h('div');
            var c = api.h('div', { class: 'controls' });
            api.select(c, { label: 'Grammar state', options: keys, value: sel, onChange: function (v) { sel = v; draw(); } });
            api.toggle(c, { label: 'Apply grammar mask', value: true, onChange: function (v) { on = v; draw(); } });
            el.appendChild(c); el.appendChild(out);
            function draw() {
              var st = states[sel], mask = toks.map(function (t) { return st.allow.indexOf(t) >= 0; });
              var lg = raw.map(function (v, i) { return on && !mask[i] ? -Infinity : v; });
              var p = M.softmax(lg), ord = order(raw);
              out.innerHTML = '';
              api.html(out, '<div class="output-sampling-state">Generated so far: <code>' + api.esc(st.prefix) + '</code> ▮</div>');
              api.table(out, { headers: ['Token', 'mask ' + api.type('mask'), 'logit', 'probability'], rows: ord.map(function (i) {
                return [api.tok(toks[i]), mask[i] ? '<span class="output-sampling-ok">allowed</span>' : '<span class="output-sampling-no">blocked</span>', api.fmt(lg[i], 1), pct(api, p[i])]; }) });
              api.html(out, '<div class="output-sampling-note">' + (on ? 'Blocked tokens get logit −∞, so softmax assigns them probability exactly 0; the remaining ones are renormalised.' : 'Without the mask the model\'s favourite is "Sure" — which would break the JSON.') + ' A grammar engine tracks the parser state and recomputes the mask at every step.</div>');
            }
            draw();
            api.callout(el, 'This is how structured-output / JSON modes and many tool-calling stacks can guarantee parseable output (' + api.link('tool-calling/valid-json', 'valid JSON') + ', ' + api.link('tool-calling/tool-definition', 'tool definitions') + '). Providers differ in what they support; check current docs.', 'info');
            api.illustrative(el, 'Toy vocabulary and logits.');
          } }
        ]
      }
    ]
  });
})();
