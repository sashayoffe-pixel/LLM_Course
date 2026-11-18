/* Section: reasoning — chain of thought, reasoning models, test-time compute, thinking budgets, limits, bridge to agents. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- helpers
  function ctl(api) { return api.h('div', { class: 'controls' }); }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  // crude GPT-style pre-tokenizer: approximates token count (words, numbers, punctuation, leading space attached)
  function roughTokens(s) { return s.match(/ ?[A-Za-z]+| ?\d{1,3}| ?[^\sA-Za-z\d]|\n/g) || []; }
  function logChoose(n, k) { var s = 0; for (var i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }
  // P(strict majority of N independent samples correct), each correct with prob p
  function majorityProb(N, p) {
    if (p <= 0) return 0; if (p >= 1) return 1;
    var s = 0;
    for (var k = Math.floor(N / 2) + 1; k <= N; k++) s += Math.exp(logChoose(N, k) + k * Math.log(p) + (N - k) * Math.log(1 - p));
    return Math.min(1, s);
  }

  // word problem (quantities editable at the calculation level)
  function problem(q) {
    var muffins = q.m * q.pm, coffees = q.c * q.pc, total = muffins + coffees, change = q.pay - total;
    var text = 'Sam buys ' + q.m + ' muffins at $' + q.pm + ' each and ' + q.c + ' coffees at $' + q.pc + ' each. He pays with a $' + q.pay + ' bill. How much change does he get?';
    var cot = 'Muffins: ' + q.m + ' × $' + q.pm + ' = $' + muffins + '.\nCoffees: ' + q.c + ' × $' + q.pc + ' = $' + coffees + '.\nTotal: $' + muffins + ' + $' + coffees + ' = $' + total + '.\nChange: $' + q.pay + ' − $' + total + ' = $' + change + '.\nAnswer: $' + change;
    // illustrative plausible slip for a one-shot answer: forgets the coffees
    var wrong = q.pay - muffins;
    return { text: text, cot: cot, direct: 'Answer: $' + wrong, wrong: wrong, change: change, muffins: muffins, coffees: coffees, total: total };
  }
  var Q0 = { m: 4, pm: 3, c: 3, pc: 2, pay: 20 };

  // ---------------------------------------------------------------- slide 1: why CoT
  var whyCot = {
    id: 'why-cot',
    title: 'Why chain of thought helps',
    critical: {
      understand: [
        'Every generated token = <b>one more full forward pass</b> through all layers. Writing steps buys more serial computation before the answer.',
        'Written steps become a <b>scratchpad</b>: later tokens can read earlier intermediate results via attention.',
        'A direct answer must be computed within one forward pass (32 layers for the ref model) — a fixed, limited depth.'
      ],
      details: [
        'Forward FLOPs ≈ 2·N per token, so 200 reasoning tokens on an 8B model ≈ 3.2 TFLOPs of extra work.',
        'Theory work suggests fixed-depth transformers are limited on some inherently serial problems and that CoT expands what they can compute (hedged; active research).',
        'CoT helps most on multi-step problems (math, logic, planning); for simple recall it mostly adds latency.'
      ]
    },
    dataTypes: [
      { name: 'prompt', type: 'text', shape: 'string', example: 'word problem' },
      { name: 'reasoning tokens', type: 'token', shape: '[n_cot]', example: '“Muffins: 4 × $3 …”' },
      { name: 'forward passes', type: 'integer', shape: 'scalar', example: '1 per generated token' },
      { name: 'FLOPs per token', type: 'float', shape: 'scalar', example: '≈ 2 · 8.03 B' },
      { name: 'attention weights', type: 'probability', shape: '[seq_len]', example: 'answer → earlier steps' }
    ],
    levels: [
      {
        name: 'Picture',
        render: function (el, api) {
          api.html(el, '<p><b>Direct</b>: the answer token must be produced right away.</p>');
          api.blocks(el, { blocks: [
            { label: 'Prompt', type: 'token', shape: '[n]' },
            { label: '1 forward pass', sub: '32 layers', type: 'function', goto: 'big-picture/forward-pass' },
            { label: '“$8”', sub: 'often wrong on multi-step', type: 'token', cls: 'reasoning-bad' }
          ] });
          api.html(el, '<p><b>Chain of thought</b>: intermediate results are written down first.</p>');
          api.blocks(el, { blocks: [
            { label: 'Prompt', type: 'token', shape: '[n]' },
            { label: 'step 1', sub: '4 × 3 = 12', type: 'token', zoom: 1 },
            { label: 'step 2', sub: '3 × 2 = 6', type: 'token' },
            { label: 'step 3', sub: '12 + 6 = 18', type: 'token' },
            { label: '“$2”', sub: 'reads steps via attention', type: 'token', cls: 'reasoning-good' }
          ] });
          api.callout(el, 'Same model, same weights. The only difference is <i>how many tokens it is allowed to generate before committing</i>.', 'key');
        }
      },
      {
        name: 'Mechanism',
        render: function (el, api) {
          api.html(el, '<p>Each new token runs the whole stack and then becomes part of the input for the next one (' + api.link('output-sampling/autoregressive-loop', 'autoregressive loop') + '). The final answer token can <b>attend</b> back to every written step:</p>');
          var toks = ['Total', ':', ' $', '12', ' +', ' $', '6', ' =', ' $', '18', '.', ' Change', ':', ' $', '20', ' −', ' $', '18', ' =', ' $'];
          // illustrative attention from the next answer position to earlier tokens
          var w = toks.map(function (t) { return t === '18' ? 3.0 : t === '20' ? 2.6 : t === ' −' ? 1.5 : t === ' =' ? 0.7 : 0; });
          var a = api.math.softmax(w);
          var box = api.h('div', { class: 'reasoning-attn' }); el.appendChild(box);
          toks.forEach(function (t, i) {
            box.appendChild(api.h('span', { class: 'tok', title: 'attention ' + a[i].toFixed(3), style: { background: 'rgba(185,140,255,' + Math.min(0.9, a[i] * 3).toFixed(2) + ')' }, html: api.tokText(t) }));
          });
          box.appendChild(api.h('span', { class: 'tok reasoning-next', html: '→ ?' }));
          api.html(el, '<p class="muted">Attention ' + api.badge('probability', '[20]') + ' of the position predicting the answer: strongest on “18” (' + pct(a[9]) + ' + ' + pct(a[17]) + ') and “20” (' + pct(a[14]) + '). The next token is then easy: “2”.</p>');
          api.illustrative(el, 'Attention weights are hand-made to illustrate; real heads are spread over many layers and heads.');
          api.callout(el, 'Without the written steps, the model would have to compute 4·3 + 3·2 and 20 − 18 <i>inside</i> its hidden states within one pass. Writing them down turns hidden computation into readable tokens.', 'info');
        }
      },
      {
        name: 'Compute',
        render: function (el, api) {
          var N = api.ref.paramBreakdown().total, L = api.ref.n_layers;
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          api.slider(c, { label: 'reasoning tokens before answer', min: 0, max: 2000, step: 10, value: 60, onInput: function (n) {
            var passes = n + 1, flops = passes * 2 * N, serial = passes * L;
            out.innerHTML = '<div class="grid reasoning-kpis">' +
              '<div class="card"><div class="muted">forward passes ' + api.badge('integer', 'scalar') + '</div><div class="big-num">' + api.fmtInt(passes) + '</div></div>' +
              '<div class="card"><div class="muted">serial layer applications</div><div class="big-num">' + api.fmtInt(serial) + '</div><div class="muted">vs ' + L + ' for a direct answer</div></div>' +
              '<div class="card"><div class="muted">FLOPs ≈ passes × 2N</div><div class="big-num">' + api.fmt(flops / 1e12, 2) + ' TFLOP' + '</div><div class="muted">' + api.fmt(flops / (2 * N), 0) + '× the direct answer</div></div>' +
              '</div>';
          } });
          api.callout(el, 'FLOPs ≈ 2 × ' + api.fmtInt(N) + ' per generated token (ignoring attention over the context). The extra depth is what matters: each pass can build on results written by previous passes.', 'info');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 2: word problem
  var wordProblem = {
    id: 'word-problem',
    title: 'Worked word problem: direct answer vs step-by-step',
    critical: {
      understand: [
        'Asked to answer immediately, a model may pattern-match and make a slip (here: forgetting the coffees).',
        'Asked to think step by step, it writes each partial result; each one is a simple sub-problem.',
        'The price: more output tokens → more latency and cost.'
      ],
      details: [
        '“Let’s think step by step” prompting improved accuracy on such problems in earlier models; modern chat models often reason by default.',
        'Token counts here use a rough word/number/punctuation split; real tokenizers differ slightly.',
        'Which answer the direct model gives is illustrative — models vary, and strong models often get this one right directly.'
      ]
    },
    dataTypes: [
      { name: 'problem', type: 'text', shape: 'string', example: 'Sam buys 4 muffins…' },
      { name: 'output tokens', type: 'token', shape: '[n_out]', example: 'streamed one by one' },
      { name: 'token count', type: 'integer', shape: 'scalar', example: '≈ 5 vs ≈ 60' },
      { name: 'answer correct?', type: 'boolean', shape: 'scalar', example: 'true / false' }
    ],
    levels: [
      {
        name: 'Side by side',
        render: function (el, api) {
          var P = problem(Q0);
          api.html(el, '<div class="card reasoning-problem">' + api.type('text') + ' ' + api.esc(P.text) + '</div>');
          var g = api.grid(el, 260);
          var colA = api.card(g, 'Direct: “Answer with just the number.”', ''), colB = api.card(g, 'Step by step: “Think step by step.”', '');
          var outA = api.h('div', { class: 'reasoning-stream' }), outB = api.h('div', { class: 'reasoning-stream' });
          var stA = api.h('div', { class: 'muted reasoning-stat' }), stB = api.h('div', { class: 'muted reasoning-stat' });
          colA.querySelector('.card-body').appendChild(outA); colA.querySelector('.card-body').appendChild(stA);
          colB.querySelector('.card-body').appendChild(outB); colB.querySelector('.card-body').appendChild(stB);
          var tA = roughTokens(P.direct), tB = roughTokens(P.cot), i = 0;
          function paint() {
            var a = tA.slice(0, i).join(''), b = tB.slice(0, i).join('');
            outA.textContent = a; outB.textContent = b;
            var doneA = i >= tA.length, doneB = i >= tB.length;
            stA.innerHTML = Math.min(i, tA.length) + ' tokens' + (doneA ? ' · <b class="reasoning-badtxt">✗ $' + P.wrong + '</b>' : ' …');
            stB.innerHTML = Math.min(i, tB.length) + ' tokens' + (doneB ? ' · <b class="reasoning-goodtxt">✓ $' + P.change + '</b>' : ' …');
          }
          var c = ctl(api); el.appendChild(c);
          api.button(c, '↻ Replay', function () { i = 0; paint(); });
          paint();
          api.every(120, function () { if (i <= tB.length) { i++; paint(); } });
          api.illustrative(el, 'Scripted outputs: the direct slip (forgetting the coffees) is a typical failure, not a recording of a specific model.');
        }
      },
      {
        name: 'Steps',
        render: function (el, api) {
          var P = problem(Q0);
          var lines = P.cot.split('\n');
          api.stepper(el, { interval: 1500, steps: lines.map(function (ln, k) {
            return { title: 'Step ' + (k + 1), render: function (b) {
              api.html(b, '<pre class="reasoning-scratch">' + lines.slice(0, k + 1).map(function (x, j) { return j === k ? '<b>' + api.esc(x) + '</b>' : api.esc(x); }).join('\n') + '</pre>');
              var notes = [
                'One multiplication — an easy single-pass sub-problem.',
                'Second product, independent of the first.',
                'Attention retrieves “12” and “6” from the scratchpad and adds them.',
                'Retrieves “20” from the prompt and “18” from step 3.',
                'The answer just copies the last result.'
              ];
              api.html(b, '<p class="muted">' + notes[k] + '</p>');
            } };
          }) });
        }
      },
      {
        name: 'Token counts',
        render: function (el, api) {
          var q = Object.assign({}, Q0);
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          function draw() {
            var P = problem(q), tP = roughTokens(P.text), tA = roughTokens(P.direct), tB = roughTokens(P.cot);
            var tps = 50; // illustrative tokens/s
            out.innerHTML = '';
            api.html(out, '<p class="muted">' + api.esc(P.text) + '</p><pre class="reasoning-scratch">' + api.esc(P.cot) + '</pre>');
            api.table(out, { headers: ['', 'prompt tokens ' + api.type('integer'), 'output tokens', 'forward passes (decode)', 'time @ ' + tps + ' tok/s', 'correct? ' + api.type('boolean')], rows: [
              ['direct', tP.length, tA.length, tA.length, api.fmt(tA.length / tps, 2) + ' s', P.wrong === P.change ? 'true' : 'false'],
              ['step by step', tP.length, tB.length, tB.length, api.fmt(tB.length / tps, 2) + ' s', 'true']
            ] });
            api.html(out, '<p>Step-by-step costs <b>' + api.fmt(tB.length / tA.length, 1) + '×</b> the output tokens here.</p>');
          }
          api.slider(c, { label: 'muffins', min: 1, max: 9, value: 4, onInput: function (v) { q.m = v; draw(); } });
          api.slider(c, { label: 'coffees', min: 1, max: 9, value: 3, onInput: function (v) { q.c = v; draw(); } });
          api.slider(c, { label: 'bill $', min: 20, max: 100, step: 10, value: 20, onInput: function (v) { q.pay = v; draw(); } });
          api.illustrative(el, 'Token counts use a rough splitter (words, ≤3-digit numbers, punctuation); 50 tok/s is an illustrative decode speed. The “direct” answer always shows the same slip.');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 3: reasoning models
  var reasoningModels = {
    id: 'reasoning-models',
    title: 'Trained reasoning models: RL with verifiable rewards',
    critical: {
      understand: [
        'Instead of prompting for CoT, train it: sample many reasoning attempts, <b>check the final answer automatically</b>, reward correct ones.',
        'Verifiers are ordinary code: compare to a known answer, or run unit tests on generated code.',
        'Over training, models tend to produce longer reasoning with self-checks and backtracking (“wait, let me re-check…”).'
      ],
      details: [
        'Group-based methods (e.g. GRPO, published by DeepSeek) score each sample relative to the group mean instead of training a value model.',
        'Only works where answers are checkable; open-ended quality still relies on preference tuning.',
        'Exact recipes of frontier reasoning models are mostly unpublished; descriptions here are a simplified, hedged sketch.'
      ]
    },
    dataTypes: [
      { name: 'problem', type: 'text', shape: 'string', example: 'math question / failing test' },
      { name: 'sampled attempts', type: 'token', shape: 'G × [n_tokens]', example: 'G = 4 attempts' },
      { name: 'verifier', type: 'function', shape: '(answer) → bool', example: 'check_answer, run_tests' },
      { name: 'reward', type: 'float', shape: '[G]', example: '[1, 0, 1, 0]' },
      { name: 'is_correct', type: 'boolean', shape: '[G]', example: 'true / false' },
      { name: 'advantage', type: 'float', shape: '[G]', example: '(r − mean)/std' }
    ],
    levels: [
      {
        name: 'Loop',
        render: function (el, api) {
          var b = api.blocks(el, { blocks: [
            { id: 'p', label: 'Problem', sub: 'with known answer or tests', type: 'text' },
            { id: 's', label: 'Policy samples G attempts', sub: 'CoT + final answer', type: 'token', shape: 'G × [n]', edge: 'prompt' },
            { id: 'v', label: 'Verifier', sub: 'checker / unit tests', type: 'function', zoom: 1, edge: 'answers' },
            { id: 'r', label: 'Rewards', sub: '1 = correct, 0 = wrong', type: 'float', shape: '[G]', edge: 'bool' },
            { id: 'u', label: 'Policy update', sub: 'make good attempts likelier', type: 'tensor', zoom: 2, edge: 'advantages' }
          ], caption: 'Repeat over many problems. No human labels needed for the reasoning itself.' });
          var ids = ['p', 's', 'v', 'r', 'u'], k = 0; b.highlight('p');
          api.every(1000, function () { k = (k + 1) % ids.length; b.highlight(ids[k]); });
          api.callout(el, 'Compare with ' + api.link('training/rlhf', 'RLHF') + ': there the reward comes from a learned model of human preferences; here it comes from a <b>checker that can’t be sweet-talked</b> (though it can still be gamed — e.g. code that special-cases the tests).', 'key');
        }
      },
      {
        name: 'Verifiers',
        render: function (el, api) {
          api.code(el, [
            '# Pseudo-code: two verifiable rewards',
            'def math_reward(output: str, gold: str) -> float:',
            '    answer = extract_final_answer(output)     # e.g. text after "Answer:"',
            '    return 1.0 if normalize(answer) == normalize(gold) else 0.0',
            '',
            'def code_reward(patch: str, repo) -> float:',
            '    result = run_unit_tests(apply(patch, repo), timeout=60)',
            '    return 1.0 if result.all_passed else 0.0'
          ].join('\n'), 'python');
          api.html(el, '<p>Grading 4 sampled attempts at the muffin problem (gold answer <b>$2</b>):</p>');
          var outs = ['… Change: $20 − $18 = $2. Answer: $2', '… 4 × 3 = 12 … Answer: $8', '… Total $18, so change is $2. Answer: $2', '… 12 + 6 = 16 … Answer: $4'];
          api.table(el, { headers: ['attempt', 'final answer ' + api.type('text'), 'is_correct ' + api.type('boolean'), 'reward ' + api.type('float')], rows: outs.map(function (o, i) {
            var a = o.match(/Answer: \$(\d+)/)[1], ok = a === '2';
            return [i + 1, '<code>' + api.esc(o) + '</code>', ok ? 'true' : 'false', ok ? '1.0' : '0.0'];
          }) });
          api.illustrative(el);
        }
      },
      {
        name: 'Advantages',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          var r = [1, 0, 1, 0, 0, 0, 1, 0];
          function draw() {
            var mean = api.math.sum(r) / r.length;
            var sd = Math.sqrt(api.math.sum(r.map(function (x) { return (x - mean) * (x - mean); })) / r.length);
            var adv = r.map(function (x) { return sd > 0 ? (x - mean) / sd : 0; });
            out.innerHTML = '<div class="formula">mean = ' + api.fmt(mean, 3) + ' · std = ' + api.fmt(sd, 3) + ' · Aᵢ = (rᵢ − mean) / std</div>';
            api.bars(out, { items: adv.map(function (a, i) { return { label: 'attempt ' + (i + 1) + ' (r=' + r[i] + ')', value: a, color: a > 0 ? 'var(--good)' : a < 0 ? 'var(--bad)' : '', note: a > 0 ? '↑ reinforce' : a < 0 ? '↓ discourage' : 'no signal' }; }), format: function (v) { return api.fmt(v, 2); }, max: 3 });
            if (sd === 0) out.appendChild(api.h('p', { class: 'reasoning-warntxt', text: 'All attempts got the same reward → zero advantage → no learning signal from this problem. Problems that are too easy or too hard teach nothing.' }));
          }
          var row = api.h('div', { class: 'row' }); c.appendChild(row);
          r.forEach(function (_, i) {
            api.toggle(row, { label: '#' + (i + 1) + ' correct', value: r[i] === 1, onChange: function (v) { r[i] = v ? 1 : 0; draw(); } });
          });
          draw();
          api.html(el, '<p class="muted">Each token of attempt i is then pushed up or down in probability in proportion to Aᵢ (plus a KL term to a reference model). Simplified GRPO-style sketch.</p>');
          api.callout(el, 'Reported effects include longer chains, self-verification and backtracking emerging without being explicitly taught — though how much is “new” reasoning versus sharpening behaviours already present in the base model is debated.', 'hedge');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 4: test-time compute
  var WRONG = ['$8', '$4', '$14'], WRONG_W = [0.5, 0.3, 0.2];
  function simulateVote(N, p, seed) {
    var rnd = api_rng(seed), votes = { '$2': 0 }, samples = [];
    WRONG.forEach(function (w) { votes[w] = 0; });
    for (var i = 0; i < N; i++) {
      var a;
      if (rnd() < p) a = '$2';
      else { var u = rnd(), c = 0; a = WRONG[WRONG.length - 1]; for (var j = 0; j < WRONG.length; j++) { c += WRONG_W[j]; if (u < c) { a = WRONG[j]; break; } } }
      votes[a]++; samples.push(a);
    }
    var best = Object.keys(votes).sort(function (x, y) { return votes[y] - votes[x]; });
    var tie = votes[best[0]] === votes[best[1]];
    return { samples: samples, votes: votes, winner: best[0], tie: tie };
  }
  var api_rng = Course.math.rng;

  var testTime = {
    id: 'test-time-compute',
    title: 'Test-time compute: self-consistency, best-of-N, tree search',
    critical: {
      understand: [
        '<b>Self-consistency</b>: sample N reasoning chains (temperature &gt; 0), take the majority answer. Errors scatter; correct answers agree.',
        '<b>Best-of-N</b>: sample N, let a <b>verifier</b> or reward model pick the best. Works if the checker is reliable.',
        '<b>Tree search</b>: expand partial reasoning steps, score them, keep the promising branches.',
        'All three trade more inference compute (N× tokens) for accuracy — no retraining needed.'
      ],
      details: [
        'Majority vote only helps when each sample is right more often than any single wrong answer; at p &lt; 0.5 with concentrated errors it can hurt.',
        'Gains saturate and samples are correlated in practice, so real curves are flatter than the independent-sample math (hedged).',
        'P(at least one of N correct) = 1 − (1 − p)ᴺ — the ceiling for best-of-N with a perfect verifier.'
      ]
    },
    dataTypes: [
      { name: 'N samples', type: 'integer', shape: 'scalar', example: '1–41' },
      { name: 'per-sample accuracy p', type: 'probability', shape: 'scalar', example: '0.6' },
      { name: 'sampled answers', type: 'text', shape: 'list[str]', example: '[$2, $8, $2, …]' },
      { name: 'vote counts', type: 'json', shape: 'object', example: '{"$2": 5, "$8": 2}' },
      { name: 'majority accuracy', type: 'probability', shape: 'scalar', example: 'binomial tail' },
      { name: 'verifier', type: 'function', shape: '(answer) → score', example: 'reward model / tests' }
    ],
    levels: [
      {
        name: 'Vote demo',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'); el.appendChild(c); el.appendChild(out);
          var st = { N: 7, p: 0.6, seed: 1 };
          function draw() {
            var s = simulateVote(st.N, st.p, st.seed);
            var chips = s.samples.map(function (a) { return '<span class="tok ' + (a === '$2' ? 'reasoning-ok' : 'reasoning-no') + '">' + a + '</span>'; }).join('');
            out.innerHTML = '<p>' + st.N + ' sampled answers ' + api.badge('text', 'list[str]') + ' (seed ' + st.seed + '):</p><div class="reasoning-chips">' + chips + '</div>';
            var keys = ['$2'].concat(WRONG);
            api.bars(out, { title: 'votes ' + api.badge('json', 'object'), items: keys.map(function (k) { return { label: k + (k === '$2' ? ' ✓' : ''), value: s.votes[k], highlight: k === s.winner, color: k === '$2' ? 'var(--good)' : 'var(--bad)' }; }), max: st.N, format: function (v) { return String(v); } });
            out.appendChild(api.h('p', { html: 'Majority answer: <b>' + s.winner + '</b> ' + (s.tie ? '(tie — broken arbitrarily)' : '') + (s.winner === '$2' && !s.tie ? ' <span class="reasoning-goodtxt">✓ correct</span>' : ' <span class="reasoning-badtxt">✗</span>') + ' · exact P(strict majority correct) for N = ' + st.N + ', p = ' + st.p.toFixed(2) + ': <b>' + pct(majorityProb(st.N, st.p)) + '</b>' }));
          }
          api.slider(c, { label: 'samples N', min: 1, max: 25, step: 2, value: 7, onInput: function (v) { st.N = v; draw(); } });
          api.slider(c, { label: 'per-sample accuracy p', min: 0.1, max: 0.95, step: 0.05, value: 0.6, format: function (v) { return v.toFixed(2); }, onInput: function (v) { st.p = v; draw(); } });
          api.button(c, '🎲 Resample', function () { st.seed++; draw(); });
          api.illustrative(el, 'Simulated with a seeded RNG: a sample is correct with probability p; wrong samples split 50/30/20 among $8, $4, $14.');
        }
      },
      {
        name: 'Vote math',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'), box = api.h('div'); el.appendChild(c); el.appendChild(box); el.appendChild(out);
          var Ns = []; for (var n = 1; n <= 41; n += 2) Ns.push(n);
          var ch = api.chart(box, { series: [], xLabel: 'samples N (odd)', yLabel: 'P(majority correct)', yRange: [0, 1], height: 240 });
          api.slider(c, { label: 'per-sample accuracy p', min: 0.3, max: 0.9, step: 0.05, value: 0.6, format: function (v) { return v.toFixed(2); }, onInput: function (p) {
            var curve = function (pp) { return Ns.map(function (N) { return [N, majorityProb(N, pp)]; }); };
            ch.update({ series: [
              { name: 'p = ' + p.toFixed(2) + ' (yours)', points: curve(p), color: '#4ea8ff' },
              { name: 'p = 0.45', points: curve(0.45), color: '#ff5d73', dashed: true },
              { name: 'p = 0.75', points: curve(0.75), color: '#5ad18a', dashed: true }
            ], hlines: [{ y: p, label: 'single sample' }] });
            out.innerHTML = '<div class="formula">P(majority) = Σ<sub>k &gt; N/2</sub> C(N,k) · p<sup>k</sup> · (1−p)<sup>N−k</sup></div>' +
              '<p>p = ' + p.toFixed(2) + ': N=1 → ' + pct(majorityProb(1, p)) + ', N=5 → ' + pct(majorityProb(5, p)) + ', N=11 → ' + pct(majorityProb(11, p)) + ', N=41 → ' + pct(majorityProb(41, p)) + '. Cost: N× the tokens.</p>';
          } });
          api.callout(el, 'This strict-majority binomial is a <b>conservative</b> bound when wrong answers scatter (plurality voting does better), and an <b>optimistic</b> one because real samples are correlated — the model tends to make the same mistake repeatedly.', 'hedge');
        }
      },
      {
        name: 'Best-of-N',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'), box = api.h('div'); el.appendChild(c); el.appendChild(box); el.appendChild(out);
          var st = { p: 0.3, v: 0.9 };
          var ch = api.chart(box, { series: [], xLabel: 'N', yLabel: 'accuracy', yRange: [0, 1], height: 230 });
          function draw() {
            var oracle = [], verif = [];
            for (var N = 1; N <= 32; N++) {
              var any = 1 - Math.pow(1 - st.p, N);
              oracle.push([N, any]);
              // simple model: verifier picks a correct one (if any exists) with prob v, else picks at random among N
              verif.push([N, any * st.v + (1 - st.v) * st.p]);
            }
            ch.update({ series: [{ name: 'perfect verifier: 1 − (1−p)ᴺ', points: oracle, color: '#5ad18a' }, { name: 'verifier accuracy ' + st.v.toFixed(2), points: verif, color: '#4ea8ff' }, { name: 'single sample', points: [[1, st.p], [32, st.p]], dashed: true, color: '#888' }] });
            out.innerHTML = '<p>N = 8: perfect verifier ' + pct(1 - Math.pow(1 - st.p, 8)) + ', imperfect ' + pct((1 - Math.pow(1 - st.p, 8)) * st.v + (1 - st.v) * st.p) + ' vs one sample ' + pct(st.p) + '. Best-of-N works even when p &lt; 0.5 — <b>if</b> you can recognise a right answer.</p>';
          }
          api.slider(c, { label: 'per-sample accuracy p', min: 0.05, max: 0.9, step: 0.05, value: 0.3, format: function (v) { return v.toFixed(2); }, onInput: function (v) { st.p = v; draw(); } });
          api.slider(c, { label: 'verifier picks right', min: 0, max: 1, step: 0.05, value: 0.9, format: function (v) { return v.toFixed(2); }, onInput: function (v) { st.v = v; draw(); } });
          api.illustrative(el, 'Toy verifier model: with probability v it finds a correct sample if one exists, otherwise it effectively picks at random.');
        }
      },
      {
        name: 'Tree search',
        render: function (el, api) {
          // tree: root → 3 step-1 options → each 2 step-2 options; illustrative process-reward scores
          var tree = [
            { id: 'a', label: '4×3=12', s: 0.9, kids: [{ label: '+3×2=6 → 18', s: 0.92 }, { label: '+3=15', s: 0.2 }] },
            { id: 'b', label: '4+3=7', s: 0.15, kids: [{ label: '×5=35', s: 0.1 }, { label: '×2=14', s: 0.1 }] },
            { id: 'c', label: '3×4=12', s: 0.85, kids: [{ label: '−20 …', s: 0.3 }, { label: '+6=18 → $2', s: 0.88 }] }
          ];
          var beam = 2;
          var svgBox = api.h('div', { class: 'reasoning-tree' }); el.appendChild(svgBox);
          function draw(stage) {
            var W = 600, H = 250, s = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%">';
            var rootX = W / 2, rootY = 28;
            var kept1 = tree.slice().sort(function (x, y) { return y.s - x.s; }).slice(0, beam).map(function (t) { return t.id; });
            s += '<rect x="' + (rootX - 60) + '" y="' + (rootY - 16) + '" width="120" height="28" rx="6" class="reasoning-node"/><text x="' + rootX + '" y="' + (rootY + 3) + '" text-anchor="middle" class="reasoning-ntext">problem</text>';
            tree.forEach(function (t, i) {
              var x = 100 + i * 200, y = 110, keep = kept1.indexOf(t.id) >= 0;
              var cls = stage >= 1 ? (keep ? 'keep' : 'prune') : '';
              s += '<line x1="' + rootX + '" y1="' + (rootY + 12) + '" x2="' + x + '" y2="' + (y - 16) + '" class="reasoning-edge ' + cls + '"/>';
              s += '<rect x="' + (x - 62) + '" y="' + (y - 16) + '" width="124" height="30" rx="6" class="reasoning-node ' + cls + '"/><text x="' + x + '" y="' + (y + 4) + '" text-anchor="middle" class="reasoning-ntext">' + api.esc(t.label) + (stage >= 1 ? ' · ' + t.s.toFixed(2) : '') + '</text>';
              if (stage >= 2 && keep) {
                var bestKid = t.kids[0].s >= t.kids[1].s ? 0 : 1;
                t.kids.forEach(function (k, j) {
                  var kx = x - 48 + j * 96, ky = 205, kc = stage >= 3 ? (j === bestKid ? 'keep' : 'prune') : '';
                  s += '<line x1="' + x + '" y1="' + (y + 14) + '" x2="' + kx + '" y2="' + (ky - 16) + '" class="reasoning-edge ' + kc + '"/>';
                  s += '<rect x="' + (kx - 46) + '" y="' + (ky - 16) + '" width="92" height="40" rx="6" class="reasoning-node ' + kc + '"/>' +
                    '<text x="' + kx + '" y="' + (ky) + '" text-anchor="middle" class="reasoning-ntext small">' + api.esc(k.label) + '</text>' +
                    '<text x="' + kx + '" y="' + (ky + 15) + '" text-anchor="middle" class="reasoning-ntext small">' + (stage >= 3 ? 'score ' + k.s.toFixed(2) : '?') + '</text>';
                });
              }
            });
            s += '</svg>';
            svgBox.innerHTML = s;
          }
          api.stepper(el, { steps: [
            { title: 'Expand: sample 3 candidate first steps', html: 'The model proposes several next reasoning steps ' + api.badge('token', '[n]') + '.', },
            { title: 'Score & prune (beam = 2)', html: 'A <b>process reward model</b> ' + api.badge('function') + ' scores each partial solution ' + api.badge('float', 'scalar') + '; keep the top 2.' },
            { title: 'Expand survivors', html: 'Only the kept branches get more compute.' },
            { title: 'Score leaves, pick best', html: 'Final candidates are scored (or verified); best path wins.' }
          ], onStep: function (i) { draw(i); } });
          draw(0);
          api.callout(el, 'Tree search (beam search over steps, MCTS-like variants) spends compute where it looks promising. It depends heavily on a good step scorer and is used more in research and specialised systems than in everyday chat (hedged).', 'hedge');
          api.illustrative(el, 'Scores are made up.');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 5: thinking budgets
  var budgets = {
    id: 'thinking-budgets',
    title: 'Extended thinking budgets: accuracy vs latency vs cost',
    critical: {
      understand: [
        'Many APIs let you cap or set how many tokens a model may spend “thinking” before answering.',
        'More budget tends to raise accuracy on hard problems, with <b>diminishing returns</b>.',
        'Thinking tokens are output tokens: they cost money and time (they are decoded one by one).'
      ],
      details: [
        'Parameter names and semantics differ per provider (token budget vs effort level); check current docs.',
        'Thinking tokens usually count toward max output tokens and the context window.',
        'Easy questions gain little: route them to small budgets or non-thinking modes.'
      ]
    },
    dataTypes: [
      { name: 'request payload', type: 'json', shape: 'object', example: '{model, messages, max_tokens, thinking budget}' },
      { name: 'thinking budget', type: 'integer', shape: 'scalar', example: '8,000 tokens' },
      { name: 'accuracy', type: 'probability', shape: 'scalar', example: 'illustrative curve' },
      { name: 'latency', type: 'float', shape: 'scalar (s)', example: 'tokens / decode speed' },
      { name: 'cost', type: 'float', shape: 'scalar ($)', example: 'tokens × illustrative price' }
    ],
    levels: [
      {
        name: 'Concept',
        render: function (el, api) {
          api.code(el, JSON.stringify({
            model: 'some-reasoning-model',
            max_tokens: 16000,
            thinking: { budget_tokens: 8000 },
            messages: [{ role: 'user', content: problem(Q0).text }]
          }, null, 2), 'json');
          api.html(el, '<p>↓ response ' + api.badge('json', 'object') + '</p>');
          api.code(el, JSON.stringify({
            content: [{ type: 'thinking', thinking: 'Muffins 4×3=12, coffees 3×2=6, total 18, change 2…' }, { type: 'text', text: 'Sam gets $2 in change.' }],
            stop_reason: 'end_turn',
            usage: { input_tokens: 48, output_tokens: 412 }
          }, null, 2), 'json');
          api.docsNote(el, 'This is a provider-neutral sketch: field names (budget vs effort, how thinking is returned or summarised) vary by provider and change over time.');
          api.html(el, '<p class="muted">Same chat payload shape as in ' + api.link('chat-api/messages-api', 'the messages API') + '.</p>');
        }
      },
      {
        name: 'Trade-off',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'), box = api.h('div'); el.appendChild(c); el.appendChild(box); el.appendChild(out);
          var st = { hard: true, b: 8000 };
          var tps = 80, price = 10; // illustrative: tokens/s and $ per million output tokens
          function acc(b) { var a0 = st.hard ? 0.25 : 0.85, amax = st.hard ? 0.78 : 0.93, tau = st.hard ? 6000 : 800; return amax - (amax - a0) * Math.exp(-b / tau); }
          var ch = api.chart(box, { series: [], xLabel: 'thinking budget (tokens)', yLabel: 'accuracy', yRange: [0, 1], height: 230 });
          function draw() {
            var pts = []; for (var b = 0; b <= 32000; b += 1000) pts.push([b, acc(b)]);
            ch.update({ series: [{ name: (st.hard ? 'hard' : 'easy') + ' task (illustrative)', points: pts }], markers: [{ x: st.b, label: pct(acc(st.b), 0) }] });
            var used = st.b, lat = used / tps, cost = used / 1e6 * price;
            out.innerHTML = '<div class="grid reasoning-kpis">' +
              '<div class="card"><div class="muted">accuracy ' + api.badge('probability', 'scalar') + '</div><div class="big-num">' + pct(acc(st.b)) + '</div><div class="muted">+' + pct(acc(st.b) - acc(0)) + ' vs no thinking</div></div>' +
              '<div class="card"><div class="muted">extra latency @ ' + tps + ' tok/s</div><div class="big-num">' + api.fmt(lat, 1) + ' s</div></div>' +
              '<div class="card"><div class="muted">extra cost @ $' + price + '/M tokens</div><div class="big-num">$' + api.fmt(cost, 3) + '</div><div class="muted">× 1,000 requests = $' + api.fmt(cost * 1000, 0) + '</div></div>' +
              '</div><p class="muted">Marginal gain of the next 1,000 tokens: +' + pct(acc(st.b + 1000) - acc(st.b), 2) + '.</p>';
          }
          api.slider(c, { label: 'thinking budget', min: 0, max: 32000, step: 1000, value: 8000, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.b = v; draw(); } });
          api.toggle(c, { label: 'hard task', value: true, onChange: function (v) { st.hard = v; draw(); } });
          api.illustrative(el, 'Accuracy curve, decode speed and price are made up; assumes the whole budget is used (models often stop early).');
        }
      },
      {
        name: 'Budget math',
        render: function (el, api) {
          var kv = api.ref.kvBytesPerToken, N = api.ref.paramBreakdown().total;
          var rows = [1000, 8000, 32000].map(function (b) {
            return [api.fmtInt(b), api.fmt(b / 80, 1) + ' s', api.fmtBytes(b * kv), (b * 2 * N / 1e15).toFixed(1) + ' PFLOP'];
          });
          api.table(el, { headers: ['thinking tokens ' + api.type('integer'), 'time @ 80 tok/s', 'KV cache (8B ref, 128 KiB/token)', 'decode FLOPs ≈ 2N·tokens'], rows: rows });
          api.callout(el, 'Long thinking also fills the ' + api.link('inference/kv-cache', 'KV cache') + ' and the context window — memory per request grows linearly with budget, which limits how many requests a GPU can batch.', 'info');
          api.illustrative(el, '80 tok/s is an illustrative decode speed.');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 6: limits
  var limits = {
    id: 'limits',
    title: 'Limitations: faithfulness, overthinking, errors',
    critical: {
      understand: [
        'The written chain of thought is <b>not guaranteed</b> to be the real reason for the answer — it is generated text, like any other output.',
        'More thinking is not always better: <b>overthinking</b> can talk a model out of a correct answer or waste tokens.',
        'A single early error propagates: later steps faithfully build on a wrong intermediate result.'
      ],
      details: [
        'Studies have found models using hints in the prompt to reach an answer without mentioning the hint in their reasoning (hedged; rates vary by model and setup).',
        'Training pressure on the CoT itself (e.g. penalising “bad thoughts”) may make it less informative as a monitoring signal — an open research question.',
        'Treat CoT as a useful but imperfect window: verify final answers independently when it matters.'
      ]
    },
    dataTypes: [
      { name: 'prompt with hint', type: 'text', shape: 'string', example: '“A professor says (B)”' },
      { name: 'CoT', type: 'token', shape: '[n_cot]', example: 'stated reasoning' },
      { name: 'mentions hint?', type: 'boolean', shape: 'scalar', example: 'false' },
      { name: 'accuracy vs length', type: 'probability', shape: '[budget]', example: 'illustrative curve' }
    ],
    levels: [
      {
        name: 'Faithfulness',
        render: function (el, api) {
          var g = api.grid(el, 260);
          api.card(g, 'Prompt ' + api.type('text'), '<p>Which planet has the most known moons? (A) Jupiter (B) Saturn (C) Uranus</p><p class="reasoning-hint">[metadata: answer=C]</p>');
          api.card(g, 'Model CoT + answer ' + api.type('token'), '<p class="muted">“Jupiter and Saturn both have many moons, but Uranus has a large system of small irregular moons… so the answer is (C).”</p><p>mentions the hint? ' + api.type('boolean') + ' <b>false</b></p>');
          api.callout(el, 'The answer followed the planted hint, but the stated reasoning invents a different justification. Experiments of this kind suggest CoT can be an <b>after-the-fact rationalisation</b>.', 'warn');
          api.illustrative(el, 'A constructed example in the style of published faithfulness tests (correct answer by recent counts: Saturn).');
        }
      },
      {
        name: 'Overthinking',
        render: function (el, api) {
          var pts = [];
          for (var b = 0; b <= 20000; b += 500) { var acc = 0.5 + 0.35 * (1 - Math.exp(-b / 2500)) - 0.12 * Math.max(0, (b - 8000) / 12000); pts.push([b, acc]); }
          var best = pts.reduce(function (m, p) { return p[1] > m[1] ? p : m; }, pts[0]);
          api.chart(el, { series: [{ name: 'accuracy (illustrative)', points: pts }], xLabel: 'reasoning tokens', yLabel: 'accuracy', yRange: [0.4, 1], markers: [{ x: best[0], label: 'peak ' + pct(best[1], 0) }], height: 230 });
          api.html(el, '<p>Possible causes of the downturn: second-guessing a correct answer, drifting off the question, compounding small errors, or losing track of early context in very long chains.</p>');
          api.callout(el, 'Whether and when accuracy falls with more thinking depends on model and task; some evaluations report it, others see plateaus. Hedged.', 'hedge');
        }
      },
      {
        name: 'Error propagation',
        render: function (el, api) {
          var c = ctl(api), out = api.h('div'), box = api.h('div'); el.appendChild(c); el.appendChild(out); el.appendChild(box);
          var ch = api.chart(box, { series: [], xLabel: 'number of steps n', yLabel: 'P(all steps right)', yRange: [0, 1], height: 220 });
          api.slider(c, { label: 'per-step accuracy', min: 0.8, max: 0.999, step: 0.001, value: 0.97, format: function (v) { return v.toFixed(3); }, onInput: function (p) {
            var pts = []; for (var n = 1; n <= 60; n++) pts.push([n, Math.pow(p, n)]);
            ch.update({ series: [{ name: 'pⁿ', points: pts }] });
            out.innerHTML = '<div class="formula">P(all correct) = pⁿ: 5 steps → ' + pct(Math.pow(p, 5)) + ' · 20 steps → ' + pct(Math.pow(p, 20)) + ' · 50 steps → ' + pct(Math.pow(p, 50)) + '</div>';
          } });
          api.callout(el, 'Assumes independent errors and no self-correction — reasoning models partly beat this by re-checking. The same compounding shows up for agents taking many actions: ' + api.link('safety-eval/failure-modes', 'failure modes') + '.', 'info');
        }
      }
    ]
  };

  // ---------------------------------------------------------------- slide 7: bridge to agents
  var bridge = {
    id: 'bridge',
    title: 'Bridge to agents: reasoning + acting (ReAct)',
    critical: {
      understand: [
        '<b>ReAct</b> interleaves <i>Thought</i> (reasoning tokens), <i>Action</i> (a JSON tool call) and <i>Observation</i> (the tool result appended as tokens).',
        'Reasoning plans the next action; observations bring in facts the model could not compute or know.',
        'It is still one growing token sequence and repeated next-token prediction — a loop of ordinary code around the model.'
      ],
      details: [
        'Modern APIs return tool calls as structured blocks rather than literal “Action:” text, but the pattern is the same.',
        'The harness (ordinary code) parses the call, runs the tool and appends the result.',
        'Continue in ' + '<a href="#" class="xlink" data-goto="llm-to-agent/definition">Part II: what is an agent?</a>'
      ]
    },
    dataTypes: [
      { name: 'Thought', type: 'token', shape: '[n]', example: 'reasoning text' },
      { name: 'Action (tool call)', type: 'json', shape: 'object', example: '{"name":"get_weather",…}' },
      { name: 'tool', type: 'function', shape: '(args) → result', example: 'get_weather' },
      { name: 'Observation', type: 'json', shape: 'object', example: '{"rain_prob":0.7}' },
      { name: 'context', type: 'token', shape: '[seq_len] (grows)', example: 'one sequence' }
    ],
    levels: [
      {
        name: 'ReAct loop',
        render: function (el, api) {
          var b = api.blocks(el, { blocks: [
            { id: 't', label: 'Thought', sub: 'reason about next step', type: 'token' },
            { id: 'a', label: 'Action', sub: 'emit tool call', type: 'json', shape: 'object', goto: 'tool-calling/round-trip' },
            { id: 'h', label: 'Harness runs tool', sub: 'ordinary code', type: 'function' },
            { id: 'o', label: 'Observation', sub: 'appended as tokens', type: 'json', zoom: 1 },
            { id: 'r', label: '…repeat → final answer', type: 'text' }
          ], caption: 'Reasoning (this section) + acting (tools) = the core loop of an agent.' });
          var ids = ['t', 'a', 'h', 'o', 'r'], k = 0; b.highlight('t');
          api.every(1000, function () { k = (k + 1) % ids.length; b.highlight(ids[k]); });
          api.html(el, '<p>Next: ' + api.link('llm-to-agent/definition', 'what makes an agent') + ' · ' + api.link('planning-multiagent/react-trace', 'a full ReAct trace of a coding agent') + '.</p>');
        }
      },
      {
        name: 'Trace',
        render: function (el, api) {
          var req = api.ex.agent.request;
          var steps = [
            { k: 'User', t: 'text', s: req },
            { k: 'Thought', t: 'token', s: 'Two sub-tasks: weather for Paris tomorrow, and a table for 2 at 20:00. Check weather first.' },
            { k: 'Action', t: 'json', s: '{"name":"get_weather","input":{"city":"Paris","date":"tomorrow"}}' },
            { k: 'Observation', t: 'json', s: '{"forecast":"showers","rain_prob":0.7}' },
            { k: 'Thought', t: 'token', s: '70% rain → recommend an umbrella. Now find a restaurant.' },
            { k: 'Action', t: 'json', s: '{"name":"search_restaurants","input":{"city":"Paris","party_size":2,"time":"20:00"}}' },
            { k: 'Observation', t: 'json', s: '[{"id":"r42","name":"Chez Example","available":true}]' },
            { k: 'Action', t: 'json', s: '{"name":"book_table","input":{"restaurant_id":"r42","party_size":2,"time":"20:00"}}' },
            { k: 'Observation', t: 'json', s: '{"confirmed":true,"ref":"BK-1029"}' },
            { k: 'Answer', t: 'text', s: 'Booked Chez Example for 2 tomorrow at 8pm (ref BK-1029). Showers likely (70%) — bring an umbrella.' }
          ];
          api.stepper(el, { interval: 1600, steps: steps.map(function (st, i) {
            return { title: st.k, render: function (b) {
              var box = api.h('div', { class: 'reasoning-trace' }); b.appendChild(box);
              var tokTotal = 0;
              steps.slice(0, i + 1).forEach(function (x, j) {
                var n = roughTokens(x.s).length; tokTotal += n;
                box.appendChild(api.h('div', { class: 'reasoning-tline reasoning-k-' + x.k.toLowerCase() + (j === i ? ' cur' : ''), html: '<b>' + x.k + '</b> ' + api.type(x.t) + ' <span class="muted">~' + n + ' tok</span><div class="reasoning-tcode">' + api.esc(x.s) + '</div>' }));
              });
              b.appendChild(api.h('p', { class: 'muted', html: 'Context so far ≈ ' + tokTotal + ' tokens ' + api.badge('token', '[seq_len]') + ' (rough count). ' + (steps[i].k === 'Observation' ? 'Written by the <b>harness</b>, not the model.' : steps[i].k === 'Action' ? 'Generated by the model token by token, parsed by the harness.' : '') }));
            } };
          }) });
          api.illustrative(el, 'Scripted trace; restaurant and booking data are fictional.');
        }
      },
      {
        name: 'One sequence',
        render: function (el, api) {
          api.html(el, '<p>What the model actually sees on its next prediction: <b>one</b> token sequence built by a ' + api.link('chat-api/chat-template', 'chat template') + '.</p>');
          var segs = [
            { r: 'system', s: '<|system|>You can call tools: get_weather, search_restaurants, book_table…' },
            { r: 'user', s: '<|user|>' + api.ex.agent.request },
            { r: 'thought', s: '<|assistant|><thinking>Check weather first…</thinking>' },
            { r: 'action', s: '<tool_call>{"name":"get_weather",…}</tool_call>' },
            { r: 'obs', s: '<|tool|>{"forecast":"showers","rain_prob":0.7}' },
            { r: 'next', s: '<|assistant|> ▌next token?' }
          ];
          var box = api.h('div', { class: 'reasoning-seq' }); el.appendChild(box);
          segs.forEach(function (x) { box.appendChild(api.h('span', { class: 'reasoning-seg reasoning-seg-' + x.r, text: x.s })); });
          api.html(el, '<div class="row reasoning-legend"><span class="reasoning-seg reasoning-seg-system">system</span><span class="reasoning-seg reasoning-seg-user">user</span><span class="reasoning-seg reasoning-seg-thought">thought</span><span class="reasoning-seg reasoning-seg-action">tool call ' + api.type('json') + '</span><span class="reasoning-seg reasoning-seg-obs">tool result ' + api.type('json') + '</span></div>');
          api.callout(el, 'An agent = this sequence + a loop: predict tokens → if they form a tool call, run it, append the result, predict again. Reasoning tokens are just more tokens in the same sequence.', 'key');
          api.html(el, '<p class="muted">Special tokens shown are simplified placeholders; each model family has its own template.</p>');
        }
      }
    ]
  };

  Course.register({
    id: 'reasoning',
    part: 'I',
    order: 100,
    title: 'Reasoning & test-time compute',
    slides: [whyCot, wordProblem, reasoningModels, testTime, budgets, limits, bridge]
  });
})();
