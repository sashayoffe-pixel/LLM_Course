(function () {
  'use strict';

  var MSGS = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hi! What is the capital of France?' },
    { role: 'assistant', content: 'The capital of France is Paris.' },
    { role: 'user', content: 'And its population?' }
  ];

  function roleChip(r) { return '<span class="chat-api-role ' + r + '">' + r + '</span>'; }

  // Two illustrative template styles (real models ship their own template with the tokenizer).
  var STYLES = {
    chatml: {
      label: 'ChatML-style',
      start: function (r) { return { text: '<|im_start|>' + r + '\n', sp: ['<|im_start|>'] }; },
      end: '<|im_end|>\n',
      gen: '<|im_start|>assistant\n',
      endTok: '<|im_end|>'
    },
    llama: {
      label: 'Llama-3-style',
      start: function (r) { return { text: '<|start_header_id|>' + r + '<|end_header_id|>\n\n', sp: ['<|start_header_id|>', '<|end_header_id|>'] }; },
      end: '<|eot_id|>',
      gen: '<|start_header_id|>assistant<|end_header_id|>\n\n',
      endTok: '<|eot_id|>'
    }
  };

  function applyTemplate(msgs, styleKey, genPrompt) {
    var s = STYLES[styleKey], out = '';
    msgs.forEach(function (m) { out += s.start(m.role).text + m.content + s.end; });
    if (genPrompt) out += s.gen;
    return out;
  }

  // Illustrative tokenizer: specials get fixed high IDs, other pieces get a hash-based ID.
  var SPECIAL_RE = /(<\|[a-z_]+\|>)/;
  function hashId(s) { var h = 7; for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 50000; return h + 100; }
  var spIds = {};
  function specialId(s) { if (!spIds[s]) spIds[s] = 128000 + Object.keys(spIds).length; return spIds[s]; }
  function toyTokenize(text) {
    var out = [];
    text.split(SPECIAL_RE).forEach(function (part) {
      if (!part) return;
      if (SPECIAL_RE.test(part)) { out.push({ text: part, id: specialId(part), sp: true }); return; }
      var re = /\n+| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]| /g, m;
      while ((m = re.exec(part))) {
        var piece = m[0];
        if (piece.length > 7 && /[A-Za-z]/.test(piece)) { out.push({ text: piece.slice(0, 6), id: hashId(piece.slice(0, 6)) }); out.push({ text: piece.slice(6), id: hashId(piece.slice(6)) }); }
        else out.push({ text: piece, id: hashId(piece) });
      }
    });
    return out;
  }
  function tplHtml(api, text) {
    return api.esc(text).replace(/&lt;\|[a-z_]+\|&gt;/g, function (m) { return '<span class="chat-api-sp">' + m + '</span>'; });
  }

  // ---- cost model
  function costModel(turns, perTurn, sysTok, cachedMult, writeMult, priceIn, priceOut, outPerTurn) {
    // turn t input = sysTok + t * perTurn (history incl. previous answers) ; output = outPerTurn
    var rows = [], cumNo = 0, cumC = 0, cumTok = 0, cumTokCached = 0;
    for (var t = 1; t <= turns; t++) {
      var inTok = sysTok + (t - 1) * (perTurn + outPerTurn) + perTurn;
      var prefix = t === 1 ? 0 : sysTok + (t - 1) * (perTurn + outPerTurn);
      var fresh = inTok - prefix;
      var costNo = inTok * priceIn / 1e6 + outPerTurn * priceOut / 1e6;
      var costC = (prefix * cachedMult + fresh * (t === 1 ? writeMult : writeMult)) * priceIn / 1e6 + outPerTurn * priceOut / 1e6;
      cumNo += costNo; cumC += costC; cumTok += inTok;
      rows.push({ t: t, inTok: inTok, prefix: prefix, fresh: fresh, cumTok: cumTok, cumNo: cumNo, cumC: cumC, costNo: costNo, costC: costC });
    }
    return rows;
  }

  Course.register({
    id: 'chat-api',
    part: 'I',
    order: 95,
    title: 'Chat APIs & Cost',
    slides: [
      // ---------------------------------------------------------------- messages-api
      {
        id: 'messages-api',
        title: 'The messages API: stateless request → response',
        critical: {
          understand: [
            'A chat API call is <b>stateless</b>: the server remembers nothing, so the client resends the <b>whole history</b> every turn.',
            'You send <code>{model, system, messages, max_tokens, temperature, …}</code>; you get back <code>{content, stop_reason, usage}</code>.',
            '<code>stop_reason</code> tells you <i>why</i> generation ended — in agents, <code>tool_use</code> means "run a tool and call me again".'
          ],
          details: [
            'Field names differ between providers (e.g. where the system prompt lives, how tool calls are encoded). Treat the JSON here as provider-neutral and check current docs.',
            '<code>usage</code> counts tokens, which is what you are billed for — see ' + '<span data-goto="chat-api/cost-caching" class="xlink">cost &amp; caching</span>.'
          ]
        },
        dataTypes: [
          { name: 'request', type: 'json', shape: 'object', example: '{model, system, messages, …}' },
          { name: 'messages', type: 'json', shape: 'list[{role, content}]', example: '4 items' },
          { name: 'max_tokens', type: 'integer', shape: 'scalar', example: '256' },
          { name: 'temperature', type: 'float', shape: 'scalar', example: '0.7' },
          { name: 'stop_reason', type: 'text', shape: 'string', example: '"end_turn"' },
          { name: 'usage', type: 'json', shape: '{input_tokens, output_tokens}', example: '{57, 12}' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'client', label: 'Your code', sub: 'keeps the history', type: 'function' },
              { id: 'req', label: 'Request', sub: 'system + ALL messages + params', type: 'json', shape: 'object', edge: 'HTTP POST', zoom: true },
              { id: 'model', label: 'Model server', sub: 'stateless: template → tokens → generate', type: 'function', edge: 'tokens in' },
              { id: 'resp', label: 'Response', sub: 'content + stop_reason + usage', type: 'json', shape: 'object', edge: 'tokens out', zoom: true }
            ], caption: 'Each call is independent. The "conversation" exists only in the list your code keeps.' });
            api.callout(el, 'Foreshadowing: an <b>agent</b> is this call in a loop. The harness appends the model\'s answer (and tool results) to <code>messages</code> and calls again — see ' + api.link('llm-to-agent/agent-loop', 'the agent loop') + '.', 'key');
          } },
          { name: 'Request & response', render: function (el, api) {
            var g = api.h('div', { class: 'chat-api-split' }); el.appendChild(g);
            var a = api.h('div'), b = api.h('div'); g.appendChild(a); g.appendChild(b);
            a.appendChild(api.h('div', { class: 'chat-api-h', text: 'Request' }));
            var state = { max: 256, temp: 0.7, stop: 'end_turn' };
            var reqBox = api.h('div'), respBox = api.h('div');
            a.appendChild(reqBox);
            b.appendChild(api.h('div', { class: 'chat-api-h', text: 'Response' })); b.appendChild(respBox);
            var tools = api.h('div', { class: 'controls' }); el.appendChild(tools);
            api.slider(tools, { init: false, label: 'max_tokens', min: 8, max: 1024, step: 8, value: 256, format: function (v) { return String(v); }, onInput: function (v) { state.max = v; draw(); } });
            api.slider(tools, { init: false, label: 'temperature', min: 0, max: 1.5, step: 0.1, value: 0.7, format: function (v) { return v.toFixed(1); }, onInput: function (v) { state.temp = v; draw(); } });
            api.select(tools, { init: false, label: 'stop_reason in response', options: ['end_turn', 'max_tokens', 'stop_sequence', 'tool_use'], value: 'end_turn', onChange: function (v) { state.stop = v; draw(); } });
            var meaning = api.h('div', { class: 'muted' }); el.appendChild(meaning);
            api.docsNote(el, 'Parameter names and response shapes vary by provider (e.g. <code>max_tokens</code> vs <code>max_completion_tokens</code>; some put the system prompt inside <code>messages</code>).');
            var MEAN = {
              end_turn: 'The model emitted its end-of-turn special token: a natural finish.',
              max_tokens: 'Hit your <code>max_tokens</code> cap mid-answer; the text may be cut off.',
              stop_sequence: 'Output contained one of your <code>stop_sequences</code>; generation halted there.',
              tool_use: 'The model produced a tool call instead of (or besides) text. Your harness must run it and call again.'
            };
            function draw() {
              reqBox.innerHTML = ''; respBox.innerHTML = '';
              var req = {
                model: 'some-model-name', system: 'You are a helpful assistant.',
                messages: [{ role: 'user', content: 'What is the capital of France?' }],
                max_tokens: state.max, temperature: state.temp, stop_sequences: ['###'],
                tools: [{ name: 'get_weather', description: '…', input_schema: { type: 'object' } }]
              };
              api.code(reqBox, JSON.stringify(req, null, 2), 'json');
              var content = state.stop === 'tool_use'
                ? [{ type: 'tool_use', id: 'call_1', name: 'get_weather', input: { city: 'Paris' } }]
                : [{ type: 'text', text: state.stop === 'max_tokens' ? 'The capital of France is Pa' : 'The capital of France is Paris.' }];
              var outTok = state.stop === 'max_tokens' ? Math.min(state.max, 8) : 9;
              var inTok = 31;
              var resp = { content: content, stop_reason: state.stop, usage: { input_tokens: inTok, output_tokens: outTok } };
              api.code(respBox, JSON.stringify(resp, null, 2), 'json');
              meaning.innerHTML = '<b>' + state.stop + '</b>: ' + MEAN[state.stop] + ' Token counts shown are illustrative.';
            }
            draw();
          } },
          { name: 'Resending history', render: function (el, api) {
            var turns = [
              { u: 'What is the capital of France?', a: 'Paris.' },
              { u: 'And its population?', a: 'About 2.1 million in the city proper.' },
              { u: 'Thanks! Any must-see places?', a: 'The Louvre, Notre-Dame, and the Eiffel Tower.' }
            ];
            var box = api.h('div');
            api.stepper(el, { steps: turns.map(function (t, i) { return { title: 'Turn ' + (i + 1) }; }), interval: 2200, onStep: function (i) {
              box.innerHTML = '';
              var total = 0;
              for (var k = 0; k <= i; k++) {
                var d = api.h('div', { class: 'chat-api-turn' + (k === i ? ' chat-api-new' : '') });
                d.appendChild(api.h('div', { class: 'muted', html: 'Request ' + (k + 1) + ' sends:' }));
                var n = 0;
                for (var j = 0; j <= k; j++) {
                  d.appendChild(api.h('div', { class: 'chat-api-msg', html: roleChip('user') + '<span>' + api.esc(turns[j].u) + '</span>' }));
                  n += turns[j].u.split(/\s+/).length;
                  if (j < k) { d.appendChild(api.h('div', { class: 'chat-api-msg', html: roleChip('assistant') + '<span>' + api.esc(turns[j].a) + '</span>' })); n += turns[j].a.split(/\s+/).length; }
                }
                total += n;
                d.appendChild(api.h('div', { class: 'muted', html: '≈ ' + n + ' words sent this call; ' + total + ' words sent in total so far.' }));
                box.appendChild(d);
              }
            } });
            el.appendChild(box);
            api.callout(el, 'The assistant\'s earlier replies are in the request only because <b>your code</b> stored and re-sent them. Edit or drop them and the model will not know the difference.', 'info');
          } }
        ]
      },

      // ---------------------------------------------------------------- chat-template
      {
        id: 'chat-template',
        title: 'Chat template: messages JSON → ONE token sequence',
        critical: {
          understand: [
            'The model never sees JSON roles. A <b>chat template</b> flattens <code>system</code>, <code>messages</code> (and tools) into <b>one string</b> with special role-marker tokens.',
            'That string is tokenized into <b>one sequence of IDs</b>; the model just predicts the next token after the "assistant" marker.',
            'Tool definitions and tool results get serialized into the same sequence — this is the root of the agent loop.'
          ],
          details: [
            'Each model family has its own template (shipped with its tokenizer); using the wrong one degrades quality.',
            'Role markers are <i>special tokens</i>, see ' + '<span data-goto="tokenization/special-tokens" class="xlink">special tokens</span>.',
            'Why this matters for agents: ' + '<span data-goto="llm-to-agent/loop-internals" class="xlink">loop internals</span>.'
          ]
        },
        dataTypes: [
          { name: 'messages', type: 'json', shape: 'list[{role, content}]', example: '4 items' },
          { name: 'templated prompt', type: 'text', shape: 'string', example: '"<|im_start|>system\\n…"' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[128000, 9125, …]' },
          { name: 'role markers', type: 'token', shape: 'special tokens', example: '<|im_start|>' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'msgs', label: 'messages', sub: 'role + content', type: 'json', shape: 'list[{role, content}]' },
              { id: 'tpl', label: 'Chat template', sub: 'adds role markers', type: 'function', edge: 'render', zoom: true },
              { id: 'str', label: 'Prompt string', type: 'text', shape: 'string', edge: 'string' },
              { id: 'tok', label: 'Tokenizer', type: 'function', edge: 'encode', zoom: 2 },
              { id: 'ids', label: 'Token IDs', type: 'integer', shape: '[seq_len]', edge: 'ids' },
              { id: 'llm', label: 'Model', sub: 'next-token prediction', type: 'function', goto: 'big-picture/next-token' }
            ], caption: 'JSON is for your code; the model only ever sees one flat sequence of token IDs.' });
            api.callout(el, 'THE key connection: agent = loop around next-token prediction. Tools and results ride in this same sequence — see ' + api.link('llm-to-agent/loop-internals', 'loop internals') + ' and ' + api.link('tokenization/special-tokens', 'special tokens') + '.', 'key');
          } },
          { name: 'Template', render: function (el, api) {
            var style = 'chatml', gen = true, msgs = MSGS.slice();
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.select(c, { init: false, label: 'template style', options: [{ value: 'chatml', label: STYLES.chatml.label }, { value: 'llama', label: STYLES.llama.label }], value: 'chatml', onChange: function (v) { style = v; draw(); } });
            api.toggle(c, { init: false, label: 'add generation prompt', value: true, onChange: function (v) { gen = v; draw(); } });
            api.slider(c, { init: false, label: 'messages', min: 1, max: MSGS.length, step: 1, value: MSGS.length, onInput: function (v) { msgs = MSGS.slice(0, v); draw(); } });
            var g = api.h('div', { class: 'chat-api-split' }); el.appendChild(g);
            var l = api.h('div'), r = api.h('div'); g.appendChild(l); g.appendChild(r);
            l.appendChild(api.h('div', { class: 'chat-api-h', html: 'messages ' + api.badge('json', 'list[{role, content}]') }));
            r.appendChild(api.h('div', { class: 'chat-api-h', html: 'templated string ' + api.badge('text', 'string') }));
            var lb = api.h('div'), rb = api.h('div', { class: 'chat-api-tpl' }); l.appendChild(lb); r.appendChild(rb);
            var info = api.h('div', { class: 'muted' }); el.appendChild(info);
            api.illustrative(el, 'These two templates are simplified look-alikes; real models ship exact templates (including whitespace details) with their tokenizer.');
            function draw() {
              lb.innerHTML = '';
              api.code(lb, JSON.stringify(msgs, null, 2), 'json');
              var s = applyTemplate(msgs, style, gen);
              rb.innerHTML = tplHtml(api, s);
              info.innerHTML = 'The model continues after the final marker. It stops when it emits <span class="chat-api-sp">' + api.esc(STYLES[style].endTok) + '</span>, the end-of-turn token (this is what becomes <code>stop_reason: end_turn</code>).';
            }
            draw();
          } },
          { name: 'Token IDs', render: function (el, api) {
            var s = applyTemplate(MSGS, 'chatml', true);
            var toks = toyTokenize(s);
            var nSp = toks.filter(function (t) { return t.sp; }).length;
            api.html(el, '<div class="chat-api-h">' + api.badge('text', 'string') + ' → ' + api.badge('integer', '[' + toks.length + ']') + '</div>');
            var box = api.h('div', { class: 'chat-api-ids' }); el.appendChild(box);
            toks.forEach(function (t) {
              var vis = t.text.replace(/\n/g, '⏎').replace(/ /g, '␣');
              box.appendChild(api.h('span', { class: 'chat-api-id' + (t.sp ? ' sp' : ''), html: api.esc(vis) + ' <b>' + t.id + '</b>' }));
            });
            api.html(el, '<p>Sequence length: <b>' + toks.length + '</b> tokens, of which <b>' + nSp + '</b> are special role/boundary tokens (orange). This is what <code>usage.input_tokens</code> counts (plus any tool definitions).</p>');
            api.illustrative(el, 'This is a toy tokenizer with hash-based IDs, not real BPE (see ' + api.link('tokenization/text-to-ids', 'text → IDs') + '). Real counts differ.');
          } }
        ]
      },

      // ---------------------------------------------------------------- cost-caching
      {
        id: 'cost-caching',
        title: 'Cost grows every turn — and prompt caching',
        critical: {
          understand: [
            'Because the history is resent, input tokens per turn grow linearly, so the <b>cumulative</b> input tokens over a conversation grow roughly <b>quadratically</b>.',
            '<b>Prompt caching</b> lets the provider reuse the already-processed prefix (system prompt, tools, history) at a discounted rate.',
            'Agents are hit hardest: many turns, large tool results, long prefixes.'
          ],
          details: [
            'Caches generally match on an exact identical <i>prefix</i>; changing something early invalidates everything after it. Keep stable content first.',
            'Discounts, write surcharges, and cache lifetimes differ per provider — the prices below are illustrative.',
            'Related: the server-side ' + '<span data-goto="inference/kv-cache" class="xlink">KV cache</span> is what makes reusing a prefix cheap.'
          ]
        },
        dataTypes: [
          { name: 'turns', type: 'integer', shape: 'scalar', example: '10' },
          { name: 'tokens per turn', type: 'integer', shape: 'scalar', example: '300' },
          { name: 'input tokens at turn t', type: 'integer', shape: 'scalar', example: 'system + t × (user+assistant)' },
          { name: 'price', type: 'float', shape: 'scalar', example: '$/M tokens (illustrative)' },
          { name: 'cached-read multiplier', type: 'float', shape: 'scalar', example: '0.1' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var turns = 6, per = 200, sys = 500;
            var rows = costModel(turns, per, sys, 1, 1, 1, 1, per);
            var items = rows.map(function (r) { return { label: 'turn ' + r.t, value: r.inTok, note: api.fmtInt(r.inTok) + ' input tokens' }; });
            api.html(el, '<p>Each turn re-sends everything so far. With a ' + sys + '-token system prompt and ' + per + ' tokens per user message and per answer:</p>');
            api.bars(el, { items: items, format: function (v) { return api.fmtInt(v); }, title: 'input tokens per turn ' });
            api.html(el, '<p>Total input over ' + turns + ' turns: <b>' + api.fmtInt(rows[rows.length - 1].cumTok) + '</b> tokens — versus only ' + api.fmtInt(turns * per * 2) + ' tokens of "new" text. Most of the bill is re-reading old context.</p>');
            api.illustrative(el, 'Token counts are made up for the example.');
          } },
          { name: 'Calculator', render: function (el, api) {
            var st = { turns: 20, per: 300, sys: 1500, out: 300, mult: 0.1, write: 1.25, pin: 3, pout: 15, cache: true };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { init: false, label: 'turns', min: 1, max: 60, step: 1, value: 20, onInput: function (v) { st.turns = v; draw(); } });
            api.slider(c, { init: false, label: 'user tokens/turn', min: 20, max: 2000, step: 20, value: 300, onInput: function (v) { st.per = v; draw(); } });
            api.slider(c, { init: false, label: 'answer tokens/turn', min: 20, max: 2000, step: 20, value: 300, onInput: function (v) { st.out = v; draw(); } });
            api.slider(c, { init: false, label: 'cached read × price', min: 0.05, max: 1, step: 0.05, value: 0.1, format: function (v) { return v.toFixed(2); }, onInput: function (v) { st.mult = v; draw(); } });
            var chartBox = api.h('div'), out = api.h('div', { class: 'row' }); el.appendChild(chartBox); el.appendChild(out);
            var ch = null;
            function draw() {
              var rows = costModel(st.turns, st.per, st.sys, st.mult, st.write, st.pin, st.pout, st.out);
              var pts1 = rows.map(function (r) { return [r.t, r.cumTok]; });
              var lin = rows.map(function (r) { return [r.t, r.t * (st.per + st.out)]; });
              var spec = { series: [{ name: 'cumulative input tokens (resend)', points: pts1, color: '#ff7a7a' }, { name: 'new text only (linear)', points: lin, color: '#7ab8ff', dashed: true }], xLabel: 'turn', yLabel: 'tokens', height: 240 };
              if (ch) ch.update(spec); else { ch = api.chart(chartBox, spec); }
              var last = rows[rows.length - 1];
              out.innerHTML = '';
              api.html(out, '<div class="card"><div class="muted">cumulative input tokens</div><div class="big-num">' + api.fmtInt(last.cumTok) + '</div></div>' +
                '<div class="card"><div class="muted">cost without caching</div><div class="big-num">$' + api.fmt(last.cumNo, 2) + '</div></div>' +
                '<div class="card"><div class="muted">cost with caching</div><div class="big-num">$' + api.fmt(last.cumC, 2) + '</div></div>' +
                '<div class="card"><div class="muted">saving</div><div class="big-num">' + api.fmt(100 * (1 - last.cumC / last.cumNo), 0) + '%</div></div>');
            }
            draw();
            api.illustrative(el, 'Illustrative prices: $' + st.pin + ' per M input tokens, $' + st.pout + ' per M output tokens, cache writes at ' + st.write + '× and reads at the slider multiplier. Real prices and rules differ; check your provider.');
          } },
          { name: 'Per-turn numbers', render: function (el, api) {
            var st = { turns: 8, per: 300, sys: 1500, out: 300, mult: 0.1, write: 1.25, pin: 3, pout: 15 };
            var rows = costModel(st.turns, st.per, st.sys, st.mult, st.write, st.pin, st.pout, st.out);
            api.table(el, {
              headers: ['turn', 'input tokens', 'cached prefix', 'fresh tokens', 'cost (no cache)', 'cost (cache)'],
              rows: rows.map(function (r) { return [String(r.t), api.fmtInt(r.inTok), api.fmtInt(r.prefix), api.fmtInt(r.fresh), '$' + api.fmt(r.costNo, 4), '$' + api.fmt(r.costC, 4)]; })
            });
            api.html(el, '<div class="formula">input(t) = system + t·user + (t−1)·answer<br>cached part = everything before this turn\'s new user message<br>cost = (cached·r_read + fresh·r_write)·price_in + output·price_out</div>');
            api.html(el, '<p>Sum over turns of a linear function ⇒ <b>O(T²)</b> total tokens. Caching changes the constant, not the shape — but the constant is large.</p>');
            api.illustrative(el, 'Prices: $' + st.pin + '/M in, $' + st.pout + '/M out, read ×' + st.mult + ', write ×' + st.write + ' (made up). First turn writes the cache; later turns read the prefix.');
            api.docsNote(el, 'Caching is enabled differently per provider (explicit markers vs automatic prefix matching) and has minimum prefix lengths and expiry times.');
          } }
        ]
      }
    ]
  });
})();
