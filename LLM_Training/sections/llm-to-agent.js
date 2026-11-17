(function () {
  'use strict';

  // ---------- shared scenario (Paris booking) ----------
  var TOOLS = [
    { name: 'get_weather', description: 'Get the forecast for a city on a date.',
      input_schema: { type: 'object', properties: { city: { type: 'string' }, date: { type: 'string', description: 'YYYY-MM-DD' } }, required: ['city', 'date'] } },
    { name: 'search_restaurants', description: 'Find restaurants with free tables.',
      input_schema: { type: 'object', properties: { city: { type: 'string' }, party_size: { type: 'integer' }, time: { type: 'string', description: 'HH:MM' } }, required: ['city', 'party_size'] } },
    { name: 'book_table', description: 'Book a table. Has side effects.',
      input_schema: { type: 'object', properties: { restaurant_id: { type: 'string' }, party_size: { type: 'integer' }, datetime: { type: 'string' } }, required: ['restaurant_id', 'party_size', 'datetime'] } }
  ];
  var SYSTEM = 'You are a travel assistant. Use tools when you need facts or actions. Be concise.';
  function tokEst(s) { return Math.ceil(s.length / 4); }

  var CALLS = [
    { call: { id: 'tu_1', name: 'get_weather', input: { city: 'Paris', date: '2026-11-18' } },
      result: { city: 'Paris', date: '2026-11-18', condition: 'light rain', temp_c: 9, precip_prob: 0.7 } },
    { call: { id: 'tu_2', name: 'search_restaurants', input: { city: 'Paris', party_size: 2, time: '20:00' } },
      result: { results: [{ id: 'r_204', name: 'Chez Louise', rating: 4.6, free_slots: ['19:30', '20:00'] }, { id: 'r_377', name: 'Le Petit Zinc', rating: 4.3, free_slots: ['20:30'] }] } },
    { call: { id: 'tu_3', name: 'book_table', input: { restaurant_id: 'r_204', party_size: 2, datetime: '2026-11-18T20:00' } },
      result: { status: 'confirmed', confirmation: 'BK-48213' } }
  ];
  var FINAL = 'Done: a table for 2 at Chez Louise tomorrow at 20:00 (confirmation BK-48213). Light rain is likely (70%, 9 C), so bring an umbrella.';

  // The context window before model call n (0..3) and what the model emits in call n.
  function context(n, api) {
    var segs = [
      { k: 'system', role: 'system', text: SYSTEM },
      { k: 'tools', role: 'tool definitions', text: JSON.stringify(TOOLS) },
      { k: 'user', role: 'user', text: api.ex.agent.request }
    ];
    for (var i = 0; i < n; i++) {
      segs.push({ k: 'asst', role: 'assistant (tool_use)', text: JSON.stringify({ type: 'tool_use', id: CALLS[i].call.id, name: CALLS[i].call.name, input: CALLS[i].call.input }) });
      segs.push({ k: 'result', role: 'tool_result', text: JSON.stringify({ type: 'tool_result', tool_use_id: CALLS[i].call.id, content: JSON.stringify(CALLS[i].result) }) });
    }
    segs.forEach(function (s) { s.tokens = tokEst(s.text); });
    return segs;
  }
  function output(n) {
    if (n < CALLS.length) return { role: 'assistant (tool_use)', k: 'asst', text: JSON.stringify({ type: 'tool_use', id: CALLS[n].call.id, name: CALLS[n].call.name, input: CALLS[n].call.input }) };
    return { role: 'assistant (final text)', k: 'asst', text: FINAL };
  }
  function sumTok(segs) { return segs.reduce(function (a, s) { return a + s.tokens; }, 0); }
  function clip(s, n) { return s.length > n ? s.slice(0, n) + ' …' : s; }
  function J(o) { return JSON.stringify(o, null, 2); }

  function segHtml(api, s, isNew, full) {
    return '<div class="llm-to-agent-seg k-' + s.k + (isNew ? ' new' : '') + '"><div class="llm-to-agent-seg-h"><span><b>' + api.esc(s.role) + '</b>' + (isNew ? ' · new' : '') + '</span><span>≈ ' + s.tokens + ' tokens</span></div><div class="llm-to-agent-seg-t">' + api.esc(full ? s.text : clip(s.text, 150)) + '</div></div>';
  }
  var COLORS = { system: 'var(--t-text)', tools: 'var(--t-function)', user: 'var(--t-token)', asst: 'var(--t-json)', result: 'var(--t-probability)' };

  var SEC = {
    id: 'llm-to-agent', part: 'II', order: 110, title: 'From LLM to Agent',
    slides: [

      // ------------------------------------------------------------ definition
      {
        id: 'definition', title: 'What is an agent? LLM + loop + tools + memory + goal',
        critical: {
          understand: ['An <b>agent</b> is not a new kind of model. It is an ordinary LLM wrapped in a <b>loop</b> that can call <b>tools</b>, keeps <b>memory</b> (the context), and works toward a <b>goal</b>.',
            'The model only ever does one thing: predict the next token. The loop around it turns that into behaviour.'],
          details: ['The code around the model is called the <b>harness</b> (or runtime / scaffold). It is plain software: it parses, executes, appends and decides when to stop.',
            'Memory has layers: the context window (short-term) plus anything the harness stores outside and re-injects (files, databases, retrieval).' ]
        },
        dataTypes: [
          { name: 'goal / user request', type: 'text', shape: 'string', example: 'Book me a table for 2 in Paris…' },
          { name: 'context (messages)', type: 'json', shape: 'list[message]', example: 'system + tools + user + …' },
          { name: 'LLM', type: 'function', shape: 'tokens → next-token distribution', example: 'frozen weights' },
          { name: 'tool', type: 'function', shape: 'json → json', example: 'get_weather(city, date)' },
          { name: 'done?', type: 'boolean', shape: 'scalar', example: 'stop_reason == "end_turn"' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.callout(el, 'Agent = <b>LLM</b> + <b>loop</b> + <b>tools</b> + <b>memory</b> + <b>goal</b>. Click a block to dive in.', 'key');
            api.blocks(el, { rows: [
              [{ id: 'goal', label: 'Goal', sub: 'user request', type: 'text', shape: 'string', zoom: 1 },
               { id: 'llm', label: 'LLM', sub: 'predicts next tokens', type: 'function', goto: 'big-picture/next-token' },
               { id: 'tools', label: 'Tools', sub: 'functions the model may request', type: 'function', shape: 'json → json', zoom: 1 }],
              [{ id: 'mem', label: 'Memory', sub: 'context window + stores', type: 'json', shape: 'list[message]', zoom: 1 },
               { id: 'loop', label: 'Loop (harness)', sub: 'while not done: call → parse → run → append', type: 'function', zoom: 1 },
               { id: 'out', label: 'Result', sub: 'final answer / side effects', type: 'text', shape: 'string' }]
            ], caption: 'The LLM proposes; the harness disposes. The loop repeats until the model answers instead of asking for a tool.' });
            api.html(el, '<p class="muted">Related: ' + api.link('big-picture/next-token', 'the next-token loop') + ' (Part I) is the <i>inner</i> loop; this section adds the <i>outer</i> loop.</p>');
          } },
          { name: 'Five parts', render: function (el, api) {
            var g = api.grid(el, 230);
            api.card(g, 'Goal', 'The task, stated in the user message (and the system prompt). The model has no goals of its own: the goal is just <b>text in the context</b>.');
            api.card(g, 'LLM', 'The frozen network. It reads the whole context and emits tokens: either prose or a structured tool call. See ' + api.link('output-sampling/autoregressive-loop', 'autoregressive loop') + '.');
            api.card(g, 'Tools', 'Ordinary functions (search, run code, book a table). The model never runs them; it only <i>asks</i>, in JSON. See ' + api.link('tool-calling/tool-definition', 'tool definitions') + '.');
            api.card(g, 'Memory', 'Short-term: the messages so far (the context window). Long-term: files, databases, vector stores that the harness reads and writes. See ' + api.link('memory-rag/memory-types', 'memory types') + '.');
            api.card(g, 'Loop', 'The harness: call model, check <code>stop_reason</code>, execute tool calls, append results, repeat. Also enforces budgets. See the loop on the ' + api.link('llm-to-agent/agent-loop', 'agent loop slide') + '.');
          } },
          { name: 'In code', render: function (el, api) {
            api.callout(el, 'Pseudo-code: every agent framework is a variation of these ~12 lines. Each of the five parts is marked.', 'info');
            api.code(el, [
              'goal     = "Book me a table for 2 in Paris tomorrow at 8pm…"   # GOAL',
              'messages = [system_prompt, user(goal)]                         # MEMORY (short-term)',
              'tools    = [get_weather, search_restaurants, book_table]       # TOOLS (schemas)',
              '',
              'while True:                                                    # LOOP',
              '    reply = llm(messages, tools)          # LLM: one forward pass per generated token',
              '    messages.append(reply)',
              '    if reply.stop_reason == "end_turn":    # model chose to answer',
              '        return reply.text',
              '    for call in reply.tool_calls:          # model chose to act',
              '        result = run_tool(call.name, call.input)   # ordinary code',
              '        messages.append(tool_result(call.id, result))'
            ].join('\n'), 'python');
            api.callout(el, 'Note the asymmetry: <code>llm()</code> is a neural network; everything else is regular, testable, deterministic code you control.', 'key');
          } }
        ]
      },

      // ------------------------------------------------------------ spectrum
      {
        id: 'spectrum', title: 'Chatbot vs workflow vs agent: who decides the control flow?',
        critical: {
          understand: ['The dividing line is <b>who decides what happens next</b>: you (code), or the model.',
            'A chatbot answers once. A workflow follows a path you wrote. An agent chooses its own steps at runtime.'],
          details: ['More autonomy buys flexibility on open-ended tasks but costs predictability, latency, tokens and safety margin. A common recommendation is to use the simplest pattern that works.',
            'Real systems mix them: a fixed workflow with one agentic step inside.']
        },
        dataTypes: [
          { name: 'control flow', type: 'function', shape: 'code path | model choice', example: 'if/else vs next tool call' },
          { name: 'LLM calls per task', type: 'integer', shape: 'scalar', example: '1 / fixed N / variable' },
          { name: 'tool call', type: 'json', shape: 'object', example: '{name, input}' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var g = api.grid(el, 240);
            api.card(g, '1 · Chatbot', '<p>One model call per user turn. No tools, no loop.</p><p class="muted">Control flow: <b>none</b> (single call).</p><p>' + api.badge('text', 'string') + ' → LLM → ' + api.badge('text', 'string') + '</p>');
            api.card(g, '2 · Workflow', '<p>Several LLM calls and tools wired in a <b>fixed</b> sequence you designed (extract → look up → draft).</p><p class="muted">Control flow: <b>your code</b>.</p>');
            api.card(g, '3 · Agent', '<p>The model decides <b>which tool next, and when to stop</b>, in a loop.</p><p class="muted">Control flow: <b>the model</b> (within harness limits).</p>');
            api.callout(el, 'Ask: <i>"Could I draw the exact flowchart in advance?"</i> If yes, build a workflow. If the path depends on what the tools return, an agent may fit.', 'tip');
          } },
          { name: 'Same task, 3 ways', render: function (el, api) {
            api.callout(el, 'Task: "' + api.esc(api.ex.agent.request) + '"', 'info');
            var sel = { v: 'chatbot' };
            var out = api.h('div');
            var traces = {
              chatbot: { who: 'nobody: single call', steps: ['LLM: writes a plausible-sounding answer from its weights', '(cannot check weather or book anything)'], res: 'Cannot actually book. May hallucinate availability.' },
              workflow: { who: 'your code', steps: ['code: call get_weather("Paris", tomorrow)', 'code: call search_restaurants(...)', 'LLM: pick one from results (fixed prompt)', 'code: call book_table(top pick)', 'LLM: write summary'], res: 'Works when the task matches the pipeline. If no tables are free, the flowchart has no branch for it unless you coded one.' },
              agent: { who: 'the model, each turn', steps: ['LLM: "I need weather" → tool_use get_weather', 'LLM: "now restaurants" → tool_use search_restaurants', 'LLM: "20:00 is free at Chez Louise" → tool_use book_table', 'LLM: end_turn with final answer'], res: 'Adapts: if 20:00 were full it could pick 19:30 or ask you. Costs more tokens and is less predictable.' }
            };
            function draw() {
              var t = traces[sel.v];
              out.innerHTML = '<p>Control flow decided by: <b>' + t.who + '</b></p><ol>' + t.steps.map(function (s) { return '<li>' + api.esc(s) + '</li>'; }).join('') + '</ol><p>' + api.esc(t.res) + '</p>';
            }
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.select(c, { label: 'Pattern', options: [{ value: 'chatbot', label: 'Chatbot' }, { value: 'workflow', label: 'Workflow' }, { value: 'agent', label: 'Agent' }], value: 'chatbot', onChange: function (v) { sel.v = v; draw(); } });
            el.appendChild(out); draw();
          } },
          { name: 'Trade-offs', render: function (el, api) {
            api.table(el, { headers: ['', 'Chatbot', 'Workflow', 'Agent'], rows: [
              ['Who decides next step', 'n/a', 'Your code', 'The model'],
              ['LLM calls per task', '1', 'Fixed, N', 'Variable (loop)'],
              ['Predictability', 'High', 'High', 'Lower'],
              ['Handles unforeseen paths', 'No', 'Only coded ones', 'Often, within its tools'],
              ['Cost / latency', 'Lowest', 'Moderate', 'Highest, grows with turns'],
              ['Main risk', 'Hallucination', 'Brittle pipeline', 'Compounding errors, unsafe actions']
            ] });
            api.html(el, '<p>Compounding errors are quantified in ' + api.link('safety-eval/failure-modes', 'failure modes') + '; workflow patterns are in ' + api.link('planning-multiagent/workflow-patterns', 'workflow patterns') + '.</p>');
          } }
        ]
      },

      // ------------------------------------------------------------ agent loop
      {
        id: 'agent-loop', title: 'The agent loop: the context window grows every turn',
        critical: {
          understand: ['Each loop iteration is a <b>fresh model call over the whole context so far</b>. The model has no hidden state between calls; the context <i>is</i> the memory.',
            'Every tool call and result is appended, so the context (and the input-token bill) grows turn by turn.'],
          details: ['Token counts here use a rough chars/4 estimate on the actual JSON shown. Real tokenizers differ.',
            'Because the same prefix is re-sent each turn, prompt caching can cut the cost of the repeated part considerably.' ]
        },
        dataTypes: [
          { name: 'context (messages)', type: 'json', shape: 'list[message]', example: 'grows each turn' },
          { name: 'tool_use', type: 'json', shape: 'object {id, name, input}', example: '{"name":"get_weather",…}' },
          { name: 'tool_result', type: 'json', shape: 'object {tool_use_id, content}', example: '{"condition":"light rain"}' },
          { name: 'input tokens per call', type: 'integer', shape: 'scalar', example: 'e.g. 300 → 560' },
          { name: 'stop_reason', type: 'text', shape: 'string', example: '"tool_use" | "end_turn"' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.callout(el, 'Request: "' + api.esc(api.ex.agent.request) + '"', 'info');
            api.blocks(el, { rows: [
              [{ id: 'ctx', label: 'Context', sub: 'system + tools + user + history', type: 'json', shape: 'list[message]' },
               { id: 'llm', label: 'LLM call', sub: 'next-token prediction', type: 'function', edge: 'tokens', zoom: 1 },
               { id: 'dec', label: 'stop_reason?', sub: 'tool_use or end_turn', type: 'boolean', edge: 'JSON or text' }],
              [{ id: 'run', label: 'Harness runs tool', sub: 'get_weather / search / book', type: 'function', zoom: 1 },
               { id: 'append', label: 'Append result', sub: 'tool_result message', type: 'json', edge: 'result' },
               { id: 'ans', label: 'Final answer', sub: 'end_turn', type: 'text', shape: 'string' }]
            ], caption: 'Loop until the model stops asking for tools. 4 model calls for this request (3 tool calls + final answer).' });
          } },
          { name: 'Turn by turn', render: function (el, api) {
            var BUDGET = 1000;
            api.illustrative(el, ' Token counts are estimated as characters / 4 on the JSON shown, and the 1,000-token gauge is an illustrative harness budget (real windows hold thousands).');
            var cum = 0, cums = [];
            for (var n = 0; n < 4; n++) { cum += sumTok(context(n, api)); cums.push(cum); }
            var steps = [];
            for (var k = 0; k < 4; k++) (function (n) {
              steps.push({ title: n < 3 ? 'Call ' + (n + 1) + ': model requests <code>' + CALLS[n].call.name + '</code>' : 'Call 4: model writes the final answer', render: function (body) {
                var ctx = context(n, api), prev = n ? context(n - 1, api).length : 0;
                var inTok = sumTok(ctx), out = output(n), outTok = tokEst(out.text);
                var kp = api.h('div', { class: 'llm-to-agent-kpis' });
                kp.innerHTML = '<div><b>' + inTok + '</b><small>input tokens this call</small></div><div><b>' + outTok + '</b><small>output tokens</small></div><div><b>' + cums[n] + '</b><small>total input billed so far</small></div>';
                body.appendChild(kp);
                var over = inTok > BUDGET;
                var gauge = api.h('div', { class: 'llm-to-agent-gauge', title: 'context / 1,000-token demo budget' });
                var segs = ctx, used = 0;
                segs.forEach(function (s) { var w = Math.min(100 - used, s.tokens / BUDGET * 100); used += w; gauge.appendChild(api.h('span', { style: { width: w + '%', background: COLORS[s.k] } })); });
                body.appendChild(gauge);
                body.appendChild(api.h('div', { class: 'muted', html: 'Context fill: ' + Math.round(inTok / BUDGET * 100) + '% of the demo budget' + (over ? ' (over!)' : '') }));
                var h1 = api.h('div', { html: '<h4>Input to the model (' + ctx.length + ' messages)</h4>' }); body.appendChild(h1);
                ctx.forEach(function (s, i) { h1.insertAdjacentHTML('beforeend', segHtml(api, s, n > 0 && i >= ctx.length - 2, false)); });
                body.appendChild(api.h('div', { html: '<h4>Model output (this call)</h4>' + segHtml(api, { k: out.k, role: out.role, text: out.text, tokens: outTok }, true, true) }));
                body.appendChild(api.h('div', { class: 'muted', html: n < 3 ? 'Harness parses the JSON, runs <code>' + CALLS[n].call.name + '</code>, and appends the <code>tool_result</code>. Next call re-sends everything above plus these two messages.' : '<code>stop_reason = "end_turn"</code>: the loop ends and the harness returns this text.' }));
              } });
            })(k);
            api.stepper(el, { steps: steps, interval: 2600 });
          } },
          { name: 'Token gauge', render: function (el, api) {
            api.illustrative(el, ' chars/4 estimate; the shape (quadratic total) is the point, not the exact numbers.');
            var rows = [], cum = 0, ins = [];
            for (var n = 0; n < 4; n++) {
              var i1 = sumTok(context(n, api)), o1 = tokEst(output(n).text); cum += i1; ins.push(i1);
              rows.push(['Call ' + (n + 1), api.fmtInt(i1), api.fmtInt(o1), api.fmtInt(cum), n < 3 ? 'tool_use' : 'end_turn']);
            }
            api.table(el, { headers: ['Call', 'Input tokens', 'Output tokens', 'Cumulative input billed', 'stop_reason'], rows: rows });
            api.bars(el, { title: 'Input tokens per call (context size)', items: ins.map(function (v, i) { return { label: 'Call ' + (i + 1), value: v, highlight: i === ins.length - 1 }; }), format: function (v) { return api.fmtInt(v); } });
            var once = ins[ins.length - 1];
            api.callout(el, 'The final context is only <b>' + once + '</b> tokens, but across the whole task the provider processed <b>' + cum + '</b> input tokens (<b>' + api.fmt(cum / once, 1) + '×</b>). Over many turns the total grows roughly quadratically, which is why ' + api.link('chat-api/cost-caching', 'prompt caching') + ' and ' + api.link('memory-rag/context-management', 'context management') + ' matter.', 'key');
          } },
          { name: 'Raw context', render: function (el, api) {
            api.callout(el, 'This is what the model really receives: one text sequence (then token IDs). The marker syntax below is <b>illustrative</b>; each model family has its own chat template (' + 'see ' + api.link('chat-api/chat-template', 'chat template') + ').', 'illustrative');
            var box = api.h('div'); var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            function draw(v) {
              var n = +v; box.innerHTML = '';
              var ctx = context(n, api), lines = [];
              ctx.forEach(function (s) {
                var tag = s.k === 'tools' ? 'tools' : s.k === 'asst' ? 'assistant' : s.k === 'result' ? 'tool' : s.k;
                lines.push('<|' + tag + '|>\n' + s.text);
              });
              lines.push('<|assistant|>   ← the model continues from here, token by token');
              api.code(box, lines.join('\n'), 'text');
              box.appendChild(api.h('div', { class: 'muted', html: 'Before call ' + (n + 1) + ': ≈ ' + sumTok(ctx) + ' tokens.' }));
            }
            api.select(c, { label: 'Before model call', options: [{ value: '0', label: '1' }, { value: '1', label: '2' }, { value: '2', label: '3' }, { value: '3', label: '4' }], value: '0', onChange: draw, init: false });
            el.appendChild(box); draw('0');
          } }
        ]
      },

      // ------------------------------------------------------------ loop internals
      {
        id: 'loop-internals', title: 'Loop internals: data types at each hop',
        critical: {
          understand: ['The agent is a <b>loop around repeated next-token prediction</b>. Text becomes token IDs, the model produces logits, a sampler picks tokens, and those tokens <i>happen to spell</i> a JSON tool call.',
            'The harness is <b>ordinary code</b>: it parses that JSON, calls a real function, and turns the result back into tokens for the next prediction.'],
          details: ['There are two nested loops: the inner one (one token per forward pass) and the outer agent loop (one tool round-trip per iteration).', 'The tool-call syntax is trained into the model; formats differ per provider, but the principle is the same.']
        },
        dataTypes: [
          { name: 'prompt text', type: 'text', shape: 'string', example: 'Book me a table…' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[464, 3797, …]' },
          { name: 'logits', type: 'float', shape: '[vocab]', example: '[128,256] floats' },
          { name: 'next-token probabilities', type: 'probability', shape: '[vocab]', example: 'sums to 1' },
          { name: 'tool call', type: 'json', shape: 'object', example: '{"name":"get_weather",…}' },
          { name: 'harness / tool', type: 'function', shape: 'json → json', example: 'run_tool()' },
          { name: 'tool result', type: 'json', shape: 'object | string', example: '{"condition":"light rain"}' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.callout(el, 'One trip around the outer loop, with the <b>type of data on every arrow</b>. Click blocks to jump to the related lesson.', 'key');
            api.blocks(el, { rows: [
              [{ id: 'text', label: 'Messages', sub: 'text', type: 'text', shape: 'list[message]' },
               { id: 'tmpl', label: 'Chat template', sub: 'serialize to ONE sequence', type: 'function', edge: 'text', goto: 'chat-api/chat-template' },
               { id: 'tok', label: 'Tokenizer', sub: 'text → IDs', type: 'function', edge: 'one string', goto: 'tokenization/text-to-ids' },
               { id: 'model', label: 'Model', sub: 'transformer forward pass', type: 'function', edge: 'IDs [integer]', goto: 'big-picture/next-token' }],
              [{ id: 'logits', label: 'Logits', sub: 'one score per vocab entry', type: 'float', shape: '[vocab]', goto: 'output-sampling/unembedding' },
               { id: 'samp', label: 'Sampler', sub: 'temperature, top-p, constraints', type: 'function', edge: 'floats', goto: 'output-sampling/sampling-playground' },
               { id: 'json', label: 'Tool call', sub: 'sampled tokens spell JSON', type: 'json', edge: 'token IDs → text', goto: 'tool-calling/round-trip' },
               { id: 'harn', label: 'Harness', sub: 'ordinary code', type: 'function', edge: 'parsed object' }],
              [{ id: 'res', label: 'Tool result', sub: 'json / text', type: 'json', edge: 'return value' },
               { id: 'app', label: 'Append', sub: 'new message → new tokens', type: ['text', 'integer'], edge: 'serialize again' },
               { id: 'again', label: 'Predict again', sub: 'back to the top', type: 'function', zoom: 2 }]
            ], caption: 'Only the grey-matter steps (model, logits) are neural. Template, sampler, parser, tool and append are ordinary code.' });
            api.html(el, '<p class="muted">Inner loop: ' + api.link('output-sampling/autoregressive-loop', 'autoregressive generation') + '. Outer loop: this section.</p>');
          } },
          { name: 'Each hop', render: function (el, api) {
            api.illustrative(el, ' Token IDs, logits and the template markers below are made-up for illustration (real values depend on the model).');
            var ill = [{ text: 'Book', id: 40133 }, { text: ' me', id: 502 }, { text: ' a', id: 257 }, { text: ' table', id: 3084 }, { text: ' …', id: 986 }];
            var cands = [{ t: 'weather', z: 7.1 }, { t: 'forecast', z: 3.2 }, { t: 'search', z: 2.6 }, { t: 'book', z: 1.9 }, { t: 'the', z: 0.2 }];
            var probs = api.math.softmax(cands.map(function (c) { return c.z; }));
            var steps = [
              { title: 'Text in', render: function (b) { b.innerHTML = '<p>' + api.badge('text', 'string') + ' The harness holds a list of messages (system, tool definitions, user…).</p>'; api.code(b, 'user: "' + api.ex.agent.request + '"', 'text'); } },
              { title: 'Chat template → one string', render: function (b) { b.innerHTML = '<p>' + api.badge('text', 'string') + ' Messages and tool definitions are flattened into <b>one</b> string with special marker tokens (template is model-specific).</p>'; api.code(b, '<|system|> You are a travel assistant…\n<|tools|> [{"name":"get_weather",…}, …]\n<|user|> Book me a table for 2 in Paris…\n<|assistant|>', 'text'); } },
              { title: 'Tokenize → integer IDs', render: function (b) { b.innerHTML = '<p>' + api.badge('integer', '[seq_len]') + ' Each piece becomes an ID (shown: only the first few, illustrative).</p>'; api.tokens(b, ill); } },
              { title: 'Model → logits', render: function (b) { b.innerHTML = '<p>One forward pass over all IDs yields ' + api.badge('float', '[vocab]') + ' logits for the <i>next</i> position (128,256 numbers for the reference model). Top candidates for the token after <code>"name": "get_</code>:</p>'; api.bars(b, { items: cands.map(function (c) { return { label: api.esc(c.t) + ' <small class="muted">logit ' + api.fmt(c.z, 1) + '</small>', value: c.z, highlight: c === cands[0] }; }), max: 8, format: function (v) { return api.fmt(v, 1); } }); } },
              { title: 'Softmax + sample → token', render: function (b) { b.innerHTML = '<p>' + api.badge('probability', '[vocab]') + ' softmax turns logits into probabilities; the sampler picks one (here the most likely: <b>' + api.esc(cands[0].t) + '</b>, p = ' + api.fmt(probs[0], 2) + '). Repeat per token to spell the whole call.</p>'; api.bars(b, { items: cands.map(function (c, i) { return { label: api.esc(c.t), value: probs[i], highlight: i === 0 }; }), max: 1, format: function (v) { return api.fmt(v, 2); } }); } },
              { title: 'Tokens spell a JSON tool call', render: function (b) { b.innerHTML = '<p>' + api.badge('json', 'object') + ' After many forward passes, the generated tokens decode to text that is valid JSON. Nothing magic: the model learned to emit this format.</p>'; api.code(b, J({ type: 'tool_use', id: 'tu_1', name: 'get_weather', input: { city: 'Paris', date: '2026-11-18' } }), 'json'); } },
              { title: 'Harness executes (ordinary code)', render: function (b) { b.innerHTML = '<p>' + api.badge('function', 'json → json') + ' The harness parses the JSON, looks the name up in a registry and <b>calls a real function</b>. No neural network involved.</p>'; api.code(b, 'call = json.loads(model_output)\nfn = REGISTRY[call["name"]]          # plain dict lookup\nresult = fn(**call["input"])         # real HTTP call to a weather API', 'python'); } },
              { title: 'Result → appended tokens', render: function (b) { b.innerHTML = '<p>' + api.badge('json', 'object') + ' The result is wrapped in a message, serialized by the template and tokenized into new ' + api.badge('integer', '[n_new]') + ' IDs appended to the sequence. Then the model predicts again.</p>'; api.code(b, J({ type: 'tool_result', tool_use_id: 'tu_1', content: '{"condition":"light rain","temp_c":9,"precip_prob":0.7}' }), 'json'); } }
            ];
            api.stepper(el, { steps: steps, interval: 2600 });
          } },
          { name: 'Two nested loops', render: function (el, api) {
            api.callout(el, 'The agent is <b>a loop around repeated next-token prediction</b>.', 'key');
            var o = api.h('div', { class: 'llm-to-agent-layer' }); el.appendChild(o);
            o.innerHTML = '<b>Outer loop (harness, ordinary code):</b> one iteration per tool round-trip - typically a handful, bounded by a max-iteration limit.';
            var inn = api.h('div', { class: 'llm-to-agent-layer' }); o.appendChild(inn);
            inn.innerHTML = '<b>Inner loop (decoding):</b> one iteration per generated token - dozens for a tool call, hundreds for an answer. Each iteration = forward pass → logits → sample → append token (' + api.link('output-sampling/autoregressive-loop', 'details') + ').';
            api.code(el, [
              'while not done:                              # outer: agent loop',
              '    ids = tokenize(chat_template(messages, tools))',
              '    out = []',
              '    while True:                               # inner: next-token loop',
              '        logits = model(ids + out)[-1]         # float[vocab]',
              '        t = sample(logits)                    # integer',
              '        if t == EOS: break',
              '        out.append(t)',
              '    msg = parse(detokenize(out))              # text/JSON',
              '    messages.append(msg)',
              '    if msg.has_tool_calls: messages += run_tools(msg)',
              '    else: done = True'
            ].join('\n'), 'python');
            api.html(el, '<p>Why the loop helps: the model\'s own output (a tool call) and the tool\'s result land back in the context where attention can read them. Compare with chain-of-thought as a scratchpad in ' + api.link('reasoning/why-cot', 'reasoning') + '. See the protocol side in ' + api.link('tool-calling/round-trip', 'the tool round-trip') + '.</p>');
          } }
        ]
      },

      // ------------------------------------------------------------ stop conditions
      {
        id: 'stop-conditions', title: 'Stop conditions and budgets',
        critical: {
          understand: ['The normal exit is the model choosing to answer: <code>stop_reason = "end_turn"</code> instead of <code>"tool_use"</code>.',
            'A harness must <b>also</b> impose hard limits: max iterations, token/cost budget and wall-clock time, because a model can loop, stall or wander.'],
          details: ['Other stop reasons exist (output cut off by <code>max_tokens</code>, a refusal, a stop sequence); the harness should handle each explicitly.',
            'When a limit trips, return a partial result or ask a human instead of silently failing.']
        },
        dataTypes: [
          { name: 'stop_reason', type: 'text', shape: 'string', example: '"tool_use" | "end_turn" | "max_tokens"' },
          { name: 'max_iterations', type: 'integer', shape: 'scalar', example: '8' },
          { name: 'token budget', type: 'integer', shape: 'scalar', example: '10,000 input tokens' },
          { name: 'budget exceeded?', type: 'boolean', shape: 'scalar', example: 'true / false' },
          { name: 'elapsed time', type: 'float', shape: 'scalar (seconds)', example: '42.5' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ id: 'call', label: 'Model call', type: 'function' },
               { id: 'sr', label: 'stop_reason?', sub: 'what did the model emit?', type: 'text', edge: 'response' },
               { id: 'run', label: 'tool_use → run tools', sub: 'append results, loop', type: 'json', zoom: 1 }],
              [{ id: 'end', label: 'end_turn → return answer', sub: 'normal exit', type: 'text' },
               { id: 'bud', label: 'Budget check', sub: 'iterations · tokens · time', type: 'boolean', zoom: 1 },
               { id: 'abort', label: 'Abort / hand off', sub: 'partial result or ask a human', type: 'text' }]
            ], caption: 'Two ways out: the model finishes (end_turn), or the harness cuts it off (budget).' });
            api.html(el, '<p class="muted">Compounding failure across many steps: ' + api.link('safety-eval/failure-modes', 'failure modes') + '. Human approval gates: ' + api.link('safety-eval/autonomy-levels', 'autonomy levels') + '.</p>');
          } },
          { name: 'Pseudo-code', render: function (el, api) {
            api.callout(el, 'Pseudo-code, not a specific SDK.', 'info');
            api.code(el, [
              'MAX_ITERS, MAX_INPUT_TOKENS, MAX_SECONDS = 8, 50_000, 120',
              'spent_tokens, t0 = 0, now()',
              '',
              'for i in range(MAX_ITERS):                       # budget 1: iterations',
              '    reply = llm(messages, tools, max_tokens=1024)',
              '    spent_tokens += reply.usage.input_tokens      # budget 2: tokens (cost)',
              '    messages.append(reply)',
              '',
              '    if reply.stop_reason == "end_turn":           # normal exit',
              '        return reply.text',
              '    if reply.stop_reason == "max_tokens":         # output was cut off',
              '        messages.append(user("Please continue.")); continue',
              '    for call in reply.tool_calls:',
              '        messages.append(run_tool_safely(call))    # errors returned as data',
              '',
              '    if spent_tokens > MAX_INPUT_TOKENS: break      # budget 2',
              '    if now() - t0 > MAX_SECONDS:        break      # budget 3: wall-clock',
              '',
              'return give_up_summary(messages)                  # partial result / escalate'
            ].join('\n'), 'python');
          } },
          { name: 'Budget simulator', render: function (el, api) {
            api.illustrative(el, ' Simulated agent: context starts at 350 tokens and grows by 140 per iteration (tool call + result). Nothing is called.');
            var st = { need: 6, maxIt: 8, tokB: 4000, secB: 60 };
            var out = api.h('div');
            function draw() {
              var cum = 0, rows = [], reason = 'end_turn: the model finished', stopped = false;
              for (var i = 1; i <= st.need + 1; i++) {
                var inp = 350 + (i - 1) * 140, secs = i * 6; cum += inp;
                var final = i === st.need + 1, status = final ? 'end_turn' : 'tool_use';
                var trip = null;
                if (i > st.maxIt) trip = 'max iterations (' + st.maxIt + ')';
                else if (cum > st.tokB) trip = 'token budget (' + api.fmtInt(st.tokB) + ')';
                else if (secs > st.secB) trip = 'time budget (' + st.secB + ' s)';
                if (trip) { rows.push([i, api.fmtInt(inp), api.fmtInt(cum), secs + ' s', '<span class="llm-to-agent-trip">STOP: ' + trip + '</span>']); reason = 'harness stopped it: ' + trip; stopped = true; break; }
                rows.push([i, api.fmtInt(inp), api.fmtInt(cum), secs + ' s', '<span class="llm-to-agent-ok">' + status + '</span>']);
              }
              out.innerHTML = '';
              api.table(out, { headers: ['Model call', 'Input tokens', 'Cumulative input', 'Elapsed (≈6 s/call)', 'Outcome'], rows: rows });
              api.callout(out, 'Result: <b>' + api.esc(reason) + '</b>' + (stopped ? ' - the task is left unfinished; a good harness returns a partial result or asks a human.' : '.'), stopped ? 'warn' : 'tip');
            }
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'Tool rounds the task needs', min: 1, max: 14, step: 1, value: 6, format: function (v) { return v; }, onInput: function (v) { st.need = v; draw(); } });
            api.slider(c, { label: 'max iterations', min: 2, max: 12, step: 1, value: 8, format: function (v) { return v; }, onInput: function (v) { st.maxIt = v; draw(); } });
            api.slider(c, { label: 'token budget', min: 1000, max: 12000, step: 500, value: 4000, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.tokB = v; draw(); } });
            api.slider(c, { label: 'time budget (s)', min: 10, max: 120, step: 5, value: 60, format: function (v) { return v + ' s'; }, onInput: function (v) { st.secB = v; draw(); } });
            el.appendChild(out); draw();
          } }
        ]
      }
    ]
  };
  // Workaround for core layout: in column mode (narrow screens) .slide-content shrink-wraps, so cap the width.
  SEC.slides.forEach(function (sl) { sl.levels.forEach(function (lv) { var r = lv.render; lv.render = function (el, api) { el.classList.add('llm-to-agent-wrap'); return r(el, api); }; }); });
  Course.register(SEC);
})();
