(function () {
  'use strict';

  // ReAct trace of a coding agent (illustrative content)
  var TRACE = [
    { t: 'The user says test_parse_price fails. I should run the tests first to see the actual error.',
      a: { name: 'run_tests', arguments: { path: 'tests/test_price.py' } },
      o: 'FAILED tests/test_price.py::test_parse_price\n  assert parse_price("$1,299.50") == 1299.5\n  E  ValueError: could not convert string to float: \'1,299.50\'\n1 failed, 4 passed' },
    { t: 'The comma is not handled. Let me read the implementation of parse_price.',
      a: { name: 'read_file', arguments: { path: 'src/price.py' } },
      o: '1  def parse_price(s):\n2      s = s.strip().lstrip("$")\n3      return float(s)' },
    { t: 'Line 3 calls float() directly on a string with a thousands separator. Strip commas first. Keep the change minimal.',
      a: { name: 'edit_file', arguments: { path: 'src/price.py', old: 'return float(s)', new: 'return float(s.replace(",", ""))' } },
      o: 'ok: 1 replacement in src/price.py' },
    { t: 'Re-run the same test file to verify the fix and make sure nothing else broke.',
      a: { name: 'run_tests', arguments: { path: 'tests/test_price.py' } },
      o: '5 passed in 0.04s' },
    { t: 'All tests pass. I can stop and report what I changed.',
      a: null,
      o: null, final: 'Fixed: parse_price now strips thousands separators before float(). Tests in tests/test_price.py pass (5/5).' }
  ];

  function est(s) { return Math.max(1, Math.round(s.length / 4)); }   // rough chars/4 estimate

  Course.register({
    id: 'planning-multiagent', part: 'II', order: 140, title: 'Planning & Multi-Agent',
    slides: [
      // ================================================================ react-trace
      {
        id: 'react-trace', title: 'ReAct: a coding agent fixes a failing test',
        critical: {
          understand: [
            'ReAct interleaves <b>Thought → Action → Observation</b>: reason in text, call a tool, read the result, repeat.',
            'Planning is not a separate module: it is the model writing the next step as tokens, conditioned on everything so far.',
            'The harness (plain code) runs each action and appends the observation; the loop ends when the model answers without a tool call.'
          ],
          details: [
            'This trace is hand-written to show the structure; a real run varies from sample to sample.',
            'Modern APIs often hide or summarize the "thought" part, but the loop shape is the same.'
          ]
        },
        dataTypes: [
          { name: 'thought', type: 'text', shape: 'string', example: '"The comma is not handled"' },
          { name: 'action', type: 'json', shape: 'object', example: '{name, arguments}' },
          { name: 'observation', type: 'text', shape: 'string', example: 'test output, file contents' },
          { name: 'context', type: 'integer', shape: '[n_tokens]', example: 'grows every step' },
          { name: 'is_final', type: 'boolean', shape: 'scalar', example: 'no tool call → stop' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.html(el, '<p>Task: <i>"' + api.esc(api.ex.agent.coding) + '"</i> with tools <code>run_tests</code>, <code>read_file</code>, <code>edit_file</code>.</p>');
            api.blocks(el, { rows: [[
              { id: 'th', label: 'Thought', type: 'text', shape: 'string', sub: 'model writes tokens', zoom: 1, color: '#b98cff' },
              { id: 'ac', label: 'Action', type: 'json', shape: 'object', sub: 'tool call as JSON tokens', zoom: 1, color: '#ff9a4d', edge: 'next-token prediction' },
              { id: 'ob', label: 'Observation', type: 'text', shape: 'string', sub: 'harness runs the tool', zoom: 1, color: '#5ad18a', edge: 'ordinary code' },
              { id: 'ctx', label: 'Appended to context', type: 'integer', shape: '[n_tokens]', zoom: 2, edge: 'then predict again' }
            ]], caption: 'The loop repeats until the model emits a plain answer instead of a tool call. ' + '(' + 'see ' + api.link('llm-to-agent/agent-loop', 'the agent loop') + ')' });
          } },
          { name: 'Trace', render: function (el, api) {
            api.html(el, '<p>Step through the run: <b>run tests → read file → edit → re-run → pass</b>.</p>');
            api.stepper(el, { interval: 2600, steps: TRACE.map(function (s, i) {
              return { title: s.final ? 'Final answer' : 'Step ' + (i + 1) + ': ' + s.a.name, render: function (b) {
                api.html(b, '<div class="planning-multiagent-tao t"><b>Thought</b> ' + api.esc(s.t) + ' ' + api.type('text') + '</div>');
                if (s.a) {
                  api.html(b, '<div class="planning-multiagent-tao a"><b>Action</b> ' + api.type('json', 'tool call') + '<pre>' + api.esc(JSON.stringify(s.a)) + '</pre></div>');
                  api.html(b, '<div class="planning-multiagent-tao o"><b>Observation</b> ' + api.type('text') + '<pre>' + api.esc(s.o) + '</pre></div>');
                } else {
                  api.html(b, '<div class="planning-multiagent-tao o"><b>Answer</b> ' + api.type('text') + ' ' + api.type('boolean', 'is_final = true') + '<pre>' + api.esc(s.final) + '</pre></div>');
                }
              } };
            }) });
            api.illustrative(el, 'A hand-written, idealized trace. Real agents may read more files, run extra commands or take wrong turns first.');
          } },
          { name: 'Context grows', render: function (el, api) {
            var base = 'system: You are a coding agent. Tools: run_tests, read_file, edit_file.\nuser: ' + api.ex.agent.coding + ': test_parse_price fails.\n';
            var seqs = [], acc = base;
            TRACE.forEach(function (s) {
              var add = s.a ? 'assistant: ' + s.t + '\nassistant.tool_call: ' + JSON.stringify(s.a) + '\ntool: ' + s.o + '\n' : 'assistant: ' + s.final + '\n';
              seqs.push({ before: acc, add: add });
              acc += add;
            });
            var out = api.h('div');
            function draw(k) {
              out.innerHTML = '';
              var tot = est(seqs[k].before + seqs[k].add), prev = est(seqs[k].before);
              api.html(out, '<p>After step ' + (k + 1) + ' the model has been fed ' + api.badge('integer', '[n_tokens]') + ' ≈ <b>' + api.fmtInt(tot) + '</b> tokens (chars/4 estimate), of which <b>' + api.fmtInt(tot - prev) + '</b> are new. Every prediction re-reads the whole sequence (often cheaper thanks to ' + api.link('chat-api/cost-caching', 'prompt caching') + ').</p>');
              api.html(out, '<div class="planning-multiagent-seq">' + api.esc(seqs[k].before) + '<span class="m-new">' + api.esc(seqs[k].add) + '</span></div>');
              api.html(out, '<p class="muted">Rendered by a chat template into <b>one</b> token sequence (' + api.link('chat-api/chat-template', 'chat template') + '); the tool call is just JSON-shaped tokens the model predicted.</p>');
            }
            var c = api.h('div', { class: 'controls' });
            api.slider(c, { label: 'After step', min: 1, max: TRACE.length, value: 2, format: function (v) { return v + ' / ' + TRACE.length; }, onInput: function (v) { draw(v - 1); } });
            el.appendChild(c); el.appendChild(out);
          } }
        ]
      },

      // ================================================================ workflow-patterns
      {
        id: 'workflow-patterns', title: 'Workflow patterns: chaining, routing, parallel, orchestrator, evaluator',
        critical: {
          understand: [
            'Many tasks need a <b>workflow</b> (code decides the path) rather than a free-roaming agent (the model decides): simpler, cheaper, more predictable.',
            'Five common patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.',
            'Start with the simplest one that works; add agency only when fixed paths cannot cover the task.'
          ],
          details: [
            'Names follow common usage (e.g. in public agent-design guides); other sources use different labels.',
            'Frameworks such as ' + '<a href="#" class="xlink" data-goto="langgraph/chains-vs-graphs">LangGraph</a>' + ' express these as graphs with nodes and edges.'
          ]
        },
        dataTypes: [
          { name: 'LLM call', type: 'function', shape: 'text → text', example: 'one model invocation' },
          { name: 'intermediate output', type: 'text', shape: 'string', example: 'passed to the next step' },
          { name: 'route label', type: 'text', shape: 'string', example: '"refund" | "bug" | "other"' },
          { name: 'critique / score', type: 'float', shape: 'scalar', example: 'evaluator output' },
          { name: 'subtask list', type: 'json', shape: 'list[object]', example: 'made by the orchestrator' }
        ],
        levels: [
          { name: 'Pattern picker', render: function (el, api) {
            var P = {
              chain: { n: 'Prompt chaining', d: 'Fixed sequence; each call consumes the previous output; optional code checks (gates) between steps.',
                rows: [[{ label: 'LLM 1', sub: 'outline', type: 'text', shape: 'string' }, { label: 'Gate', sub: 'code check', type: 'boolean', shape: 'scalar' }, { label: 'LLM 2', sub: 'draft', type: 'text', shape: 'string' }, { label: 'LLM 3', sub: 'translate', type: 'text', shape: 'string' }]] },
              route: { n: 'Routing', d: 'A classifier call (or code) picks which specialised prompt, model or tool handles the input.',
                rows: [[{ label: 'Input', type: 'text', shape: 'string' }, { label: 'Router', sub: 'classify', type: 'text', shape: 'label', edge: 'label' }], [{ label: 'Refund flow', type: 'function' }, { label: 'Bug flow', type: 'function' }, { label: 'General Q&A', type: 'function' }]] },
              par: { n: 'Parallelization', d: 'Independent subtasks run at the same time (sectioning), or the same task runs several times and results are voted/merged.',
                rows: [[{ label: 'Input', type: 'text', shape: 'string' }], [{ label: 'LLM A', type: 'function' }, { label: 'LLM B', type: 'function' }, { label: 'LLM C', type: 'function' }], [{ label: 'Aggregate', sub: 'code or vote', type: 'text', shape: 'string' }]] },
              orch: { n: 'Orchestrator-workers', d: 'A lead model decides the subtasks <i>at run time</i>, delegates to workers, then synthesises. Use when subtasks cannot be listed in advance.',
                rows: [[{ label: 'Orchestrator', sub: 'plans subtasks', type: 'json', shape: 'list[object]' }], [{ label: 'Worker 1', type: 'function' }, { label: 'Worker 2', type: 'function' }, { label: 'Worker n', type: 'function' }], [{ label: 'Synthesize', type: 'text', shape: 'string' }]] },
              eval: { n: 'Evaluator-optimizer', d: 'One call produces, another critiques against criteria; loop until accepted or a budget is hit.',
                rows: [[{ label: 'Generator', type: 'text', shape: 'string' }, { label: 'Evaluator', sub: 'rubric', type: 'float', shape: 'scalar', edge: 'draft' }, { label: 'Accept?', type: 'boolean', shape: 'scalar', edge: 'score' }, { label: 'Output', type: 'text', shape: 'string', edge: 'yes' }]] }
            };
            var cur = 'chain', out = api.h('div'), tabs = api.h('div', { class: 'planning-multiagent-tabs' });
            function draw() {
              out.innerHTML = '';
              [].forEach.call(tabs.children, function (b) { b.classList.toggle('on', b.getAttribute('data-k') === cur); });
              api.html(out, '<p><b>' + P[cur].n + '.</b> ' + P[cur].d + '</p>');
              api.blocks(out, { rows: P[cur].rows, caption: cur === 'eval' ? 'Evaluator says "no" → feedback goes back to the generator.' : null });
            }
            Object.keys(P).forEach(function (k) { var b = api.button(tabs, P[k].n, function () { cur = k; draw(); }); b.setAttribute('data-k', k); });
            el.appendChild(tabs); el.appendChild(out); draw();
          } },
          { name: 'When to use', render: function (el, api) {
            api.table(el, { headers: ['Pattern', 'Use when', 'Cost / risk', 'Example'], rows: [
              ['Chaining', 'Task splits into fixed steps', 'Latency adds up; errors propagate (' + api.link('safety-eval/failure-modes', 'pⁿ') + ')', 'Outline → draft → translate'],
              ['Routing', 'Distinct input categories', 'Misrouting sends work down the wrong path', 'Support triage'],
              ['Parallelization', 'Independent subtasks, or want a vote', 'More tokens; same wall-clock as the slowest branch', 'Review code for several issue types'],
              ['Orchestrator-workers', 'Subtasks unknown in advance', 'Most tokens and moving parts (' + api.link('planning-multiagent/multi-agent', 'cost') + ')', 'Multi-file code change'],
              ['Evaluator-optimizer', 'Clear criteria; iteration helps', 'Loops cost tokens; judge bias (' + api.link('safety-eval/evals', 'LLM-as-judge') + ')', 'Translation polishing']
            ] });
            api.callout(el, 'A tool-using agent loop (' + api.link('planning-multiagent/react-trace', 'ReAct') + ') is the open-ended alternative: the model chooses the path. Prefer the least agency that solves the task.', 'key');
          } },
          { name: 'Evaluator loop', render: function (el, api) {
            var st = { p: 60, max: 4 }, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              out.innerHTML = '';
              var p = st.p / 100, rows = [], cum = 0, calls = 0;
              for (var i = 1; i <= st.max; i++) {
                var pr = Math.pow(1 - p, i - 1) * p; cum += pr; calls += 2;
                rows.push([i, api.fmt(pr * 100, 1) + '%', api.fmt(cum * 100, 1) + '%', calls]);
              }
              var expected = 0; for (var j = 1; j <= st.max; j++) expected += (j < st.max ? Math.pow(1 - p, j - 1) * p * j : Math.pow(1 - p, j - 1) * j);
              api.table(out, { headers: ['Round', 'Accepted exactly here', 'Cumulative accepted', 'Model calls so far (generate + evaluate)'], rows: rows });
              api.html(out, '<div class="formula">E[rounds] = Σ rounds × P = <b>' + api.fmt(expected, 2) + '</b> (capped at ' + st.max + ') → ≈ ' + api.fmt(2 * expected, 1) + ' model calls</div>');
              api.callout(out, 'This assumes each round is an independent chance <i>p</i> of passing the evaluator. In practice, rounds are correlated and the evaluator itself can be wrong or biased, so treat this as a budgeting sketch.', 'hedge');
            }
            api.slider(c, { label: 'P(accept per round)', min: 10, max: 95, step: 5, value: st.p, format: function (v) { return v + '%'; }, onInput: function (v) { st.p = v; draw(); }, init: false });
            api.slider(c, { label: 'Max rounds', min: 1, max: 8, value: st.max, format: function (v) { return v; }, onInput: function (v) { st.max = v; draw(); }, init: false });
            el.appendChild(c); el.appendChild(out); draw();
            api.illustrative(el, 'Acceptance probabilities are assumptions you set.');
          } }
        ]
      },

      // ================================================================ multi-agent
      {
        id: 'multi-agent', title: 'Multi-agent systems and what they cost',
        critical: {
          understand: [
            'A multi-agent system runs several model loops, each with its <b>own context window</b>, coordinated by an orchestrator or by message passing.',
            'Benefits: parallel work, separation of concerns, smaller focused contexts. Costs: more tokens in total, more latency in coordination, more failure modes.',
            'Public reports from teams building such systems describe token use several times higher than a single chat (figures vary widely); decide by measuring, not by fashion.'
          ],
          details: [
            'Information is lost at every hand-off: workers only know what the orchestrator wrote into their prompt.',
            'Whether it is cheaper or dearer than one long-context agent depends on how big the shared context is and how independent the subtasks are: use the calculator.'
          ]
        },
        dataTypes: [
          { name: 'n_agents N', type: 'integer', shape: 'scalar', example: '3' },
          { name: 'context per agent', type: 'integer', shape: 'scalar', example: 'tokens' },
          { name: 'total input tokens', type: 'integer', shape: 'scalar', example: 'summed over all calls' },
          { name: 'token multiplier', type: 'float', shape: 'scalar', example: 'multi ÷ single' },
          { name: 'worker report', type: 'text', shape: 'string', example: 'summary back to orchestrator' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ id: 'user', label: 'User task', type: 'text', shape: 'string' }, { id: 'orch', label: 'Orchestrator agent', sub: 'own loop + context', type: 'json', shape: 'tool calls', zoom: 1, edge: 'delegate' }],
              [{ id: 'w1', label: 'Worker 1', sub: 'own context', type: 'integer', shape: '[n_tokens]', zoom: 1 }, { id: 'w2', label: 'Worker 2', sub: 'own context', type: 'integer', shape: '[n_tokens]', zoom: 1 }, { id: 'w3', label: 'Worker 3', sub: 'own context', type: 'integer', shape: '[n_tokens]', zoom: 1 }],
              [{ id: 'rep', label: 'Reports back', sub: 'short summaries', type: 'text', shape: 'list[str]', edge: 'results' }, { id: 'ans', label: 'Final answer', type: 'text', shape: 'string' }]
            ], caption: 'Each box is a full agent loop (' + api.link('llm-to-agent/agent-loop', 'agent loop') + '). Workers can be exposed to the orchestrator as <i>tools</i> (' + api.link('tool-calling/round-trip', 'tool round-trip') + ').' });
          } },
          { name: 'Token calculator', render: function (el, api) {
            var st = { n: 3, tw: 6, c0: 3000, g: 800, r: 600, to: 4 }, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function sumInput(turns, c0, g) { return turns * c0 + g * turns * (turns - 1) / 2; }  // input tokens re-read each call
            function draw() {
              out.innerHTML = '';
              var S = st.n * st.tw;
              var single = sumInput(S, st.c0, st.g);
              var workers = st.n * sumInput(st.tw, st.c0, st.g);
              var orch = sumInput(st.to, st.c0 + st.n * st.r, st.r);
              var multi = workers + orch;
              var g2 = api.h('div', { class: 'planning-multiagent-cmp' }); out.appendChild(g2);
              api.card(g2, 'One agent, ' + S + ' turns', '<div class="planning-multiagent-big">' + api.fmtInt(single) + '</div>input tokens (context grows from ' + api.fmtInt(st.c0) + ' by ' + api.fmtInt(st.g) + '/turn)');
              api.card(g2, st.n + ' workers + orchestrator', '<div class="planning-multiagent-big">' + api.fmtInt(multi) + '</div>workers ' + api.fmtInt(workers) + ' + orchestrator ' + api.fmtInt(orch));
              api.card(g2, 'Multiplier', '<div class="planning-multiagent-big">' + api.fmt(multi / single, 2) + '×</div>multi ÷ single');
              api.bars(out, { items: [{ label: 'single', value: single, color: '#4ea8ff' }, { label: 'multi', value: multi, color: '#ff9a4d' }], format: function (v) { return api.fmtInt(v); } });
              api.html(out, '<div class="formula">input tokens ≈ Σ per call (context size) = turns × C₀ + g × turns × (turns−1) / 2</div>');
              api.callout(out, multi > single ? 'Here the multi-agent setup uses more input tokens. Shrinking the shared base context C₀ or lengthening each worker\'s run changes that: try it.' : 'Here splitting wins on input tokens because each worker\'s context stays short, while one long agent re-reads an ever-growing history. This ignores output tokens, caching, retries and quality differences.', 'hedge');
            }
            api.slider(c, { label: 'Agents N', min: 1, max: 10, value: st.n, format: function (v) { return v; }, onInput: function (v) { st.n = v; draw(); }, init: false });
            api.slider(c, { label: 'Turns / worker', min: 1, max: 20, value: st.tw, format: function (v) { return v; }, onInput: function (v) { st.tw = v; draw(); }, init: false });
            api.slider(c, { label: 'Base context C₀', min: 500, max: 20000, step: 500, value: st.c0, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.c0 = v; draw(); }, init: false });
            api.slider(c, { label: 'Growth g / turn', min: 100, max: 3000, step: 100, value: st.g, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.g = v; draw(); }, init: false });
            api.slider(c, { label: 'Report size', min: 100, max: 2000, step: 100, value: st.r, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.r = v; draw(); }, init: false });
            el.appendChild(c); el.appendChild(out); draw();
            api.illustrative(el, 'A simplified model: every worker repeats the same base context C₀, grows linearly, and the orchestrator reads N reports. Real costs also include output tokens and caching discounts.');
          } },
          { name: 'Worth it?', render: function (el, api) {
            var g = api.grid(el, 260);
            api.card(g, 'Tends to help when', '<ul><li>Subtasks are <b>independent</b> and can run in parallel</li><li>Work exceeds one context window</li><li>Different roles need different tools or permissions</li><li>Value of the answer justifies several times the tokens</li></ul>');
            api.card(g, 'Tends to hurt when', '<ul><li>Agents need <b>the same shared context</b> (copying it N times)</li><li>Steps are tightly coupled or sequential</li><li>Coordination text outweighs the work</li><li>Errors compound across hand-offs (' + api.link('safety-eval/failure-modes', 'pⁿ') + ')</li></ul>');
            api.callout(el, 'Hedge: evidence is still emerging and task-dependent. A well-built single agent with good tools often matches a crowd of agents; add agents when an <b>eval</b> (' + api.link('safety-eval/evals', 'evals') + ') shows a gain that pays for the extra cost.', 'hedge');
            api.html(el, '<p class="muted">Framework patterns (supervisor, handoffs): ' + api.link('langgraph/multi-agent-patterns', 'multi-agent in LangGraph') + '.</p>');
          } }
        ]
      },

      // ================================================================ protocols
      {
        id: 'protocols', title: 'Protocols: MCP, agent-to-agent (A2A) and skills',
        critical: {
          understand: [
            '<b>MCP</b> (Model Context Protocol) standardises how an agent app connects to <b>tools and data</b> servers: discover tools, call them, read resources.',
            '<b>A2A</b> (agent-to-agent) protocols aim to let <b>agents talk to other agents</b> across vendors/frameworks; the space is young and still shifting.',
            '<b>Skills</b> are packaged instructions (plus files/scripts) loaded on demand: they change <i>what the model knows how to do</i>, not what it can connect to.'
          ],
          details: [
            'All three end up as <b>text in the context</b>: tool schemas, tool results, skill instructions. None of them changes the model weights.',
            'MCP messages use JSON-RPC 2.0; check the current spec for transports, auth and versions.'
          ]
        },
        dataTypes: [
          { name: 'tools/list response', type: 'json', shape: 'list[object]', example: 'names + JSON schemas' },
          { name: 'tools/call request', type: 'json', shape: 'object', example: '{method, params}' },
          { name: 'tool result', type: 'json', shape: 'object', example: 'content blocks' },
          { name: 'skill', type: 'text', shape: 'SKILL.md + files', example: 'instructions loaded on demand' },
          { name: 'agent card', type: 'json', shape: 'object', example: 'describes an agent\'s skills/endpoint' }
        ],
        levels: [
          { name: 'How they relate', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ id: 'host', label: 'Agent app (host + harness)', sub: 'runs the loop', type: 'function', zoom: 1 }],
              [{ id: 'mcp', label: 'MCP', sub: 'agent ↔ tools / data', type: 'json', shape: 'JSON-RPC', goto: 'tool-calling/mcp', color: '#4ea8ff' },
               { id: 'a2a', label: 'A2A', sub: 'agent ↔ agent', type: 'json', shape: 'messages / tasks', zoom: 2, color: '#ff9a4d' },
               { id: 'skill', label: 'Skills', sub: 'instructions on demand', type: 'text', shape: 'SKILL.md', goto: 'skills/skill-anatomy', color: '#5ad18a' }]
            ], caption: 'Click MCP / Skills to jump to their sections. Tools: ' + api.link('tool-calling/tool-definition', 'tool definitions') + '.' });
            api.html(el, '<p>Mental model: <b>MCP = the plug</b> (what the agent can reach), <b>skills = the know-how</b> (how to do a kind of task), <b>A2A = the phone line</b> between independent agents.</p>');
          } },
          { name: 'Comparison', render: function (el, api) {
            api.table(el, { headers: ['', 'MCP', 'A2A', 'Skills'], rows: [
              ['Connects', 'Agent ↔ tool/data server', 'Agent ↔ another agent', 'Agent ↔ packaged instructions'],
              ['Typical content', 'Tool schemas, calls, resources, prompts', 'Agent descriptions, tasks, messages', 'SKILL.md, scripts, reference files'],
              ['Lives in context as', 'Tool definitions + results', 'Messages from a remote agent', 'Instructions loaded when relevant'],
              ['Trust concern', 'Untrusted results; server permissions', 'Remote agent may be wrong or hostile', 'Skill content is instructions: review it'],
              ['Maturity (as of writing)', 'Widely adopted, evolving', 'Early; several proposals', 'Emerging convention']
            ] });
            api.html(el, '<p>Deep dives: ' + api.link('tool-calling/mcp', 'MCP intro') + ' · ' + api.link('skills/skill-anatomy', 'skill anatomy') + ' · ' + api.link('skills/comparison', 'skills vs tools vs MCP') + ' · ' + api.link('safety-eval/prompt-injection', 'security') + '.</p>');
            api.docsNote(el, 'Protocol names, versions and adoption change quickly; verify against the current specs.');
          } },
          { name: 'Payloads', render: function (el, api) {
            var cur = 'mcp', out = api.h('div'), tabs = api.h('div', { class: 'planning-multiagent-tabs' });
            var P = {
              mcp: ['MCP: discover then call (JSON-RPC 2.0)', [
                '// client → server\n{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n\n// server → client\n{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"get_weather","description":"Weather for a city","inputSchema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}}',
                '// client → server (after the model emitted a matching tool call)\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Paris"}}}\n\n// server → client\n{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"Rain, 11°C"}],"isError":false}}'].join('\n\n'), 'json'],
              a2a: ['A2A: describe an agent, then send it a task (shape is illustrative)', '// an "agent card" published by a remote agent (field names vary by spec version)\n{"name":"travel-agent","description":"Plans trips","url":"https://agents.example.com/travel","skills":[{"id":"plan_trip","description":"Itinerary for a city"}]}\n\n// a task message sent to it\n{"method":"message/send","params":{"message":{"role":"user","parts":[{"text":"3 days in Paris, vegetarian"}]}}}', 'json'],
              skill: ['Skill: a folder the agent loads on demand', '---\nname: restaurant-booking\ndescription: Use when the user wants to book a table; covers confirmation rules.\n---\n# Instructions\n1. Search restaurants, check the weather for outdoor seating.\n2. Ask for explicit approval before calling book_table.\n(Only name + description sit in context up front; the body loads when the skill is chosen.)', 'markdown']
            };
            function draw() {
              out.innerHTML = '';
              [].forEach.call(tabs.children, function (b) { b.classList.toggle('on', b.getAttribute('data-k') === cur); });
              api.html(out, '<p><b>' + P[cur][0] + '</b> ' + api.badge(cur === 'skill' ? 'text' : 'json', cur === 'skill' ? 'string' : 'object') + '</p>');
              api.code(out, P[cur][1], P[cur][2]);
            }
            [['mcp', 'MCP'], ['a2a', 'A2A'], ['skill', 'Skill']].forEach(function (k) { var b = api.button(tabs, k[1], function () { cur = k[0]; draw(); }); b.setAttribute('data-k', k[0]); });
            el.appendChild(tabs); el.appendChild(out); draw();
            api.docsNote(el, 'These payloads are simplified sketches. Exact field names and methods differ by spec version; the A2A example in particular is schematic.');
            api.illustrative(el, 'Example agent, URL and weather result are invented.');
          } }
        ]
      }
    ]
  });
})();
