(function () {
  'use strict';

  var SKILL_MD = [
    '---',
    'name: expense-policy',
    'description: Check expense reports against the company policy and draft',
    '  approval notes. Use when the user mentions expenses, receipts,',
    '  reimbursement, per-diem limits or travel claims.',
    '---',
    '',
    '# Expense policy checker',
    '',
    '## Steps',
    '1. Read the report (CSV or PDF) the user provides.',
    '2. For every line, look up the limit in `references/limits.md`.',
    '3. Run `python scripts/check.py report.csv` to flag violations.',
    '4. Write a short approval note using `assets/note-template.md`.',
    '',
    '## Rules',
    '- Never approve a line without a receipt.',
    '- If a limit is ambiguous, ask the user instead of guessing.'
  ].join('\n');

  // illustrative token sizes
  var TOK = { meta: 55, body: 1800, extra: 3500, window: 200000 };

  var LIB = [
    { name: 'expense-policy', desc: 'Check expense reports against company policy; use for receipts, reimbursement, per-diem.', kw: ['expense', 'receipt', 'reimburse', 'policy', 'per-diem', 'travel', 'claim'] },
    { name: 'pdf-report', desc: 'Build a formatted PDF report from data; use when asked for a PDF, report or summary document.', kw: ['pdf', 'report', 'document', 'summary', 'export'] },
    { name: 'sql-helper', desc: 'Write and review SQL against our warehouse schema; use for queries, joins, slow queries.', kw: ['sql', 'query', 'join', 'database', 'table', 'warehouse'] },
    { name: 'brand-voice', desc: 'Rewrite text in the company brand voice; use for marketing copy, emails, announcements.', kw: ['rewrite', 'brand', 'marketing', 'copy', 'email', 'tone', 'announcement'] },
    { name: 'code-review', desc: 'Review a diff for bugs and style using our checklist; use for pull requests and reviews.', kw: ['review', 'diff', 'pull', 'bug', 'style', 'pr', 'code'] }
  ];

  var REQUESTS = [
    { text: 'Here is my Berlin trip receipts, can you check them against our per-diem limits?', best: 'expense-policy' },
    { text: 'Turn this sales table into a PDF report for the board.', best: 'pdf-report' },
    { text: 'This query takes 40 s, can you speed up the join?', best: 'sql-helper' },
    { text: 'What is the capital of Australia?', best: null },
    { text: 'Rewrite this launch announcement so it sounds like us.', best: 'brand-voice' }
  ];

  function words(s) { return s.toLowerCase().replace(/[^a-z0-9\- ]/g, ' ').split(/\s+/).filter(Boolean); }
  function score(req, skill) {
    var w = words(req), s = 0;
    skill.kw.forEach(function (k) { w.forEach(function (x) { if (x === k || (x.length > 3 && k.indexOf(x) === 0) || (k.length > 3 && x.indexOf(k) === 0)) s++; }); });
    return s;
  }

  var DEF = {
    id: 'skills', part: 'II', order: 143, title: 'Agent Skills (SKILL.md)',
    slides: [

      // ------------------------------------------------------------ anatomy
      {
        id: 'skill-anatomy', title: 'Anatomy of a skill: a folder with a SKILL.md',
        critical: {
          understand: ['A skill is just a <b>folder</b>: a <code>SKILL.md</code> file plus optional scripts, reference docs and assets.',
            'The frontmatter (<code>name</code>, <code>description</code>) is what the model sees up front; the body is instructions loaded later.'],
          details: ['Skills package <b>know-how as files</b>, not as model weights: you can edit, version and share them like code.',
            'Exact frontmatter fields and folder conventions vary by product and version; check the current docs.']
        },
        dataTypes: [
          { name: 'SKILL.md', type: 'text', shape: 'string (markdown + YAML)', example: '~ 2 KB' },
          { name: 'name', type: 'text', shape: 'string', example: '"expense-policy"' },
          { name: 'description', type: 'text', shape: 'string', example: 'what + when to use' },
          { name: 'scripts/check.py', type: 'function', shape: 'executable file', example: 'run by a tool' },
          { name: 'references/, assets/', type: 'text', shape: 'files', example: 'loaded on demand' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.html(el, '<p>A <b>skill</b> teaches the agent a repeatable procedure: "how we check expenses", "how we build our PDF reports". It lives in a folder.</p>');
            api.blocks(el, { rows: [[
              { id: 'f', label: 'skill folder', sub: 'expense-policy/', type: 'text', zoom: true },
              { id: 'm', label: 'SKILL.md', sub: 'frontmatter + instructions', type: 'text', zoom: true }
            ], [
              { id: 'x', label: 'scripts / references / assets', sub: 'optional extras', type: 'function', zoom: 2 }
            ]], caption: 'Click a block to zoom into it.' });
            api.docsNote(el, 'Skill file formats and loading rules differ between products.');
          } },
          { name: 'SKILL.md file', render: function (el, api) {
            api.html(el, '<p>Two parts: <b>YAML frontmatter</b> (metadata the model always sees) and a <b>markdown body</b> (instructions, read only when the skill is used).</p>');
            api.code(el, SKILL_MD, 'markdown');
            var g = api.grid(el, 240);
            api.card(g, 'Frontmatter ' + api.type('text'), '<code>name</code> is a short id. <code>description</code> says <b>what it does and when to use it</b>: this is the only text used for matching.');
            api.card(g, 'Body ' + api.type('text'), 'Plain markdown: steps, rules, pointers to other files. It can reference <code>scripts/check.py</code> by path.');
            api.docsNote(el);
          } },
          { name: 'Folder contents', render: function (el, api) {
            var files = {
              'SKILL.md': ['text', 'Entry point. Keep it short; point at the other files.'],
              'scripts/check.py': ['function', 'Deterministic code. The agent runs it with a shell/code tool; only its <i>output</i> enters the context, not its source.'],
              'references/limits.md': ['text', 'Long reference tables (e.g. per-diem limits per country). Read only when a line item needs it.'],
              'assets/note-template.md': ['text', 'Templates or static files the output should reuse.']
            };
            var tree = api.h('div', { class: 'skills-tree' });
            var info = api.h('div', { class: 'card' });
            var lines = [['expense-policy/', null, 0], ['SKILL.md', 'SKILL.md', 1], ['scripts/check.py', 'scripts/check.py', 1], ['references/limits.md', 'references/limits.md', 1], ['assets/note-template.md', 'assets/note-template.md', 1]];
            function sel(k) {
              Array.prototype.forEach.call(tree.children, function (c) { c.classList.toggle('on', c.getAttribute('data-k') === k); });
              info.innerHTML = '<b>' + api.esc(k) + '</b> ' + api.type(files[k][0]) + '<p>' + files[k][1] + '</p>';
            }
            lines.forEach(function (l) {
              var d = api.h('span', { class: 'skills-f', 'data-k': l[1] || '', html: (l[2] ? '&nbsp;&nbsp;├─ ' : '📁 ') + api.esc(l[0]) });
              if (l[1]) d.onclick = function () { sel(l[1]); };
              tree.appendChild(d);
            });
            el.appendChild(tree); el.appendChild(info);
            sel('SKILL.md');
            api.html(el, '<p class="muted">Click a file. Which of these are read into the context, and when, is the next slide: ' + api.link('skills/progressive-disclosure', 'progressive disclosure') + '.</p>');
            api.docsNote(el);
          } }
        ]
      },

      // ------------------------------------------------------------ progressive disclosure
      {
        id: 'progressive-disclosure', title: 'Progressive disclosure: pay tokens only when needed',
        critical: {
          understand: ['<b>Level 1</b>: only name + description of every installed skill sit in the context (tens of tokens each).',
            '<b>Level 2</b>: when a task matches, the full SKILL.md is read. <b>Level 3</b>: extra files or scripts are touched only if the steps call for them.'],
          details: ['Context is a limited, costly budget: longer prompts cost more and tend to degrade accuracy (see the context window slide).',
            'That is why you can install dozens of skills without drowning the model: the unused ones cost almost nothing.']
        },
        dataTypes: [
          { name: 'skill metadata', type: 'text', shape: 'N × (name + description)', example: '≈ 55 tokens each (illustrative)' },
          { name: 'SKILL.md body', type: 'text', shape: 'string', example: '≈ 1,800 tokens (illustrative)' },
          { name: 'extra file / script output', type: 'text', shape: 'string', example: '≈ 3,500 tokens (illustrative)' },
          { name: 'tokens used', type: 'integer', shape: 'scalar', example: 'computed below' },
          { name: 'N installed skills', type: 'integer', shape: 'scalar', example: '1 to 60' }
        ],
        levels: [
          { name: 'Three levels', render: function (el, api) {
            api.html(el, '<p>Think of a library: the <b>catalogue cards</b> are always on the desk; you fetch a <b>book</b> only for a question, and a <b>chapter appendix</b> only if needed.</p>');
            api.blocks(el, { rows: [[
              { id: 'l1', label: 'Level 1: metadata', sub: 'name + description, always in context', type: 'text', shape: '~55 tok × N', zoom: true }
            ], [
              { id: 'l2', label: 'Level 2: SKILL.md', sub: 'read when the task matches', type: 'text', shape: '~1.8k tok', zoom: true }
            ], [
              { id: 'l3', label: 'Level 3: files & scripts', sub: 'read / run only if needed', type: ['text', 'function'], shape: 'on demand', zoom: true }
            ]] });
            api.illustrative(el, 'Token counts are round illustrative numbers (≈ 55 / 1,800 / 3,500); real sizes depend on your files and tokenizer.');
            api.docsNote(el);
          } },
          { name: 'Token gauge', render: function (el, api) {
            var N = 20, bodies = 0, extras = 0;
            var bar = api.h('div', { class: 'controls' });
            el.appendChild(bar);
            var stat = api.h('div', { class: 'skills-stat' });
            var gauge = api.h('div', { class: 'skills-gauge' });
            var ctx = api.h('div', { class: 'skills-ctx' });
            var note = api.h('div', { class: 'muted' });
            var g;
            api.slider(bar, { label: 'Installed skills', min: 1, max: 60, step: 1, value: N, format: function (v) { return String(v); }, onInput: function (v) { N = v; if (bodies > N) bodies = N; draw(); } });
            api.button(bar, 'Task matches a skill (+SKILL.md)', function () { if (bodies < N) bodies++; draw(); });
            api.button(bar, 'It needs a reference file (+extra)', function () { if (bodies === 0) bodies = 1; extras++; draw(); });
            api.button(bar, 'Reset', function () { bodies = 0; extras = 0; draw(); });
            el.appendChild(stat); el.appendChild(ctx); el.appendChild(gauge); el.appendChild(note);
            g = api.bars(gauge, { items: [], format: function (v) { return api.fmtInt(v) + ' tok'; } });
            function draw() {
              if (!g) return;
              var prog = N * TOK.meta + bodies * TOK.body + extras * TOK.extra;
              var eager = N * (TOK.meta + TOK.body);
              var all = N * (TOK.meta + TOK.body + TOK.extra);
              stat.innerHTML = '<div><div class="big-num">' + api.fmtInt(prog) + '</div><div class="muted">tokens in context now</div></div>' +
                '<div><div class="big-num">' + (100 * prog / TOK.window).toFixed(1) + '%</div><div class="muted">of a ' + api.fmtInt(TOK.window) + '-token window</div></div>' +
                '<div><div class="big-num">' + Math.round(100 * (1 - prog / eager)) + '%</div><div class="muted">saved vs loading all SKILL.md files</div></div>';
              var s = '<span>system + user prompt</span><span>' + N + ' × (name + description)</span>';
              for (var i = 0; i < bodies; i++) s += '<span class="b">SKILL.md #' + (i + 1) + '</span>';
              for (var j = 0; j < extras; j++) s += '<span class="x">extra file #' + (j + 1) + '</span>';
              ctx.innerHTML = s;
              g.update([
                { label: 'Progressive (now)', value: prog, color: '#5ad18a', highlight: true },
                { label: 'All SKILL.md up front', value: eager, color: '#ff9a4d' },
                { label: 'Everything up front', value: all, color: '#ff5d73' }
              ]);
              note.innerHTML = 'Level 1 alone costs ' + api.fmtInt(N * TOK.meta) + ' tokens for ' + N + ' skills; loading every SKILL.md eagerly would cost ' + api.fmtInt(eager) + '.';
            }
            draw();
            api.illustrative(el, 'Sizes: metadata 55, SKILL.md 1,800, extra file 3,500 tokens per item. The window size is a round example.');
          } },
          { name: 'The arithmetic', render: function (el, api) {
            var N = 20, k = 2, e = 1;
            var prog = N * TOK.meta + k * TOK.body + e * TOK.extra;
            var eager = N * (TOK.meta + TOK.body);
            api.html(el, '<p>Worked example: ' + N + ' skills installed, the task needs ' + k + ' SKILL.md files and ' + e + ' extra file.</p>');
            api.html(el, '<div class="formula">progressive = N·meta + k·body + e·extra\n            = ' + N + '·' + TOK.meta + ' + ' + k + '·' + TOK.body + ' + ' + e + '·' + TOK.extra + '\n            = ' + api.fmtInt(prog) + ' tokens\n\neager     = N·(meta + body) = ' + N + '·' + (TOK.meta + TOK.body) + ' = ' + api.fmtInt(eager) + ' tokens\nsaving    = ' + api.fmtInt(eager - prog) + ' tokens (' + Math.round(100 * (1 - prog / eager)) + '%)</div>');
            api.html(el, '<p>The gap widens with N: the progressive cost grows by only ' + TOK.meta + ' tokens per extra installed skill, the eager cost by ' + api.fmtInt(TOK.meta + TOK.body) + '. Long contexts also tend to make the model less accurate and requests slower and more expensive (see ' + api.link('inference/context-window', 'context window') + ').</p>');
            api.illustrative(el);
          } }
        ]
      },

      // ------------------------------------------------------------ selection
      {
        id: 'skill-selection', title: 'How the model picks a skill',
        critical: {
          understand: ['There is no special skill router: the descriptions are <b>text in the context</b>, and the model predicts the next tokens, which may be a tool call that reads a skill file.',
            'It is the same loop as tool calling: model emits a call, the harness runs it, the result is appended.'],
          details: ['Matching quality depends on the description wording; vague descriptions lead to missed or wrong skills.',
            'The word-overlap scorer in the demo is a toy; a real model weighs meaning through attention, not keyword counts.']
        },
        dataTypes: [
          { name: 'user request', type: 'text', shape: 'string', example: '"check my receipts"' },
          { name: 'skill descriptions', type: 'text', shape: 'list[str]', example: '5 entries in context' },
          { name: 'tool call (read file)', type: 'json', shape: 'object', example: '{name:"read_file", ...}' },
          { name: 'match score (toy)', type: 'float', shape: 'scalar', example: 'keyword overlap' },
          { name: 'SKILL.md text', type: 'text', shape: 'string', example: 'appended as tool result' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.html(el, '<p>The agent harness puts skill <b>names and descriptions</b> into the system prompt. When a request arrives, the model decides: answer directly, or read a skill first.</p>');
            api.blocks(el, { rows: [[
              { label: 'system prompt', sub: 'tools + skill descriptions', type: 'text', zoom: true },
              { label: 'user request', type: 'text', shape: 'string' }
            ], [
              { label: 'LLM', sub: 'next-token prediction', type: 'function' },
              { label: 'read_file call', type: 'json', edge: 'parsed by harness' }
            ], [
              { label: 'SKILL.md → context', type: 'text', zoom: 2 }
            ]] });
            api.html(el, '<p class="muted">Same machinery as ' + api.link('tool-calling/round-trip', 'tool calling') + ': skills add <i>knowledge to load</i>, not a new mechanism.</p>');
            api.docsNote(el);
          } },
          { name: 'Matching demo', render: function (el, api) {
            var cur = 0;
            var bar = api.h('div', { class: 'controls' }); el.appendChild(bar);
            var out = api.h('div'); 
            api.select(bar, { label: 'Request', options: REQUESTS.map(function (r, i) { return { value: i, label: (i + 1) + '. ' + r.text.slice(0, 24) + '…' }; }), value: 0, onChange: function (v) { cur = +v; draw(); } });
            el.appendChild(out);
            api.html(el, '<p class="muted">Installed (Level 1 only, shown to the model):</p>');
            var tb = api.h('div'); el.appendChild(tb);
            api.table(tb, { headers: ['name', 'description'], rows: LIB.map(function (s) { return ['<code>' + s.name + '</code>', api.esc(s.desc)]; }) });
            function draw() {
              var r = REQUESTS[cur];
              var sc = LIB.map(function (s) { return { s: s, v: score(r.text, s) }; });
              var best = sc.reduce(function (a, b) { return b.v > a.v ? b : a; }, sc[0]);
              var pick = best.v > 0 ? best.s.name : null;
              var html = '<p>Request: <i>' + api.esc(r.text) + '</i></p><p class="muted">Toy score = number of description keywords found in the request:</p>';
              sc.forEach(function (x) { html += '<div class="skills-match' + (x.s.name === pick ? ' best' : '') + '"><span class="n">' + x.s.name + '</span><span>' + '█'.repeat(x.v) + (x.v ? '' : '·') + ' ' + x.v + '</span></div>'; });
              html += pick ? '<p>→ The model would likely emit a call to read <code>' + pick + '/SKILL.md</code>.</p>' : '<p>→ No description fits: the model answers directly, no skill is loaded.</p>';
              out.innerHTML = html;
            }
            draw();
            api.illustrative(el, 'The keyword scorer is a teaching toy. A real model matches by meaning and may also pick two skills or ask a clarifying question.');
          } },
          { name: 'Token sequence', render: function (el, api) {
            api.html(el, '<p>What the model actually sees: one growing token sequence. The tool call is ordinary generated text in a JSON shape.</p>');
            var steps = [
              { title: '1. Context before the request', html: 'System prompt lists tools and skills:<pre class="code skills-pre"><code>Available skills:\n- expense-policy: Check expense reports against company policy; use for receipts, reimbursement, per-diem.\n- pdf-report: Build a formatted PDF report ...\n(... 3 more ...)</code></pre>' },
              { title: '2. User request appended', html: 'User: <i>"Here are my Berlin receipts, check them against our per-diem limits."</i>' },
              { title: '3. Model generates a tool call', html: api.type('json') + ' tokens predicted one at a time:<pre class="code skills-pre"><code>{"name": "read_file",\n "arguments": {"path": "skills/expense-policy/SKILL.md"}}</code></pre>' },
              { title: '4. Harness runs it, appends the result', html: 'The file text (≈ 1,800 tokens, illustrative) becomes new context tokens. Level 2 is now loaded.' },
              { title: '5. Model continues, following the steps', html: 'It may call <code>read_file references/limits.md</code> or a shell tool to run <code>scripts/check.py</code> (Level 3), then writes the final answer.' }
            ];
            api.stepper(el, { steps: steps, interval: 2400 });
            api.docsNote(el, 'Real products use their own tool names and loading mechanism.');
          } }
        ]
      },

      // ------------------------------------------------------------ comparison
      {
        id: 'comparison', title: 'Skills vs tools vs MCP vs prompts vs sub-agents vs fine-tuning',
        critical: {
          understand: ['Each mechanism answers a different question: <b>what to know</b> (skill, prompt), <b>what to do</b> (tool, MCP), <b>who does it</b> (sub-agent), <b>how the model behaves</b> (fine-tuning).',
            'They combine well: a skill can tell the agent which tools to call and in which order.'],
          details: ['Rule of thumb: start with the cheapest, most editable option (prompt → skill → tool) and only fine-tune if that is not enough.',
            'The table is a simplification; real products blur these boundaries.']
        },
        dataTypes: [
          { name: 'mechanism', type: 'text', shape: 'string', example: 'skill, tool, MCP, ...' },
          { name: 'tool', type: 'function', shape: 'name + JSON schema', example: 'get_weather' },
          { name: 'MCP server', type: 'json', shape: 'protocol endpoint', example: 'tools/list' },
          { name: 'scenario', type: 'text', shape: 'string', example: 'chosen below' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.html(el, '<p>Ways to extend an agent, from "just words" to "change the weights":</p>');
            api.blocks(el, { rows: [[
              { label: 'Prompt', sub: 'always-on text', type: 'text' },
              { label: 'Skill', sub: 'loaded on demand', type: 'text', zoom: true }
            ], [
              { label: 'Tool', sub: 'callable function', type: 'function', goto: 'tool-calling/tool-definition' },
              { label: 'MCP', sub: 'standard tool protocol', type: 'json', goto: 'tool-calling/mcp' }
            ], [
              { label: 'Sub-agent', sub: 'own context + loop', type: 'function', goto: 'planning-multiagent/multi-agent' },
              { label: 'Fine-tune', sub: 'changes weights', type: 'matrix', goto: 'training/lora' }
            ]] });
          } },
          { name: 'Comparison table', render: function (el, api) {
            api.table(el, {
              headers: ['Mechanism', 'What it is', 'Cost in context', 'Needs code?', 'Best for'],
              rows: [
                ['System prompt', 'Fixed instructions', 'Always paid', 'No', 'Global rules, persona'],
                ['Skill', 'Folder of instructions + files, loaded on demand', 'Tiny until used', 'Optional scripts', 'Repeatable procedures, domain know-how'],
                ['Tool', 'Function + JSON schema the model can call', 'Schema always paid', 'Yes', 'Actions and fresh data (API, DB)'],
                ['MCP server', 'Standard protocol exposing tools/resources', 'Schemas of exposed tools', 'Yes (server)', 'Reusing integrations across apps'],
                ['Sub-agent', 'Own loop with own context', 'Only its summary returns', 'Orchestration', 'Large or parallel side tasks'],
                ['Fine-tuning', 'Train weights (e.g. LoRA)', 'None at inference', 'Training pipeline', 'Style, format, narrow skills at scale']
              ]
            });
            api.html(el, '<p class="muted">See ' + api.link('planning-multiagent/protocols', 'protocols') + ' for how MCP and skills sit side by side.</p>');
          } },
          { name: 'When to use', render: function (el, api) {
            var recs = [
              { s: 'Every answer must follow a short style rule', r: 'System prompt', why: 'Small, always relevant; no need to load on demand.' },
              { s: 'A 12-step procedure for our monthly close, used a few times a week', r: 'Skill', why: 'Long, only sometimes relevant: progressive disclosure keeps it out of the way.' },
              { s: 'Fetch live order status from our API', r: 'Tool', why: 'Needs a real action and fresh data, which text cannot provide.' },
              { s: 'Ten different apps should all reach the same CRM', r: 'MCP server', why: 'One standard integration reused by many hosts.' },
              { s: 'Research 8 competitors in parallel without bloating my context', r: 'Sub-agents', why: 'Each gets a clean context and returns a summary.' },
              { s: 'A 1M-example dataset, need consistent classification at low latency', r: 'Fine-tuning', why: 'Behaviour learned in weights; no prompt tokens per call.' }
            ];
            var out = api.h('div', { class: 'card' });
            var bar = api.h('div', { class: 'controls' }); el.appendChild(bar);
            api.select(bar, { label: 'Scenario', options: recs.map(function (x, i) { return { value: i, label: (i + 1) + '. ' + x.s.slice(0, 24) + '…' }; }), value: 1, onChange: function (v) { var x = recs[+v]; out.innerHTML = '<p><i>' + api.esc(x.s) + '</i></p><b>Suggested: ' + x.r + '</b><p>' + x.why + '</p>'; }, init: true });
            el.appendChild(out);
            api.html(el, '<p class="muted">These are rules of thumb, and several mechanisms often work together (a skill that calls tools through MCP).</p>');
          } }
        ]
      },

      // ------------------------------------------------------------ authoring & safety
      {
        id: 'authoring-safety', title: 'Authoring good skills, and using them safely',
        critical: {
          understand: ['The <b>description</b> decides whether the skill is ever used: say what it does <i>and when</i>, with the words users would say.',
            'Skill scripts run with the <b>agent\'s permissions</b>. Install only skills you trust, and read them first.'],
          details: ['Keep SKILL.md lean; move long tables and examples into references/ so they load only when needed.',
            'Skill text is instructions the model follows: a malicious skill is like a malicious script or a prompt injection (see the safety section).']
        },
        dataTypes: [
          { name: 'description', type: 'text', shape: 'string', example: '1-3 sentences' },
          { name: 'lint score', type: 'integer', shape: 'scalar', example: '0 to 4' },
          { name: 'script', type: 'function', shape: 'file', example: 'scripts/check.py' },
          { name: 'needs approval?', type: 'boolean', shape: 'scalar', example: 'true for risky actions' }
        ],
        levels: [
          { name: 'Descriptions', render: function (el, api) {
            api.html(el, '<p>Try editing the description. A toy checker looks for four properties of a good one.</p>');
            var out = api.h('div', { class: 'card' });
            api.input(el, { label: 'description:', multiline: true, rows: 3, value: 'Helps with documents.', onInput: function (v) {
              var t = v.toLowerCase();
              var checks = [
                ['Says what it does (verb + object)', /\b(check|build|write|review|convert|extract|draft|generate|rewrite)\b/.test(t)],
                ['Says when to use it ("use when / for")', /\buse (when|for)\b|\bwhen the user\b/.test(t)],
                ['Mentions concrete trigger words (3+ commas or "or")', (t.match(/,| or /g) || []).length >= 2],
                ['Long enough but concise (40 to 300 chars)', v.length >= 40 && v.length <= 300]
              ];
              var n = checks.filter(function (c) { return c[1]; }).length;
              out.innerHTML = '<b>Score ' + n + ' / 4</b> ' + api.type('integer') + '<ul class="skills-ok">' + checks.map(function (c) { return '<li class="' + (c[1] ? 'skills-good' : 'skills-badc') + '">' + (c[1] ? '✔ ' : '✘ ') + c[0] + '</li>'; }).join('') + '</ul>';
            } });
            el.appendChild(out);
            api.html(el, '<p><b>Weak:</b> "Helps with documents." <b>Better:</b> "Build a formatted PDF report from tabular data. Use when the user asks for a PDF, report, board summary or export."</p>');
            api.illustrative(el, 'The checker is a heuristic for teaching, not an official validator.');
            api.docsNote(el);
          } },
          { name: 'Keep it lean', render: function (el, api) {
            var g = api.grid(el, 250);
            api.card(g, '<span class="skills-good">Do</span>', '<ul><li>One skill = one job.</li><li>Steps as a short numbered list.</li><li>Long tables in <code>references/</code>, linked by path.</li><li>Deterministic work in <code>scripts/</code> (output enters context, not the code).</li><li>Say what to do when unsure.</li></ul>');
            api.card(g, '<span class="skills-badc">Avoid</span>', '<ul><li>A 20-page SKILL.md (it is all loaded when used).</li><li>Overlapping descriptions across skills (the model may pick the wrong one).</li><li>Secrets or API keys inside the folder.</li><li>Vague "helps with anything" descriptions.</li></ul>');
            api.html(el, '<p>Test with real requests: does the skill load when it should, and stay out when it should not?</p>');
          } },
          { name: 'Safety', render: function (el, api) {
            api.html(el, '<p>A skill is <b>instructions plus possibly code</b>. Whatever it tells the agent to run, the agent can run, with the permissions you gave the agent.</p>');
            api.blocks(el, { rows: [[
              { label: 'Skill folder', sub: 'from where?', type: 'text' },
              { label: 'Agent', sub: 'follows instructions', type: 'function' },
              { label: 'scripts/', sub: 'run with YOUR permissions', type: 'function', edge: 'shell tool' },
              { label: 'Your files, network, secrets', type: 'json', edge: 'reachable' }
            ]] });
            var g = api.grid(el, 240);
            api.card(g, 'Before installing', '<ul><li>Install only from sources you trust.</li><li>Read SKILL.md and every script.</li><li>Look for network calls, <code>rm</code>, credential access.</li></ul>');
            api.card(g, 'While running', '<ul><li>Least-privilege sandbox and tokens.</li><li>Require approval for risky tool calls (' + api.link('safety-eval/autonomy-levels', 'autonomy levels') + ').</li><li>Treat text inside skills as able to steer the model (' + api.link('safety-eval/prompt-injection', 'prompt injection') + ').</li></ul>');
            api.callout(el, 'Skills make an agent more capable, and also widen what a bad instruction can do. Treat them like code you run.', 'warn');
            api.docsNote(el);
          } }
        ]
      }
    ]
  };
  DEF.slides.forEach(function (sl) { sl.levels.forEach(function (l) { var r = l.render; l.render = function (el, api) { el.style.minWidth = '0'; el.style.maxWidth = '100%'; if (el.parentElement) el.parentElement.style.maxWidth = '100%'; return r(el, api); }; }); });
  Course.register(DEF);
})();
