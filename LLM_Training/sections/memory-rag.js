(function () {
  'use strict';

  // ---------------------------------------------------------------- toy retrieval engine (shared by several levels)
  var DOCS = [
    'Refunds are issued within 14 days of purchase if the item is unused.',
    'To reset your password, click Forgot password on the login page.',
    'Our Paris restaurant is open from noon to 11pm every day except Monday.',
    'The API rate limit is 60 requests per minute per key.',
    'Premium plans include priority support and a 99.9 percent uptime guarantee.',
    'Umbrellas are available at the front desk when it rains in Paris.',
    'Invoices can be downloaded as PDF from the billing page.',
    'Rate limit errors return HTTP 429; retry after the delay in the header.'
  ];
  var STOP = 'a an the of to is are be in on at for from and or if it its your our per when can as by with after every except within'.split(' ');
  var SYN = { rain: 'weather', umbrella: 'weather', raining: 'weather', money: 'refund', return: 'refund', pay: 'billing',
    invoice: 'billing', receipt: 'billing', cost: 'billing', signin: 'login', password: 'login', login: 'login',
    throttle: 'ratelimit', limit: 'ratelimit', '429': 'ratelimit', open: 'hours', hour: 'hours', late: 'hours', noon: 'hours',
    food: 'restaurant', eat: 'restaurant', dinner: 'restaurant', paris: 'restaurant', support: 'help', help: 'help' };

  function words(s) {
    return s.toLowerCase().split(/[^a-z0-9]+/).filter(function (w) { return w && STOP.indexOf(w) < 0; })
      .map(function (w) { return w.length > 3 && w.charAt(w.length - 1) === 's' ? w.slice(0, -1) : w; });
  }
  function features(s) {            // words plus coarse "concept" features: our stand-in for learned semantics
    var f = [];
    words(s).forEach(function (w) { f.push('w:' + w); if (SYN[w]) f.push('c:' + SYN[w]); });
    return f;
  }
  var FEATS = DOCS.map(features), WORDS = DOCS.map(words);
  var VOCAB = [];
  FEATS.forEach(function (fs) { fs.forEach(function (f) { if (VOCAB.indexOf(f) < 0) VOCAB.push(f); }); });
  VOCAB.sort();
  var DF = VOCAB.map(function (v) { return FEATS.filter(function (fs) { return fs.indexOf(v) >= 0; }).length; });
  var IDF = DF.map(function (d) { return Math.log((1 + DOCS.length) / (1 + d)) + 1; });
  function embed(s) {               // text -> vector [|V|] (tf-idf bag of words+concepts)
    var f = features(s), v = VOCAB.map(function () { return 0; });
    f.forEach(function (x) { var i = VOCAB.indexOf(x); if (i >= 0) v[i] += 1; });
    return v.map(function (c, i) { return c * IDF[i]; });
  }
  var DVEC = DOCS.map(embed);
  function rankDense(q, M) {
    var qv = embed(q);
    return DOCS.map(function (d, i) { return { i: i, s: M.cosine(qv, DVEC[i]) }; }).sort(function (a, b) { return b.s - a.s || a.i - b.i; });
  }
  var AVGLEN = WORDS.reduce(function (a, w) { return a + w.length; }, 0) / WORDS.length;
  function rankBM25(q) {
    var qw = words(q), k1 = 1.5, b = 0.75, N = DOCS.length;
    return DOCS.map(function (d, i) {
      var s = 0;
      qw.forEach(function (t) {
        var n = WORDS.filter(function (ws) { return ws.indexOf(t) >= 0; }).length;
        if (!n) return;
        var tf = WORDS[i].filter(function (x) { return x === t; }).length;
        var idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
        s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * WORDS[i].length / AVGLEN));
      });
      return { i: i, s: s };
    }).sort(function (a, b2) { return b2.s - a.s || a.i - b2.i; });
  }
  function rrf(lists, k) {
    var sc = {};
    lists.forEach(function (l) { l.forEach(function (r, rank) { sc[r.i] = (sc[r.i] || 0) + 1 / (k + rank + 1); }); });
    return Object.keys(sc).map(function (i) { return { i: +i, s: sc[i] }; }).sort(function (a, b) { return b.s - a.s || a.i - b.i; });
  }
  function buildPrompt(q, ids) {
    var ctx = ids.map(function (i, n) { return '[' + (n + 1) + '] ' + DOCS[i]; }).join('\n');
    return 'Answer using ONLY the context. Cite like [1].\n\nContext:\n' + ctx + '\n\nQuestion: ' + q + '\nAnswer:';
  }
  function approxTokens(s) { return Math.max(1, Math.round(s.length / 4)); }   // rough rule of thumb, not a tokenizer

  function gauge(el, parts, window_, api, thresholdPct) {
    var box = api.h('div', { class: 'memory-rag-gauge' });
    var bar = api.h('div', { class: 'memory-rag-gauge-bar' });
    var used = 0;
    parts.forEach(function (p) { used += p.v; bar.appendChild(api.h('span', { class: p.cls, style: { width: Math.min(100, p.v / window_ * 100) + '%' }, title: p.name + ': ' + api.fmtInt(p.v) + ' tokens' })); });
    var leg = api.h('div', { class: 'memory-rag-gauge-legend' });
    parts.forEach(function (p) { leg.appendChild(api.h('span', { html: '<i class="' + p.cls + '"></i>' + p.name + ' ' + api.fmtInt(p.v) })); });
    leg.appendChild(api.h('span', { html: '<b>used ' + api.fmtInt(used) + ' / ' + api.fmtInt(window_) + ' (' + api.fmt(used / window_ * 100, 0) + '%)</b>' }));
    box.appendChild(bar);
    if (thresholdPct != null) box.appendChild(api.h('div', { class: 'memory-rag-thresh' }, api.h('b', { style: { left: thresholdPct + '%' }, title: 'compaction trigger' })));
    box.appendChild(leg);
    el.appendChild(box);
    return used;
  }

  Course.register({
    id: 'memory-rag', part: 'II', order: 130, title: 'Memory & RAG',
    slides: [
      // ================================================================ memory-types
      {
        id: 'memory-types', title: 'Memory types: what an agent can "remember"',
        critical: {
          understand: [
            'The model itself is stateless between calls: <b>all</b> "memory" is text that the harness puts back into the context window.',
            'Working memory = the current context. Long-term memory = an <b>external store</b> the agent reads and writes through tools or the harness.',
            'Long-term stores are often split into semantic (facts), episodic (past events) and procedural (how-to) memory, a loose analogy to human memory.'
          ],
          details: [
            'These categories are a design vocabulary, not a standard; products name and implement them differently.',
            'Writing to long-term memory is a side effect: decide what gets saved, and treat recalled text as untrusted input (see ' + '<a href="#" class="xlink" data-goto="safety-eval/prompt-injection">prompt injection</a>).'
          ]
        },
        dataTypes: [
          { name: 'context window', type: 'tensor', shape: '[n_tokens]', example: 'tokens the model sees this turn' },
          { name: 'messages history', type: 'json', shape: 'list[object]', example: '[{role, content}, …]' },
          { name: 'memory record', type: 'json', shape: 'object', example: '{text, kind, created_at, embedding}' },
          { name: 'memory embedding', type: 'vector', shape: '[d_emb]', example: 'for similarity search' },
          { name: 'recalled memory', type: 'text', shape: 'string', example: '"User prefers vegetarian places"' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.html(el, '<p>An LLM call is a pure function of its input tokens. "Memory" is therefore a question of <b>what gets put into the next prompt</b>, and where it is kept in between.</p>');
            api.blocks(el, {
              rows: [
                [{ id: 'ctx', label: 'Context window', sub: 'working memory: system + tools + this conversation', type: 'integer', shape: '[n_tokens]', zoom: 1, color: '#4ea8ff' }],
                [{ id: 'hist', label: 'Conversation store', sub: 'short-term: messages kept by the app between turns', type: 'json', shape: 'list[object]', zoom: 1, color: '#5ad18a' },
                 { id: 'sem', label: 'Semantic', sub: 'facts & preferences', type: 'text', zoom: 1, color: '#ff9a4d' },
                 { id: 'epi', label: 'Episodic', sub: 'past runs & events', type: 'text', zoom: 1, color: '#ff9a4d' },
                 { id: 'proc', label: 'Procedural', sub: 'rules, skills, how-to', type: 'text', goto: 'skills/skill-anatomy', color: '#ff9a4d' }]
              ],
              caption: 'Click a block. Orange blocks are long-term stores that live outside the model (files, a database, a vector index).'
            });
          } },
          { name: 'Where each lives', render: function (el, api) {
            var kinds = {
              ctx: ['In-context (working)', 'Inside the model call: the token sequence the model attends to right now.', 'Lost when the call ends. Bounded by the context window. Costs tokens (and latency) every turn.', 'The system prompt, tool schemas, the last N messages, tool results.'],
              conv: ['Short-term (conversation)', 'In your application: a list of messages (database row, in-process list, session store).', 'Lives as long as the session. Re-sent to the stateless API each turn; compacted when long.', 'The chat transcript for this thread.'],
              sem: ['Long-term: semantic', 'External store: key-value, document store or vector index.', 'Persists across sessions. Retrieved by key or similarity search; written explicitly.', '"User is vegetarian", "Prod DB is Postgres 15".'],
              epi: ['Long-term: episodic', 'External store of past episodes: logs, trajectories, summaries of earlier sessions.', 'Persists; usually retrieved by recency or similarity to the current situation.', '"Last time deploy failed because the migration was not run."'],
              proc: ['Long-term: procedural', 'Instruction files, skills, rules, learned prompts. Often plain files in the repo.', 'Persists; loaded on demand. Changing it changes the agent\'s behaviour without retraining.', 'A CLAUDE.md-style file, a SKILL.md, a runbook.']
            };
            var cur = 'ctx', out = api.h('div');
            var tabs = api.h('div', { class: 'memory-rag-tabs' });
            function draw() {
              out.innerHTML = '';
              var k = kinds[cur];
              api.card(out, '<b>' + k[0] + '</b>', '<p><b>Lives in:</b> ' + k[1] + '</p><p><b>Lifetime / cost:</b> ' + k[2] + '</p><p><b>Example:</b> <i>' + k[3] + '</i></p>');
              [].forEach.call(tabs.children, function (b) { b.classList.toggle('on', b.getAttribute('data-k') === cur); });
            }
            Object.keys(kinds).forEach(function (key) {
              var b = api.button(tabs, kinds[key][0], function () { cur = key; draw(); });
              b.setAttribute('data-k', key);
            });
            el.appendChild(tabs); el.appendChild(out); draw();
            api.callout(el, 'Rule of thumb: <b>only the context window is "seen" by the model</b>. Everything else must be <i>retrieved</i> and pasted into it. Cost grows with what you paste back, see ' + api.link('chat-api/cost-caching', 'cost & caching') + '.', 'key');
          } },
          { name: 'Memory as tools', render: function (el, api) {
            api.html(el, '<p>A common pattern: expose memory as <b>tools</b>, so the model decides when to read and write. The harness executes them like any other tool call (' + api.link('tool-calling/round-trip', 'round trip') + ').</p>');
            api.code(el, '// model output (tokens that parse as JSON)\n{"name": "memory_write", "arguments": {"kind": "semantic", "text": "User is vegetarian"}}\n\n// harness result appended to the context\n{"ok": true, "id": "mem_0042"}\n\n// later, another session\n{"name": "memory_search", "arguments": {"query": "dietary restrictions", "k": 3}}\n// -> [{"id": "mem_0042", "text": "User is vegetarian", "score": 0.81}]', 'json');
            api.illustrative(el, 'Tool names, fields and the score are made up; real memory products differ.');
            api.docsNote(el, 'Memory features of specific products and frameworks change often.');
          } }
        ]
      },

      // ================================================================ context-management
      {
        id: 'context-management', title: 'Context management and compaction',
        critical: {
          understand: [
            'Every turn re-sends the whole history, so the context <b>fills up</b> as tool results and messages accumulate.',
            'Compaction replaces old turns with a short <b>summary</b> and keeps recent turns verbatim, freeing room to keep working.',
            'Compaction is lossy: details not in the summary are gone unless they were saved to external memory first.'
          ],
          details: [
            'Triggers are usually a fraction of the window (e.g. ~80%); the exact policy varies by product.',
            'Alternatives: truncate oldest turns, drop bulky old tool results, or offload to files and re-read on demand.'
          ]
        },
        dataTypes: [
          { name: 'context budget', type: 'integer', shape: 'scalar', example: 'window size in tokens' },
          { name: 'used tokens', type: 'integer', shape: 'scalar', example: 'sum over all context parts' },
          { name: 'summary', type: 'text', shape: 'string', example: 'written by the model itself' },
          { name: 'compaction ratio', type: 'float', shape: 'scalar', example: '≈ 0.1 (illustrative)' },
          { name: 'messages', type: 'json', shape: 'list[object]', example: 'before and after' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'turns', label: 'Turns pile up', sub: 'user + assistant + tool results', type: 'json', shape: 'list[object]' },
              { id: 'full', label: 'Nearly full?', sub: 'used ≥ threshold × window', type: 'boolean', shape: 'scalar', edge: 'harness checks' },
              { id: 'sum', label: 'Summarize old turns', sub: 'a model call writes the summary', type: 'text', shape: 'string', zoom: 1 },
              { id: 'new', label: 'New, shorter context', sub: 'system + summary + recent turns', type: 'integer', shape: '[n_tokens]', zoom: 1 }
            ], caption: 'Click "Summarize old turns" to play with the numbers.' });
          } },
          { name: 'Token gauge', render: function (el, api) {
            var st = { turns: 40, per: 700, win: 32000, keep: 6, ratio: 10 };
            var out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              out.innerHTML = '';
              var sys = 800, tools = 1500, trig = 0.8;
              var turnsTok = st.turns * st.per, before = sys + tools + turnsTok;
              var needs = before >= trig * st.win;
              var keep = Math.min(st.keep, st.turns), old = (st.turns - keep) * st.per;
              var summary = needs ? Math.round(old * st.ratio / 100) : 0;
              var after = needs ? sys + tools + summary + keep * st.per : before;
              api.html(out, '<b>Before compaction</b>');
              gauge(out, [{ name: 'system', v: sys, cls: 'memory-rag-seg-sys' }, { name: 'tool schemas', v: tools, cls: 'memory-rag-seg-tools' }, { name: st.turns + ' turns', v: turnsTok, cls: 'memory-rag-seg-turns' }], st.win, api, trig * 100);
              api.html(out, '<b>After compaction</b> ' + (needs ? '' : '<span class="muted">(not triggered: below ' + api.fmt(trig * 100, 0) + '% of the window)</span>'));
              gauge(out, [{ name: 'system', v: sys, cls: 'memory-rag-seg-sys' }, { name: 'tool schemas', v: tools, cls: 'memory-rag-seg-tools' }, { name: 'summary', v: summary, cls: 'memory-rag-seg-sum' }, { name: keep + ' recent turns', v: needs ? keep * st.per : turnsTok, cls: 'memory-rag-seg-turns' }], st.win, api);
              api.html(out, '<div class="formula">after = system + tools + summary + recent = ' + api.fmtInt(sys) + ' + ' + api.fmtInt(tools) + ' + ' + api.fmtInt(summary) + ' + ' + api.fmtInt(needs ? keep * st.per : turnsTok) + ' = <b>' + api.fmtInt(after) + '</b> tokens (was ' + api.fmtInt(before) + ')</div>');
              if (before > st.win) api.callout(out, 'Without compaction this request would <b>exceed the window</b> and fail or be truncated.', 'warn');
              api.illustrative(out, 'System/tool sizes, tokens per turn and the summary ratio are assumptions; real summaries vary.');
            }
            api.slider(c, { label: 'Turns so far', min: 1, max: 120, value: st.turns, format: function (v) { return v; }, onInput: function (v) { st.turns = v; draw(); }, init: false });
            api.slider(c, { label: 'Tokens / turn', min: 200, max: 3000, step: 100, value: st.per, format: function (v) { return api.fmtInt(v); }, onInput: function (v) { st.per = v; draw(); }, init: false });
            api.select(c, { label: 'Window', options: [{ value: 8192, label: '8k' }, { value: 32000, label: '32k' }, { value: 128000, label: '128k' }, { value: 200000, label: '200k' }], value: st.win, onChange: function (v) { st.win = +v; draw(); } });
            api.slider(c, { label: 'Keep verbatim', min: 2, max: 20, value: st.keep, format: function (v) { return v + ' turns'; }, onInput: function (v) { st.keep = v; draw(); }, init: false });
            api.slider(c, { label: 'Summary size', min: 3, max: 30, value: st.ratio, format: function (v) { return v + '% of old'; }, onInput: function (v) { st.ratio = v; draw(); }, init: false });
            el.appendChild(c); el.appendChild(out); draw();
          } },
          { name: 'Messages', render: function (el, api) {
            api.html(el, '<p>What the harness actually does to the <code>messages</code> list (' + api.badge('json', 'list[object]') + '):</p>');
            var g = api.grid(el, 260);
            api.card(g, 'Before (' + 14 + ' messages)', '<div class="memory-rag-vec">user: "fix failing test"<br>assistant: tool_call run_tests<br>tool: 212 lines of output<br>assistant: tool_call read_file<br>tool: 340 lines of source<br>… 8 more …<br>assistant: "patching foo.py"<br>user: "also add a changelog line"</div>');
            api.card(g, 'After (4 messages)', '<div class="memory-rag-vec">system: …<br><b>user: [summary] Goal: fix failing test_parse. Done: found off-by-one in parse(); ran tests (2 fail). Open: patch foo.py. Constraints: no API changes.</b><br>assistant: "patching foo.py"<br>user: "also add a changelog line"</div>');
            api.callout(el, 'The summary is written by a model call, so it can omit or distort details. Good practice: have the agent write key facts (decisions, file paths, TODOs) to a file or memory store <i>before</i> compaction.', 'hedge');
            api.html(el, '<p class="muted">Related: ' + api.link('inference/context-window', 'context windows and "lost in the middle"') + ', ' + api.link('chat-api/cost-caching', 'prompt caching') + '.</p>');
          } }
        ]
      },

      // ================================================================ rag-pipeline
      {
        id: 'rag-pipeline', title: 'RAG pipeline: from documents to an augmented prompt',
        critical: {
          understand: [
            'RAG = <b>R</b>etrieval-<b>A</b>ugmented <b>G</b>eneration: look up relevant text first, paste it into the prompt, then let the model answer.',
            'Documents are <b>chunked</b>, each chunk is <b>embedded</b> into a vector, and vectors go into an index. A query is embedded the same way and compared by cosine similarity.',
            'The model never "queries the database": it only sees the retrieved text as ordinary tokens in the prompt.'
          ],
          details: [
            'Indexing happens ahead of time (offline); only the query embedding and the top-k search happen at question time.',
            'The query and the chunks must use the <b>same</b> embedding model, or the vectors are not comparable.',
            'Chunk size and overlap are tuning knobs; there is no universally best value.'
          ]
        },
        dataTypes: [
          { name: 'document', type: 'text', shape: 'string', example: 'a page, a PDF, a ticket' },
          { name: 'chunk', type: 'text', shape: 'string', example: '≈ 200–800 tokens (typical, tunable)' },
          { name: 'chunk embedding', type: 'vector', shape: '[d_emb]', example: 'e.g. d_emb = 384…3072' },
          { name: 'vector index', type: 'matrix', shape: '[n_chunks × d_emb]', example: 'rows = chunks' },
          { name: 'similarity scores', type: 'float', shape: '[n_chunks]', example: 'cosine ∈ [−1, 1]' },
          { name: 'top-k indices', type: 'integer', shape: '[k]', example: '[5, 0, 3]' },
          { name: 'augmented prompt', type: 'text', shape: 'string', example: 'instructions + context + question' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ id: 'docs', label: 'Documents', type: 'text', shape: 'list[str]', sub: 'offline: indexing' },
               { id: 'chunk', label: 'Chunk', type: 'text', shape: 'list[str]', zoom: 1, edge: 'split' },
               { id: 'emb', label: 'Embed', type: 'vector', shape: '[n_chunks × d_emb]', zoom: 1, edge: 'embedding model' },
               { id: 'db', label: 'Vector DB', type: 'matrix', shape: '[n_chunks × d_emb]', zoom: 1, edge: 'store' }],
              [{ id: 'q', label: 'Query', type: 'text', shape: 'string', sub: 'online: per question' },
               { id: 'qe', label: 'Embed query', type: 'vector', shape: '[d_emb]', edge: 'same model' },
               { id: 'topk', label: 'Top-k cosine', type: ['float', 'integer'], shape: '[n_chunks] → [k]', zoom: 2, edge: 'compare' },
               { id: 'prompt', label: 'Augmented prompt', type: 'text', shape: 'string', zoom: 2, edge: 'paste' },
               { id: 'llm', label: 'LLM', type: 'token', shape: '[n_tokens]', goto: 'big-picture/next-token', edge: 'generate' }]
            ], caption: 'Top row runs ahead of time; bottom row runs for every question. Click a block to dive in.' });
          } },
          { name: 'Stage by stage', render: function (el, api) {
            var doc = 'Refunds are issued within 14 days of purchase if the item is unused. To start a refund, open the order page and press Return. Refunds go back to the original payment method.';
            var sents = doc.split(/(?<=\.)\s+/);
            var chunks = [sents[0] + ' ' + sents[1], sents[1] + ' ' + sents[2]];
            api.stepper(el, { interval: 2400, steps: [
              { title: 'Chunk: split text into overlapping pieces', render: function (b) {
                api.html(b, '<p>Input ' + api.badge('text', 'string') + ' → output ' + api.badge('text', 'list[str]') + '. Overlap keeps a sentence from being cut off from its context.</p>');
                chunks.forEach(function (c, i) { api.html(b, '<div class="memory-rag-chunk"><b>chunk ' + i + '</b> ' + api.esc(c) + '</div>'); });
                api.html(b, '<p class="muted">Here the "overlap" is the middle sentence, which appears in both chunks.</p>');
              } },
              { title: 'Embed: each chunk becomes a vector', render: function (b) {
                api.html(b, '<p>An embedding model maps ' + api.badge('text', 'string') + ' → ' + api.badge('vector', '[d_emb]') + '. Similar meaning ⇒ nearby vectors (see ' + api.link('embeddings/geometry', 'embedding geometry') + ').</p>');
                chunks.forEach(function (c, i) { var v = embed(c); api.html(b, '<div class="memory-rag-vec">chunk ' + i + ' → [' + v.slice(0, 6).map(function (x) { return api.fmt(x, 2); }).join(', ') + ', … ] (' + v.length + ' dims in this toy)</div>'); });
                api.illustrative(b, 'This toy embedding is a bag of words, not a neural model; real d_emb is typically hundreds to a few thousand.');
              } },
              { title: 'Store: vectors + original text go into an index', render: function (b) {
                api.html(b, '<p>A vector DB keeps a ' + api.badge('matrix', '[n_chunks × d_emb]') + ' plus the chunk text and metadata (source, page, date). Large indexes use approximate nearest-neighbour search (ANN) instead of comparing against every row.</p>');
                api.code(b, '{"id": 17, "text": "Refunds are issued …", "embedding": [0.12, -0.03, …], "metadata": {"source": "policy.pdf", "page": 3}}', 'json');
              } },
              { title: 'Retrieve: embed the query, score, take top-k', render: function (b) {
                var q = 'how do I get my money back', r = rankDense(q, api.math).slice(0, 3);
                api.html(b, '<p>Query ' + api.badge('text', 'string') + ' <i>"' + q + '"</i> → vector ' + api.badge('vector', '[d_emb]') + ' → cosine against every chunk → ' + api.badge('float', '[n_chunks]') + ' → top-k ' + api.badge('integer', '[k]') + '.</p>');
                api.bars(b, { items: r.map(function (x) { return { label: 'doc ' + x.i, value: x.s, note: api.esc(DOCS[x.i].slice(0, 36)) + '…' }; }), max: 1, format: function (v) { return api.fmt(v, 3); } });
              } },
              { title: 'Augment & generate: paste text into the prompt', render: function (b) {
                api.html(b, '<p>The retrieved ' + api.badge('text', 'string') + ' becomes ordinary prompt tokens (' + api.link('chat-api/chat-template', 'one token sequence') + ').</p>');
                api.html(b, '<div class="memory-rag-prompt">' + api.esc(buildPrompt('how do I get my money back', rankDense('how do I get my money back', api.math).slice(0, 2).map(function (x) { return x.i; }))) + '</div>');
              } }
            ] });
          } },
          { name: 'Cosine math', render: function (el, api) {
            var q = 'rate limit 429', qv = embed(q), best = rankDense(q, api.math)[0], dv = DVEC[best.i];
            var idx = []; VOCAB.forEach(function (_, i) { if (qv[i] > 0 && dv[i] > 0) idx.push(i); });
            api.html(el, '<p>Query <i>"' + q + '"</i> vs best chunk (doc ' + best.i + ': <i>' + api.esc(DOCS[best.i]) + '</i>).</p>');
            api.table(el, { headers: ['feature', 'query q [d_emb]', 'chunk c [d_emb]', 'q·c term'],
              rows: idx.map(function (i) { return [VOCAB[i], api.fmt(qv[i], 2), api.fmt(dv[i], 2), api.fmt(qv[i] * dv[i], 3)]; }).concat([['<i>other dims</i>', '…', '…', '…']]) });
            var dot = api.math.dot(qv, dv), nq = api.math.norm(qv), nd = api.math.norm(dv);
            api.html(el, '<div class="formula">cos(q, c) = (q · c) / (‖q‖ · ‖c‖) = ' + api.fmt(dot, 3) + ' / (' + api.fmt(nq, 3) + ' × ' + api.fmt(nd, 3) + ') = <b>' + api.fmt(dot / (nq * nd), 3) + '</b></div>');
            api.html(el, '<p>' + api.badge('float', 'scalar') + ' Repeat for all ' + DOCS.length + ' chunks, sort descending, keep the first <i>k</i> ' + api.badge('integer', '[k]') + '. Many vector DBs normalise vectors so that cosine is just a dot product.</p>');
            api.html(el, '<p class="muted">Compare with ' + api.link('embeddings/cosine', 'cosine step by step') + '. Try your own question in the ' + api.link('memory-rag/mini-rag', 'mini-RAG playground') + '.</p>');
          } }
        ]
      },

      // ================================================================ mini-rag
      {
        id: 'mini-rag', title: 'Mini-RAG playground',
        critical: {
          understand: [
            'Type a question: the page embeds it, scores all 8 tiny documents by cosine similarity, picks the top-k and builds the augmented prompt.',
            'The toy embedding counts words (plus a few hand-made "concept" features), so it only matches vocabulary it knows: real embedding models are learned and far more flexible.'
          ],
          details: [
            'Raising k adds context (more recall) but also more tokens and possibly more noise.',
            'If nothing is relevant, top-k still returns k items: real systems add a score threshold or let the model say "not found".'
          ]
        },
        dataTypes: [
          { name: 'query', type: 'text', shape: 'string', example: 'typed by you' },
          { name: 'query vector', type: 'vector', shape: '[d_emb]', example: 'd_emb = vocabulary size in this toy' },
          { name: 'cosine scores', type: 'float', shape: '[8]', example: '0 … 1 here (no negative weights)' },
          { name: 'k', type: 'integer', shape: 'scalar', example: '1–5' },
          { name: 'retrieved ids', type: 'integer', shape: '[k]', example: '[5, 2]' },
          { name: 'augmented prompt', type: 'text', shape: 'string', example: 'built from top-k' }
        ],
        levels: [
          { name: 'Playground', render: function (el, api) {
            var st = { q: 'will I need an umbrella in Paris?', k: 2 }, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              out.innerHTML = '';
              var r = rankDense(st.q, api.math), top = r.slice(0, st.k).filter(function (x) { return x.s > 0; });
              var g = api.h('div', { class: 'memory-rag-two' }); out.appendChild(g);
              var left = api.h('div'), right = api.h('div'); g.appendChild(left); g.appendChild(right);
              api.html(left, '<b>Scores</b> ' + api.badge('float', '[8]'));
              r.forEach(function (x, rank) {
                var hit = rank < st.k && x.s > 0;
                api.html(left, '<div class="memory-rag-chunk' + (hit ? ' hit' : ' dim') + '"><div class="memory-rag-doc"><span class="memory-rag-id">#' + x.i + '</span><span>' + api.esc(DOCS[x.i]) + '</span></div><span class="memory-rag-score">cos = <b>' + api.fmt(x.s, 3) + '</b>' + (hit ? ' &nbsp;✔ top-' + (rank + 1) : '') + '</span></div>');
              });
              api.html(right, '<b>Augmented prompt</b> ' + api.badge('text', 'string') + ' <span class="muted">≈ ' + api.fmtInt(approxTokens(buildPrompt(st.q, top.map(function (x) { return x.i; })))) + ' tokens (chars/4 estimate)</span>');
              api.html(right, '<div class="memory-rag-prompt">' + api.esc(buildPrompt(st.q, top.map(function (x) { return x.i; }))) + '</div>');
              if (!top.length) api.callout(right, 'No chunk shares any vocabulary with the query: score 0 everywhere. A real system would answer "not found" instead of guessing.', 'hedge');
            }
            api.input(c, { label: 'Question', value: st.q, placeholder: 'e.g. how do I get a refund?', onInput: function (v) { st.q = v; draw(); }, init: false });
            api.slider(c, { label: 'top-k', min: 1, max: 5, value: st.k, format: function (v) { return 'k = ' + v; }, onInput: function (v) { st.k = v; draw(); }, init: false });
            el.appendChild(c); el.appendChild(out); draw();
            api.html(el, '<p class="muted">Try: <i>how do I get my money back</i>, <i>throttle errors</i>, <i>dinner Monday</i>, <i>forgot sign-in</i>.</p>');
            api.illustrative(el, 'Toy bag-of-words embedding computed in your browser; real embeddings are learned vectors.');
          } },
          { name: 'Vectors', render: function (el, api) {
            var st = { q: 'throttle errors per minute', doc: 3 }, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              out.innerHTML = '';
              var qv = embed(st.q), dv = DVEC[st.doc];
              var cols = []; VOCAB.forEach(function (_, i) { if (qv[i] > 0 || dv[i] > 0) cols.push(i); });
              if (!cols.length) { api.callout(out, 'Query has no known words.', 'hedge'); return; }
              api.matrix(out, { data: [cols.map(function (i) { return qv[i]; }), cols.map(function (i) { return dv[i]; })], rowLabels: ['query', 'doc ' + st.doc], colLabels: cols.map(function (i) { return VOCAB[i]; }), decimals: 2, title: 'Non-zero dims only', type: 'vector', shape: '[2 × ' + cols.length + ' of ' + VOCAB.length + ']' });
              api.html(out, '<div class="formula">cos = ' + api.fmt(api.math.dot(qv, dv), 3) + ' / (' + api.fmt(api.math.norm(qv), 3) + ' × ' + api.fmt(api.math.norm(dv), 3) + ') = <b>' + api.fmt(api.math.cosine(qv, dv), 3) + '</b></div>');
              api.html(out, '<p class="muted">Features starting with <code>c:</code> are hand-made concept groups (e.g. throttle ≈ limit ≈ 429) that stand in for what a learned embedding captures automatically.</p>');
            }
            api.input(c, { label: 'Question', value: st.q, onInput: function (v) { st.q = v; draw(); }, init: false });
            api.select(c, { label: 'Compare with', options: DOCS.map(function (d, i) { return { value: i, label: '#' + i + ' ' + d.slice(0, 28) + '…' }; }), value: st.doc, onChange: function (v) { st.doc = +v; draw(); } });
            el.appendChild(c); el.appendChild(out); draw();
          } }
        ]
      },

      // ================================================================ advanced-rag
      {
        id: 'advanced-rag', title: 'Hybrid search, reranking and agentic RAG',
        critical: {
          understand: [
            'Vector search finds <b>meaning</b>, keyword search (BM25) finds <b>exact terms</b> (names, error codes, IDs). Hybrid search combines both.',
            'Reciprocal rank fusion (RRF) merges ranked lists using only <b>ranks</b>, so scores from different systems need no calibration.',
            'A reranker re-scores the top candidates with a heavier model; agentic RAG lets the model call retrieval as a <b>tool</b>, decide when to search, and refine queries.'
          ],
          details: [
            'Typical recipe (not a rule): retrieve ~50 candidates cheaply, rerank to ~5, then prompt.',
            'RRF constant k = 60 is a common default from the original paper; it is tunable.',
            'Agentic RAG can improve hard questions but adds latency, cost and more ways to loop; measure it (see ' + '<a href="#" class="xlink" data-goto="safety-eval/evals">evals</a>).'
          ]
        },
        dataTypes: [
          { name: 'BM25 score', type: 'float', shape: '[n_chunks]', example: 'keyword relevance' },
          { name: 'dense score', type: 'float', shape: '[n_chunks]', example: 'cosine' },
          { name: 'rank', type: 'integer', shape: 'scalar', example: '1 = best' },
          { name: 'RRF score', type: 'float', shape: '[n_chunks]', example: 'Σ 1/(60 + rank)' },
          { name: 'search tool call', type: 'json', shape: 'object', example: '{"name":"search_docs", …}' },
          { name: 'tool result', type: 'json', shape: 'list[object]', example: 'chunks + ids' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [
              [{ id: 'q', label: 'Query', type: 'text', shape: 'string' },
               { id: 'bm', label: 'Keyword (BM25)', type: 'float', shape: '[n_chunks]', zoom: 1, edge: 'exact terms' },
               { id: 'vec', label: 'Vector (cosine)', type: 'float', shape: '[n_chunks]', zoom: 1, edge: 'meaning' },
               { id: 'fuse', label: 'Fuse (RRF)', type: ['integer', 'float'], shape: '[n_chunks]', zoom: 1 },
               { id: 'rr', label: 'Rerank', type: 'float', shape: '[~50] → [k]', sub: 'cross-encoder or LLM', note: 'A heavier model scores (query, chunk) pairs jointly.' }],
              [{ id: 'agent', label: 'Agent loop', type: 'json', shape: 'tool call', sub: 'search_docs as a tool', zoom: 2 },
               { id: 'res', label: 'Tool result', type: 'json', shape: 'list[object]', sub: 'chunks appended to context', goto: 'tool-calling/round-trip' },
               { id: 'ans', label: 'Answer or search again', type: 'text', shape: 'string' }]
            ], caption: 'Top: a better retriever. Bottom: retrieval becomes one tool among others.' });
          } },
          { name: 'Hybrid + RRF', render: function (el, api) {
            var st = { q: 'rate limit 429 error', k: 60 }, out = api.h('div'), c = api.h('div', { class: 'controls' });
            function draw() {
              out.innerHTML = '';
              var bm = rankBM25(st.q).filter(function (x) { return x.s > 0; }), de = rankDense(st.q, api.math).filter(function (x) { return x.s > 0; });
              var fu = rrf([bm, de], st.k).slice(0, 5);
              var rb = {}, rd = {};
              bm.forEach(function (x, r) { rb[x.i] = r + 1; }); de.forEach(function (x, r) { rd[x.i] = r + 1; });
              api.table(out, { headers: ['doc', 'BM25 score', 'BM25 rank', 'cosine', 'dense rank', 'RRF = Σ 1/(' + st.k + '+rank)'],
                rows: fu.map(function (x) {
                  var bs = bm.filter(function (y) { return y.i === x.i; })[0], ds = de.filter(function (y) { return y.i === x.i; })[0];
                  return ['<b>#' + x.i + '</b> ' + api.esc(DOCS[x.i].slice(0, 34)) + '…', bs ? api.fmt(bs.s, 2) : '–', rb[x.i] || '–', ds ? api.fmt(ds.s, 3) : '–', rd[x.i] || '–', '<b>' + api.fmt(x.s, 4) + '</b>'];
                }) });
              if (!fu.length) api.callout(out, 'No matches in either list for this query.', 'hedge');
              api.html(out, '<p>Example: a doc ranked 1st by BM25 and 3rd by cosine gets 1/(' + st.k + '+1) + 1/(' + st.k + '+3) = <b>' + api.fmt(1 / (st.k + 1) + 1 / (st.k + 3), 4) + '</b>. Appearing high in <i>both</i> lists wins.</p>');
              api.html(out, '<p class="muted">Reranking would then take these fused candidates and re-score each (query, chunk) pair with a cross-encoder or an LLM, which reads both texts together and is slower but usually more precise (tends to help; measure on your data).</p>');
            }
            api.input(c, { label: 'Query', value: st.q, onInput: function (v) { st.q = v; draw(); }, init: false });
            api.slider(c, { label: 'RRF k', min: 1, max: 100, value: st.k, format: function (v) { return v; }, onInput: function (v) { st.k = v; draw(); }, init: false });
            el.appendChild(c); el.appendChild(out); draw();
            api.illustrative(el, 'BM25 uses k1 = 1.5, b = 0.75 on 8 toy docs; the "dense" side is the toy embedding from the previous slides.');
          } },
          { name: 'Agentic RAG', render: function (el, api) {
            api.html(el, '<p>Instead of a fixed pipeline, the model gets a <b>search tool</b>. The harness runs it and appends chunks as a tool result; the model can read them, refine the query, or answer (' + api.link('llm-to-agent/agent-loop', 'the agent loop') + ').</p>');
            api.stepper(el, { interval: 2600, steps: [
              { title: 'Model emits a tool call (JSON tokens)', render: function (b) { api.code(b, '{"name": "search_docs", "arguments": {"query": "API rate limit", "k": 3}}', 'json'); api.html(b, '<p class="muted">' + api.badge('json', 'object') + ' produced token by token, then parsed by the harness.</p>'); } },
              { title: 'Harness runs retrieval, appends a result', render: function (b) {
                var r = rankDense('API rate limit', api.math).slice(0, 2);
                api.code(b, JSON.stringify({ results: r.map(function (x) { return { id: x.i, score: +api.math.round(x.s, 3), text: DOCS[x.i] }; }) }, null, 2), 'json');
              } },
              { title: 'Model judges: enough? No, search again', render: function (b) { api.code(b, '{"name": "search_docs", "arguments": {"query": "HTTP 429 retry header", "k": 2}}', 'json'); api.html(b, '<p>The model noticed the first results did not say how to <i>recover</i>, so it rewrote the query. A fixed pipeline cannot do this.</p>'); } },
              { title: 'Final answer with citations', render: function (b) { api.html(b, '<div class="memory-rag-prompt">The limit is 60 requests per minute per key [#3]. Errors come back as HTTP 429; retry after the delay given in the header [#7].</div>'); api.callout(b, 'Retrieved text is <b>untrusted input</b>: a poisoned document can carry instructions. See ' + api.link('safety-eval/prompt-injection', 'prompt injection') + '.', 'warn'); } }
            ] });
            api.docsNote(el, 'Retrieval tool schemas and framework helpers (see ' + api.link('rag-practice/langchain-rag', 'RAG in LangChain') + ') vary by library.');
          } }
        ]
      }
    ]
  });
})();
