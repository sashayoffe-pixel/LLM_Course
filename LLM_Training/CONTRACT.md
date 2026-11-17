# LLM Training — build contract (read fully before writing a section)

An interactive, offline web course on **how LLMs work** (Part I) and **agentic AI** (Part II), plus an
appendix. Audience: smart engineers who are **not** ML specialists. Dark theme. English.

## Hard rules
* Vanilla JS, plain `<script>` files. **No build step, no ES modules, no CDNs, no `fetch()`, no network, no images from the web.**
  Must work by double-clicking `index.html` (file://). Inline SVG is fine.
* You own exactly two files: `sections/<id>.js` and `sections/<id>.css` (both already linked from `index.html`;
  create both, the CSS may be tiny). **Do not edit** `index.html`, `core/*`, `CONTRACT.md`, `tools/*` or other sections.
  If you need a core feature, implement it locally in your file and mention it in your final report.
* Wrap your file in an IIFE with `'use strict'` and call `Course.register({...})` once per section.
* **Prefix every CSS class you define with your section id** (e.g. `.attention-head-grid`, `.tool-calling-seq`).
  Reuse the core classes listed below freely.
* Use `api.every` / `api.after` / `api.onCleanup` for every timer/animation. For `requestAnimationFrame` loops, stop when `el.isConnected` is false.
* No `console.log` noise, no uncaught errors, no text rendering as `NaN`, `undefined` or `[object Object]`.
* No horizontal overflow at 390 px wide: wrap things, use `.row`/`.grid`, and put wide tables/matrices inside the core
  scrolling wrappers (`api.matrix`, `api.table`, `api.code` already scroll internally). SVGs: use `viewBox` + `width:100%`.
* Position in the course comes from `order`, not from file names.

## Section schema
```js
Course.register({
  id: 'attention',            // section id from the map below (kebab-case)
  part: 'I',                  // 'I' (How LLMs Work) | 'II' (Agentic AI) | 'A' (Appendix)
  order: 50,                  // from the map below
  title: 'Attention',
  slides: [{
    id: 'worked-example',     // kebab-case, unique within the section
    title: 'Full worked example: X → softmax(QKᵀ/√d)·V',
    critical: {               // right-hand panel; HTML strings allowed; 2–4 bullets each
      understand: ['Each token builds a weighted average of earlier tokens\' values.'],
      details:    ['Scores are divided by √d_head before softmax to keep their variance ≈ 1.']
    },
    dataTypes: [              // every datapoint shown on the slide; type ∈ the 12 types below
      { name: 'X', type: 'matrix', shape: '[seq_len × d_model]', example: '[5 × 4]' },
      { name: 'attention weights A', type: 'probability', shape: '[seq_len × seq_len]', example: 'row sums = 1' }
    ],
    levels: [                 // 2–4 ZOOM LEVELS: overview → sub-process → calculation
      { name: 'Overview',      render: function (el, api) { /* simple visual first */ } },
      { name: 'Per head',      render: function (el, api) { /* deeper */ } },
      { name: 'Numbers',       render: function (el, api) { /* step-by-step calculation */ } }
    ]
  }]
});
```
`render(el, api)` is called each time the level is shown, with a fresh empty `el`. Build DOM into `el`.
The engine renders the slide title, the level breadcrumb, ▼ Dive in / ▲ Zoom out buttons (↓ ↑ keys), the
right panel, Back/Next (← → keys), the TOC (grouped by part, with visited ✓ and ●●● = number of zoom levels),
the progress bar, and the URL hash `#/<section>/<slide>/<level>`. A "▼ Dive in: <next level name>" hint is
appended under each level automatically — don't add your own.

**Zoom levels must be real zooms** (the same topic, one level deeper), e.g. *Overview → Sub-process → Calculation*,
*Picture → Mechanism → Code/Math*. Level names are short (1–3 words).

## The 12 data types (colour-coded badges)
| type | use for |
|---|---|
| `text` | human-readable string: prompt, document, message content |
| `token` | a sub-word piece (`" cat"`) |
| `integer` | token IDs, counts, indices, dims |
| `float` | a logit, a score, a weight, a loss, a learning rate |
| `probability` | value in [0,1] / distribution summing to 1 |
| `vector` | 1-D floats, e.g. embedding `[d_model]` |
| `matrix` | 2-D floats, e.g. `X [seq_len × d_model]`, `W_q [d_model × d_model]` |
| `tensor` | 3-D+, e.g. `Q [n_heads × seq_len × d_head]`, a latent image `[4 × 64 × 64]` |
| `mask` | causal mask, loss mask, padding mask, top-k/top-p filter |
| `boolean` | flags: stop?, is_error, needs_approval |
| `json` | tool calls, tool schemas, API payloads, graph state |
| `function` | a tool, a node, an activation or scoring function, a harness |

**Shape notation:** `[seq_len × d_model]`, `[d_head]`, `[vocab]`, `scalar`, `[5 × 4]`, `string`, `object`, `list[str]`.
Always use `×` (not x) inside shapes. Label EVERY datapoint you show (in diagrams via block `type`/`shape`, inline via `api.badge`).

## Running examples (use them everywhere; available as `api.ex`)
* Prompt **"The cat sat on the"** → next token **" mat"**. GPT-2 IDs: The=464, ␣cat=3797, ␣sat=3332, ␣on=319, ␣the=262, ␣mat=2603.
  `api.ex.tokens` = `[{text:'The',id:464}, {text:' cat',id:3797}, …]`, `api.ex.nextToken` = `{text:' mat', id:2603}`.
* Toy model for hand calculations: `d_model = 4`, 2 heads, `d_head = 2`, seq_len 5. `api.ex.toyX` is an illustrative
  `[5 × 4]` input matrix (embeddings + position) — reuse it for X wherever a toy X appears so sections agree.
* Reference model ("Llama-3-8B-class", `api.ref`): vocab 128,256 · d_model 4,096 · 32 layers · 32 query heads, 8 KV heads (GQA) ·
  d_head 128 · FFN hidden 14,336 · ≈ 8.03 B params · BF16 weights ≈ 16 GB · context 8,192.
  `api.ref.paramBreakdown()` → `{wq, wk, wv, wo, attn, ffn, norms, perLayer, embed, unembed, finalNorm, total, bytesBF16}`
  (total = 8,030,261,248). `api.ref.kvBytesPerToken` = 131,072 (= 128 KiB: 2 (K,V) × 32 layers × 8 KV heads × 128 × 2 bytes).
* Agentic examples: `api.ex.agent.request` = "Book me a table for 2 in Paris tomorrow at 8pm and tell me if I need an umbrella"
  with tools `get_weather`, `search_restaurants`, `book_table`; and **a coding agent fixing a failing unit test**.
* **Compute worked numbers in code at runtime** (with `api.math`) so every number on screen is consistent with the inputs.
  Mark made-up numbers with `api.illustrative(el)`. Hedge claims that are not settled ("tends to", "often", "research suggests").
  Put `api.docsNote(el)` on every slide that shows a framework API (LangChain, LangGraph, ClearML, ComfyUI, chat APIs, MCP, skills).
* Chat-API code: keep it provider-neutral or clearly labelled; prefer showing the JSON payload shape
  (`{model, system, messages:[{role, content}], tools, max_tokens}` → `{content, stop_reason, usage}`) over SDK specifics.
  Avoid hard-coded prices; if you show cost math, use illustrative prices and say so.

## THE key connection (Part II must make it explicit, Part I should foreshadow it)
An agent is a **loop around repeated next-token prediction**. The conversation, tool definitions and tool results are
serialized into **ONE token sequence** via a chat template. The model emits tokens that form a **JSON tool call**.
The **harness** (ordinary code) parses it, runs the tool, and appends the result as new tokens; then the model predicts again.

## Helper API (`api` passed to every render)
Context: `api.section`, `api.slide`, `api.level` (index), `api.levels` (count), `api.ex`, `api.ref`, `api.math`, `api.types`.

**HTML-string helpers** (return strings, use inside your own HTML):
* `api.type(t, text?)` → coloured type badge. `api.shape('[5 × 4]')` → shape label. `api.badge(t, shape)` → both.
* `api.tok(text, id?)` → token chip (spaces shown as ␣). `api.link('attention/worked-example', 'see attention')` → cross-link.
  Cross-link targets: `'section'`, `'section/slide'` or `'section/slide/level'`. Any element with `data-goto="…"` or `data-zoom="2"` also works.
* `api.esc(s)`, `api.tokText(s)`, `api.fmt(x, decimals=2)` (−∞ for -Infinity, proper minus), `api.fmtInt(n)` (1,234), `api.fmtBytes(b)` (KiB/MiB/GiB), `api.fmtBig(n)` (8.03 B).

**Element helpers** (append to `el`, return the element or a small controller):
* `api.h(tag, attrs, ...children)` → element. attrs: `class`, `html`, `text`, `style` (object), `onclick` etc.
* `api.html(el, htmlString)` → div.
* `api.callout(el, html, kind)` — kinds: `info`, `tip`, `warn`, `key`, `illustrative`, `docs`, `hedge`.
  Shortcuts: `api.illustrative(el, extra?)`, `api.docsNote(el, extra?)`.
* `api.blocks(el, spec)` → `{el, highlight(ids), node(id)}`. Diagram of blocks with **animated flow arrows** and click-to-zoom/jump.
  `spec = { blocks:[…] }` (one row) or `{ rows:[[…],[…]] }` (rows joined by ▼), `direction:'column'`, `caption`, `animate:false`, `arrows:false`.
  Block: `{ id, label, sub, type:'matrix' | ['token','integer'], shape:'[5 × 4]', zoom: 1 | true, goto:'attention/projections', onClick(b), note, color, edge:'label on incoming arrow', arrow:false, cls }`.
  `zoom:n` dives to level n of this slide (`true` = next level); `goto` jumps to a section/slide.
* `api.stepper(el, { steps:[{title, html} | {title, render(body, api)}], interval:1800, onStep(i, body), start })` → `{go(i), index, count}`. Prev / Next / Play.
* `api.matrix(el, { data, rowLabels, colLabels, decimals, title, type, shape, causal:true, mask:[[bool]], maskText:'−∞', highlight:{row,col,cell:[r,c]}, onCell(r,c), scale:'seq'|'diverging', heat:false, small, caption })`
  → `{el, update(partialSpec)}`. Heat-mapped; `causal:true` greys out the upper triangle; `-Infinity` prints as −∞.
* Controls (append inside a `api.h('div',{class:'controls'})` bar for a nice look):
  `api.slider(el, {label, min, max, step, value, format(v), onInput(v)})` (calls onInput once immediately unless `init:false`);
  `api.select(el, {label, options:['a', {value, label}], value, onChange(v), init})`; `api.button(el, html, onClick, cls)`;
  `api.toggle(el, {label, value, onChange(bool)})`; `api.input(el, {label, value, placeholder, multiline, onInput(v)})`.
* `api.chart(el, { series:[{name, points:[[x,y]], color, dashed}], xLabel, yLabel, width, height, xRange, yRange, logY, markers:[{x,label}], hlines:[{y,label}] })` → `{update(spec)}`. SVG line chart.
* `api.bars(el, { items:[{label, value, color, highlight, dim, note}], max, format(v), title })` → `{update(items)}`. Horizontal bars (great for probabilities).
* `api.code(el, src, lang)` — lang `python`, `json`, `js`, `bash`, `yaml`, `text`, `markdown`. `api.table(el, {headers, rows})`.
* `api.tokens(el, [{text, id, color, cls}], {highlight: index, ids:false})` → token chips.
* `api.grid(el, minColPx)` → responsive grid container; `api.card(el, titleHtml, bodyHtml)`.
* Timers: `api.every(ms, fn)`, `api.after(ms, fn)`, `api.onCleanup(fn)` — auto-cleared on navigation.
* Navigation: `api.zoomTo(n)`, `api.zoomIn()`, `api.zoomOut()`, `api.goto('section/slide')`, `api.next()`, `api.prev()`.
* `api.math`: `sum, max, argmax, dot, norm, cosine, add, sub, scale, transpose, matmul, matvec, vecmat, softmax(arr, T), logSoftmax,
  rmsnorm(x, g, eps), sigmoid, silu, relu, gelu, round(x,d), range(n), linspace(a,b,n), rng(seed) → () ⇒ [0,1), sampleIndex(probs, r)`.
* Global: `Course.TYPES` (type → {label, desc}), `Course.PARTS`, `Course.EX`, `Course.REF`.

## Core CSS classes you can use
`.controls` (control bar), `.row` (flex wrap), `.grid`, `.card`, `.formula` (mono equation box), `.big-num`, `.pill`, `.muted`,
`.btn`, `.btn.primary`, `.btn.on`, `.tok`, `.dt .dt-<type>`, `.shape`, `.xlink`, `kbd`, `code`.
Core colours as CSS variables: `--bg --bg2 --panel --panel2 --border --text --muted --accent --good --bad --warn`
and per type `--t-text --t-token --t-integer --t-float --t-probability --t-vector --t-matrix --t-tensor --t-mask --t-boolean --t-json --t-function`.

## Content style
* Simple visual first, then deeper. Show the flow of data: inputs → outputs, **with types and shapes**.
* Use animated `api.blocks` diagrams, step-by-step walkthroughs (`api.stepper`), and sliders/demos that actually compute.
* Every slide: 2–4 zoom levels, `critical.understand` and `critical.details` (2–4 bullets each), `dataTypes` (every datapoint, with shape).
* Cross-link generously with `api.link(...)` / block `goto` to related sections (use only ids from the map below).
* Be accurate. Hedge unsettled claims. No invented APIs; if unsure, label code as pseudo-code.
* Keep each level fitting comfortably on ~1–1.5 screens.

## Section map (section id · order · part) — slide ids are suggestions; keep them if you can
**Part I — How LLMs Work**
* `big-picture` 10: `welcome` (how to navigate + legend of the 12 types) · `next-token` (prediction loop) · `architecture-map`
  (END-TO-END MASTER MAP, blocks link to every section, types+shapes on every block) · `forward-pass` (with numbers) ·
  `scale` (what "8B parameters" means) · `llm-vs-software` (vs software, search, databases)
* `tokenization` 20: `why-tokens` · `bpe-playground` (toy BPE tokenizer) · `bpe-training` · `text-to-ids` (text → UTF-8 bytes → merges → IDs) ·
  `special-tokens` (and chat templates) · `quirks` (spaces, numbers, "strawberry", non-English cost)
* `embeddings` 30: `lookup` (ID → row of E) · `geometry` (2-D word map, king − man + woman) · `cosine` (step by step) ·
  `why-position` · `rope` (rotation with slider) · `input-matrix` (the input matrix X)
* `transformer-block` 40: `decoder-only` · `residual-stream` ("shared whiteboard") · `rmsnorm` (worked example) ·
  `layer-roles` (what layers tend to do) · `param-count` (per layer × 32 = 8.03 B)
* `attention` 50 (THE most important): `qkv-intuition` (interactive attention view) · `projections` ·
  `worked-example` (FULL numeric: X → Q,K,V → QKᵀ → ÷√d_head → causal mask → softmax → A·V) · `softmax-scaling` (playground) ·
  `causal-mask` · `multi-head` (plus W_o) · `gqa-cost` (GQA/MQA and O(n²))
* `ffn-moe` 60: `ffn-role` (per-token) · `swiglu` (worked, d_ff = 8) · `activations` (plots) · `kv-memory` (key-value memory intuition) · `moe` (router, top-k, load balancing)
* `output-sampling` 70: `unembedding` (hidden → logits → softmax) · `sampling-playground` (temperature / top-k / top-p + sample histogram) ·
  `autoregressive-loop` · `logprobs-constraints` (logprobs, penalties, constrained decoding)
* `training` 80: `lifecycle` · `cross-entropy` (and teacher forcing) · `backprop` (one worked update, LR playground, AdamW) ·
  `scaling` (scaling laws, BF16) · `sft` (masked loss) · `rlhf` (reward model, PPO, DPO, RLAIF) · `lora` (parameter savings)
* `clearml` 85: `mlops` (where MLOps fits) · `components` (Server, SDK Task.init auto-logging, Agent + queues, Data, Pipelines, HPO, Serving) ·
  `lora-tracking` (worked LoRA fine-tune tracking flow with a loss chart) · `web-ui` (mock of the UI) · `rag-prompts` (tracking RAG/prompt experiments; vs MLflow / W&B)
* `inference` 90: `prefill-decode` · `kv-cache` (deep-dive + memory calculator; 128 KiB/token) · `context-window` ("lost in the middle") ·
  `quantization` (calculator + INT8 example) · `batching-speculative`
* `chat-api` 95: `messages-api` (stateless; roles, params, stop_reason, usage) · `chat-template` (→ one token sequence) · `cost-caching` (cost grows per turn; prompt caching)
* `reasoning` 100: `why-cot` (more serial compute + scratchpad via attention) · `word-problem` (direct vs step-by-step) ·
  `reasoning-models` (RL with verifiable rewards) · `test-time-compute` (self-consistency vote, best-of-N, tree search) ·
  `thinking-budgets` · `limits` (limitations, faithfulness) · `bridge` (bridge to agents)

**Part II — Agentic AI**
* `llm-to-agent` 110: `definition` (LLM + loop + tools + memory + goal) · `spectrum` (chatbot vs workflow vs agent) ·
  `agent-loop` (THE AGENT LOOP, context growing turn by turn) · `loop-internals` (data types at each hop) · `stop-conditions` (and budgets)
* `tool-calling` 120: `tool-definition` (JSON schema) · `round-trip` (sequence diagram with payloads, parallel calls) ·
  `valid-json` (how the model produces valid JSON) · `router-demo` · `tool-design` (and errors) · `security` (results are untrusted) · `mcp` (intro)
* `langchain` 125: `where-frameworks-fit` · `lcel` (components, prompt | model | parser) · `tools-structured` (bind_tools / structured output) ·
  `agents` (built on LangGraph; framework vs direct API)
* `memory-rag` 130: `memory-types` · `context-management` (compaction) · `rag-pipeline` (chunking → embeddings → vector DB → top-k cosine → prompt) ·
  `mini-rag` (playground) · `advanced-rag` (hybrid search, reranking, agentic RAG)
* `rag-practice` 135: `langchain-rag` (splitter, embeddings, FAISS, retriever) · `chunking-demo` (live) · `query-walkthrough` · `production-tips`
* `planning-multiagent` 140: `react-trace` (coding agent) · `workflow-patterns` (chaining, routing, parallel, orchestrator-workers, evaluator-optimizer) ·
  `multi-agent` (and their cost) · `protocols` (MCP, agent-to-agent, skills)
* `skills` 143: `skill-anatomy` (SKILL.md) · `progressive-disclosure` (token gauge) · `skill-selection` (how the model picks) ·
  `comparison` (skills vs tools vs MCP vs prompts vs sub-agents vs fine-tuning) · `authoring-safety`
* `langgraph` 145: `chains-vs-graphs` · `state-reducers` · `react-graph` (INTERACTIVE GRAPH EXECUTOR) · `checkpoint-interrupt` (interrupt() approval gate) ·
  `multi-agent-patterns` (supervisor, handoffs, Send fan-out) · `when-to-use` (LangChain vs LangGraph vs plain code)
* `comfyui` 147: `node-graph` (typed sockets) · `diffusion` (denoising simulator, CFG math, VAE latents) · `text-encoder` (transformer text encoder + CROSS-attention) ·
  `agent-tool` (ComfyUI via its HTTP API) · `ecosystem` (VRAM; DAG vs LangGraph's cyclic graph)
* `safety-eval` 150: `failure-modes` (pⁿ compounding-error chart) · `prompt-injection` · `autonomy-levels` (human-in-the-loop) ·
  `evals` (traces, LLM-as-judge) · `recap` (Part II RECAP diagram + "5 things to remember")

**Appendix**
* `glossary` 200: `data-types` (reference) · `terms` (filterable glossary ≈ 48 terms, each linking to its section) · `quiz` (12 questions)

## Verify before you finish
```bash
node --check sections/<id>.js
node tools/validate.js sections/<id>.js          # stub registration + schema test
node tools/browser-test.js --section=<id>        # real Chromium: every level, clicks steppers, moves sliders, selects, buttons
node tools/browser-test.js --section=<id> --width=390   # narrow screen: no horizontal overflow
node tools/shot.js "<id>/<slide>/<level>" /tmp/x.png    # optional screenshot to eyeball a slide (then view it)
```
All must report 0 issues. Cross-links to sections that other agents have not written yet are listed as
"(info) links to sections not yet loaded" (not counted) — expected while they work; use only ids from the map above and they will resolve.
Report: slides written, anything you could not do, any core feature you wished for.
