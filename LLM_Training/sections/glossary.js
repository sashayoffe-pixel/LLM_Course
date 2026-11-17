/* Appendix: data-type reference, filterable glossary, and quiz.
 * All Course.* data is read inside render functions (the validator loads this file with stub data).
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------ data-type reference
  // shape = canonical shape for the reference datapoint; shapes/ex = display text; where = cross-links.
  var CARDS = [
    { t: 'text', shape: 'string', shapes: 'string · list[str]', ex: '"The cat sat on the" · a chat message · a retrieved document',
      exShort: '"The cat sat on the"',
      where: [['big-picture/next-token', 'Next-token loop'], ['chat-api/messages-api', 'Messages API'], ['memory-rag/context-management', 'Context management']] },
    { t: 'token', shape: 'string', shapes: 'string (one sub-word piece)', ex: '" cat" (ID 3797) · " mat" (ID 2603)',
      exShort: '" mat"',
      where: [['tokenization/why-tokens', 'Why tokens'], ['tokenization/bpe-playground', 'BPE playground'], ['inference/prefill-decode', 'Prefill and decode']] },
    { t: 'integer', shape: 'scalar', shapes: 'scalar · [seq_len]', ex: '2603 (ID of " mat") · 8,192 (context) · 131,072 (bytes per token)',
      exShort: '2603',
      where: [['tokenization/text-to-ids', 'Text to IDs'], ['embeddings/lookup', 'Embedding lookup'], ['inference/context-window', 'Context window']] },
    { t: 'float', shape: 'scalar', shapes: 'scalar · [vocab]', ex: 'a logit · a cross-entropy loss · a learning rate',
      exShort: 'a logit',
      where: [['attention/worked-example', 'Attention worked example'], ['output-sampling/unembedding', 'Unembedding (logits)'], ['training/cross-entropy', 'Cross-entropy']] },
    { t: 'probability', shape: '[vocab]', shapes: '[vocab] (sums to 1) · [seq_len × seq_len]', ex: 'a softmax output · attention weights (each row sums to 1)',
      exShort: 'softmax row',
      where: [['attention/softmax-scaling', 'Softmax scaling'], ['output-sampling/sampling-playground', 'Sampling playground'], ['output-sampling/unembedding', 'Softmax over logits']] },
    { t: 'vector', shape: '[d_model]', shapes: '[d_model] · [4] in the toy model', ex: 'one token embedding · the residual stream at one position',
      exShort: 'embedding [4096]',
      where: [['embeddings/cosine', 'Cosine similarity'], ['transformer-block/residual-stream', 'Residual stream'], ['memory-rag/rag-pipeline', 'RAG embeddings']] },
    { t: 'matrix', shape: '[seq_len × d_model]', shapes: '[seq_len × d_model] · [d_model × d_model]', ex: 'the input X · a weight W_q · the toy X [5 × 4]',
      exShort: 'X [5 × 4]',
      where: [['embeddings/input-matrix', 'Input matrix X'], ['attention/projections', 'Q, K, V projections'], ['training/lora', 'LoRA matrices']] },
    { t: 'tensor', shape: '[n_heads × seq_len × d_head]', shapes: '[n_heads × seq_len × d_head] · [4 × 64 × 64]', ex: 'per-head Q · the KV cache · a diffusion latent',
      exShort: 'Q [2 × 5 × 2]',
      where: [['attention/multi-head', 'Multi-head attention'], ['inference/kv-cache', 'KV cache'], ['comfyui/diffusion', 'Diffusion latents']] },
    { t: 'mask', shape: '[seq_len × seq_len]', shapes: '[seq_len × seq_len] · [vocab]', ex: 'causal mask (upper triangle = −∞) · top-p filter · loss mask',
      exShort: 'causal mask',
      where: [['attention/causal-mask', 'Causal mask'], ['output-sampling/sampling-playground', 'Top-k and top-p filters'], ['training/sft', 'SFT loss mask']] },
    { t: 'boolean', shape: 'scalar', shapes: 'scalar', ex: 'stop? = true · is_error = false · needs_approval = true',
      exShort: 'stop? = true',
      where: [['llm-to-agent/stop-conditions', 'Stop conditions'], ['safety-eval/autonomy-levels', 'Autonomy levels'], ['langgraph/checkpoint-interrupt', 'Approval interrupt']] },
    { t: 'json', shape: 'object', shapes: 'object · list[object]', ex: '{"name": "get_weather", "arguments": {…}}',
      exShort: 'tool call {…}',
      where: [['tool-calling/round-trip', 'Tool-call round trip'], ['tool-calling/tool-definition', 'Tool definition (JSON schema)'], ['chat-api/messages-api', 'Messages API payload']] },
    { t: 'function', shape: '(input) → output', shapes: 'callable: (x) → y', ex: 'a tool · a graph node · the SwiGLU activation',
      exShort: 'get_weather()',
      where: [['llm-to-agent/definition', 'Agent definition'], ['ffn-moe/swiglu', 'SwiGLU activation'], ['langgraph/state-reducers', 'Graph nodes and reducers']] }
  ];

  // ------------------------------------------------------------------ glossary terms
  // [term, data type, one-line definition, section/slide target]
  var TERMS = [
    ['Token', 'token', 'Sub-word piece of text the model reads and writes; the unit of context length and of cost.', 'tokenization/why-tokens'],
    ['BPE (byte-pair encoding)', 'function', 'Tokenizer training: repeatedly merge the most frequent adjacent pair of symbols into one new token.', 'tokenization/bpe-training'],
    ['Token ID', 'integer', 'Position of a token in the vocabulary. In GPT-2, " mat" is 2603.', 'tokenization/text-to-ids'],
    ['Embedding', 'vector', 'Learned vector of length d_model that represents a token; a token ID selects one row of the embedding table.', 'embeddings/lookup'],
    ['RoPE (rotary position embedding)', 'function', 'Rotates Q and K by position-dependent angles so attention scores depend on relative distance.', 'embeddings/rope'],
    ['Attention', 'function', 'Each position builds a weighted average of value vectors; the weights come from comparing its query with other positions’ keys.', 'attention/qkv-intuition'],
    ['Query, Key, Value (Q, K, V)', 'matrix', 'Three learned projections of the input: what a token looks for, what it offers for matching, and what it passes on.', 'attention/projections'],
    ['Causal mask', 'mask', 'Blocks attention to future positions by setting those scores to −∞ before the softmax.', 'attention/causal-mask'],
    ['Multi-head attention', 'tensor', 'Several heads run in parallel, each with its own d_head; their outputs are concatenated and mixed by W_o.', 'attention/multi-head'],
    ['GQA (grouped-query attention)', 'tensor', 'Groups of query heads share one key/value head, which shrinks the KV cache. MQA is the extreme case of one shared KV head.', 'attention/gqa-cost'],
    ['Residual stream', 'vector', 'The per-token vector that every layer reads from and adds its output to; a shared whiteboard.', 'transformer-block/residual-stream'],
    ['RMSNorm', 'function', 'Scales a vector by its root-mean-square and applies a learned gain. Used before attention and the FFN in many LLMs.', 'transformer-block/rmsnorm'],
    ['FFN (feed-forward network)', 'function', 'Per-token MLP after attention that transforms each position’s vector independently; much of the parameter count sits here.', 'ffn-moe/ffn-role'],
    ['SwiGLU', 'function', 'Gated activation used in many Llama-style models: SiLU(gate) × up, then a down projection.', 'ffn-moe/swiglu'],
    ['MoE (mixture of experts)', 'function', 'Replaces the FFN with several experts; a router sends each token to the top-k of them.', 'ffn-moe/moe'],
    ['Logits', 'float', 'Unnormalised scores, one per vocabulary entry, produced by the unembedding from the last hidden state.', 'output-sampling/unembedding'],
    ['Softmax', 'probability', 'Turns scores into a probability distribution: non-negative values that sum to 1.', 'attention/softmax-scaling'],
    ['Temperature', 'float', 'Divides the logits before the softmax. Low values make sampling near-greedy; high values flatten the distribution.', 'output-sampling/sampling-playground'],
    ['Top-p (nucleus sampling)', 'mask', 'Keeps the smallest set of tokens whose probabilities add up to at least p, then renormalises.', 'output-sampling/sampling-playground'],
    ['KV cache', 'tensor', 'Stored keys and values of earlier tokens, so each new token does not recompute them.', 'inference/kv-cache'],
    ['Prefill', 'token', 'Processes the whole prompt in one parallel pass and fills the KV cache.', 'inference/prefill-decode'],
    ['Decode', 'token', 'Generates output one token per step; each step reads the KV cache.', 'inference/prefill-decode'],
    ['Quantization', 'float', 'Stores weights in fewer bits (for example INT8 or 4-bit) to save memory, usually with some accuracy cost.', 'inference/quantization'],
    ['Speculative decoding', 'token', 'A small draft model proposes several tokens and the large model checks them in one pass.', 'inference/batching-speculative'],
    ['Context window', 'integer', 'Maximum number of tokens (prompt plus output) the model can handle in one request.', 'inference/context-window'],
    ['Cross-entropy loss', 'float', 'Negative log-probability of the correct next token, averaged over positions. The pretraining objective.', 'training/cross-entropy'],
    ['Scaling laws', 'float', 'Empirical power-law trends linking loss to parameters, data and compute. They describe trends, not guarantees.', 'training/scaling'],
    ['SFT (supervised fine-tuning)', 'text', 'Training on example (prompt, ideal reply) pairs, usually with the loss computed on the reply tokens only.', 'training/sft'],
    ['RLHF', 'function', 'Fine-tuning against a reward model trained on human preference comparisons, often with PPO.', 'training/rlhf'],
    ['DPO (direct preference optimization)', 'function', 'Trains directly on preferred vs rejected replies, without a separate reward model or RL loop.', 'training/rlhf'],
    ['LoRA', 'matrix', 'Freezes the base weights and trains small low-rank matrices that are added to them.', 'training/lora'],
    ['Chain of thought', 'text', 'The model writes intermediate reasoning tokens before its answer, spending more serial compute on the problem.', 'reasoning/why-cot'],
    ['Test-time compute', 'integer', 'Extra inference work (more samples, search or longer reasoning) spent to improve an answer.', 'reasoning/test-time-compute'],
    ['Chat template', 'text', 'Formats roles and messages into the single token sequence the model actually sees, with special tokens.', 'chat-api/chat-template'],
    ['Prompt caching', 'token', 'Reuses the processed prefix of an unchanged prompt across calls. Availability, pricing and rules vary by provider.', 'chat-api/cost-caching'],
    ['Agent', 'function', 'An LLM in a loop that chooses tools, observes the results and continues until a stop condition is met.', 'llm-to-agent/definition'],
    ['Harness', 'function', 'The ordinary code around the model: builds the context, parses tool calls, runs tools and appends the results.', 'llm-to-agent/agent-loop'],
    ['Tool call', 'json', 'Structured output in which the model names a tool and gives its arguments as JSON.', 'tool-calling/round-trip'],
    ['MCP (Model Context Protocol)', 'json', 'An open protocol for exposing tools, resources and prompts to agents through a common message format.', 'tool-calling/mcp'],
    ['Skill', 'text', 'A folder with a SKILL.md file of instructions, loaded progressively when a task needs it.', 'skills/skill-anatomy'],
    ['Prompt injection', 'text', 'Instructions hidden in untrusted content (a web page, a document or a tool result) that try to override what the user asked for.', 'safety-eval/prompt-injection'],
    ['RAG (retrieval-augmented generation)', 'text', 'Retrieves relevant chunks and places them in the prompt before the model answers.', 'memory-rag/rag-pipeline'],
    ['Chunking', 'text', 'Splitting documents into pieces that are embedded and retrieved separately; chunk size is a trade-off.', 'rag-practice/chunking-demo'],
    ['Vector database', 'matrix', 'Stores embeddings and returns nearest neighbours, for example the top-k by cosine similarity.', 'memory-rag/rag-pipeline'],
    ['Reranking', 'float', 'A second-stage model rescores retrieved candidates for relevance before they reach the prompt.', 'memory-rag/advanced-rag'],
    ['LLM-as-judge', 'float', 'Uses a model with a rubric to score outputs. Practical at scale, but it needs calibration against human judgements.', 'safety-eval/evals'],
    ['LangChain', 'function', 'Framework of components (models, prompts, parsers, tools) for composing LLM applications.', 'langchain/where-frameworks-fit'],
    ['LCEL (LangChain Expression Language)', 'function', 'Composes runnables with the pipe operator, for example prompt | model | parser.', 'langchain/lcel'],
    ['LangGraph', 'function', 'Framework for stateful graphs of nodes and edges; used to control agent loops, branches and handoffs.', 'langgraph/chains-vs-graphs'],
    ['Checkpoint', 'json', 'A saved snapshot of graph state at a step, enabling resume, interrupts and inspection.', 'langgraph/checkpoint-interrupt'],
    ['ComfyUI', 'function', 'Node-graph tool for image-diffusion pipelines; it also exposes an HTTP API.', 'comfyui/node-graph'],
    ['CFG (classifier-free guidance)', 'float', 'Pushes the denoising prediction away from the unconditional one, scaled by a guidance value.', 'comfyui/diffusion'],
    ['ClearML', 'function', 'Open-source platform for experiment tracking, pipelines, data and serving.', 'clearml/mlops'],
    ['Approval gate (human in the loop)', 'boolean', 'A gate where a person must approve an action (needs_approval = true) before the harness runs it.', 'safety-eval/autonomy-levels']
  ];

  // ------------------------------------------------------------------ quiz
  var quizAnswers = {}; // question index -> chosen option index (persists across zoom levels)

  function quizData() {
    var nt = Course.EX.nextToken, kv = Course.REF.kvBytesPerToken;
    return [
      { short: 'ID of " mat" in GPT-2',
        q: 'In GPT-2’s vocabulary, what is the token ID of <code>" mat"</code>, the predicted next token after “The cat sat on the”?',
        opts: ['262', String(nt.id), '3797', '464'], a: 1,
        why: 'The IDs are The = 464, " cat" = 3797, " sat" = 3332, " on" = 319, " the" = 262 and " mat" = 2603.',
        target: 'tokenization/text-to-ids' },
      { short: 'Why divide by √d_head?',
        q: 'Why is QKᵀ divided by √d_head before the softmax?',
        opts: ['It normalises each token vector to unit length', 'It converts the scores to integers for the KV cache', 'It keeps the score variance near 1 so the softmax does not saturate into a one-hot', 'It removes the causal mask from the upper triangle'], a: 2,
        why: 'Dot products grow in variance with d_head; dividing by √d_head brings it back to about 1, so the softmax stays spread out and its gradients do not vanish.',
        target: 'attention/softmax-scaling' },
      { short: 'KV cache bytes per token',
        q: 'How much KV cache does one token need in the reference model (BF16)?',
        opts: ['32 KiB', '128 KiB', '2 MiB', '16 GiB'], a: 1,
        why: '2 (K and V) × 32 layers × 8 KV heads × 128 × 2 bytes = ' + kv.toLocaleString('en-US') + ' bytes = 128 KiB. Multiply by context length and batch size for the total.',
        target: 'inference/kv-cache' },
      { short: 'What does the harness do?',
        q: 'In an agent, what does the harness do?',
        opts: ['Ordinary code: it parses the model’s tool-call JSON, runs the tool and appends the result to the context', 'The model weights choose and run the tool themselves', 'It fine-tunes the model on every tool result', 'It stores all tool results in a vector database'], a: 0,
        why: 'The model only emits tokens. The harness, which is plain code around the model, turns those tokens into actions and feeds the results back as new tokens.',
        target: 'llm-to-agent/agent-loop' },
      { short: 'What does a causal mask do?',
        q: 'What does a causal mask do in self-attention?',
        opts: ['Hides the first token so it cannot be copied', 'Removes padding tokens from the vocabulary', 'Blocks each position from attending to later positions, by setting those scores to −∞ before the softmax', 'Limits how many heads can attend at once'], a: 2,
        why: 'Position t may only look at positions 0…t. Future scores are set to −∞, so their softmax weight is exactly 0.',
        target: 'attention/causal-mask' },
      { short: 'Query heads per KV head (GQA)',
        q: 'The reference model has 32 query heads and 8 KV heads. How many query heads share each KV head?',
        opts: ['1', '8', '32', '4'], a: 3,
        why: '32 ÷ 8 = 4. Sharing K and V this way cuts the KV cache by 4× compared with one KV head per query head.',
        target: 'attention/gqa-cost' },
      { short: 'Temperature close to 0',
        q: 'What happens to the next-token distribution as temperature approaches 0?',
        opts: ['It becomes uniform over the vocabulary', 'It concentrates on the highest-logit token, so sampling becomes near-greedy', 'It is unchanged, because temperature only affects training', 'It keeps only the lowest-probability tail'], a: 1,
        why: 'Dividing the logits by a very small T makes the largest logit dominate the softmax.',
        target: 'output-sampling/sampling-playground' },
      { short: 'Prefill vs decode',
        q: 'Why is prefill usually cheaper per prompt token than decode?',
        opts: ['Prefill processes all prompt tokens in one parallel pass, while decode generates one token per step', 'Prefill skips attention entirely', 'Decode does not use a KV cache', 'Prefill runs only on the CPU'], a: 0,
        why: 'Prefill can batch every prompt position into large matrix multiplications. Decode must wait for each new token, so it is often limited by memory reads.',
        target: 'inference/prefill-decode' },
      { short: 'What does LoRA train?',
        q: 'What does LoRA train?',
        opts: ['All 8 billion weights from scratch', 'Only the tokenizer', 'Small low-rank matrices added to frozen base weights', 'The softmax function'], a: 2,
        why: 'LoRA keeps the base weights frozen and learns a small low-rank update, which saves memory and storage.',
        target: 'training/lora' },
      { short: 'Why send tool results back?',
        q: 'Why must a tool result be written back into the context?',
        opts: ['The harness needs it to compute gradients', 'The model only predicts from its context, so the result must be serialized into tokens', 'The tool result updates the model weights immediately', 'It is optional: the model remembers tools in its weights'], a: 1,
        why: 'Each step is one more next-token prediction over the whole sequence. If the result is not in the sequence, the model cannot use it.',
        target: 'tool-calling/round-trip' },
      { short: 'DPO vs RLHF',
        q: 'How does DPO differ from RLHF with PPO?',
        opts: ['DPO is only used for image models', 'DPO replaces the transformer with a recurrent network', 'DPO uses no preference data', 'DPO optimises directly on preferred vs rejected replies, without a separate reward model and RL loop'], a: 3,
        why: 'DPO turns the preference pairs into a classification-style loss on the policy itself. Details and results vary by setup.',
        target: 'training/rlhf' },
      { short: 'What is prompt injection?',
        q: 'What is prompt injection?',
        opts: ['Instructions hidden in untrusted content (a web page, a document or a tool result) that try to override the user’s intent', 'Adding special tokens to the tokenizer', 'Using a larger embedding dimension', 'Increasing the context window at inference time'], a: 0,
        why: 'The model cannot reliably tell instructions from data in its context, so tool results and documents must be treated as untrusted input, with limited permissions and approval gates.',
        target: 'safety-eval/prompt-injection' }
    ];
  }

  // ------------------------------------------------------------------ helpers
  function secInfo(id) {
    var s = (Course.sections || []).filter(function (x) { return x.id === id; })[0];
    if (s) return { title: s.title, part: s.part, order: s.order };
    return { title: id.replace(/-/g, ' '), part: '', order: 999 };
  }
  function secOf(target) { return String(target).split('/')[0]; }

  // ------------------------------------------------------------------ slide 1: data types
  function renderTypesOverview(el, api) {
    api.callout(el, 'Every datapoint in the course carries one of these 12 types and a shape. Hover a badge for its definition.', 'info');
    var g = api.grid(el, 240);
    CARDS.forEach(function (c) {
      api.card(g, api.type(c.t), '<div class="glossary-card-desc">' + api.esc(Course.TYPES[c.t].desc) + '</div>' +
        '<div>Typical shapes: ' + api.shape(c.shapes) + '</div>' +
        '<div class="glossary-card-ex">e.g. ' + api.esc(c.ex) + '</div>', 'glossary-card');
    });
  }
  function renderTypesWhere(el, api) {
    var rows = CARDS.map(function (c) {
      return [api.type(c.t), c.where.map(function (w) { return api.link(w[0], w[1]); }).join(' · ')];
    });
    api.table(el, { headers: ['Type', 'Where it appears (click to open)'], rows: rows });
    api.html(el, '<p class="muted">Each link opens the slide where the type is used in a worked datapoint.</p>');
  }

  // ------------------------------------------------------------------ slide 2: terms
  var termsState = { q: '', part: 'all' };

  function termsRender(grouped) {
    return function (el, api) {
      var bar = api.h('div', { class: 'controls glossary-bar' });
      var count = api.h('div', { class: 'glossary-count muted' });
      var list = api.h('div', { class: 'glossary-list' });
      var entries = [], groups = [];
      el.appendChild(bar);
      el.appendChild(list);

      function refresh() {
        var q = termsState.q.trim().toLowerCase(), shown = 0;
        entries.forEach(function (e) {
          var ok = (termsState.part === 'all' || e.part === termsState.part) && (!q || e.search.indexOf(q) >= 0);
          e.el.classList.toggle('glossary-hidden', !ok);
          if (ok) shown++;
        });
        groups.forEach(function (g) {
          var any = g.entries.some(function (e) { return !e.el.classList.contains('glossary-hidden'); });
          g.el.classList.toggle('glossary-hidden', !any);
        });
        count.textContent = 'Showing ' + shown + ' of ' + entries.length + ' terms';
      }

      function makeEntry(t, showWhere) {
        var info = secInfo(secOf(t[3]));
        var box = api.h('div', { class: 'glossary-term' });
        var headHtml = '<b class="glossary-term-name">' + api.esc(t[0]) + '</b> ' + api.type(t[1]);
        if (showWhere) headHtml += ' <span class="glossary-where">' + api.link(t[3], info.title) + '</span>';
        box.appendChild(api.h('div', { class: 'glossary-term-head', html: headHtml }));
        box.appendChild(api.h('div', { class: 'glossary-term-def', text: t[2] }));
        var e = { el: box, part: info.part, search: (t[0] + ' ' + t[1] + ' ' + t[2] + ' ' + info.title).toLowerCase() };
        entries.push(e);
        return e;
      }

      if (grouped) {
        var secIds = [];
        TERMS.forEach(function (t) { var s = secOf(t[3]); if (secIds.indexOf(s) < 0) secIds.push(s); });
        secIds.sort(function (a, b) { return secInfo(a).order - secInfo(b).order; });
        secIds.forEach(function (id) {
          var info = secInfo(id);
          var g = { el: api.h('div', { class: 'glossary-group' }), entries: [] };
          g.el.appendChild(api.h('h3', { class: 'glossary-group-title', html: api.link(id, info.title) + ' <span class="muted">· ' + api.esc(Course.PARTS[info.part] || 'Appendix') + '</span>' }));
          TERMS.filter(function (t) { return secOf(t[3]) === id; }).forEach(function (t) {
            var e = makeEntry(t, false);
            g.entries.push(e);
            g.el.appendChild(e.el);
          });
          list.appendChild(g.el);
          groups.push(g);
        });
      } else {
        TERMS.slice().sort(function (a, b) { return a[0].localeCompare(b[0]); }).forEach(function (t) {
          list.appendChild(makeEntry(t, true).el);
        });
      }

      api.input(bar, {
        label: 'Filter', value: termsState.q, placeholder: 'term, definition or section…',
        onInput: function (v) { termsState.q = v; refresh(); }
      });
      api.select(bar, {
        label: 'Part', value: termsState.part,
        options: [{ value: 'all', label: 'All parts' }, { value: 'I', label: 'Part I' }, { value: 'II', label: 'Part II' }, { value: 'A', label: 'Appendix' }],
        onChange: function (v) { termsState.part = v; refresh(); }
      });
      bar.appendChild(count);
      refresh();
      api.docsNote(el, 'LangChain, LangGraph, ClearML, ComfyUI and MCP are named here only as one-line descriptions.');
    };
  }

  // ------------------------------------------------------------------ slide 3: quiz
  function quizLevel(el, api) {
    var Q = quizData();
    var blocks = [];
    var scoreEl = api.h('div', { class: 'glossary-score' });
    var bar = api.h('div', { class: 'controls glossary-bar' });
    el.appendChild(bar);

    function choose(qi, oi) { if (quizAnswers[qi] == null) { quizAnswers[qi] = oi; paint(); } }
    function paint() {
      var answered = 0, correct = 0;
      blocks.forEach(function (b, qi) {
        var a = quizAnswers[qi];
        b.btns.forEach(function (btn, oi) {
          btn.disabled = a != null;
          btn.classList.toggle('right', a != null && oi === b.q.a);
          btn.classList.toggle('wrong', a === oi && oi !== b.q.a);
        });
        if (a == null) { b.fb.className = 'glossary-feedback'; b.fb.innerHTML = ''; return; }
        answered++;
        var ok = a === b.q.a;
        if (ok) correct++;
        b.fb.className = 'glossary-feedback ' + (ok ? 'good' : 'bad');
        b.fb.innerHTML = '<b>' + (ok ? '✓ Correct.' : '✗ Not quite. The answer is: ' + api.esc(b.q.opts[b.q.a]) + '.') + '</b> ' +
          api.esc(b.q.why) + ' ' + api.link(b.q.target, 'Open section →');
      });
      scoreEl.innerHTML = 'Score: <b>' + correct + '</b> / ' + Q.length + ' correct · answered ' + answered + ' / ' + Q.length;
    }

    bar.appendChild(scoreEl);
    api.button(bar, '↺ Reset', function () { quizAnswers = {}; paint(); }, '');

    var list = api.h('div', { class: 'glossary-quiz' });
    el.appendChild(list);
    Q.forEach(function (q, qi) {
      var card = api.h('div', { class: 'card glossary-q' });
      card.appendChild(api.h('div', { class: 'glossary-q-num', text: 'Question ' + (qi + 1) + ' of ' + Q.length }));
      card.appendChild(api.h('div', { class: 'glossary-q-text', html: q.q }));
      var opts = api.h('div', { class: 'glossary-opts' });
      var btns = q.opts.map(function (o, oi) {
        return api.button(opts, api.esc(o), function () { choose(qi, oi); }, 'glossary-opt');
      });
      card.appendChild(opts);
      var fb = api.h('div', { class: 'glossary-feedback', 'aria-live': 'polite' });
      card.appendChild(fb);
      list.appendChild(card);
      blocks.push({ btns: btns, fb: fb, q: q });
    });
    paint();
  }

  function quizReview(el, api) {
    var Q = quizData();
    var answered = 0, correct = 0;
    var rows = Q.map(function (q, qi) {
      var a = quizAnswers[qi];
      var status = '—';
      if (a != null) {
        answered++;
        if (a === q.a) { correct++; status = '<span class="glossary-ok">✓ right</span>'; }
        else status = '<span class="glossary-no">✗ wrong</span>';
      }
      return [String(qi + 1), api.esc(q.short), a == null ? '<span class="muted">not yet</span>' : api.esc(q.opts[a]),
        api.esc(q.opts[q.a]), status, api.link(q.target, secInfo(secOf(q.target)).title)];
    });
    api.html(el, '<p>Score: <b>' + correct + '</b> / ' + Q.length + ' · answered ' + answered + ' / ' + Q.length +
      '. Your answers are kept while you move between zoom levels; Reset is on the Questions level.</p>');
    api.table(el, { headers: ['#', 'Topic', 'Your answer', 'Correct answer', 'Result', 'Section'], rows: rows });
  }

  // ------------------------------------------------------------------ register
  var DT_SLIDE_DATA = CARDS.map(function (c) {
    return { name: 'example: ' + c.exShort, type: c.t, shape: c.shape, example: c.exShort };
  });

  Course.register({
    id: 'glossary',
    part: 'A',
    order: 200,
    title: 'Glossary & quiz',
    slides: [
      {
        id: 'data-types',
        title: 'Data-type reference',
        critical: {
          understand: [
            'Every datapoint in the course has one of 12 types (text, token, integer, float, probability, vector, matrix, tensor, mask, boolean, json, function).',
            'The type says what kind of thing a value is; the shape says how many of them there are, for example <code>[seq_len × d_model]</code>.'
          ],
          details: [
            'Reference model: d_model = 4,096 and d_head = 128. The course’s toy model uses d_model = 4 and d_head = 2.',
            'A <code>mask</code> is a pattern over positions; a <code>boolean</code> is a single flag. They are easy to confuse.'
          ]
        },
        dataTypes: DT_SLIDE_DATA,
        levels: [
          { name: 'Overview', render: renderTypesOverview },
          { name: 'Where they appear', render: renderTypesWhere }
        ]
      },
      {
        id: 'terms',
        title: 'Glossary of terms',
        critical: {
          understand: [
            'Most of the course reduces to a few steps: tokens, vectors, attention and FFN blocks, logits, sampling, and then an agent loop around the model.',
            'Type in the filter to match a term, definition or section. Click a section link to open the slide that explains the term.'
          ],
          details: [
            'Each term carries the data type of the thing it names, so you can match it with the badges on the slides.',
            'Level 2 groups the terms by section, in course order.',
            'Framework and product names (LangChain, LangGraph, ClearML, ComfyUI, MCP) change quickly; check their current docs.'
          ]
        },
        dataTypes: [
          { name: 'ID of " mat"', type: 'integer', shape: 'scalar', example: '2603' },
          { name: 'Q, K, V of one head', type: 'matrix', shape: '[seq_len × d_head]', example: '[5 × 2]' },
          { name: 'causal mask', type: 'mask', shape: '[seq_len × seq_len]', example: 'upper triangle = −∞' },
          { name: 'softmax output', type: 'probability', shape: '[vocab]', example: 'sums to 1' },
          { name: 'KV cache for one layer', type: 'tensor', shape: '[n_kv_heads × seq_len × d_head]', example: '[8 × 8192 × 128]' },
          { name: 'tool call', type: 'json', shape: 'object', example: '{"name": "book_table", …}' },
          { name: 'needs_approval', type: 'boolean', shape: 'scalar', example: 'true' },
          { name: 'agent step', type: 'function', shape: '(state) → state', example: 'harness loop' }
        ],
        levels: [
          { name: 'Overview', render: termsRender(false) },
          { name: 'By section', render: termsRender(true) }
        ]
      },
      {
        id: 'quiz',
        title: 'Quiz',
        critical: {
          understand: [
            'The 12 questions cover both parts of the course. Each answer links to the slide that explains it.',
            'Feedback appears at once and an answer locks after one try. Reset starts the quiz again.'
          ],
          details: [
            'Numbers come from the reference model and the GPT-2 token IDs used throughout the course.',
            'Answers are kept while you move between zoom levels, so the review on level 2 matches your score.'
          ]
        },
        dataTypes: [
          { name: 'ID of " mat"', type: 'integer', shape: 'scalar', example: '2603' },
          { name: 'KV cache per token', type: 'integer', shape: 'scalar', example: '131,072 bytes' },
          { name: 'causal mask', type: 'mask', shape: '[seq_len × seq_len]', example: 'future = −∞' },
          { name: 'next-token distribution', type: 'probability', shape: '[vocab]', example: 'sums to 1' },
          { name: 'tool call', type: 'json', shape: 'object', example: '{"name": "get_weather", …}' },
          { name: 'tool', type: 'function', shape: '(input) → output', example: 'get_weather' }
        ],
        levels: [
          { name: 'Questions', render: quizLevel },
          { name: 'Answer review', render: quizReview }
        ]
      }
    ]
  });
})();
