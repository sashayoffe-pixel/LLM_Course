(function () {
  'use strict';

  // ---------------------------------------------------------- recursive splitter (JS port of the idea)
  function recursiveSplit(text, size, overlap, seps) {
    function merge(splits) {
      var docs = [], cur = [], total = 0;
      splits.forEach(function (d) {
        var l = d.length;
        if (total + l > size && cur.length) {
          var doc = cur.join('').trim(); if (doc) docs.push(doc);
          while (cur.length && (total > overlap || total + l > size)) { total -= cur[0].length; cur.shift(); }
        }
        cur.push(d); total += l;
      });
      var last = cur.join('').trim(); if (last) docs.push(last);
      return docs;
    }
    function keepSplit(t, sep) {
      if (sep === '') return t.split('');
      var parts = t.split(sep), out = [];
      parts.forEach(function (p, i) { var piece = i < parts.length - 1 ? p + sep : p; if (piece) out.push(piece); });
      return out;
    }
    function rec(t, sepList) {
      var sep = sepList[sepList.length - 1], rest = [];
      for (var i = 0; i < sepList.length; i++) {
        if (sepList[i] === '') { sep = ''; break; }
        if (t.indexOf(sepList[i]) >= 0) { sep = sepList[i]; rest = sepList.slice(i + 1); break; }
      }
      var final = [], good = [];
      keepSplit(t, sep).forEach(function (s) {
        if (s.length < size) good.push(s);
        else {
          if (good.length) { final = final.concat(merge(good)); good = []; }
          if (!rest.length) final.push(s); else final = final.concat(rec(s, rest));
        }
      });
      if (good.length) final = final.concat(merge(good));
      return final;
    }
    var chunks = rec(text, seps), out = [], prevStart = 0, prevLen = 0;
    chunks.forEach(function (c, i) {
      var from = i === 0 ? 0 : Math.max(0, prevStart + prevLen - overlap - 1);
      var idx = text.indexOf(c, from); if (idx < 0) idx = text.indexOf(c);
      if (idx < 0) idx = 0;
      out.push({ text: c, start: idx, end: idx + c.length });
      prevStart = idx; prevLen = c.length;
    });
    return out;
  }

  var DOC = 'Acme Support Handbook\n\nRefunds. Items can be returned within 14 days of purchase if they are unused. Contact support with your order number to start a return. Refunds go back to the original payment method.\n\nShipping. Standard shipping takes 3 to 5 business days. Express shipping delivers in 1 to 2 days for an extra fee. You will receive a tracking number by email.\n\nWarranty. All devices include a 2 year warranty that covers manufacturing defects. Accidental damage is not covered, but our repair service can fix it for a fee.\n\nSupport hours. Support is available Monday to Friday from 9am to 6pm. Chat support is also open on weekends.';
  var SEPSETS = {
    default: ['\n\n', '\n', ' ', ''],
    sentence: ['\n\n', '\n', '. ', ' ', '']
  };
  var COLORS = ['#7ab8ff', '#7ee0a8', '#ffc46b', '#d79bff', '#ff8f8f', '#6fe3e3', '#c9d36a', '#ff9bd0'];

  // ---------------------------------------------------------- toy "embedding" for the walkthrough
  var AXES = [
    ['refund', ['refund', 'refunds', 'return', 'returned', 'returns', 'money', 'back']],
    ['shipping', ['shipping', 'delivery', 'deliver', 'delivers', 'ships', 'ship', 'days', 'arrive', 'tracking']],
    ['account', ['password', 'login', 'reset', 'account', 'forgot', 'sign']],
    ['warranty', ['warranty', 'defect', 'defects', 'damage', 'broken', 'repair']],
    ['support', ['support', 'hours', 'chat', 'contact', 'weekend', 'weekends', 'help', 'open']],
    ['billing', ['invoice', 'invoices', 'billing', 'card', 'payment', 'pay', 'charge', 'charged']],
    ['time', ['days', 'business', 'monday', 'friday', 'weekend', 'within', 'hours', 'long', 'when']]
  ];
  function embed(text) {
    var words = text.toLowerCase().match(/[a-z]+/g) || [];
    return AXES.map(function (ax) { var n = 0; words.forEach(function (w) { if (ax[1].indexOf(w) >= 0) n++; }); return n; });
  }
  var CORPUS = [
    { id: 'refunds.md#1', text: 'Items can be returned within 14 days of purchase if they are unused. Contact support with your order number to start a return.' },
    { id: 'shipping.md#1', text: 'Standard shipping takes 3 to 5 business days. Express shipping delivers in 1 to 2 days for an extra fee.' },
    { id: 'account.md#1', text: 'To reset your password, click Forgot password on the login page and follow the emailed link.' },
    { id: 'warranty.md#1', text: 'All devices include a 2 year warranty covering manufacturing defects but not accidental damage.' },
    { id: 'support.md#1', text: 'Support is available Monday to Friday from 9am to 6pm, with chat support on weekends.' },
    { id: 'billing.md#1', text: 'Invoices are emailed monthly. You can update your credit card or billing address in account settings.' }
  ];
  CORPUS.forEach(function (c) { c.vec = embed(c.text); });
  function search(api, q, k) {
    var qv = embed(q);
    var sc = CORPUS.map(function (c, i) {
      var nz = api.math.norm(qv) * api.math.norm(c.vec);
      return { i: i, c: c, score: nz === 0 ? 0 : api.math.cosine(qv, c.vec) };
    });
    sc.sort(function (a, b) { return b.score - a.score; });
    return { qv: qv, ranked: sc, top: sc.slice(0, k) };
  }
  function vecHtml(v) { return '[' + v.join(', ') + ']'; }

  function wrap(api, el) { var d = api.h('div', { class: 'rag-practice-split' }); el.appendChild(d); var i = api.h('div'); d.appendChild(i); return i; }

  Course.register({
    id: 'rag-practice',
    part: 'II',
    order: 135,
    title: 'RAG in Practice',
    slides: [
      // ------------------------------------------------------------ langchain-rag
      {
        id: 'langchain-rag',
        title: 'RAG with LangChain: splitter → embeddings → FAISS → retriever → chain',
        critical: {
          understand: [
            'RAG has two phases: <b>index</b> (offline: load → split → embed → store) and <b>query</b> (online: embed question → top-k search → stuff chunks into the prompt → LLM answers).',
            'LangChain provides one component per box: <code>RecursiveCharacterTextSplitter</code>, an <code>Embeddings</code> class, a vector store such as <code>FAISS</code>, <code>as_retriever()</code> and <code>prompt | llm</code>.',
            'The model never "searches": retrieved text simply becomes more tokens in the prompt (see ' + '<span data-goto="memory-rag/rag-pipeline" class="xlink">RAG pipeline</span>).'
          ],
          details: [
            'FAISS is an in-process similarity-search library; any other vector store exposes the same retriever interface, so the chain does not change.',
            'Embedding the query and the chunks must use the <b>same</b> embedding model.'
          ]
        },
        dataTypes: [
          { name: 'documents', type: 'text', shape: 'list[Document]', example: 'page_content + metadata' },
          { name: 'chunks', type: 'text', shape: 'list[Document]', example: '~500 chars each' },
          { name: 'chunk embeddings', type: 'matrix', shape: '[n_chunks × d_embed]', example: '[120 × 1536]' },
          { name: 'query embedding', type: 'vector', shape: '[d_embed]', example: '[1536]' },
          { name: 'top-k docs', type: 'text', shape: 'list[Document]', example: 'k = 4' },
          { name: 'retriever / chain', type: 'function', shape: 'Runnable', example: 'retriever | prompt | llm' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [[
              { id: 'd', label: 'Documents', sub: 'loader', type: 'text', shape: 'list[Document]' },
              { id: 's', label: 'Splitter', sub: 'RecursiveCharacterTextSplitter', type: 'function', edge: 'chunks', goto: 'rag-practice/chunking-demo' },
              { id: 'e', label: 'Embeddings', type: 'function', shape: '→ [n × d_embed]', edge: 'texts' },
              { id: 'v', label: 'FAISS', sub: 'vector store', type: 'matrix', edge: 'vectors' }
            ], [
              { id: 'q', label: 'Question', type: 'text', shape: 'string' },
              { id: 'r', label: 'Retriever', sub: 'top-k similarity', type: 'function', edge: 'embed + search', goto: 'rag-practice/query-walkthrough' },
              { id: 'p', label: 'Prompt', sub: 'question + context', type: 'text', edge: 'k chunks' },
              { id: 'l', label: 'LLM', type: 'function', edge: 'one token sequence', goto: 'chat-api/chat-template' }
            ]], caption: 'Top row: index once (offline). Bottom row: every question (online).' });
          } },
          { name: 'Stages', render: function (el, api) {
            var stages = [
              ['Load', 'text', 'list[Document]', 'docs = TextLoader("handbook.txt").load()'],
              ['Split', 'text', 'list[Document]', 'splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\nchunks = splitter.split_documents(docs)'],
              ['Embed + store', 'matrix', '[n_chunks × d_embed]', 'vectorstore = FAISS.from_documents(chunks, embeddings)'],
              ['Retrieve', 'text', 'list[Document]', 'retriever = vectorstore.as_retriever(search_kwargs={"k": 4})\ndocs = retriever.invoke("How long is the warranty?")'],
              ['Stuff into prompt', 'text', 'string', 'context = "\\n\\n".join(d.page_content for d in docs)\nprompt.invoke({"context": context, "question": q})'],
              ['Generate', 'text', 'string', 'answer = (prompt | llm | StrOutputParser()).invoke(...)']
            ];
            var info = api.h('div'); var code = api.h('div');
            api.stepper(el, { steps: stages.map(function (s) { return { title: s[0] }; }), interval: 2600, onStep: function (i) {
              var s = stages[i];
              info.innerHTML = 'Output of this stage: ' + api.badge(s[1], s[2]);
              code.innerHTML = '';
              api.code(code, s[3], 'python');
            } });
            el.appendChild(info); el.appendChild(code);
            api.docsNote(el, 'Package locations move between releases (<code>langchain_text_splitters</code>, <code>langchain_community</code>, provider packages). Check your installed version.');
          } },
          { name: 'Full code', render: function (el, api) {
            var w = wrap(api, el);
            api.code(w, [
              'from langchain_community.document_loaders import TextLoader',
              'from langchain_text_splitters import RecursiveCharacterTextSplitter',
              'from langchain_community.vectorstores import FAISS',
              'from langchain_core.prompts import ChatPromptTemplate',
              'from langchain_core.output_parsers import StrOutputParser',
              'from langchain_core.runnables import RunnablePassthrough',
              '# embeddings and llm: any LangChain Embeddings / chat model, e.g.',
              '# embeddings = SomeProviderEmbeddings(model="...")',
              '# llm = init_chat_model("provider:some-model")',
              '',
              '# ---- index (offline) ----',
              'docs = TextLoader("handbook.txt").load()',
              'splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)',
              'chunks = splitter.split_documents(docs)',
              'vectorstore = FAISS.from_documents(chunks, embeddings)',
              'retriever = vectorstore.as_retriever(search_kwargs={"k": 4})',
              '',
              '# ---- query (online) ----',
              'def format_docs(docs):',
              '    return "\\n\\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))',
              '',
              'prompt = ChatPromptTemplate.from_messages([',
              '    ("system", "Answer using ONLY the context. Cite sources like [1]."),',
              '    ("human", "Context:\\n{context}\\n\\nQuestion: {question}"),',
              '])',
              'rag_chain = (',
              '    {"context": retriever | format_docs, "question": RunnablePassthrough()}',
              '    | prompt | llm | StrOutputParser()',
              ')',
              'print(rag_chain.invoke("How long is the warranty?"))'
            ].join('\n'), 'python');
            api.docsNote(el, 'Written in LangChain ≥ 1.0 style; the dict-of-runnables syntax is LCEL (' + api.link('langchain/lcel', 'LCEL') + '). FAISS needs the <code>faiss-cpu</code> package.');
          } }
        ]
      },

      // ------------------------------------------------------------ chunking-demo
      {
        id: 'chunking-demo',
        title: 'Live chunking: chunk_size, chunk_overlap, recursive separators',
        critical: {
          understand: [
            'The recursive splitter tries the <b>biggest natural boundary first</b> (paragraphs <code>\\n\\n</code>), falls back to lines, sentences, then words, only when a piece is still larger than <code>chunk_size</code>.',
            'Small pieces are <b>merged</b> back up toward <code>chunk_size</code>; <code>chunk_overlap</code> repeats the tail of one chunk at the head of the next, so facts on a boundary are not lost.',
            'Sizes here are in <b>characters</b> (LangChain\'s default length function); many setups count tokens instead.'
          ],
          details: [
            'This is a JS re-implementation of the idea, not the library code; details (separator handling, whitespace stripping) may differ slightly.',
            'Trade-off: small chunks retrieve precisely but lose context; large chunks keep context but dilute the embedding and cost more prompt tokens.'
          ]
        },
        dataTypes: [
          { name: 'document', type: 'text', shape: 'string', example: '~800 characters' },
          { name: 'chunk_size', type: 'integer', shape: 'scalar', example: '200' },
          { name: 'chunk_overlap', type: 'integer', shape: 'scalar', example: '40' },
          { name: 'separators', type: 'text', shape: 'list[str]', example: '["\\n\\n", "\\n", " ", ""]' },
          { name: 'chunks', type: 'text', shape: 'list[str]', example: '6 chunks' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var secs = [['paragraphs', '\\n\\n'], ['lines', '\\n'], ['sentences', '". "'], ['words', '" "'], ['characters', '""']];
            api.html(el, '<p>Try separators in order; recurse into any piece that is still too big:</p>');
            api.blocks(el, { blocks: secs.map(function (s, i) { return { id: 's' + i, label: s[0], sub: 'sep ' + s[1], type: 'text', edge: i ? 'still too big?' : '', zoom: i === 0 ? true : undefined }; }), caption: 'Then merge the small pieces back up to chunk_size, with overlap.' });
            api.callout(el, 'Goal: keep semantically related text together. A paragraph boundary is a better cut than the middle of a word.', 'tip');
          } },
          { name: 'Live demo', render: function (el, api) {
            var st = { text: DOC, size: 140, overlap: 40, seps: 'default' };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var out = api.h('div'), ta;
            api.slider(c, { init: false, label: 'chunk_size', min: 50, max: 600, step: 10, value: 140, onInput: function (v) { st.size = v; draw(); } });
            api.slider(c, { init: false, label: 'chunk_overlap', min: 0, max: 200, step: 5, value: 40, onInput: function (v) { st.overlap = v; draw(); } });
            api.select(c, { init: false, label: 'separators', options: [{ value: 'default', label: 'paragraph → line → word' }, { value: 'sentence', label: 'paragraph → line → sentence → word' }], value: 'default', onChange: function (v) { st.seps = v; draw(); } });
            api.input(el, { init: false, label: 'Document (edit me)', value: DOC, multiline: true, onInput: function (v) { st.text = v; draw(); } });
            el.appendChild(out);
            function draw() {
              out.innerHTML = '';
              var ov = Math.min(st.overlap, Math.max(0, st.size - 1));
              var text = st.text || '';
              if (!text.trim()) { out.appendChild(api.h('div', { class: 'muted', text: 'Type some text above.' })); return; }
              var ch = recursiveSplit(text, st.size, ov, SEPSETS[st.seps]);
              var L = text.length;
              var map = api.h('div', { class: 'rag-practice-map' });
              ch.forEach(function (k, i) {
                var lane = i % 2;
                map.appendChild(api.h('i', { style: { left: (100 * k.start / L) + '%', width: Math.max(0.5, 100 * (k.end - k.start) / L) + '%', background: COLORS[i % COLORS.length], top: lane ? '50%' : '0', height: '50%' } }));
              });
              out.appendChild(map);
              var avg = Math.round(api.math.sum(ch.map(function (k) { return k.text.length; })) / ch.length);
              var dup = api.math.sum(ch.map(function (k) { return k.text.length; })) - L;
              out.appendChild(api.h('div', { class: 'muted', html: '<b>' + ch.length + '</b> chunks · average ' + avg + ' chars · total chunk text ' + api.fmtInt(api.math.sum(ch.map(function (k) { return k.text.length; }))) + ' vs document ' + api.fmtInt(L) + ' chars (' + (dup >= 0 ? '+' : '') + api.fmtInt(dup) + ' from overlap / trimming). Top bar: where each chunk sits in the document (two lanes so overlaps are visible).' }));
              var list = api.h('div', { class: 'rag-practice-chunks' }); out.appendChild(list);
              ch.forEach(function (k, i) {
                var prev = ch[i - 1], ovLen = prev ? Math.max(0, prev.end - k.start) : 0;
                ovLen = Math.min(ovLen, k.text.length);
                var d = api.h('div', { class: 'rag-practice-chunk', style: { borderLeftColor: COLORS[i % COLORS.length] } });
                d.appendChild(api.h('div', { class: 'rag-practice-meta', text: 'chunk ' + (i + 1) + ' · chars ' + k.start + '–' + k.end + ' · ' + k.text.length + ' long' + (ovLen ? ' · ' + ovLen + ' overlap with previous (highlighted)' : '') }));
                if (ovLen) d.appendChild(api.h('span', { class: 'rag-practice-ov', text: k.text.slice(0, ovLen) }));
                d.appendChild(document.createTextNode(k.text.slice(ovLen)));
                list.appendChild(d);
              });
            }
            draw();
          } },
          { name: 'Algorithm', render: function (el, api) {
            var w = wrap(api, el);
            api.code(w, [
              'def split(text, separators):',
              '    sep = first separator that occurs in text',
              '    pieces = text.split(sep)            # keep the separator',
              '    good = []',
              '    for p in pieces:',
              '        if len(p) < chunk_size:',
              '            good.append(p)              # small enough',
              '        else:',
              '            flush(merge(good)); good = []',
              '            recurse into p with the NEXT separators',
              '    flush(merge(good))',
              '',
              'def merge(pieces):',
              '    glue pieces until adding one more would exceed chunk_size;',
              '    emit the chunk; then drop pieces from the FRONT',
              '    until the remainder is <= chunk_overlap; continue'
            ].join('\n'), 'text');
            api.html(el, '<p>That "drop from the front until the remainder is ≤ <code>chunk_overlap</code>" step is exactly what creates the repeated text between neighbouring chunks.</p>');
            api.docsNote(el, 'Pseudo-code of the library\'s behaviour; the real class has more options (<code>keep_separator</code>, <code>length_function</code>, <code>add_start_index</code>, <code>from_tiktoken_encoder</code>).');
          } }
        ]
      },

      // ------------------------------------------------------------ query-walkthrough
      {
        id: 'query-walkthrough',
        title: 'Query walkthrough: embed → search → stuff → answer',
        critical: {
          understand: [
            'The question is embedded into a vector; the store returns the <b>k chunks whose vectors are most similar</b> (cosine or inner product).',
            'The chunks are pasted into the prompt as text with source labels; the LLM then answers from that context and can cite them.',
            'Retrieval quality caps answer quality: if the right chunk is not in the top-k, the model cannot use it.'
          ],
          details: [
            'The embedding here is a toy keyword-count vector so you can see every number; real embeddings are learned, dense and hundreds to thousands of dimensions (see ' + '<span data-goto="embeddings/cosine" class="xlink">cosine</span>).',
            'Citations come from asking the model to reference the labelled chunks; verify them — models can cite incorrectly.'
          ]
        },
        dataTypes: [
          { name: 'question', type: 'text', shape: 'string', example: '"How long does delivery take?"' },
          { name: 'query vector', type: 'vector', shape: '[7]', example: 'toy: keyword counts' },
          { name: 'similarity scores', type: 'float', shape: '[n_chunks]', example: 'cosine ∈ [0,1]' },
          { name: 'top-k', type: 'integer', shape: 'scalar', example: '2' },
          { name: 'prompt', type: 'text', shape: 'string', example: 'context + question' },
          { name: 'answer', type: 'text', shape: 'string', example: '"… [1]"' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'q', label: 'query', type: 'text', shape: 'string' },
              { id: 'e', label: 'embed', type: 'vector', shape: '[d_embed]', edge: 'embedding model', zoom: true },
              { id: 's', label: 'similarity search', type: 'float', shape: '[n_chunks]', edge: 'cosine vs all chunks', zoom: true },
              { id: 'k', label: 'top-k chunks', type: 'text', shape: 'list[str]', edge: 'sort, keep k' },
              { id: 'p', label: 'prompt', type: 'text', shape: 'string', edge: 'stuff', goto: 'chat-api/chat-template' },
              { id: 'a', label: 'answer + citations', type: 'text', shape: 'string', edge: 'LLM' }
            ] });
            api.callout(el, 'Compare with the conceptual walk-through in ' + api.link('memory-rag/mini-rag', 'mini RAG') + '.', 'info');
          } },
          { name: 'Stepper', render: function (el, api) {
            var st = { q: 'How long does delivery take?', k: 2 };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var body = api.h('div'); var sp = null;
            api.select(c, { init: false, label: 'question', options: ['How long does delivery take?', 'Can I get my money back?', 'I forgot my password', 'Is accidental damage covered?', 'When is chat open on weekends?'], value: st.q, onChange: function (v) { st.q = v; build(); } });
            api.slider(c, { init: false, label: 'top-k', min: 1, max: 4, step: 1, value: 2, onInput: function (v) { st.k = v; build(); } });
            el.appendChild(body);
            function steps() {
              var r = search(api, st.q, st.k);
              return [
                { title: '1. Query', html: 'Question ' + api.badge('text', 'string') + ': <b>' + api.esc(st.q) + '</b>' },
                { title: '2. Embed', html: 'Toy embedding over axes <code>' + AXES.map(function (a) { return a[0]; }).join(', ') + '</code> ' + api.badge('vector', '[7]') + '<div class="formula">q = ' + vecHtml(r.qv) + '</div>' },
                { title: '3. Search', render: function (b) {
                  b.appendChild(api.h('div', { html: 'Cosine similarity against every chunk ' + api.badge('float', '[6]') + '; keep the top ' + st.k + ':' }));
                  api.bars(b, { items: r.ranked.map(function (x, j) { return { label: x.c.id, value: x.score, highlight: j < st.k, dim: j >= st.k }; }), max: 1, format: function (v) { return api.fmt(v, 3); } });
                } },
                { title: '4. Stuff prompt', render: function (b) {
                  var ctx = r.top.map(function (x, j) { return '[' + (j + 1) + '] (' + x.c.id + ') ' + x.c.text; }).join('\n');
                  b.appendChild(api.h('div', { class: 'rag-practice-prompt', text: 'SYSTEM: Answer using only the context. Cite sources like [1].\n\nContext:\n' + ctx + '\n\nQuestion: ' + st.q }));
                } },
                { title: '5. Answer', render: function (b) {
                  var t = r.top[0];
                  var ok = t.score > 0;
                  b.appendChild(api.h('div', { class: 'rag-practice-card hit', html: ok ? ('Scripted answer: based on the best match, <i>"' + api.esc(t.c.text) + '"</i> <span class="rag-practice-cite">[1]</span>' + (r.top.length > 1 && r.top[1].score > 0 ? ' Also relevant: ' + api.esc(r.top[1].c.id) + ' <span class="rag-practice-cite">[2]</span>.' : '')) : 'No chunk matched: a good prompt tells the model to say "I don\'t know".' }));
                } }
              ];
            }
            function build() { body.innerHTML = ''; sp = api.stepper(body, { steps: steps(), interval: 2600 }); }
            build();
            api.illustrative(el, 'Toy keyword-count embeddings and a scripted "answer" (a real LLM would phrase it itself). All scores are computed live.');
            api.docsNote(el, 'Similarity metric (cosine vs inner product vs L2) is a vector-store setting; embeddings are usually normalised so cosine and inner product agree.');
          } },
          { name: 'Numbers', render: function (el, api) {
            var q = 'How long does delivery take?';
            var r = search(api, q, 3);
            var t = r.top[0];
            api.html(el, '<p>Question: <b>' + api.esc(q) + '</b> → best chunk <b>' + api.esc(t.c.id) + '</b></p>');
            api.html(el, '<div class="formula">q = ' + vecHtml(r.qv) + '<br>c = ' + vecHtml(t.c.vec) + '<br>q·c = ' + api.math.dot(r.qv, t.c.vec) + '<br>|q| = ' + api.fmt(api.math.norm(r.qv), 3) + ' , |c| = ' + api.fmt(api.math.norm(t.c.vec), 3) + '<br>cos = q·c / (|q||c|) = ' + api.fmt(t.score, 3) + '</div>');
            var rows = r.ranked.map(function (x) { return [x.c.id, vecHtml(x.c.vec), api.math.dot(r.qv, x.c.vec).toString(), api.fmt(x.score, 3)]; });
            api.table(el, { headers: ['chunk', 'vector', 'q·c', 'cosine'], rows: rows });
            api.illustrative(el, 'Toy vectors. Note "days" appears on both the shipping and time axes, which is why the time-flavoured chunks also score above zero.');
          } }
        ]
      },

      // ------------------------------------------------------------ production-tips
      {
        id: 'production-tips',
        title: 'Production tips: sizing, metadata, evaluation, re-indexing, hybrid search',
        critical: {
          understand: [
            'Most RAG failures are <b>retrieval</b> failures (right chunk not retrieved), not generation failures — measure retrieval separately.',
            'Tune chunk size and overlap on <i>your</i> documents and questions; there is no universal best value.',
            'Combine vector search with keyword search, metadata filters and reranking when exact terms or scoping matter (see ' + '<span data-goto="memory-rag/advanced-rag" class="xlink">advanced RAG</span>).'
          ],
          details: [
            'Store metadata (source, section, date, access rights) with each chunk; filter before similarity search where the store supports it.',
            'Plan re-indexing: changing the embedding model, the chunking, or the documents requires rebuilding or incrementally updating the index; keep an index version.'
          ]
        },
        dataTypes: [
          { name: 'metadata', type: 'json', shape: 'dict per chunk', example: '{source, section, date}' },
          { name: 'recall@k', type: 'probability', shape: 'scalar', example: '0.83' },
          { name: 'filter', type: 'mask', shape: '[n_chunks]', example: 'keeps matching chunks' },
          { name: 'k', type: 'integer', shape: 'scalar', example: '4' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            var g = api.grid(el, 230);
            [
              ['Chunk sizing', 'Start around a few hundred tokens with 10–20% overlap (a common starting point, not a rule); try structure-aware splits (headings, code blocks).'],
              ['Metadata filters', 'Filter by source, date, tenant or permissions <i>before</i> ranking. Never rely on the prompt to enforce access control.'],
              ['Evaluation', 'Build a small question → expected-chunk set; track recall@k and answer quality when you change anything.'],
              ['Re-indexing', 'Version your index; re-embed when the embedding model or chunking changes. Do not mix vectors from different models.'],
              ['Hybrid search', 'Blend keyword (BM25) and vector scores; add a reranker for the top ~50. Helps with IDs, names and rare terms.']
            ].forEach(function (x) { api.card(g, x[0], x[1]); });
            api.callout(el, 'Go deeper: ' + api.link('memory-rag/advanced-rag', 'hybrid search, reranking, agentic RAG') + ' and ' + api.link('safety-eval/evals', 'evals') + '.', 'info');
          } },
          { name: 'Checklist', render: function (el, api) {
            api.table(el, { headers: ['Symptom', 'Likely cause', 'Try'], rows: [
              ['Answer ignores the right fact', 'Chunk not retrieved', 'Raise k, hybrid search, rerank, smaller chunks'],
              ['Answer is vague / lacks context', 'Chunks too small', 'Larger chunks or parent-document retrieval'],
              ['Exact IDs / names not found', 'Pure vector search', 'Add keyword search'],
              ['Outdated answers', 'Stale index', 'Incremental re-indexing, date metadata'],
              ['Hallucinated citations', 'Model free-styles', 'Require quotes, verify cited IDs in code'],
              ['Huge prompts and cost', 'k too high, chunks too big', 'Lower k, compress, rerank (see ' + api.link('chat-api/cost-caching', 'cost') + ')']
            ] });
            api.docsNote(el, 'Tool names for hybrid search and reranking (e.g. BM25 retrievers, ensemble retrievers) vary by LangChain version and vector store.');
          } },
          { name: 'Eval demo', render: function (el, api) {
            var EVAL = [
              ['How long does delivery take?', 1], ['Can I get my money back?', 0], ['I forgot my password', 2],
              ['Is accidental damage covered?', 3], ['When is chat open on weekends?', 4], ['Update my credit card', 5],
              ['How many days until I can return an item?', 0], ['What are your hours?', 4]
            ];
            var ranks = EVAL.map(function (e) {
              var r = search(api, e[0], CORPUS.length).ranked;
              for (var i = 0; i < r.length; i++) if (r[i].i === e[1]) return i + 1;
              return CORPUS.length;
            });
            var out = api.h('div'); var tab = api.h('div');
            function draw(k) {
              out.innerHTML = '';
              var hits = ranks.filter(function (rk) { return rk <= k; }).length;
              api.html(out, '<div class="card"><div class="muted">recall@' + k + ' (expected chunk in top-' + k + ')</div><div class="big-num">' + hits + ' / ' + EVAL.length + ' = ' + api.fmt(hits / EVAL.length, 2) + '</div></div>');
              tab.innerHTML = '';
              api.table(tab, { headers: ['question', 'expected chunk', 'rank', 'hit?'], rows: EVAL.map(function (e, i) { return [e[0], CORPUS[e[1]].id, String(ranks[i]), ranks[i] <= k ? '✓' : '✗']; }) });
            }
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'k', min: 1, max: 4, step: 1, value: 1, onInput: draw });
            el.appendChild(out); el.appendChild(tab);
            api.illustrative(el, 'Tiny labelled set on the toy corpus from the previous slide. Real eval sets need many more questions, and the same recall@k idea applies.');
          } }
        ]
      }
    ]
  });
})();
