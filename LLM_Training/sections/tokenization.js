(function () {
  'use strict';

  // ---------------- toy BPE (fixed merge list, byte-level fallback) ----------------
  var MERGES = [['t', 'h'], [' ', 'th'], [' th', 'e'], ['a', 't'], [' ', 'c'], [' c', 'at'], [' ', 's'], [' s', 'at'],
    ['o', 'n'], [' ', 'on'], ['i', 'n'], ['in', 'g'], [' ', 'm'], [' m', 'at'], ['e', 'r'], ['s', 't']];
  function pretok(text) { return text.match(/ ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g) || []; }
  function toSymbols(w) {
    var out = [], bytes = new TextEncoder().encode(w);
    for (var i = 0; i < bytes.length; i++) {
      var b = bytes[i];
      out.push(b < 128 ? String.fromCharCode(b) : '<0x' + (b < 16 ? '0' : '') + b.toString(16).toUpperCase() + '>');
    }
    return out;
  }
  function applyMerges(syms, log) {
    syms = syms.slice();
    MERGES.forEach(function (m, mi) {
      for (var i = 0; i < syms.length - 1;) {
        if (syms[i] === m[0] && syms[i + 1] === m[1]) { syms.splice(i, 2, m[0] + m[1]); if (log) log.push({ merge: mi, syms: syms.slice() }); } else i++;
      }
    });
    return syms;
  }
  function toyId(s) {
    var mi = -1;
    MERGES.forEach(function (m, i) { if (m[0] + m[1] === s) mi = i; });
    if (mi >= 0) return 256 + mi;
    var mm = /^<0x([0-9A-F]{2})>$/.exec(s);
    if (mm) return parseInt(mm[1], 16);
    return s.charCodeAt(0);
  }
  function toyTokenize(text) {
    var out = [];
    pretok(text).forEach(function (w) { applyMerges(toSymbols(w)).forEach(function (s) { out.push({ text: s, id: toyId(s) }); }); });
    return out;
  }
  function chips(api, toks) {
    return '<div class="tokenization-chips">' + toks.map(function (t) {
      var byte = /^<0x/.test(t.text);
      return '<span class="tokenization-chip' + (byte ? ' tokenization-byte' : '') + '">' + (byte ? api.esc(t.text) : api.tokText(t.text)) + '<small>' + t.id + '</small></span>';
    }).join('') + '</div>';
  }

  // ---------------- BPE training ----------------
  function trainBPE(corpus, nMerges) {
    var words = corpus.map(function (c) { return { syms: c[0].split('').concat(['</w>']), n: c[1] }; });
    var steps = [];
    for (var k = 0; k < nMerges; k++) {
      var counts = {};
      words.forEach(function (w) { for (var i = 0; i < w.syms.length - 1; i++) { var key = w.syms[i] + '\u0001' + w.syms[i + 1]; counts[key] = (counts[key] || 0) + w.n; } });
      var keys = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); });
      if (!keys.length) break;
      var best = keys[0].split('\u0001');
      steps.push({ counts: keys.slice(0, 6).map(function (q) { return [q.split('\u0001'), counts[q]]; }), best: best, bestN: counts[keys[0]], before: words.map(function (w) { return w.syms.slice(); }) });
      words.forEach(function (w) { for (var i = 0; i < w.syms.length - 1;) { if (w.syms[i] === best[0] && w.syms[i + 1] === best[1]) w.syms.splice(i, 2, best[0] + best[1]); else i++; } });
      steps[steps.length - 1].after = words.map(function (w) { return w.syms.slice(); });
    }
    return steps;
  }
  var CORPUS = [['low', 5], ['lower', 2], ['newest', 6], ['widest', 3]];

  function wordsHtml(api, state, corpus, hl) {
    return '<table class="tbl"><tr><th>word</th><th>count</th><th>symbols</th></tr>' + state.map(function (s, i) {
      return '<tr><td>' + api.esc(corpus[i][0]) + '</td><td>' + corpus[i][1] + '</td><td class="tokenization-pair">' + s.map(function (x) {
        var m = hl && x === hl;
        return '<span class="' + (m ? 'tokenization-hl' : '') + '">' + api.esc(x) + '</span>';
      }).join(' · ') + '</td></tr>';
    }).join('') + '</table>';
  }

  Course.register({
    id: 'tokenization', part: 'I', order: 20, title: 'Tokenization',
    slides: [
      // ---------------------------------------------------------------- why tokens
      {
        id: 'why-tokens', title: 'Why tokens? Between characters and words',
        critical: {
          understand: ['A model needs a fixed list of symbols (the <b>vocabulary</b>). Characters are too fine, whole words too many; <b>sub-words</b> are the compromise.',
            'Common words become one token; rare words are assembled from pieces, so nothing is ever "unknown".'],
          details: ['The 8B reference uses a vocabulary of 128,256 tokens; GPT-2 used 50,257.',
            'Fewer tokens per text = cheaper and faster, because cost scales with sequence length.']
        },
        dataTypes: [
          { name: 'text', type: 'text', shape: 'string', example: '"unbelievable"' },
          { name: 'tokens', type: 'token', shape: '[seq_len]', example: '"un", "believ", "able" (illustrative)' },
          { name: 'vocab size', type: 'integer', shape: 'scalar', example: '128,256' }
        ],
        levels: [
          { name: 'Trade-off', render: function (el, api) {
            var g = api.h('div', { class: 'tokenization-cols' }); el.appendChild(g);
            [['Characters', 'vocab ≈ 100–300', 'Very long sequences (every letter a step); the model must learn spelling→meaning from scratch.', 'u n b e l i e v a b l e', 12],
             ['Whole words', 'vocab 500,000+', 'Short sequences but a huge table, and any new word, typo or name is "out of vocabulary".', 'unbelievable', 1],
             ['Sub-words (BPE)', 'vocab ≈ 30K–250K', 'Frequent words stay whole, rare ones split into reusable pieces. The modern default.', 'un | believ | able', 3]
            ].forEach(function (c) {
              var d = api.h('div', { class: 'card' });
              d.innerHTML = '<b>' + c[0] + '</b><div class="muted">' + c[1] + '</div><div style="margin:6px 0"><code>' + api.esc(c[3]) + '</code></div><div>' + api.esc(c[2]) + '</div><div class="muted">≈ ' + c[4] + ' tokens for "unbelievable" (illustrative)</div>';
              g.appendChild(d);
            });
          } },
          { name: 'Measure it', render: function (el, api) {
            var out = api.h('div');
            api.input(el, { label: 'Your text', value: 'Tokenization is unbelievably useful.', onInput: draw });
            el.appendChild(out);
            function draw(t) {
              var toks = toyTokenize(t), words = (t.match(/\S+/g) || []).length, chars = t.length;
              out.innerHTML = '';
              api.bars(out, { title: 'Sequence length the model must process (' + api.badge('integer', 'scalar') + ')', items: [
                { label: 'characters', value: chars }, { label: 'toy-BPE tokens', value: toks.length, highlight: true }, { label: 'words', value: words }], format: function (v) { return String(v); } });
              out.appendChild(api.h('div', { class: 'muted', html: 'Toy tokenizer has only 16 merges, so it sits close to "characters"; a real tokenizer with ≈ 100K merges lands near the words line (about 1.3 tokens per English word is a common rule of thumb).' }));
            }
          } }
        ]
      },
      // ---------------------------------------------------------------- bpe playground
      {
        id: 'bpe-playground', title: 'Toy BPE tokenizer playground',
        critical: {
          understand: ['A trained BPE tokenizer is just an <b>ordered list of merges</b>. To tokenize: split into bytes/characters, then apply the merges in order.',
            'The result is a list of tokens, each mapped to an integer ID.'],
          details: ['This toy has 16 merges; real vocabularies have 30K–250K.',
            'Unknown characters fall back to raw bytes (shown as &lt;0x..&gt;), so any text can be encoded.']
        },
        dataTypes: [
          { name: 'text', type: 'text', shape: 'string', example: 'the cat sat on the mat' },
          { name: 'merge list', type: 'json', shape: 'list[pair]', example: '[["t","h"],[" ","th"],…]' },
          { name: 'tokens', type: 'token', shape: '[seq_len]', example: '" the", " cat", …' },
          { name: 'IDs', type: 'integer', shape: '[seq_len]', example: '[258, 261, …]' }
        ],
        levels: [
          { name: 'Playground', render: function (el, api) {
            var out = api.h('div');
            api.input(el, { label: 'Type text', value: 'the cat sat on the mat', onInput: draw });
            el.appendChild(out);
            function draw(t) {
              var toks = toyTokenize(t);
              out.innerHTML = '<div>' + api.badge('token', '[' + toks.length + ']') + ' with ' + api.badge('integer', '[' + toks.length + ']') + ' IDs (toy IDs: 0–255 = single bytes, 256+ = merges)</div>' + chips(api, toks) +
                '<div class="muted">IDs: <code>[' + toks.map(function (x) { return x.id; }).join(', ') + ']</code></div>';
            }
            api.callout(el, 'Try <code>über</code>, <code>日本</code> or <code>🙂</code>: pieces the toy never merged fall back to raw UTF-8 bytes.', 'tip');
            api.illustrative(el, 'IDs are specific to this toy; real GPT-2 IDs appear on the text → IDs slide.');
          } },
          { name: 'Merge by merge', render: function (el, api) {
            var out = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.input(c, { label: 'One word', value: ' that', onInput: draw });
            el.appendChild(out);
            function draw(w) {
              w = w || ' ';
              var log = [], base = toSymbols(w), fin = applyMerges(base, log);
              var rows = [['start', base.map(api.esc).join(' · ')]];
              log.forEach(function (l) { var m = MERGES[l.merge]; rows.push(['merge #' + (l.merge + 1) + ': ' + api.esc(m[0]) + ' + ' + api.esc(m[1]), l.syms.map(api.esc).join(' · ')]); });
              out.innerHTML = '';
              api.table(out, { headers: ['Step', 'Symbols ' + api.type('token')], rows: rows });
              out.appendChild(api.h('div', { class: 'muted', html: 'Final: ' + fin.length + ' token(s). A leading space is part of the token (␣ shown).' }));
            }
          } },
          { name: 'Merge table', render: function (el, api) {
            api.table(el, { headers: ['Rank', 'Pair', 'New token', 'ID'], rows: MERGES.map(function (m, i) { return [i + 1, '<code>' + api.esc(m[0]) + '</code> + <code>' + api.esc(m[1]) + '</code>', api.tok(m[0] + m[1]), 256 + i]; }) });
            api.html(el, '<p>Order matters: earlier merges win. Where did this list come from? From <b>counting</b> — next slide: ' + api.link('tokenization/bpe-training', 'BPE training') + '.</p>');
          } }
        ]
      },
      // ---------------------------------------------------------------- bpe training
      {
        id: 'bpe-training', title: 'How a BPE vocabulary is learned',
        critical: {
          understand: ['Start with characters. Repeatedly <b>count adjacent pairs</b>, merge the most frequent one into a new symbol, and repeat.',
            'Each merge adds one vocabulary entry; stop when the vocabulary reaches the target size.'],
          details: ['Counts are weighted by how often each word appears in the training corpus.',
            'Ties are broken arbitrarily (here: alphabetically); real implementations differ in details such as byte-level handling.']
        },
        dataTypes: [
          { name: 'corpus word counts', type: 'json', shape: 'list[(string, integer)]', example: '[("low",5), ("newest",6), …]' },
          { name: 'pair counts', type: 'integer', shape: 'dict[pair]', example: '("e","s") → 9' },
          { name: 'merge rule', type: 'function', shape: 'pair → symbol', example: 'e + s → es' }
        ],
        levels: [
          { name: 'Idea', render: function (el, api) {
            api.blocks(el, { blocks: [
              { label: 'Corpus', sub: 'words + counts', type: 'json', shape: 'list' },
              { label: 'Split to characters', type: 'token', shape: 'list[list]' },
              { label: 'Count adjacent pairs', type: 'integer', shape: 'dict' },
              { label: 'Merge the top pair', type: 'function', shape: 'new symbol' },
              { label: 'Repeat', sub: 'until vocab size', type: 'boolean', shape: 'done?', zoom: 1 }
            ], caption: 'Dive in to run it on a tiny corpus.' });
          } },
          { name: 'Walkthrough', render: function (el, api) {
            var steps = trainBPE(CORPUS, 8);
            api.html(el, '<div class="muted">Corpus (illustrative): ' + CORPUS.map(function (c) { return api.esc(c[0]) + '×' + c[1]; }).join(', ') + '. <code>&lt;/w&gt;</code> marks the end of a word.</div>');
            api.stepper(el, { interval: 2800, steps: steps.map(function (s, k) {
              return { title: 'Merge ' + (k + 1) + ': ' + api.esc(s.best[0]) + ' + ' + api.esc(s.best[1]) + ' → ' + api.esc(s.best[0] + s.best[1]) + ' (count ' + s.bestN + ')', render: function (b) {
                api.bars(b, { title: 'Top pair counts ' + api.badge('integer', 'dict'), items: s.counts.map(function (c, i) { return { label: '<span class="tokenization-pair">' + api.esc(c[0][0]) + ' + ' + api.esc(c[0][1]) + '</span>', value: c[1], highlight: i === 0 }; }), format: function (v) { return String(v); } });
                api.html(b, '<div class="muted" style="margin-top:8px">After merging (highlighted = new symbol):</div>' + wordsHtml(api, s.after, CORPUS, s.best[0] + s.best[1]));
              } };
            }) });
          } },
          { name: 'Result', render: function (el, api) {
            var steps = trainBPE(CORPUS, 8);
            api.table(el, { headers: ['#', 'Merge', 'Count'], rows: steps.map(function (s, i) { return [i + 1, '<code>' + api.esc(s.best[0]) + '</code> + <code>' + api.esc(s.best[1]) + '</code>', s.bestN]; }) });
            api.html(el, '<p>Final symbols per word: ' + steps[steps.length - 1].after.map(function (s, i) { return '<code>' + api.esc(CORPUS[i][0]) + '</code> → ' + s.map(function (x) { return api.tok(x); }).join(' '); }).join('; ') + '</p>');
            api.callout(el, 'Frequent chunks like "est" and "low" became single tokens, while a new word such as "lowest" can be built from them. Applying the learned merges is exactly the playground you saw earlier.', 'key');
            api.illustrative(el, 'Tiny corpus; real tokenizers train on huge text and do thousands of merges.');
          } }
        ]
      },
      // ---------------------------------------------------------------- text to ids
      {
        id: 'text-to-ids', title: 'Text → UTF-8 bytes → merges → IDs',
        critical: {
          understand: ['Computers store text as <b>bytes</b> (UTF-8). Byte-level BPE starts from those 256 byte values, so any string can be tokenized.',
            'Merges then glue bytes into tokens; each token is looked up to get an <b>integer ID</b>.'],
          details: ['The IDs for "The cat sat on the" in GPT-2 are 464, 3797, 3332, 319, 262 — the running example of this course.',
            'Different models have different tokenizers, so the same text has different IDs in different models.']
        },
        dataTypes: [
          { name: 'text', type: 'text', shape: 'string', example: '"The cat sat on the"' },
          { name: 'bytes', type: 'integer', shape: '[n_bytes]', example: '[84, 104, 101, …]' },
          { name: 'tokens', type: 'token', shape: '[seq_len]', example: '"The", " cat", …' },
          { name: 'token IDs', type: 'integer', shape: '[seq_len]', example: '[464, 3797, 3332, 319, 262]' }
        ],
        levels: [
          { name: 'Pipeline', render: function (el, api) {
            var T = api.ex.tokens;
            api.blocks(el, { blocks: [
              { label: 'Text', sub: '"The cat sat on the"', type: 'text', shape: 'string' },
              { label: 'UTF-8 bytes', type: 'integer', shape: '[19]', edge: 'encode' },
              { label: 'Merges', type: 'token', shape: '[5]', edge: 'BPE' },
              { label: 'IDs', sub: '[' + T.map(function (t) { return t.id; }).join(', ') + ']', type: 'integer', shape: '[5]', goto: 'embeddings/lookup', edge: 'vocab lookup' }
            ], caption: 'IDs then index rows of the embedding table (next section).' });
            api.html(el, '<div class="tokenization-chips">' + T.map(function (t) { return api.tok(t.text, t.id); }).join(' ') + '</div>');
          } },
          { name: 'Bytes', render: function (el, api) {
            var out = api.h('div');
            api.input(el, { label: 'Text', value: api.ex.prompt, onInput: draw });
            el.appendChild(out);
            function draw(t) {
              var bytes = Array.prototype.slice.call(new TextEncoder().encode(t));
              var rows = []; var chars = Array.from(t).slice(0, 24);
              chars.forEach(function (ch) {
                var bs = Array.prototype.slice.call(new TextEncoder().encode(ch));
                rows.push(['<code>' + api.tokText(ch) + '</code>', bs.join(', '), bs.map(function (b) { return ('0' + b.toString(16)).slice(-2); }).join(' '), bs.length]);
              });
              out.innerHTML = '<div>' + Array.from(t).length + ' characters → ' + api.badge('integer', '[' + bytes.length + ']') + ' bytes (ASCII = 1 byte each; accented letters, CJK and emoji take 2–4).</div>';
              api.table(out, { headers: ['Char', 'UTF-8 bytes (dec)', 'hex', 'n'], rows: rows });
            }
          } },
          { name: 'Merges → IDs', render: function (el, api) {
            var T = api.ex.tokens;
            api.html(el, '<p>In a real byte-level BPE (GPT-2 family), the pieces of "The cat sat on the" are merged until each word (with its leading space) is one token:</p>');
            api.table(el, { headers: ['Token ' + api.type('token'), 'UTF-8 bytes', 'ID ' + api.type('integer')], rows: T.map(function (t) {
              return [api.tok(t.text), Array.prototype.slice.call(new TextEncoder().encode(t.text)).join(' '), '<b>' + t.id + '</b>'];
            }).concat([[api.tok(api.ex.nextToken.text) + ' (the answer)', Array.prototype.slice.call(new TextEncoder().encode(api.ex.nextToken.text)).join(' '), '<b>' + api.ex.nextToken.id + '</b>']]) });
            api.callout(el, 'The model never sees letters or bytes — only the integers. That is the root of several quirks: ' + api.link('tokenization/quirks', 'see quirks') + '.', 'info');
          } }
        ]
      },
      // ---------------------------------------------------------------- special tokens
      {
        id: 'special-tokens', title: 'Special tokens and chat templates',
        critical: {
          understand: ['Besides text pieces, the vocabulary has <b>special tokens</b>: start of text, end of turn, role markers…',
            'A chat is flattened by a <b>template</b> into ONE token sequence; the model just continues that sequence.'],
          details: ['The exact markers differ by model family; the example below follows the Llama-3 style but is simplified.',
            'The end-of-turn token is how the model signals "I am done" — the harness stops generating when it appears.']
        },
        dataTypes: [
          { name: 'special token', type: 'token', shape: 'scalar', example: '<|eot_id|>' },
          { name: 'messages', type: 'json', shape: 'list[{role, content}]', example: '[{"role":"user",…}]' },
          { name: 'templated prompt', type: 'text', shape: 'string', example: 'one long string → tokens' }
        ],
        levels: [
          { name: 'Specials', render: function (el, api) {
            api.table(el, { headers: ['Special token (Llama-3 style)', 'Purpose'], rows: [
              ['<code>&lt;|begin_of_text|&gt;</code> (BOS)', 'marks the start of a sequence'],
              ['<code>&lt;|start_header_id|&gt;</code> … <code>&lt;|end_header_id|&gt;</code>', 'wrap a role name: system / user / assistant'],
              ['<code>&lt;|eot_id|&gt;</code>', 'end of one message/turn; generating it means "stop"'],
              ['<code>&lt;|end_of_text|&gt;</code> (EOS)', 'end of document in pre-training']
            ] });
            api.callout(el, 'These are ordinary vocabulary entries with ordinary IDs and embeddings. They matter only because training data was formatted with them, so the model learned what follows each one.', 'info');
            api.html(el, '<p>Details of the API side: ' + api.link('chat-api/chat-template', 'chat template') + ' and ' + api.link('chat-api/messages-api', 'messages API') + '.</p>');
          } },
          { name: 'One sequence', render: function (el, api) {
            var msgs = [{ role: 'system', content: 'You are a helpful assistant.' }, { role: 'user', content: 'Where did the cat sit?' }];
            var st = { gen: false };
            var out = api.h('div');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.toggle(c, { label: 'Model has generated its reply', value: false, onChange: function (v) { st.gen = v; draw(); } });
            el.appendChild(out);
            function sp(s) { return '<span class="tokenization-sp">' + api.esc(s) + '</span>'; }
            function draw() {
              var h = '';
              var all = msgs.concat(st.gen ? [{ role: 'assistant', content: 'On the mat.' }] : []);
              h += sp('<|begin_of_text|>');
              all.forEach(function (m) { h += sp('<|start_header_id|>') + '<span class="tokenization-role">' + m.role + '</span>' + sp('<|end_header_id|>') + '\n\n' + api.esc(m.content) + sp('<|eot_id|>') + '\n'; });
              if (!st.gen) h += sp('<|start_header_id|>') + '<span class="tokenization-role">assistant</span>' + sp('<|end_header_id|>') + '\n\n<span class="muted">← the model continues from here</span>';
              out.innerHTML = '<div>messages ' + api.badge('json', 'list[2]') + ' → one string ' + api.badge('text', 'string') + ' → tokens ' + api.badge('token', '[seq_len]') + '</div><div class="tokenization-seq">' + h + '</div>';
            }
            draw();
            api.illustrative(el, 'Simplified; real templates have extra details (dates, tool sections) and vary per model.');
          } },
          { name: 'Why it matters', render: function (el, api) {
            api.blocks(el, { blocks: [
              { label: 'Conversation', type: 'json', shape: 'list[message]' },
              { label: 'Chat template', type: 'function', shape: 'json → string', goto: 'chat-api/chat-template' },
              { label: 'Tokenizer', type: 'function', shape: 'string → [seq_len]' },
              { label: 'Model', type: 'function', shape: '[seq_len] → [vocab]', goto: 'transformer-block' },
              { label: 'Stop at <|eot_id|>?', type: 'boolean', shape: 'scalar' }
            ], caption: 'Tool definitions and tool results in Part II are serialized into this same one sequence.' });
            api.callout(el, 'Security angle: if untrusted text contains something that looks like a role marker, it can try to impersonate a message. Serving stacks therefore treat special tokens in user text carefully (details vary).', 'hedge');
          } }
        ]
      },
      // ---------------------------------------------------------------- quirks
      {
        id: 'quirks', title: 'Tokenizer quirks: spaces, numbers, letters, languages',
        critical: {
          understand: ['The model sees token IDs, not letters. That explains odd behaviour: counting letters, arithmetic on long numbers, trailing-space surprises.',
            'Non-English text and code often need <b>more tokens</b> per idea, so they cost more and fill the context faster.'],
          details: ['Examples here are illustrative; exact splits depend on the tokenizer.',
            'Counting letters improves when the model spells a word out step by step (a form of reasoning, see ' + '<span class="xlink" data-goto="reasoning">Reasoning</span>).']
        },
        dataTypes: [
          { name: 'tokens', type: 'token', shape: '[seq_len]', example: '"str", "aw", "berry"' },
          { name: 'token count', type: 'integer', shape: 'scalar', example: '3' },
          { name: 'text', type: 'text', shape: 'string', example: '"strawberry"' }
        ],
        levels: [
          { name: 'Spaces', render: function (el, api) {
            api.html(el, '<p>The same letters are different tokens with and without a leading space:</p>');
            api.html(el, '<div class="tokenization-chips">' + api.tok(' cat', 3797) + ' ≠ ' + api.tok('cat') + ' (a different ID; no leading space) </div>');
            api.html(el, '<p>Toy tokenizer on both (note different ID sequences):</p>');
            [' the cat', 'the cat'].forEach(function (s) { api.html(el, '<div><code>' + api.tokText(s) + '</code></div>' + chips(api, toyTokenize(s))); });
            api.callout(el, 'A prompt ending in a trailing space can hurt: the natural next token usually <i>starts</i> with a space, so a prompt ending in ␣ leaves the model in an unusual position.', 'hedge');
          } },
          { name: 'Numbers & letters', render: function (el, api) {
            api.html(el, '<p><b>Numbers</b> are chopped by the tokenizer into chunks that do not respect place value:</p>');
            api.table(el, { headers: ['Text', 'A plausible split (illustrative)', 'Problem'], rows: [
              ['12345', api.tok('123') + ' ' + api.tok('45'), 'digits grouped arbitrarily'],
              ['1234567', api.tok('123') + ' ' + api.tok('456') + ' ' + api.tok('7'), 'same number can split differently in context'],
              ['3.14159', api.tok('3') + ' ' + api.tok('.') + ' ' + api.tok('141') + ' ' + api.tok('59'), 'arithmetic must be learned on chunks']
            ] });
            api.html(el, '<p><b>Counting letters.</b> How many r in "strawberry"?</p>');
            var word = 'strawberry', split = ['str', 'aw', 'berry'];
            var count = word.split('').filter(function (c) { return c === 'r'; }).length;
            api.html(el, '<div class="tokenization-chips">' + split.map(function (s, i) { return api.tok(s, 1000 + i * 111); }).join(' ') + ' ' + api.badge('token', '[3]') + '</div>');
            api.html(el, '<div>The model gets 3 opaque IDs, not 10 letters. Ordinary code counts instantly: <code>"strawberry".count("r")</code> = <b>' + count + '</b>. Models often get this wrong unless they spell the word out first.</div>');
            api.illustrative(el, 'The split and IDs are made up; real tokenizers differ.');
          } },
          { name: 'Languages', render: function (el, api) {
            var data = [['English: "Good morning, how are you?"', 7], ['Spanish: "Buenos días, ¿cómo estás?"', 11], ['German: "Guten Morgen, wie geht es dir?"', 11], ['Hindi: "सुप्रभात, आप कैसे हैं?"', 24], ['Japanese: "おはよう、元気ですか？"', 14]];
            api.bars(el, { title: 'Tokens for the same greeting (' + api.badge('integer', 'scalar') + ', illustrative)', items: data.map(function (d, i) { return { label: api.esc(d[0]), value: d[1], highlight: i === 0 }; }), format: function (v) { return '≈ ' + v; } });
            api.callout(el, 'Why: tokenizers are trained on text that is mostly English, so English words get big tokens while other scripts fall back to smaller pieces or raw bytes. Newer tokenizers with larger vocabularies narrow the gap. Counts above are rough and vary by model.', 'hedge');
            api.illustrative(el, 'Approximate counts for illustration only; measure with your model\'s own tokenizer.');
          } }
        ]
      }
    ]
  });
})();
