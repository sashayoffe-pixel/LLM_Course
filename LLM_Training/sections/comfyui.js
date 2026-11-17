(function () {
  'use strict';

  var COL = { MODEL: '#b39ddb', CLIP: '#ffd54f', VAE: '#ff7b7b', CONDITIONING: '#ffa24a', LATENT: '#ff8de1', IMAGE: '#5fb4ff' };
  var SVGNS = 'http://www.w3.org/2000/svg';

  function sv(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }
  function legend(api, el) {
    var d = api.h('div', { class: 'comfyui-legend' });
    Object.keys(COL).forEach(function (k) {
      d.appendChild(api.h('span', null, api.h('i', { class: 'comfyui-dot', style: { background: COL[k] } }), k));
    });
    el.appendChild(d);
  }
  function gauss(r) {
    var u = Math.max(r(), 1e-9), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------------------------------------------------------- slide 1: node graph */
  var NODES = [
    { id: 'ckpt', title: 'Load Checkpoint', x: 10, y: 110, w: 170, h: 116, ins: [], outs: ['MODEL', 'CLIP', 'VAE'],
      info: 'Reads one <code>.safetensors</code> file and splits it into three parts: the denoiser (<b>MODEL</b>, U-Net or DiT), the text encoder (<b>CLIP</b>) and the pixel codec (<b>VAE</b>). Widget: <code>ckpt_name</code> (string).' },
    { id: 'pos', title: 'CLIP Text Encode (+)', x: 240, y: 10, w: 180, h: 78, ins: [['clip', 'CLIP']], outs: ['CONDITIONING'],
      info: 'Tokenizes the positive prompt and runs the text encoder. Output: CONDITIONING, e.g. a tensor <code>[77 × 768]</code> for SD-1.5-class models (see ' },
    { id: 'neg', title: 'CLIP Text Encode (−)', x: 240, y: 130, w: 180, h: 78, ins: [['clip', 'CLIP']], outs: ['CONDITIONING'],
      info: 'Same node class, different text (e.g. "blurry, low quality"). Its output is the <i>unconditional/negative</i> side of classifier-free guidance.' },
    { id: 'lat', title: 'Empty Latent Image', x: 240, y: 250, w: 180, h: 78, ins: [], outs: ['LATENT'],
      info: 'Allocates a blank latent <code>[4 × 64 × 64]</code> for a 512×512 image (widgets: width, height, batch_size). The sampler fills it with noise and denoises it.' },
    { id: 'ks', title: 'KSampler', x: 500, y: 90, w: 190, h: 150, ins: [['model', 'MODEL'], ['positive', 'CONDITIONING'], ['negative', 'CONDITIONING'], ['latent_image', 'LATENT']], outs: ['LATENT'],
      info: 'The loop: for each step, call MODEL on the latent with both conditionings, combine with CFG, update the latent. Widgets: seed, steps, cfg, sampler_name, scheduler, denoise. Note the loop is <i>inside</i> this one node.' },
    { id: 'vae', title: 'VAE Decode', x: 760, y: 60, w: 190, h: 100, ins: [['samples', 'LATENT'], ['vae', 'VAE']], outs: ['IMAGE'],
      info: 'Decodes the latent <code>[4 × 64 × 64]</code> back to pixels <code>[3 × 512 × 512]</code> (IMAGE, floats 0–1).' },
    { id: 'save', title: 'Save Image', x: 760, y: 230, w: 190, h: 74, ins: [['images', 'IMAGE']], outs: [],
      info: 'An output node: writes PNG files to the output folder. Only nodes that lead to an output node get executed.' }
  ];
  var EDGES = [
    ['ckpt', 0, 'ks', 0], ['ckpt', 1, 'pos', 0], ['ckpt', 1, 'neg', 0], ['ckpt', 2, 'vae', 1],
    ['pos', 0, 'ks', 1], ['neg', 0, 'ks', 2], ['lat', 0, 'ks', 3], ['ks', 0, 'vae', 0], ['vae', 0, 'save', 0]
  ];
  function nodeById(id) { return NODES.filter(function (n) { return n.id === id; })[0]; }
  function sockY(n, i) { return n.y + 44 + i * 22; }

  function nodeGraph(el, api, onSelect) {
    var wrap = api.h('div', { class: 'comfyui-scroll' });
    var svg = sv('svg', { viewBox: '0 0 970 345', role: 'img', 'aria-label': 'ComfyUI node graph' });
    var edgeEls = [], nodeEls = {};
    EDGES.forEach(function (e) {
      var a = nodeById(e[0]), b = nodeById(e[2]);
      var x1 = a.x + a.w, y1 = sockY(a, e[1]), x2 = b.x, y2 = sockY(b, e[3]);
      var dx = Math.max(40, (x2 - x1) / 2);
      var p = sv('path', { d: 'M' + x1 + ',' + y1 + ' C' + (x1 + dx) + ',' + y1 + ' ' + (x2 - dx) + ',' + y2 + ' ' + x2 + ',' + y2,
        class: 'comfyui-edge', stroke: COL[a.outs[e[1]]] });
      p.setAttribute('data-from', e[0]); p.setAttribute('data-to', e[2]);
      svg.appendChild(p); edgeEls.push(p);
    });
    NODES.forEach(function (n) {
      var g = sv('g', { class: 'comfyui-node', tabindex: 0 });
      g.appendChild(sv('rect', { class: 'comfyui-body', x: n.x, y: n.y, width: n.w, height: n.h, rx: 8 }));
      g.appendChild(sv('text', { class: 'comfyui-title', x: n.x + 10, y: n.y + 22 }, n.title));
      n.ins.forEach(function (s, i) {
        g.appendChild(sv('circle', { cx: n.x, cy: sockY(n, i), r: 6, fill: COL[s[1]], stroke: '#111', 'stroke-width': 1 }));
        g.appendChild(sv('text', { class: 'comfyui-sock', x: n.x + 12, y: sockY(n, i) + 4 }, s[0]));
      });
      n.outs.forEach(function (s, i) {
        g.appendChild(sv('circle', { cx: n.x + n.w, cy: sockY(n, i), r: 6, fill: COL[s], stroke: '#111', 'stroke-width': 1 }));
        g.appendChild(sv('text', { class: 'comfyui-sock', x: n.x + n.w - 12, y: sockY(n, i) + 4, 'text-anchor': 'end' }, s));
      });
      var pick = function () { select(n.id); };
      g.addEventListener('click', pick);
      g.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
      svg.appendChild(g); nodeEls[n.id] = g;
    });
    wrap.appendChild(svg); el.appendChild(wrap);
    function select(id) {
      Object.keys(nodeEls).forEach(function (k) { nodeEls[k].setAttribute('class', 'comfyui-node' + (k === id ? ' comfyui-sel' : '')); });
      edgeEls.forEach(function (p) {
        var on = p.getAttribute('data-from') === id || p.getAttribute('data-to') === id;
        p.setAttribute('class', 'comfyui-edge' + (on ? '' : ' comfyui-dim'));
      });
      if (onSelect) onSelect(nodeById(id));
    }
    return { select: select };
  }

  /* ---------------------------------------------------------- slide 2 helpers: toy image */
  var G = 16;
  function targetImage() { // a ring with a dot, values in [-1,1]
    var t = [];
    for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
      var dx = x - 7.5, dy = y - 7.5, r = Math.sqrt(dx * dx + dy * dy);
      var v = (Math.abs(r - 5.2) < 1.2 || r < 1.6) ? 1 : -1;
      t.push(v);
    }
    return t;
  }
  function abar(i, N) { var s = i / N; var a = Math.sin(s * Math.PI / 2); return a * a; } // 0 = pure noise ... 1 = clean

  /* ---------------------------------------------------------- register */
  Course.register({
    id: 'comfyui', part: 'II', order: 147, title: 'ComfyUI & Diffusion',
    slides: [

      /* ======================================================= node-graph */
      {
        id: 'node-graph', title: 'ComfyUI: a typed node graph',
        critical: {
          understand: [
            'ComfyUI is a visual <b>dataflow graph</b>: every box is a function, every wire carries a typed value.',
            'Sockets are typed (MODEL, CLIP, VAE, CONDITIONING, LATENT, IMAGE); you can only wire matching colours, like type-checked function arguments.',
            'The graph is saved as JSON, so it is data an agent can read, edit and submit.'
          ],
          details: [
            'Only nodes that lead to an <b>output node</b> (Save Image, Preview) are executed; unchanged upstream results are cached between runs.',
            'Socket colours here follow ComfyUI\'s defaults only approximately; the type names are what matter.'
          ]
        },
        dataTypes: [
          { name: 'MODEL (denoiser)', type: 'function', shape: 'U-Net / DiT', example: 'ε-predictor' },
          { name: 'CLIP (text encoder)', type: 'function', shape: 'tokens → [77 × 768]', example: 'object' },
          { name: 'CONDITIONING', type: 'tensor', shape: '[77 × 768]', example: 'SD-1.5-class' },
          { name: 'LATENT', type: 'tensor', shape: '[4 × 64 × 64]', example: '512×512 image' },
          { name: 'IMAGE', type: 'tensor', shape: '[3 × H × W]', example: 'floats in 0–1' },
          { name: 'workflow', type: 'json', shape: 'object', example: 'node id → {class_type, inputs}' }
        ],
        levels: [
          { name: 'Graph', render: function (el, api) {
            api.html(el, 'A text-to-image workflow, as ComfyUI draws it. <b>Click a node</b> to highlight its wires.');
            legend(api, el);
            var info = api.h('div', { class: 'comfyui-info' });
            var g = nodeGraph(el, api, function (n) {
              info.innerHTML = '<b>' + api.esc(n.title) + '</b>: ' + n.info + (n.id === 'pos' ? api.link('comfyui/text-encoder', 'text encoder') + ').' : '');
            });
            el.appendChild(info);
            g.select('ks');
            api.callout(el, 'This is a <b>DAG</b> (directed acyclic graph): data flows left to right once. Contrast with an agent graph that <i>loops</i> (' + api.link('langgraph/chains-vs-graphs', 'chains vs graphs') + ').', 'key');
          } },
          { name: 'Sockets = types', render: function (el, api) {
            api.html(el, 'Each socket type maps to one of our 12 data types. The <b>shapes are for SD-1.5-class models at 512×512</b>; other model families differ.');
            legend(api, el);
            api.table(el, {
              headers: ['Socket', 'Our data type', 'Shape / example', 'Produced by → consumed by'],
              rows: [
                ['<b style="color:' + COL.MODEL + '">MODEL</b>', api.type('function'), 'a callable: ε = f(latent, t, cond)', 'Load Checkpoint → KSampler'],
                ['<b style="color:' + COL.CLIP + '">CLIP</b>', api.type('function'), 'tokenizer + text transformer', 'Load Checkpoint → CLIP Text Encode'],
                ['<b style="color:' + COL.VAE + '">VAE</b>', api.type('function'), 'encoder + decoder networks', 'Load Checkpoint → VAE Decode'],
                ['<b style="color:' + COL.CONDITIONING + '">CONDITIONING</b>', api.type('tensor'), api.shape('[77 × 768]') + ' (+ metadata)', 'CLIP Text Encode → KSampler'],
                ['<b style="color:' + COL.LATENT + '">LATENT</b>', api.type('tensor'), api.shape('[4 × 64 × 64]'), 'Empty Latent → KSampler → VAE Decode'],
                ['<b style="color:' + COL.IMAGE + '">IMAGE</b>', api.type('tensor'), api.shape('[3 × 512 × 512]') + ' floats 0–1', 'VAE Decode → Save Image'],
                ['widgets (seed, steps, cfg, text)', api.type('integer') + ' ' + api.type('float') + ' ' + api.type('text'), 'typed inputs set by hand', 'become entries in the JSON'],
                ['the whole graph', api.type('json'), 'object', 'saved file / HTTP payload']
              ]
            });
            api.callout(el, 'Mnemonic: <b>CLIP</b> turns text into CONDITIONING, <b>MODEL</b> turns noise + CONDITIONING into a clean LATENT, <b>VAE</b> turns LATENT into IMAGE.', 'tip');
          } },
          { name: 'As JSON', render: function (el, api) {
            api.html(el, 'Under the hood the same graph is a JSON object. A wire is written <code>["source_node_id", output_index]</code>; a widget is a plain value.');
            var wf = {
              '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'model.safetensors' } },
              '6': { class_type: 'CLIPTextEncode', inputs: { text: 'a cat on a mat', clip: ['4', 1] } },
              '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry', clip: ['4', 1] } },
              '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
              '3': { class_type: 'KSampler', inputs: { seed: 42, steps: 20, cfg: 7, sampler_name: 'euler', scheduler: 'normal', denoise: 1, model: ['4', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0] } }
            };
            api.code(el, JSON.stringify(wf, null, 2) + '\n// … plus VAEDecode "8" and SaveImage "9"', 'json');
            api.docsNote(el, 'Node class names and input names above follow ComfyUI\'s built-in nodes as commonly documented; check your installed version.');
          } }
        ]
      },

      /* ======================================================= diffusion */
      {
        id: 'diffusion', title: 'Diffusion: from noise to image',
        critical: {
          understand: [
            'A diffusion model is trained to <b>predict the noise</b> in a noisy image. Generation runs it repeatedly, removing a little noise each step.',
            'Classifier-free guidance runs the model twice (with and without the prompt) and <b>extrapolates</b> toward the prompt: ε = ε_u + s·(ε_c − ε_u).',
            'To stay cheap, this all happens in a small <b>latent space</b> (48× fewer numbers than pixels), then a VAE decodes to pixels.'
          ],
          details: [
            'Like an LLM, generation is <b>iterative</b> and the same network is called many times; unlike an LLM, all positions are refined in parallel, not left to right.',
            'Step counts (≈20–30 typical) and guidance scales (≈5–8 typical) are model- and sampler-dependent habits, not laws.'
          ]
        },
        dataTypes: [
          { name: 'noise ε', type: 'tensor', shape: '[4 × 64 × 64]', example: 'seeded Gaussian' },
          { name: 'timestep t / sigma', type: 'float', shape: 'scalar', example: 'from the scheduler' },
          { name: 'steps', type: 'integer', shape: 'scalar', example: '20' },
          { name: 'guidance scale s (cfg)', type: 'float', shape: 'scalar', example: '7.0' },
          { name: 'latent z', type: 'tensor', shape: '[4 × 64 × 64]', example: '16,384 floats' },
          { name: 'pixels', type: 'tensor', shape: '[3 × 512 × 512]', example: '786,432 values' }
        ],
        levels: [
          { name: 'Denoising', render: function (el, api) {
            api.html(el, 'A toy 16×16 "image". Step 0 is pure seeded noise; each scheduler step removes noise until the picture emerges.');
            api.illustrative(el, ' Real models predict the noise with a neural network; here the "model" is a perfect oracle that knows the target, so you only see the <i>schedule</i>.');
            var N = 20, tgt = targetImage();
            var st = { step: 0, seed: 7, noise: null };
            var cv = api.h('canvas', { class: 'comfyui-canvas' }); cv.width = G; cv.height = G;
            var ctx = cv.getContext('2d');
            var info = api.h('div', { class: 'comfyui-info' });
            function mkNoise() { var r = api.math.rng(st.seed); st.noise = tgt.map(function () { return gauss(r); }); }
            function draw() {
              var i = st.step, ab = abar(i, N), a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
              var img = ctx.createImageData(G, G);
              for (var k = 0; k < G * G; k++) {
                var x = a * tgt[k] + b * st.noise[k];
                var v = Math.max(0, Math.min(1, (x + 1) / 2)) * 255;
                img.data[4 * k] = v; img.data[4 * k + 1] = v; img.data[4 * k + 2] = v; img.data[4 * k + 3] = 255;
              }
              ctx.putImageData(img, 0, 0);
              info.innerHTML = 'step <b>' + i + ' / ' + N + '</b> · signal weight √ᾱ = <b>' + api.fmt(a, 2) + '</b> · noise weight √(1−ᾱ) = <b>' + api.fmt(b, 2) + '</b><br>' +
                '<span class="muted">x<sub>t</sub> = √ᾱ·x₀ + √(1−ᾱ)·ε &nbsp; (' + api.type('tensor') + ' ' + api.shape('[16 × 16]') + ' in this toy)</span>';
            }
            mkNoise();
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            api.slider(c, { label: 'scheduler step', min: 0, max: N, step: 1, value: 0, format: function (v) { return v + ' / ' + N; }, onInput: function (v) { st.step = v; draw(); } });
            api.slider(c, { label: 'seed', min: 1, max: 20, step: 1, value: 7, format: function (v) { return String(v); }, onInput: function (v) { st.seed = v; mkNoise(); draw(); } });
            var playing = false;
            api.button(c, '▶ Run all steps', function () {
              if (playing) return; playing = true; st.step = 0; draw();
              var t = api.every(250, function () {
                if (st.step >= N) { playing = false; return; }
                st.step++; draw(); var s = c.querySelector('input[type=range]'); if (s) s.value = st.step;
              });
            }, 'btn primary');
            var pair = api.h('div', { class: 'comfyui-pair' });
            pair.appendChild(api.h('div', { class: 'comfyui-side' }, cv));
            pair.appendChild(api.h('div', { class: 'comfyui-side' }, info,
              api.h('div', { class: 'muted', style: { marginTop: '8px', fontSize: '13px' }, html: 'Same seed → same starting noise → same image. That is why ComfyUI\'s <code>seed</code> widget makes runs reproducible (given identical settings and hardware, usually).' })));
            el.appendChild(pair);
            draw();
          } },
          { name: 'Guidance (CFG)', render: function (el, api) {
            api.html(el, 'At every step the model is called twice: once with the prompt (<b>conditional</b>) and once with an empty/negative prompt (<b>unconditional</b>). Guidance pushes the result away from "generic" and toward "the prompt".');
            api.html(el, '<div class="formula">ε = ε_uncond + s · (ε_cond − ε_uncond)</div>');
            var epsU = [0.20, -0.10, 0.05, 0.30], epsC = [0.50, -0.30, 0.15, 0.10];
            api.illustrative(el, ' Four made-up noise-prediction values stand in for the thousands in a real latent.');
            var out = api.h('div'); var c = api.h('div', { class: 'controls' });
            el.appendChild(c); el.appendChild(out);
            api.slider(c, { label: 'guidance scale s', min: 0, max: 15, step: 0.5, value: 7, format: function (v) { return api.fmt(v, 1); }, onInput: function (s) {
              out.innerHTML = '';
              var res = epsU.map(function (u, i) { return u + s * (epsC[i] - u); });
              api.table(out, {
                headers: ['element', 'ε_uncond', 'ε_cond', 'ε_cond − ε_uncond', 's × diff', 'ε (guided)'],
                rows: epsU.map(function (u, i) {
                  var d = epsC[i] - u;
                  return [i, api.fmt(u, 2), api.fmt(epsC[i], 2), api.fmt(d, 2), api.fmt(s * d, 2), '<b>' + api.fmt(res[i], 2) + '</b>'];
                })
              });
              var d0 = epsC[0] - epsU[0];
              api.html(out, '<div class="comfyui-info">Element 0: ' + api.fmt(epsU[0], 2) + ' + ' + api.fmt(s, 1) + ' × (' + api.fmt(epsC[0], 2) + ' − ' + api.fmt(epsU[0], 2) + ') = ' + api.fmt(epsU[0], 2) + ' + ' + api.fmt(s * d0, 2) + ' = <b>' + api.fmt(res[0], 2) + '</b></div>');
              api.html(out, '<span class="muted">s = 0 → ignore the prompt (pure unconditional) · s = 1 → plain conditional prediction · s &gt; 1 → extrapolate beyond it.</span>');
            } });
            api.callout(el, 'Cost: each step needs <b>two</b> model evaluations (often batched together), so CFG roughly doubles the compute per step. Very high s tends to give over-saturated, harsh images; very low s tends to ignore the prompt (typical sweet spots are model-dependent, often cited around 5–8).', 'hedge');
          } },
          { name: 'VAE latents', render: function (el, api) {
            api.html(el, 'Diffusing 786,432 pixel values would be slow. A <b>VAE</b> encoder squeezes the image into a small <b>latent</b>; diffusion happens there; a VAE decoder paints the pixels back.');
            api.blocks(el, { blocks: [
              { id: 'px', label: 'Pixels', type: 'tensor', shape: '[3 × 512 × 512]' },
              { id: 'enc', label: 'VAE encoder', type: 'function', edge: 'only for img2img' },
              { id: 'z', label: 'Latent z', type: 'tensor', shape: '[4 × 64 × 64]' },
              { id: 'dec', label: 'VAE decoder', type: 'function', edge: 'after sampling' },
              { id: 'px2', label: 'Image', type: 'tensor', shape: '[3 × 512 × 512]' }
            ] });
            var out = api.h('div'); var c = api.h('div', { class: 'controls' });
            el.appendChild(c); el.appendChild(out);
            api.slider(c, { label: 'image side (px)', min: 256, max: 1536, step: 256, value: 512, format: function (v) { return v + ' × ' + v; }, onInput: function (side) {
              var f = 8, ch = 4, lat = side / f;
              var pix = side * side * 3, lt = lat * lat * ch;
              out.innerHTML = '';
              api.table(out, { headers: ['', 'shape', 'numbers'], rows: [
                ['pixels', api.shape('[3 × ' + side + ' × ' + side + ']'), api.fmtInt(pix) + ' = 3 × ' + side + '²'],
                ['latent (8× smaller per side, 4 channels)', api.shape('[' + ch + ' × ' + lat + ' × ' + lat + ']'), api.fmtInt(lt) + ' = 4 × ' + lat + '²'],
                ['compression', '', '<b>' + api.fmt(pix / lt, 0) + '×</b> = ' + api.fmtInt(pix) + ' ÷ ' + api.fmtInt(lt)]
              ] });
              api.html(out, '<span class="muted">Compression = (3 ÷ 4) × 8² = 0.75 × 64 = 48, independent of image size. The 8× spatial factor and 4 channels are the SD-1.5/SDXL-class values; newer model families (e.g. Flux-class) reportedly use more latent channels.</span>');
            } });
          } }
        ]
      },

      /* ======================================================= text-encoder */
      {
        id: 'text-encoder', title: 'The LLM link: text encoder and cross-attention',
        critical: {
          understand: [
            'Your prompt is tokenized and run through a <b>transformer text encoder</b> (CLIP, sometimes T5): tokens → embeddings, e.g. <code>[77 × 768]</code>.',
            'The image denoiser consumes them via <b>cross-attention</b>: Q from the image latent, K and V from the text embeddings.',
            'It is the same attention formula as in an LLM; only <i>where Q, K and V come from</i> changes.'
          ],
          details: [
            'Self-attention: Q, K, V all from one sequence (+ causal mask in an LLM). Cross-attention: Q from sequence A (image patches), K,V from sequence B (prompt); the score matrix is rectangular and needs no causal mask.',
            'Encoder details vary: SD-1.5-class uses a CLIP text model with 77 token slots; SDXL-class uses two CLIP encoders; Flux-class models are reported to add a T5 encoder. Treat 77×768 as one concrete example.'
          ]
        },
        dataTypes: [
          { name: 'prompt', type: 'text', shape: 'string', example: '"a cat on a mat"' },
          { name: 'token IDs', type: 'integer', shape: '[77]', example: 'padded / truncated to 77' },
          { name: 'text embeddings', type: 'matrix', shape: '[77 × 768]', example: 'CONDITIONING' },
          { name: 'Q (from image latent)', type: 'matrix', shape: '[n_patches × d]', example: '[3 × 2] toy' },
          { name: 'K, V (from text)', type: 'matrix', shape: '[n_tokens × d]', example: '[4 × 2] toy' },
          { name: 'cross-attention weights', type: 'probability', shape: '[n_patches × n_tokens]', example: 'rows sum to 1' }
        ],
        levels: [
          { name: 'Pipeline', render: function (el, api) {
            api.html(el, 'Text goes in, a stack of vectors comes out, and the denoiser <b>reads</b> them at every layer and every step.');
            api.blocks(el, { rows: [[
              { id: 't', label: 'Prompt', type: 'text', shape: 'string' },
              { id: 'tok', label: 'Tokenizer', type: 'function', goto: 'tokenization' },
              { id: 'ids', label: 'Token IDs', type: 'integer', shape: '[77]' },
              { id: 'enc', label: 'Text encoder (CLIP / T5)', type: 'function', goto: 'transformer-block' },
              { id: 'emb', label: 'Embeddings', type: 'matrix', shape: '[77 × 768]' }
            ], [
              { id: 'lat', label: 'Noisy latent', type: 'tensor', shape: '[4 × 64 × 64]' },
              { id: 'ca', label: 'U-Net / DiT with cross-attention', type: 'function', zoom: true },
              { id: 'eps', label: 'Noise prediction ε', type: 'tensor', shape: '[4 × 64 × 64]' }
            ]] });
            api.callout(el, 'This is the same machinery as an LLM\'s front end: ' + api.link('tokenization', 'tokenization') + ' → ' + api.link('embeddings', 'embeddings') + ' → transformer layers (' + api.link('transformer-block', 'transformer block') + '). A text encoder is just a transformer that <i>reads</i> the prompt (bidirectionally, in CLIP\'s case) instead of predicting the next token. The prompt is encoded <b>once</b>; the denoiser then re-reads it every step.', 'key');
          } },
          { name: 'Self vs cross', render: function (el, api) {
            api.html(el, 'Same formula, <code>softmax(Q·Kᵀ/√d)·V</code>; different sources:');
            var g = api.grid(el, 260);
            api.card(g, 'Self-attention (LLM)', 'Q, K, V all come from <b>the same</b> sequence X.<br>Scores: ' + api.shape('[seq × seq]') + ', causal mask hides the future.<br>Tokens look at <i>other tokens</i>. See ' + api.link('attention/worked-example', 'the worked example') + '.');
            api.card(g, 'Cross-attention (diffusion)', 'Q from <b>image latent patches</b>; K, V from <b>text embeddings</b>.<br>Scores: ' + api.shape('[patches × text_tokens]') + ' (rectangular), no causal mask.<br>Each image location asks: "which words matter for me?"');
            api.table(el, { headers: ['', 'Self-attention', 'Cross-attention'], rows: [
              ['Q from', 'sequence X', 'image latent'],
              ['K, V from', 'sequence X', 'text embeddings (fixed during denoising)'],
              ['Score matrix', api.shape('[n × n]'), api.shape('[n_patches × 77]')],
              ['Mask', 'causal (decoder LLM)', 'none (padding aside)'],
              ['Output', 'mixed values per token', 'text-informed update per image patch']
            ] });
            api.callout(el, 'Recent DiT-style models often <b>concatenate</b> text and image tokens into one sequence and use joint self-attention instead; cross-attention is the classic U-Net design. Either way, text steers the image through attention.', 'hedge');
          } },
          { name: 'Numbers', render: function (el, api) {
            var M = api.math;
            api.html(el, 'A tiny cross-attention: 3 image patches ask questions of 4 prompt tokens. <code>d = 2</code>.');
            api.illustrative(el, ' All numbers here are invented for readability; real d is 40–160 per head.');
            var words = ['a', 'red', 'cat', 'mat'];
            var Q = [[1.0, 0.0], [0.2, 0.9], [0.9, 0.8]];       // patches: sky-ish, red-blob, cat-body
            var K = [[0.1, 0.1], [1.2, -0.2], [0.6, 1.0], [-0.3, 0.4]];
            var V = [[0.0, 0.1], [1.0, 0.0], [0.2, 1.0], [0.5, 0.5]];
            var d = 2, sq = Math.sqrt(d);
            var S = M.matmul(Q, M.transpose(K));
            var Ssc = S.map(function (r) { return r.map(function (x) { return x / sq; }); });
            var A = Ssc.map(function (r) { return M.softmax(r); });
            var O = M.matmul(A, V);
            api.matrix(el, { data: Q, rowLabels: ['patch 0', 'patch 1', 'patch 2'], colLabels: ['q1', 'q2'], title: 'Q from image latent', type: 'matrix', shape: '[3 × 2]', decimals: 1 });
            api.matrix(el, { data: K, rowLabels: words, colLabels: ['k1', 'k2'], title: 'K from text embeddings', type: 'matrix', shape: '[4 × 2]', decimals: 1 });
            api.matrix(el, { data: V, rowLabels: words, colLabels: ['v1', 'v2'], title: 'V from text embeddings', type: 'matrix', shape: '[4 × 2]', decimals: 1 });
            api.matrix(el, { data: Ssc, rowLabels: ['patch 0', 'patch 1', 'patch 2'], colLabels: words, title: 'Q·Kᵀ / √2  (rectangular, no mask)', type: 'matrix', shape: '[3 × 4]', decimals: 2, scale: 'diverging' });
            api.matrix(el, { data: A, rowLabels: ['patch 0', 'patch 1', 'patch 2'], colLabels: words, title: 'softmax → cross-attention weights', type: 'probability', shape: '[3 × 4]', decimals: 2, heat: true });
            var p = 2, best = M.argmax(A[p]);
            api.html(el, '<div class="comfyui-info">Patch 2, output = Σ weight × V = ' + A[p].map(function (a, i) { return api.fmt(a, 2) + '·v(' + words[i] + ')'; }).join(' + ') + ' = <b>[' + O[p].map(function (x) { return api.fmt(x, 2); }).join(', ') + ']</b>. It attends most to "<b>' + words[best] + '</b>" (' + api.fmt(A[p][best], 2) + ').</div>');
            api.matrix(el, { data: O, rowLabels: ['patch 0', 'patch 1', 'patch 2'], colLabels: ['o1', 'o2'], title: 'Output: text-informed update per patch', type: 'matrix', shape: '[3 × 2]', decimals: 2 });
            api.callout(el, 'Compare with the full self-attention calculation in ' + api.link('attention/worked-example', 'attention → worked example') + ': identical steps, but here the matrix is 3 × 4 not square, and there is no causal mask.', 'info');
          } }
        ]
      },

      /* ======================================================= agent-tool */
      {
        id: 'agent-tool', title: 'ComfyUI as an agent tool (HTTP API)',
        critical: {
          understand: [
            'ComfyUI runs as a local server. An agent can generate images by <b>POSTing a workflow JSON</b> to it: the workflow is the tool\'s "program".',
            'The agent\'s tool (<code>generate_image</code>) is ordinary harness code: fill a template, submit, poll, fetch the file, return a path. The LLM only sees a small JSON schema.',
            'This is exactly the tool-calling loop from ' + 'Part II, with a diffusion pipeline behind the tool.'
          ],
          details: [
            'Export the workflow with "Save (API Format)" (needs dev mode enabled in the UI); the API format differs from the UI save format.',
            'Generation takes seconds to minutes: poll (or use the websocket) rather than block, and return a file path or URL, not raw pixels, to keep the context small.'
          ]
        },
        dataTypes: [
          { name: 'tool call (generate_image)', type: 'json', shape: 'object', example: '{prompt, width, height}' },
          { name: 'workflow (API format)', type: 'json', shape: 'object', example: 'node id → {class_type, inputs}' },
          { name: 'prompt_id', type: 'text', shape: 'string', example: 'UUID returned by POST /prompt' },
          { name: 'finished?', type: 'boolean', shape: 'scalar', example: 'id present in /history' },
          { name: 'image file', type: 'text', shape: 'path / bytes', example: 'ComfyUI_00001_.png' },
          { name: 'generate_image', type: 'function', shape: 'harness code', example: 'wraps the 3 HTTP calls' }
        ],
        levels: [
          { name: 'Round trip', render: function (el, api) {
            api.html(el, 'The agent never talks to ComfyUI directly: the <b>harness</b> does.');
            api.blocks(el, { rows: [[
              { id: 'llm', label: 'LLM emits tool call', type: 'json', goto: 'tool-calling/round-trip' },
              { id: 'h', label: 'Harness: generate_image()', type: 'function', zoom: 2 },
              { id: 'post', label: 'POST /prompt', type: 'json', zoom: true, edge: 'workflow JSON' }
            ], [
              { id: 'q', label: 'ComfyUI queue + executor', type: 'function' },
              { id: 'poll', label: 'GET /history/{prompt_id}', type: 'json', edge: 'poll until present' },
              { id: 'view', label: 'GET /view?filename=…', type: 'text', edge: 'download PNG' },
              { id: 'res', label: 'Tool result: file path', type: 'text' }
            ]] });
            api.callout(el, 'The returned path (or a short description) becomes a <b>tool result</b> appended to the conversation, as in ' + api.link('llm-to-agent/agent-loop', 'the agent loop'), 'key');
            api.docsNote(el, 'Endpoint names (<code>/prompt</code>, <code>/history</code>, <code>/view</code>, <code>/ws</code>) reflect ComfyUI\'s server as commonly documented; verify against your version.');
          } },
          { name: 'HTTP payloads', render: function (el, api) {
            api.html(el, 'Edit the inputs; the JSON that the harness would POST is rebuilt below.');
            var st = { text: 'a cat sitting on a mat, watercolor', steps: 20, cfg: 7, seed: 42 };
            var out = api.h('div');
            function build() {
              var wf = {
                prompt: {
                  '3': { class_type: 'KSampler', inputs: { seed: st.seed, steps: st.steps, cfg: st.cfg, sampler_name: 'euler', scheduler: 'normal', denoise: 1, model: ['4', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0] } },
                  '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'model.safetensors' } },
                  '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
                  '6': { class_type: 'CLIPTextEncode', inputs: { text: st.text, clip: ['4', 1] } },
                  '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry, low quality', clip: ['4', 1] } },
                  '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
                  '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'agent', images: ['8', 0] } }
                },
                client_id: 'my-agent-1'
              };
              out.innerHTML = '';
              api.html(out, '<b>1. POST /prompt</b> ' + api.badge('json', 'object'));
              api.code(out, JSON.stringify(wf, null, 2), 'json');
              api.html(out, '<b>→ response</b> (illustrative id)');
              api.code(out, JSON.stringify({ prompt_id: 'a1b2c3d4-…', number: 0, node_errors: {} }, null, 2), 'json');
              api.html(out, '<b>2. GET /history/a1b2c3d4-…</b> once finished (an empty object <code>{}</code> means "not done yet")');
              api.code(out, JSON.stringify({ 'a1b2c3d4-…': { outputs: { '9': { images: [{ filename: 'agent_00001_.png', subfolder: '', type: 'output' }] } }, status: { completed: true } } }, null, 2), 'json');
              api.html(out, '<b>3. GET /view?filename=agent_00001_.png&amp;subfolder=&amp;type=output</b> → PNG bytes');
            }
            var c = api.h('div', { class: 'controls' }); el.appendChild(c); el.appendChild(out);
            api.input(c, { label: 'prompt text', value: st.text, onInput: function (v) { st.text = v; build(); }, init: false });
            api.slider(c, { label: 'steps', min: 4, max: 50, step: 1, value: 20, format: function (v) { return String(v); }, onInput: function (v) { st.steps = v; build(); } });
            api.slider(c, { label: 'cfg', min: 1, max: 15, step: 0.5, value: 7, format: function (v) { return api.fmt(v, 1); }, onInput: function (v) { st.cfg = v; build(); } });
            api.slider(c, { label: 'seed', min: 0, max: 99, step: 1, value: 42, format: function (v) { return String(v); }, onInput: function (v) { st.seed = v; build(); } });
            build();
            api.docsNote(el);
          } },
          { name: 'Tool definition', render: function (el, api) {
            api.html(el, 'What the <b>LLM</b> sees is only this small schema ' + api.badge('json', 'object') + '; everything above is hidden behind it. See ' + api.link('tool-calling/tool-definition', 'tool definition') + '.');
            var tool = {
              name: 'generate_image',
              description: 'Generate an image from a text prompt using a local ComfyUI server. Returns the saved file path. Takes 5–60 s.',
              input_schema: {
                type: 'object',
                properties: {
                  prompt: { type: 'string', description: 'What the image should show' },
                  negative_prompt: { type: 'string', description: 'What to avoid (optional)' },
                  width: { type: 'integer', enum: [512, 768, 1024] },
                  height: { type: 'integer', enum: [512, 768, 1024] },
                  seed: { type: 'integer', description: 'Optional, for reproducibility' }
                },
                required: ['prompt']
              }
            };
            api.code(el, JSON.stringify(tool, null, 2), 'json');
            api.html(el, '<b>Harness side</b> (pseudo-code, not a specific SDK):');
            api.code(el, [
              'def generate_image(prompt, negative_prompt="", width=512, height=512, seed=None):',
              '    wf = load_json("workflow_api.json")          # exported in API format',
              '    wf["6"]["inputs"]["text"] = prompt            # fill the template',
              '    wf["7"]["inputs"]["text"] = negative_prompt',
              '    wf["5"]["inputs"].update(width=width, height=height)',
              '    wf["3"]["inputs"]["seed"] = seed if seed is not None else random_seed()',
              '    pid = http_post(f"{BASE}/prompt", {"prompt": wf})["prompt_id"]',
              '    while pid not in (hist := http_get(f"{BASE}/history/{pid}")):',
              '        sleep(1)                                   # or listen on /ws; add a timeout!',
              '    img = first_image(hist[pid]["outputs"])',
              '    return {"path": download(f"{BASE}/view", img)} # small result, not raw pixels'
            ].join('\n'), 'python');
            api.callout(el, 'Design notes: validate arguments (the model may invent a size), add a timeout, return a short path, and treat generated prompts as untrusted input (see ' + api.link('tool-calling/security', 'tool security') + ').', 'tip');
            api.docsNote(el, 'The tool schema is generic; the field name varies by provider (<code>input_schema</code> vs <code>parameters</code>).');
          } }
        ]
      },

      /* ======================================================= ecosystem */
      {
        id: 'ecosystem', title: 'Ecosystem, VRAM, and DAG vs agent graph',
        critical: {
          understand: [
            'ComfyUI\'s power comes from its ecosystem: swappable checkpoints, LoRAs, ControlNets, upscalers and <b>custom nodes</b> (Python plugins).',
            'The binding constraint is usually <b>GPU memory</b>: weights alone ≈ parameters × bytes per parameter, plus activations and the text encoder / VAE.',
            'A ComfyUI workflow is a <b>DAG executed once</b>; an agent graph (LangGraph) has <b>cycles</b> and decisions made by an LLM at runtime.'
          ],
          details: [
            'VRAM numbers here are rough weight-only estimates; real needs depend on resolution, offloading, attention optimisations and the UI version.',
            'You can combine them: an agent node loops and decides, and one of its tools runs a ComfyUI DAG.'
          ]
        },
        dataTypes: [
          { name: 'parameters', type: 'integer', shape: 'scalar', example: '≈ 1 B – 12 B' },
          { name: 'bytes per parameter', type: 'float', shape: 'scalar', example: '2 (FP16) / 1 (FP8)' },
          { name: 'weights in VRAM', type: 'float', shape: 'scalar', example: 'GB' },
          { name: 'custom node', type: 'function', shape: 'Python class', example: 'INPUT_TYPES, FUNCTION' },
          { name: 'workflow graph', type: 'json', shape: 'object', example: 'DAG, no cycles' },
          { name: 'agent state', type: 'json', shape: 'object', example: 'messages + scratch' }
        ],
        levels: [
          { name: 'Ecosystem', render: function (el, api) {
            api.html(el, 'A workflow is mostly <b>swapping typed parts</b>:');
            var g = api.grid(el, 240);
            api.card(g, 'Checkpoints &amp; LoRAs', 'Different MODEL / CLIP / VAE bundles (SD-1.5-class, SDXL-class, Flux-class, …). A LoRA node patches the MODEL socket with small low-rank updates, the same idea as in ' + api.link('training/lora', 'LoRA') + '.');
            api.card(g, 'ControlNet &amp; adapters', 'Extra conditioning (edges, pose, depth) wired into the sampler alongside the text CONDITIONING.');
            api.card(g, 'Custom nodes', 'Python classes dropped into <code>custom_nodes/</code>: they declare input types and a function, and appear as new boxes. Quality and safety vary: you run third-party code.');
            api.card(g, 'Caching', 'ComfyUI generally re-runs only nodes whose inputs changed, so editing the prompt does not reload the checkpoint.');
            api.callout(el, 'Custom nodes are executable code from the internet. For an agent that edits workflows or installs nodes, this is an attack surface: pin versions and review them.', 'warn');
            api.docsNote(el, 'Node names and folder layout follow ComfyUI conventions as commonly documented; check the current docs.');
          } },
          { name: 'VRAM maths', render: function (el, api) {
            api.html(el, 'Rule of thumb: <b>weights ≈ parameters × bytes per parameter</b>. Activations, text encoders and the VAE come on top.');
            api.illustrative(el, ' Parameter counts below are approximate public figures (≈ order of magnitude) and may differ by variant.');
            var models = [
              { n: 'SD-1.5-class (U-Net ≈ 0.9 B)', p: 0.86 },
              { n: 'SDXL-class (U-Net ≈ 2.6 B)', p: 2.6 },
              { n: 'Flux-class (DiT ≈ 12 B)', p: 12 }
            ];
            var out = api.h('div'); var c = api.h('div', { class: 'controls' }); el.appendChild(c); el.appendChild(out);
            api.select(c, { label: 'precision', value: '2', options: [{ value: '4', label: 'FP32 (4 B)' }, { value: '2', label: 'FP16/BF16 (2 B)' }, { value: '1', label: 'FP8 / INT8 (1 B)' }, { value: '0.5', label: '4-bit (0.5 B)' }],
              onChange: function (v) { draw(parseFloat(v)); }, init: false });
            function draw(bpp) {
              out.innerHTML = '';
              api.bars(out, { items: models.map(function (m) { return { label: m.n, value: m.p * bpp, note: api.fmt(m.p * bpp, 1) + ' GB weights' }; }), max: 50, format: function (v) { return api.fmt(v, 1) + ' GB'; }, title: 'Denoiser weights only (GB) = params(B) × bytes' });
              api.html(out, '<span class="muted">Example: 12 B × ' + bpp + ' bytes = <b>' + api.fmt(12 * bpp, 1) + ' GB</b>. Compare an 8 B LLM in BF16: ' + api.fmtBytes(api.ref.paramBreakdown().bytesBF16) + ' (' + api.link('inference/quantization', 'quantization') + ').</span>');
            }
            draw(2);
            api.callout(el, 'Practical hedge: SD-1.5-class workflows are often run on roughly 4–6 GB GPUs, SDXL-class on roughly 8–12 GB, Flux-class typically needs more (≈ 16–24 GB, or less with quantization / offloading, at a speed cost). Treat these as rough community experience, not guarantees.', 'hedge');
          } },
          { name: 'DAG vs loop', render: function (el, api) {
            api.html(el, 'Two kinds of graph, easy to confuse:');
            var wrap = api.h('div', { class: 'comfyui-pair' });
            function svgBox(title, loop) {
              var s = sv('svg', { viewBox: '0 0 260 150', class: 'comfyui-svgbox', role: 'img', 'aria-label': title });
              var defs = sv('defs'); var mk = sv('marker', { id: 'cfarrow' + (loop ? 'b' : 'a'), viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' });
              mk.appendChild(sv('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#9aa' })); defs.appendChild(mk); s.appendChild(defs);
              var ref = 'url(#cfarrow' + (loop ? 'b' : 'a') + ')';
              function box(x, y, t, fill) { s.appendChild(sv('rect', { x: x, y: y, width: 64, height: 28, rx: 6, fill: fill, stroke: '#667' })); s.appendChild(sv('text', { x: x + 32, y: y + 18, 'text-anchor': 'middle', fill: '#fff', 'font-size': 11, style: 'fill:#eee' }, t)); }
              function arrow(d) { s.appendChild(sv('path', { d: d, fill: 'none', stroke: '#9aa', 'stroke-width': 1.8, 'marker-end': ref })); }
              if (!loop) {
                box(6, 60, 'Load', '#2a3550'); box(98, 20, 'Encode', '#2a3550'); box(98, 100, 'Latent', '#2a3550'); box(190, 60, 'Sample', '#2a3550');
                arrow('M70,70 L98,38'); arrow('M70,80 L98,112'); arrow('M162,38 L190,68'); arrow('M162,114 L190,82');
              } else {
                box(6, 60, 'LLM', '#3a2a50'); box(98, 20, 'Tool: image', '#2a3550'); box(98, 100, 'Tool: search', '#2a3550'); box(190, 60, 'Observe', '#3a2a50');
                arrow('M70,66 L98,38'); arrow('M70,82 L98,110'); arrow('M162,38 L190,66'); arrow('M162,114 L190,84');
                arrow('M222,88 C222,142 38,142 38,90');
                s.appendChild(sv('text', { x: 130, y: 140, 'text-anchor': 'middle', class: 'comfyui-m', style: 'fill:#9aa', 'font-size': 10 }, 'loop until done'));
              }
              return api.h('div', { class: 'comfyui-side' }, api.h('b', { text: title }), s);
            }
            wrap.appendChild(svgBox('ComfyUI: DAG, runs once', false));
            wrap.appendChild(svgBox('LangGraph agent: cycles', true));
            el.appendChild(wrap);
            api.table(el, { headers: ['', 'ComfyUI workflow', 'LangGraph / agent graph'], rows: [
              ['Shape', 'directed <b>acyclic</b>', 'may contain <b>cycles</b>'],
              ['Who decides the path', 'the author, in advance', 'the LLM / conditional edges at runtime'],
              ['Node = ', 'deterministic-ish function (given seed)', 'LLM call, tool, or code'],
              ['Edge carries', 'typed tensors (LATENT, IMAGE…)', 'a shared state object ' + api.type('json')],
              ['Ends when', 'all outputs computed', 'a stop condition or budget is hit']
            ] });
            api.callout(el, 'Compare ' + api.link('langgraph/chains-vs-graphs', 'chains vs graphs') + ' and the ' + api.link('langgraph/react-graph', 'ReAct graph') + '. The KSampler\'s internal step loop is hidden inside a single node: from the graph\'s point of view it is still acyclic. In an agent system, the agent loops <i>around</i> ComfyUI: it generates, looks at the result (a vision-capable model can critique it), and tries again.', 'key');
          } }
        ]
      }
    ]
  });
})();
