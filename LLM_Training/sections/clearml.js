(function () {
  'use strict';

  // ---------- seeded loss curves (illustrative) ----------
  function curves(lr, steps, seed, rank) {
    var rnd = Course.math.rng(seed || 7);
    var lr0 = 2e-4, ratio = lr / lr0;
    var k = 0.012 * Math.sqrt(ratio);
    var floor = 1.05 + 0.18 * Math.abs(Math.log(ratio) / Math.LN10) - 0.012 * Math.log(rank || 16) / Math.LN2;
    var overfit = Math.max(40, Math.round(steps * 0.6));
    var tr = [], va = [];
    for (var s = 0; s <= steps; s += 10) {
      var base = floor + (2.4 - floor) * Math.exp(-k * s);
      var t = base + (rnd() - 0.5) * 0.06 * Math.exp(-s / (steps * 1.2));
      var gap = 0.04 + 0.35 * Math.pow(Math.max(0, s - overfit) / steps, 1.3) + 0.03 * (rnd() - 0.5);
      tr.push([s, t]); va.push([s, t + gap]);
    }
    return { train: tr, val: va };
  }
  function best(c) { var bi = 0; c.val.forEach(function (p, i) { if (p[1] < c.val[bi][1]) bi = i; }); return c.val[bi]; }

  var RUNS = [
    { id: 'a1f3', name: 'lora-r16-lr2e-4', status: 'done', lr: 2e-4, rank: 16, steps: 300, seed: 3, user: 'ana', dur: '24m' },
    { id: 'b72c', name: 'lora-r8-lr2e-4', status: 'done', lr: 2e-4, rank: 8, steps: 300, seed: 5, user: 'ana', dur: '21m' },
    { id: 'c90d', name: 'lora-r16-lr1e-3', status: 'done', lr: 1e-3, rank: 16, steps: 300, seed: 9, user: 'ben', dur: '24m' },
    { id: 'd4e8', name: 'lora-r32-lr5e-5', status: 'fail', lr: 5e-5, rank: 32, steps: 120, seed: 11, user: 'ben', dur: '9m' },
    { id: 'e15a', name: 'lora-r16-lr2e-4-b', status: 'run', lr: 2e-4, rank: 16, steps: 200, seed: 13, user: 'ana', dur: '14m' }
  ];

  var COMPONENTS = [
    { id: 'server', label: 'ClearML Server', sub: 'UI + API + stores', type: 'function',
      text: 'The central service (self-hosted or hosted) that stores experiments, metrics, artifacts metadata and serves the web UI. Everything else talks to it with credentials from <code>clearml-init</code>.',
      lang: 'bash', code: '# self-hosted (see the official docs for the current compose file)\ndocker compose up -d\n# on each machine, once:\nclearml-init   # paste server URLs + API credentials' },
    { id: 'sdk', label: 'SDK: Task.init', sub: 'auto-logging', type: 'function',
      text: 'Two lines in your script create a <b>Task</b> (one experiment run). The SDK then hooks common frameworks and captures argparse, git state, installed packages, console output and framework metrics automatically.',
      lang: 'python', code: 'from clearml import Task\n\ntask = Task.init(\n    project_name="llm-finetune",\n    task_name="lora-r16-lr2e-4",\n)' },
    { id: 'agent', label: 'Agent + queues', sub: 'run jobs remotely', type: 'function',
      text: 'A <b>ClearML Agent</b> runs on a GPU box and pulls tasks from a named <b>queue</b>. It rebuilds the environment (repo commit, packages, optionally Docker) and runs your code there.',
      lang: 'bash', code: '# on the GPU machine\nclearml-agent daemon --queue gpu --docker\n\n# in your script: stop locally, enqueue instead\n# task.execute_remotely(queue_name="gpu")' },
    { id: 'data', label: 'ClearML Data', sub: 'dataset versions', type: 'function',
      text: 'Versioned datasets: add files, upload, finalize. Child versions record only changes, and a Task can pull an exact dataset version so runs are reproducible.',
      lang: 'python', code: 'from clearml import Dataset\n\nds = Dataset.create(dataset_name="support-chats",\n                    dataset_project="llm-finetune")\nds.add_files("data/")\nds.upload(); ds.finalize()\n\n# later, in training:\npath = Dataset.get(dataset_name="support-chats",\n                   dataset_project="llm-finetune").get_local_copy()' },
    { id: 'pipe', label: 'Pipelines', sub: 'chained steps', type: 'function',
      text: 'A <b>PipelineController</b> chains tasks (prepare data → train → evaluate) as a DAG; each step is a Task that runs on an agent queue, with outputs passed between steps.',
      lang: 'python', code: 'from clearml import PipelineController\n\npipe = PipelineController(name="lora-pipeline",\n                          project="llm-finetune", version="1.0.0")\npipe.add_step(name="prep",  base_task_project="llm-finetune", base_task_name="prep-data")\npipe.add_step(name="train", parents=["prep"],\n              base_task_project="llm-finetune", base_task_name="lora-train")\npipe.start(queue="services")' },
    { id: 'hpo', label: 'HPO', sub: 'hyperparameter search', type: 'function',
      text: 'The optimizer clones a base Task many times with different hyperparameters, queues them for agents, and ranks them by a reported scalar (e.g. validation loss).',
      lang: 'python', code: 'from clearml.automation import (HyperParameterOptimizer,\n    UniformParameterRange, DiscreteParameterRange, RandomSearch)\n\nopt = HyperParameterOptimizer(\n    base_task_id=BASE_TASK_ID,\n    hyper_parameters=[\n        UniformParameterRange("General/lr", min_value=5e-5, max_value=1e-3),\n        DiscreteParameterRange("General/lora_r", values=[8, 16, 32]),\n    ],\n    objective_metric_title="loss", objective_metric_series="val",\n    objective_metric_sign="min",\n    optimizer_class=RandomSearch, execution_queue="gpu",\n    max_number_of_concurrent_tasks=2)\nopt.start()' },
    { id: 'serving', label: 'Serving', sub: 'deploy models', type: 'function',
      text: '<b>ClearML Serving</b> deploys registered models behind an inference endpoint with versioning and metrics. For LLMs it is commonly paired with an engine such as vLLM; check the docs for what is currently supported.',
      lang: 'bash', code: '# pseudo-CLI sketch; flags differ by version, see the docs\nclearml-serving create --name "support-llm"\nclearml-serving --id <service-id> model add \\\n   --engine <engine> --endpoint "chat" --model-id <model-id>' }
  ];

  function tabs(el, api, names, onSel, first) {
    var bar = api.h('div', { class: 'clearml-tabs' }); el.appendChild(bar);
    var btns = names.map(function (n, i) {
      return api.button(bar, n, function () { sel(i); }, '');
    });
    function sel(i) { btns.forEach(function (b, j) { b.classList.toggle('on', i === j); }); onSel(i); }
    sel(first || 0);
  }

  var DEF = {
    id: 'clearml', part: 'I', order: 85, title: 'MLOps with ClearML',
    slides: [

      // ------------------------------------------------------------ mlops
      {
        id: 'mlops', title: 'Where MLOps fits in the LLM lifecycle',
        critical: {
          understand: ['MLOps is the <b>bookkeeping and plumbing</b> around training: which code, data and settings produced which model, and how to rerun or ship it.',
            'Without it, "which run was the good one?" ends up in filenames like <code>final_v3_REAL.pt</code>.'],
          details: ['ClearML is one open-source platform covering tracking, data versioning, orchestration and serving; similar jobs are done by other tools.',
            'Even a small LoRA fine-tune benefits: many runs differ only by learning rate, rank or data version.']
        },
        dataTypes: [
          { name: 'experiment (Task)', type: 'json', shape: 'object', example: 'code + params + metrics' },
          { name: 'hyperparameters', type: 'json', shape: 'dict[str → value]', example: '{lr: 2e-4, rank: 16}' },
          { name: 'metric series', type: 'float', shape: '[steps]', example: 'loss per step' },
          { name: 'dataset version', type: 'text', shape: 'string (id)', example: 'support-chats v3' },
          { name: 'model artifact', type: 'matrix', shape: 'weights file', example: 'LoRA adapter' }
        ],
        levels: [
          { name: 'Lifecycle', render: function (el, api) {
            api.html(el, '<p>MLOps wraps the training stage of the lifecycle shown in ' + api.link('training/lifecycle', 'training') + ' with four jobs:</p>');
            api.blocks(el, { rows: [[
              { id: 'd', label: 'Data', sub: 'versioned datasets', type: 'text', zoom: true },
              { id: 't', label: 'Train / tune', sub: 'LoRA, SFT runs', type: 'function', goto: 'training/lora' }
            ], [
              { id: 'e', label: 'Track & compare', sub: 'metrics, params, artifacts', type: 'json', zoom: true },
              { id: 's', label: 'Serve', sub: 'deploy chosen model', type: 'function', goto: 'inference/prefill-decode' }
            ]], caption: 'Orchestration (queues, pipelines) schedules these steps on GPUs.' });
            api.docsNote(el, 'ClearML names and APIs shown in this section may have changed.');
          } },
          { name: 'Pain → fix', render: function (el, api) {
            var rows = [
              ['Which settings made the best run?', 'Experiment tracking', 'Hyperparameters and git commit stored per run'],
              ['Which data was this trained on?', 'Data versioning', 'Dataset id / version recorded in the run'],
              ['My laptop has no GPU', 'Orchestration', 'Queue + agent runs the job on a GPU box'],
              ['Rerun prep → train → eval by hand', 'Pipelines', 'DAG of tasks, rerun end to end'],
              ['Which weights are in production?', 'Model registry + serving', 'Output model linked to its run']
            ];
            api.table(el, { headers: ['Pain', 'MLOps job', 'What is recorded'], rows: rows });
            api.docsNote(el);
            api.html(el, '<p class="muted">Next slides: the ' + api.link('clearml/components', 'components') + ', then a worked ' + api.link('clearml/lora-tracking', 'LoRA tracking flow') + '.</p>');
          } },
          { name: 'A run record', render: function (el, api) {
            api.html(el, '<p>Conceptually, one experiment is a record that makes the run reproducible. Simplified sketch (not ClearML\'s exact schema):</p>');
            api.code(el, JSON.stringify({
              id: 'a1f3', project: 'llm-finetune', name: 'lora-r16-lr2e-4',
              code: { repo: 'git@…/finetune.git', commit: '9c1e4d2', uncommitted_diff: false },
              environment: { python: '3.11', packages: ['torch', 'transformers', 'peft'] },
              hyperparameters: { lr: 0.0002, lora_r: 16, epochs: 3 },
              input_datasets: ['support-chats@v3'],
              scalars: { 'loss/train': '[…]', 'loss/val': '[…]' },
              output_models: ['lora-adapter.safetensors']
            }, null, 2), 'json');
            api.illustrative(el, 'All ids, names and values here are made up.');
            api.docsNote(el);
          } }
        ]
      },

      // ------------------------------------------------------------ components
      {
        id: 'components', title: 'ClearML components: Server, SDK, Agent, Data, Pipelines, HPO, Serving',
        critical: {
          understand: ['<b>SDK</b> (<code>Task.init</code>) logs from your code; the <b>Server</b> stores and shows it; <b>Agents</b> execute queued tasks; Data, Pipelines, HPO and Serving build on tasks.',
            'Everything is a <b>Task</b>: a training run, a pipeline step, an HPO trial.'],
          details: ['Click a component to see its role and a short snippet. Snippets are sketches; check current docs for exact arguments.',
            'Agents are what make "clone this experiment, change lr, run on the GPU box" possible from the UI.']
        },
        dataTypes: [
          { name: 'Task', type: 'json', shape: 'object', example: 'one run' },
          { name: 'project_name / task_name', type: 'text', shape: 'string', example: '"llm-finetune"' },
          { name: 'queue', type: 'text', shape: 'string (name)', example: '"gpu"' },
          { name: 'Dataset', type: 'function', shape: 'versioned files', example: 'support-chats v3' },
          { name: 'job count in queue', type: 'integer', shape: 'scalar', example: 'simulated' }
        ],
        levels: [
          { name: 'Diagram', render: function (el, api) {
            var out = api.h('div', { class: 'card' });
            var rows = [[COMPONENTS[1], COMPONENTS[0]], [COMPONENTS[2], COMPONENTS[3]], [COMPONENTS[4], COMPONENTS[5]], [COMPONENTS[6]]];
            var spec = rows.map(function (r) { return r.map(function (c) { return { id: c.id, label: c.label, sub: c.sub, type: c.type, onClick: function () { show(c); } }; }); });
            var bl = api.blocks(el, { rows: spec, caption: 'Click a component.' });
            function show(c) {
              out.innerHTML = '';
              api.html(out, '<b>' + c.label + '</b><p>' + c.text + '</p>');
              api.code(out, c.code, c.lang);
              if (bl && bl.highlight) bl.highlight([c.id]);
            }
            el.appendChild(out);
            show(COMPONENTS[1]);
            api.docsNote(el, 'Exact class names and parameters can differ between ClearML versions.');
          } },
          { name: 'Auto-logging', render: function (el, api) {
            api.html(el, '<p>After <code>Task.init(...)</code>, the SDK instruments your process. What gets captured without extra code (typical behaviour, depends on frameworks used):</p>');
            var g = api.grid(el, 230);
            api.card(g, 'Code & env', 'git repo, commit, uncommitted diff, entry script, python packages, OS / GPU info');
            api.card(g, 'Config', '<code>argparse</code> arguments, plus anything you pass to <code>task.connect(dict)</code>');
            api.card(g, 'Metrics', 'Scalars reported by TensorBoard, or by frameworks hooked automatically; manual ones via <code>logger.report_scalar</code>');
            api.card(g, 'Output', 'stdout / stderr console, saved model files, plots and images');
            api.code(el, 'task = Task.init(project_name="llm-finetune", task_name="lora-r16")\nparams = task.connect({"lr": 2e-4, "lora_r": 16, "epochs": 3})\nlogger = task.get_logger()\nlogger.report_scalar(title="loss", series="train", value=1.84, iteration=10)', 'python');
            api.docsNote(el);
          } },
          { name: 'Queue + agents', render: function (el, api) {
            api.html(el, '<p>Simulation: enqueue jobs and watch two agents pull them from the <code>gpu</code> queue (one job takes 4 ticks).</p>');
            var jobs = [], n = 0, ag = [null, null];
            var bar = api.h('div', { class: 'controls' }); el.appendChild(bar);
            api.button(bar, 'Enqueue a job', function () { n++; jobs.push({ id: n, left: 4, st: 'queued' }); draw(); }, 'primary');
            api.button(bar, 'Enqueue 3', function () { for (var i = 0; i < 3; i++) { n++; jobs.push({ id: n, left: 4, st: 'queued' }); } draw(); });
            api.button(bar, 'Reset', function () { jobs = []; n = 0; ag = [null, null]; draw(); });
            var qd = api.h('div'), ad = api.h('div', { class: 'row' }), stat = api.h('div', { class: 'muted' });
            el.appendChild(qd); el.appendChild(ad); el.appendChild(stat);
            function tick() {
              for (var a = 0; a < 2; a++) {
                if (ag[a]) { ag[a].left--; if (ag[a].left <= 0) { ag[a].st = 'done'; ag[a] = null; } }
                if (!ag[a]) { var j = jobs.filter(function (x) { return x.st === 'queued'; })[0]; if (j) { j.st = 'run'; ag[a] = j; } }
              }
              draw();
            }
            function draw() {
              var q = jobs.filter(function (x) { return x.st === 'queued'; });
              qd.innerHTML = '<b>Queue "gpu"</b> ' + api.type('integer') + ' ' + q.length + ' waiting<div class="clearml-q">' + (q.length ? q.map(function (x) { return '<span class="clearml-job">task #' + x.id + '</span>'; }).join('') : '<span class="muted">empty</span>') + '</div>';
              ad.innerHTML = ag.map(function (j, i) { return '<div class="clearml-ag"><b>agent ' + (i + 1) + '</b><br>' + (j ? '<span class="clearml-job run">running #' + j.id + ' (' + j.left + ' left)</span>' : '<span class="muted">idle, polling queue</span>') + '</div>'; }).join('');
              stat.innerHTML = 'Finished: ' + jobs.filter(function (x) { return x.st === 'done'; }).length + ' of ' + jobs.length;
            }
            api.every(900, tick);
            draw();
            api.docsNote(el);
            api.illustrative(el, 'Toy simulation of queue semantics; real agents poll the server and set up each environment first.');
          } }
        ]
      },

      // ------------------------------------------------------------ lora-tracking
      {
        id: 'lora-tracking', title: 'Worked flow: tracking a LoRA fine-tune',
        critical: {
          understand: ['Add <code>Task.init</code>, <code>task.connect</code> (hyperparameters) and <code>logger.report_scalar</code> (loss) to your training loop; ClearML records the rest.',
            'Train and validation curves together show <b>overfitting</b>: validation stops improving while training keeps falling.'],
          details: ['Curves on this slide come from a seeded formula, not a real training run. Drag the sliders to see how settings might shift them.',
            'The best checkpoint is chosen from validation loss; the run keeps its settings, so you can compare it with other runs.']
        },
        dataTypes: [
          { name: 'train loss', type: 'float', shape: '[steps]', example: '2.4 → ~1.2 (illustrative)' },
          { name: 'val loss', type: 'float', shape: '[steps]', example: 'U-shaped when overfitting' },
          { name: 'lr, lora_r', type: 'json', shape: 'dict', example: '{lr: 2e-4, lora_r: 16}' },
          { name: 'iteration', type: 'integer', shape: 'scalar', example: 'step 0, 10, …' },
          { name: 'adapter checkpoint', type: 'matrix', shape: 'LoRA A, B per layer', example: 'artifact' }
        ],
        levels: [
          { name: 'Code', render: function (el, api) {
            api.html(el, '<p>A condensed LoRA training script with ClearML added (training code itself is elided).</p>');
            api.code(el, [
              'from clearml import Task',
              '',
              'task = Task.init(project_name="llm-finetune",',
              '                 task_name="lora-r16-lr2e-4")',
              'cfg = task.connect({"lr": 2e-4, "lora_r": 16, "epochs": 3,',
              '                    "base_model": "my-8b-base"})',
              'logger = task.get_logger()',
              '',
              'for step, batch in enumerate(loader):',
              '    loss = train_step(model, batch, lr=cfg["lr"])',
              '    if step % 10 == 0:',
              '        logger.report_scalar("loss", "train", value=loss, iteration=step)',
              '    if step % 50 == 0:',
              '        val = evaluate(model, val_loader)',
              '        logger.report_scalar("loss", "val", value=val, iteration=step)',
              '',
              'model.save_pretrained("adapter/")',
              'task.upload_artifact("adapter", artifact_object="adapter/")'
            ].join('\n'), 'python');
            api.docsNote(el, 'Verify <code>connect</code>, <code>report_scalar</code> and <code>upload_artifact</code> signatures in the current ClearML docs.');
          } },
          { name: 'Loss curves', render: function (el, api) {
            var lr = 2e-4, steps = 300, rank = 16;
            var bar = api.h('div', { class: 'controls' }); el.appendChild(bar);
            var ch, info = api.h('div', { class: 'muted' });
            var chBox = api.h('div'); 
            api.slider(bar, { label: 'learning rate (log10)', min: -4.5, max: -2.5, step: 0.1, value: Math.log(lr) / Math.LN10, format: function (v) { return Math.pow(10, v).toExponential(1); }, onInput: function (v) { lr = Math.pow(10, v); draw(); } });
            api.slider(bar, { label: 'steps', min: 100, max: 600, step: 50, value: steps, onInput: function (v) { steps = v; draw(); } });
            api.slider(bar, { label: 'LoRA rank', min: 4, max: 64, step: 4, value: rank, onInput: function (v) { rank = v; draw(); } });
            el.appendChild(chBox); el.appendChild(info);
            function draw() {
              if (!chBox) return;
              var c = curves(lr, steps, 7, rank), b = best(c);
              var spec = { series: [{ name: 'train loss', points: c.train, color: '#4ea8ff' }, { name: 'val loss', points: c.val, color: '#ff9a4d' }], xLabel: 'iteration (step)', yLabel: 'loss', width: 560, height: 260, markers: [{ x: b[0], label: 'best val' }] };
              if (ch) ch.update(spec); else ch = api.chart(chBox, spec);
              info.innerHTML = 'Best validation loss ' + api.type('float') + ' <b>' + api.fmt(b[1], 3) + '</b> at step ' + b[0] + '. Final gap (val − train): ' + api.fmt(c.val[c.val.length - 1][1] - c.train[c.train.length - 1][1], 3) + (c.val[c.val.length - 1][1] > b[1] + 0.03 ? ' · validation rose after the best step: likely overfitting, so keep the earlier checkpoint.' : '.');
            }
            draw();
            api.illustrative(el, 'Curves = seeded exponential decay + noise + an overfitting term. They only mimic typical shapes.');
          } },
          { name: 'What is logged', render: function (el, api) {
            var steps = [
              { title: '1. Task.init', html: 'Creates the Task on the server; records git commit, diff, packages, command line. Status: <b>running</b>.<pre class="code clearml-pre"><code>{"project":"llm-finetune","name":"lora-r16-lr2e-4","status":"in_progress"}</code></pre>' },
              { title: '2. task.connect(cfg)', html: 'Hyperparameters appear in the UI and become <b>editable when the task is cloned</b>.<pre class="code clearml-pre"><code>{"lr":0.0002,"lora_r":16,"epochs":3}</code></pre>' },
              { title: '3. report_scalar every 10 steps', html: 'Points are buffered and sent in the background: <code>(loss, train, iter=10, value=2.12)</code>. They show up as live curves in the Scalars tab.' },
              { title: '4. Validation every 50 steps', html: 'A second series <code>(loss, val)</code> lets you spot overfitting and pick the best checkpoint.' },
              { title: '5. upload_artifact / output model', html: 'The adapter files are uploaded to storage; the run links to them, so "which weights came from which run" is answerable.' },
              { title: '6. Task closes', html: 'Status becomes <b>completed</b> (or <b>failed</b>/<b>aborted</b>). The run is now comparable with every other run in the project.' }
            ];
            api.stepper(el, { steps: steps, interval: 2400 });
            api.docsNote(el);
          } }
        ]
      },

      // ------------------------------------------------------------ web-ui
      {
        id: 'web-ui', title: 'The web UI (a mock): compare runs, scalars, hyperparameters, artifacts',
        critical: {
          understand: ['The UI lists experiments in a table; select several to <b>overlay their curves</b> and diff their hyperparameters.',
            'Open one run to see Scalars, Hyperparameters, Artifacts and Console tabs.'],
          details: ['<b>This is a hand-built mock, not a real ClearML screenshot</b>; layout and labels differ from the product.',
            'Everything shown is derived from the same seeded formula as the previous slide.']
        },
        dataTypes: [
          { name: 'experiment row', type: 'json', shape: 'object', example: 'name, status, params' },
          { name: 'status', type: 'text', shape: 'enum', example: 'done / running / failed' },
          { name: 'best val loss', type: 'float', shape: 'scalar', example: 'computed' },
          { name: 'selected runs', type: 'integer', shape: 'scalar', example: '0 to 5' }
        ],
        levels: [
          { name: 'Experiments table', render: function (el, api) {
            var selected = { 'a1f3': true, 'c90d': true };
            var wrap = api.h('div', { class: 'clearml-mock' });
            wrap.appendChild(api.h('span', { class: 'clearml-mock-tag', text: 'MOCK UI (not real ClearML)' }));
            var tw = api.h('div', { class: 'clearml-wrapx' }); wrap.appendChild(tw);
            var chBox = api.h('div'); wrap.appendChild(chBox);
            var ch;
            el.appendChild(wrap);
            function draw() {
              var h = '<table class="clearml-tbl"><tr><th></th><th>name</th><th>status</th><th>lr</th><th>rank</th><th>best val</th><th>user</th><th>time</th></tr>';
              RUNS.forEach(function (r, i) {
                var c = curves(r.lr, r.steps, r.seed, r.rank), b = best(c);
                h += '<tr class="' + (selected[r.id] ? 'sel' : '') + '" data-i="' + i + '"><td><input type="checkbox" ' + (selected[r.id] ? 'checked' : '') + ' data-id="' + r.id + '"></td><td>' + r.name + '</td><td><span class="clearml-st ' + (r.status === 'done' ? 'done' : r.status === 'run' ? 'run' : 'fail') + '">' + (r.status === 'done' ? 'completed' : r.status === 'run' ? 'running' : 'failed') + '</span></td><td>' + r.lr.toExponential(0) + '</td><td>' + r.rank + '</td><td>' + api.fmt(b[1], 3) + '</td><td>' + r.user + '</td><td>' + r.dur + '</td></tr>';
              });
              tw.innerHTML = h + '</table>';
              Array.prototype.forEach.call(tw.querySelectorAll('input'), function (cb) { cb.onchange = function () { selected[cb.getAttribute('data-id')] = cb.checked; draw(); }; });
              var pal = ['#4ea8ff', '#ff9a4d', '#5ad18a', '#b98cff', '#ff5d73'];
              var series = [];
              RUNS.forEach(function (r, i) { if (selected[r.id]) series.push({ name: r.name, points: curves(r.lr, r.steps, r.seed, r.rank).val, color: pal[i] }); });
              chBox.innerHTML = '<div class="muted">Compare: validation loss of ' + series.length + ' selected run(s)</div>';
              var cb = api.h('div'); chBox.appendChild(cb);
              if (series.length) api.chart(cb, { series: series, xLabel: 'iteration', yLabel: 'val loss', width: 560, height: 240 });
              else cb.innerHTML = '<span class="muted">Select runs with the checkboxes.</span>';
            }
            draw();
            api.illustrative(el, 'Run names, statuses and loss values are invented for this mock.');
          } },
          { name: 'One run', render: function (el, api) {
            var r = RUNS[0], c = curves(r.lr, r.steps, r.seed, r.rank);
            var wrap = api.h('div', { class: 'clearml-mock' });
            wrap.appendChild(api.h('span', { class: 'clearml-mock-tag', text: 'MOCK UI (not real ClearML)' }));
            wrap.appendChild(api.h('div', { html: '<b>' + r.name + '</b> <span class="clearml-st done">completed</span> · project llm-finetune · id ' + r.id }));
            var body = api.h('div'); 
            el.appendChild(wrap);
            tabs(wrap, api, ['Scalars', 'Hyperparameters', 'Artifacts', 'Console'], function (i) {
              body.innerHTML = '';
              if (i === 0) api.chart(body, { series: [{ name: 'train', points: c.train, color: '#4ea8ff' }, { name: 'val', points: c.val, color: '#ff9a4d' }], xLabel: 'iteration', yLabel: 'loss', width: 560, height: 230 });
              if (i === 1) api.table(body, { headers: ['section', 'name', 'value'], rows: [['General', 'lr', '0.0002'], ['General', 'lora_r', '16'], ['General', 'epochs', '3'], ['General', 'base_model', 'my-8b-base'], ['Args', 'seed', '3']] });
              if (i === 2) api.table(body, { headers: ['artifact', 'type', 'size (illustrative)'], rows: [['adapter', 'output model', '~ 42 MiB'], ['val_samples.csv', 'artifact', '~ 180 KiB'], ['train_config.json', 'artifact', '~ 1 KiB']] });
              if (i === 3) api.code(body, 'step 10  train_loss=2.05\nstep 20  train_loss=1.78\nstep 50  train_loss=1.52  val_loss=1.61\n...\nsaved adapter/ (rank 16)', 'text');
            }, 0);
            wrap.appendChild(body);
            api.illustrative(el, 'Sizes and console lines are made up.');
          } }
        ]
      },

      // ------------------------------------------------------------ rag-prompts
      {
        id: 'rag-prompts', title: 'Tracking RAG and prompt experiments, and how ClearML compares',
        critical: {
          understand: ['Prompt templates, retrieval settings (chunk size, top-k) and eval scores are just <b>parameters and metrics</b>: log them like hyperparameters.',
            'Then "which prompt + retrieval config scored best on our eval set?" becomes a table sort.'],
          details: ['No weights change in RAG or prompt tuning, but results still depend on many knobs; tracking keeps them honest.',
            'MLflow, Weights &amp; Biases and ClearML overlap a lot; differences below are broad generalisations, so verify against current docs.']
        },
        dataTypes: [
          { name: 'prompt template', type: 'text', shape: 'string', example: '"Answer using only: {context}"' },
          { name: 'chunk_size, top_k', type: 'integer', shape: 'scalar', example: '512, 4' },
          { name: 'recall@k', type: 'probability', shape: 'scalar', example: '0.82 (illustrative)' },
          { name: 'faithfulness score', type: 'probability', shape: 'scalar', example: 'LLM-as-judge' },
          { name: 'eval set', type: 'text', shape: 'list[question, reference]', example: '200 rows' }
        ],
        levels: [
          { name: 'What to log', render: function (el, api) {
            api.html(el, '<p>A RAG experiment (see ' + api.link('memory-rag/rag-pipeline', 'the RAG pipeline') + ') has more knobs than a weight update. Each becomes a tracked parameter or metric.</p>');
            api.blocks(el, { rows: [[
              { label: 'Prompt template', sub: 'task.connect', type: 'text' },
              { label: 'Retrieval params', sub: 'chunk_size, top_k, embedder', type: 'json' }
            ], [
              { label: 'Run eval set', sub: 'answers + retrieved ids', type: 'function' },
              { label: 'Scores', sub: 'recall@k, faithfulness', type: 'probability', edge: 'report_scalar' }
            ]] });
            api.code(el, [
              'task = Task.init(project_name="support-rag", task_name="chunk512-k4-v2")',
              'task.connect({"chunk_size": 512, "top_k": 4,',
              '              "embedder": "my-embed-model", "reranker": None})',
              'task.set_parameter("prompt/template",',
              '    "Answer using only the context. If unsure, say so.\\n{context}\\n\\nQ: {question}")',
              '',
              'scores = run_eval(eval_set)   # your code',
              'lg = task.get_logger()',
              'lg.report_scalar("eval", "recall@k", scores.recall, iteration=0)',
              'lg.report_scalar("eval", "faithfulness", scores.faith, iteration=0)',
              'lg.report_table("bad answers", "sample", table_plot=scores.worst_df)'
            ].join('\n'), 'python');
            api.docsNote(el, 'Check parameter and table-reporting method names against the current ClearML SDK.');
          } },
          { name: 'Sweep table', render: function (el, api) {
            var key = 'faith';
            var bar = api.h('div', { class: 'controls' }); el.appendChild(bar);
            var box = api.h('div'); 
            var cfg = [];
            [256, 512, 1024].forEach(function (cs) { [2, 4, 8].forEach(function (k) {
              // illustrative formulas: recall grows with k, shrinks with large chunks; faithfulness falls with noisy extra context
              var recall = Math.min(0.97, 0.45 + 0.13 * Math.log(k) / Math.LN2 * (cs === 256 ? 1.1 : cs === 512 ? 1 : 0.8) + (cs === 512 ? 0.04 : 0));
              var faith = 0.9 - 0.012 * k * (cs / 256) + 0.05 * (recall - 0.7);
              cfg.push({ cs: cs, k: k, recall: recall, faith: faith, tokens: cs * k });
            }); });
            api.select(bar, { label: 'Sort by', options: [{ value: 'faith', label: 'faithfulness' }, { value: 'recall', label: 'recall@k' }, { value: 'tokens', label: 'context tokens (cost)' }], value: key, onChange: function (v) { key = v; draw(); } });
            el.appendChild(box);
            function draw() {
              var rows = cfg.slice().sort(function (a, b) { return key === 'tokens' ? a.tokens - b.tokens : b[key] - a[key]; });
              box.innerHTML = '';
              api.table(box, { headers: ['chunk_size', 'top_k', 'recall@k', 'faithfulness', 'context tokens'], rows: rows.map(function (r) { return [r.cs, r.k, api.fmt(r.recall, 2), api.fmt(r.faith, 2), api.fmtInt(r.tokens)]; }) });
            }
            draw();
            api.html(el, '<p class="muted">Nine runs, one table: more context raises recall but costs tokens and can lower faithfulness in this toy model. In ClearML this is the experiments table with custom columns.</p>');
            api.illustrative(el, 'Scores come from made-up formulas; real values need your own eval set.');
          } },
          { name: 'Vs MLflow / W&B', render: function (el, api) {
            api.table(el, {
              headers: ['', 'ClearML', 'MLflow', 'Weights &amp; Biases'],
              rows: [
                ['Core idea', 'Tracking + orchestration + data in one platform', 'Tracking, model registry, packaging; widely adopted', 'Hosted experiment tracking and reports; strong visualisation'],
                ['Run jobs remotely', 'Built-in agents and queues', 'Generally via separate tooling', 'Sweeps and launch features; check docs'],
                ['Data versioning', 'ClearML Data', 'Datasets tracking; not a full data VCS', 'Artifacts with lineage'],
                ['Hosting', 'Self-host or hosted', 'Self-host common', 'Primarily hosted; self-host options exist'],
                ['Typical pull', 'One stack for the whole loop', 'Open standard, many integrations', 'Polished UI and team collaboration']
              ]
            });
            api.callout(el, 'Hedged, high-level comparison: all three evolve quickly and overlap. Pick based on your hosting needs, team habits and whether you want orchestration included. Verify features in current documentation.', 'hedge');
            api.docsNote(el);
          } }
        ]
      }
    ]
  };
  DEF.slides.forEach(function (sl) { sl.levels.forEach(function (l) { var r = l.render; l.render = function (el, api) { el.style.minWidth = '0'; el.style.maxWidth = '100%'; if (el.parentElement) el.parentElement.style.maxWidth = '100%'; return r(el, api); }; }); });
  Course.register(DEF);
})();
