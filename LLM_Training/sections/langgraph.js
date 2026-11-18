/* LangGraph section: chains vs graphs, state/reducers, ReAct graph executor, checkpointing, multi-agent, when to use. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- shared helpers
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  // spec: {w,h,nodes:[{id,x,y,w,h,label,sub,cls}], edges:[{id,d,label,lx,ly,cls}], notes:[{x,y,t}]}
  function svgGraph(spec, p) {
    var s = '<svg class="langgraph-svg" viewBox="0 0 ' + spec.w + ' ' + spec.h + '" role="img" aria-label="' + esc(spec.aria || 'graph diagram') + '">';
    s += '<defs><marker id="' + p + 'ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ar" d="M0,0 L10,5 L0,10 z"/></marker>' +
         '<marker id="' + p + 'arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arA" d="M0,0 L10,5 L0,10 z"/></marker></defs>';
    (spec.edges || []).forEach(function (e) {
      s += '<path class="lg-edge ' + (e.cls || '') + '" data-id="' + e.id + '" d="' + e.d + '" marker-end="url(#' + p + 'ar)"/>';
      if (e.label) s += '<text class="lg-elabel" x="' + e.lx + '" y="' + e.ly + '">' + esc(e.label) + '</text>';
    });
    (spec.nodes || []).forEach(function (n) {
      s += '<g class="lg-node ' + (n.cls || 'fn') + '" data-id="' + n.id + '"><rect x="' + n.x + '" y="' + n.y + '" width="' + n.w + '" height="' + n.h + '" rx="8"/>' +
           '<text x="' + (n.x + n.w / 2) + '" y="' + (n.y + (n.sub ? n.h / 2 - 1 : n.h / 2 + 4)) + '">' + esc(n.label) + '</text>' +
           (n.sub ? '<text class="sub" x="' + (n.x + n.w / 2) + '" y="' + (n.y + n.h / 2 + 12) + '">' + esc(n.sub) + '</text>' : '') + '</g>';
    });
    (spec.notes || []).forEach(function (n) { s += '<text class="lg-note" x="' + n.x + '" y="' + n.y + '">' + esc(n.t) + '</text>'; });
    return s + '</svg>';
  }
  function setActive(root, p, nodes, edges, doneNodes) {
    var i;
    var all = root.querySelectorAll('.lg-node');
    for (i = 0; i < all.length; i++) {
      var id = all[i].getAttribute('data-id');
      all[i].classList.toggle('active', nodes.indexOf(id) >= 0);
      all[i].classList.toggle('done', !!doneNodes && doneNodes.indexOf(id) >= 0 && nodes.indexOf(id) < 0);
    }
    var es = root.querySelectorAll('.lg-edge');
    for (i = 0; i < es.length; i++) {
      var eid = es[i].getAttribute('data-id'), on = edges.indexOf(eid) >= 0;
      es[i].classList.toggle('active', on);
      es[i].setAttribute('marker-end', 'url(#' + p + (on ? 'arA' : 'ar') + ')');
    }
  }
  function codeBlock(el, api, src) { var w = document.createElement('div'); w.className = 'langgraph-cw'; el.appendChild(w); api.code(w, src, 'python'); }
  function box(el, html) { var d = document.createElement('div'); d.innerHTML = html; el.appendChild(d); return d; }

  // ---- ReAct graph geometry (reused by several slides)
  var REACT = {
    w: 540, h: 250, aria: 'ReAct graph: START to agent, conditional edge to tools or END, tools back to agent',
    nodes: [
      { id: 'START', x: 10, y: 100, w: 80, h: 40, label: 'START', cls: 'term' },
      { id: 'agent', x: 160, y: 95, w: 120, h: 50, label: 'agent', sub: 'LLM call · function' },
      { id: 'END', x: 410, y: 15, w: 80, h: 40, label: 'END', cls: 'term' },
      { id: 'tools', x: 390, y: 180, w: 120, h: 50, label: 'tools', sub: 'ToolNode · function' }
    ],
    edges: [
      { id: 'start', d: 'M90,120 L158,120' },
      { id: 'toEnd', d: 'M262,95 C290,50 340,38 408,35', cls: 'cond', label: 'no tool_calls', lx: 330, ly: 28 },
      { id: 'toTools', d: 'M280,135 C320,175 350,200 388,205', cls: 'cond', label: 'tool_calls', lx: 290, ly: 190 },
      { id: 'back', d: 'M450,230 C440,248 230,255 215,147', label: 'add_edge("tools","agent")', lx: 310, ly: 246 }
    ]
  };

  // ---- Paris example messages
  var H = { type: 'human', id: 'm1', content: 'Book me a table for 2 in Paris tomorrow at 8pm and tell me if I need an umbrella' };
  var A1 = { type: 'ai', id: 'm2', content: '', tool_calls: [
    { id: 'c1', name: 'get_weather', args: { city: 'Paris', day: 'tomorrow' } },
    { id: 'c2', name: 'search_restaurants', args: { city: 'Paris', party: 2, time: '20:00' } }] };
  var T1 = { type: 'tool', id: 'm3', name: 'get_weather', tool_call_id: 'c1', content: '{"rain_prob":0.7,"temp_c":11}' };
  var T2 = { type: 'tool', id: 'm4', name: 'search_restaurants', tool_call_id: 'c2', content: '[{"name":"Le Petit Zinc","free_20_00":true}]' };
  var A2 = { type: 'ai', id: 'm5', content: '', tool_calls: [{ id: 'c3', name: 'book_table', args: { restaurant: 'Le Petit Zinc', time: '20:00', party: 2 } }] };
  var T3 = { type: 'tool', id: 'm6', name: 'book_table', tool_call_id: 'c3', content: '{"status":"confirmed","ref":"A7F3"}' };
  var A3 = { type: 'ai', id: 'm7', content: 'Booked Le Petit Zinc at 20:00 for 2 (ref A7F3). 70% chance of rain: bring an umbrella.', tool_calls: [] };

  function brief(m) {
    if (m.type === 'human') return esc(m.content);
    if (m.type === 'ai') return m.tool_calls && m.tool_calls.length ?
      'tool_calls: ' + m.tool_calls.map(function (c) { return '<code>' + esc(c.name) + '</code>'; }).join(', ') : esc(m.content);
    return '<code>' + esc(m.name) + '</code> → ' + esc(m.content);
  }
  function msgList(el, msgs, newFrom) {
    var d = document.createElement('div'); d.className = 'langgraph-msgs';
    msgs.forEach(function (m, i) {
      var r = document.createElement('div');
      r.className = 'langgraph-msg ' + m.type + (i >= newFrom ? ' new' : '');
      r.innerHTML = '<b>' + m.type + '</b><span>' + brief(m) + '</span>';
      d.appendChild(r);
    });
    el.appendChild(d); return d;
  }
  function stateJson(msgs, extra) {
    var o = { messages: msgs.map(function (m) {
      var c = { type: m.type, content: m.content };
      if (m.type === 'ai') c.tool_calls = m.tool_calls;
      if (m.type === 'tool') { c.name = m.name; c.tool_call_id = m.tool_call_id; }
      return c;
    }) };
    if (extra) Object.keys(extra).forEach(function (k) { o[k] = extra[k]; });
    return JSON.stringify(o, null, 1);
  }

  var REACT_CODE = `from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

tools = [get_weather, search_restaurants, book_table]   # @tool functions
llm = model.bind_tools(tools)

def agent(state: MessagesState):                  # node: function(state) -> partial update
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))        # runs every tool call in the last AI message
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)   # -> "tools" or END
builder.add_edge("tools", "agent")                # the cycle
graph = builder.compile()

graph.invoke({"messages": [("user", "Book me a table for 2 in Paris ...")]})`;

  Course.register({
    id: 'langgraph', part: 'II', order: 145, title: 'LangGraph',
    slides: [

      // ================================================================= 1. chains vs graphs
      {
        id: 'chains-vs-graphs',
        title: 'Chains vs graphs: why agents need cycles',
        critical: {
          understand: [
            'A <b>chain</b> is a fixed pipeline (a DAG): data flows one way and each step runs once.',
            'An agent must <b>loop</b>: call the model, maybe run tools, call the model again. That needs a <b>cycle</b> and a <b>conditional edge</b> deciding "again or stop?".',
            'LangGraph models this as a graph of <b>nodes</b> (functions) and <b>edges</b> over a shared <b>state</b>. It is the agent loop from ' + '<span class="xlink" data-goto="llm-to-agent/agent-loop">the agent loop</span>, made explicit.'
          ],
          details: [
            'The router (conditional edge) is ordinary code: usually "does the last AI message have tool_calls?". The LLM only supplies that text; the harness decides what runs next.',
            'Because a cycle can run forever, LangGraph enforces a <code>recursion_limit</code> (a step budget).'
          ]
        },
        dataTypes: [
          { name: 'state', type: 'json', shape: 'object', example: '{"messages": [...]}' },
          { name: 'node', type: 'function', shape: 'state → partial state', example: 'agent, tools' },
          { name: 'routing decision', type: 'boolean', shape: 'scalar', example: 'has tool_calls?' },
          { name: 'tool rounds k', type: 'integer', shape: 'scalar', example: '2' }
        ],
        levels: [
          { name: 'Chain vs graph', render: function (el, api) {
            var g = api.grid(el, 300);
            var chain = { w: 440, h: 110, aria: 'chain: prompt, model, parser, done',
              nodes: [{ id: 'p', x: 8, y: 30, w: 90, h: 44, label: 'prompt' }, { id: 'm', x: 140, y: 30, w: 90, h: 44, label: 'model' },
                      { id: 'o', x: 272, y: 30, w: 90, h: 44, label: 'parser' }, { id: 'e', x: 392, y: 34, w: 44, h: 36, label: 'out', cls: 'term' }],
              edges: [{ id: 'a', d: 'M98,52 L138,52' }, { id: 'b', d: 'M230,52 L270,52' }, { id: 'c', d: 'M362,52 L390,52' }],
              notes: [{ x: 8, y: 100, t: 'DAG: every node runs once, left to right' }] };
            var c1 = api.h('div', { class: 'card' }); c1.innerHTML = '<h4>Chain (LCEL-style pipeline)</h4>' + svgGraph(chain, 'lgc1');
            var c2 = api.h('div', { class: 'card' }); c2.innerHTML = '<h4>Graph with a cycle</h4>' + svgGraph(REACT, 'lgc2');
            g.appendChild(c1); g.appendChild(c2);
            api.callout(el, 'An agent decides <b>at run time</b> how many tool rounds it needs. A fixed chain cannot express "repeat until the model stops asking for tools".', 'key');
            setActive(c2, 'lgc2', [], [], []);
          } },
          { name: 'Conditional edge', render: function (el, api) {
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            var k = 2, fixed = 2;
            function draw() {
              out.innerHTML = '';
              var calls = k + 1;
              var chainOk = k <= fixed - 1;
              var rows = [['Chain with ' + fixed + ' fixed steps (model → tool → model)', chainOk ? '<span class="langgraph-ok">completes</span>' : '<span class="langgraph-no">stops early: needs ' + (k - (fixed - 1)) + ' more tool round(s) it has no node for</span>', fixed + ' LLM-ish steps'],
                          ['Graph with cycle', '<span class="langgraph-ok">completes</span> (limit: <code>recursion_limit=25</code> steps)', calls + ' LLM calls, ' + k + ' tool super-steps']];
              api.table(out, { headers: ['Design', 'Task needing k=' + k + ' tool rounds', 'Work done'], rows: rows });
              var steps = [];
              for (var i = 0; i < k; i++) steps.push('agent → <b>tool_calls?</b> ' + api.type('boolean', 'true') + ' → tools');
              steps.push('agent → <b>tool_calls?</b> ' + api.type('boolean', 'false') + ' → END');
              box(out, '<div class="langgraph-route">' + steps.join('<br>') + '</div>');
            }
            api.slider(ctl, { label: 'Tool rounds the task needs (k)', min: 0, max: 6, step: 1, value: 2, format: function (v) { return String(v); }, onInput: function (v) { k = v; draw(); } });
            api.callout(el, 'The routing decision is a plain <b>boolean</b> computed by your code from the state. Ten lines of Python, not an LLM call.', 'info');
          } },
          { name: 'Code', render: function (el, api) {
            codeBlock(el, api, `# Chain: fixed pipeline, one pass
chain = prompt | model | parser
chain.invoke({"question": "..."})

# Graph: cycle + conditional edge
def route(state) -> str:                       # routing = boolean-ish decision
    return "tools" if state["messages"][-1].tool_calls else END

builder.add_conditional_edges("agent", route, ["tools", END])
builder.add_edge("tools", "agent")             # <- the cycle`);
            api.docsNote(el, 'See ' + api.link('langchain/lcel', 'LCEL') + ' for the chain side.');
          } }
        ]
      },

      // ================================================================= 2. state + reducers
      {
        id: 'state-reducers',
        title: 'State and reducers: how node updates merge',
        critical: {
          understand: [
            'The graph has one shared <b>state</b> (a TypedDict or Pydantic model). Each field is a <b>channel</b>.',
            'A node returns a <b>partial update</b>, not the whole state. A <b>reducer</b> per channel decides how the update merges in.',
            'Without a reducer a channel is <b>overwritten</b>. With <code>add_messages</code> the new messages are <b>appended</b> (or replace a message with the same id).'
          ],
          details: [
            'This is why the conversation grows turn by turn: the messages channel is exactly the list that is serialized into one token sequence for the next model call.',
            'Reducers also make parallel branches safe: two nodes writing the same channel in one super-step get <i>combined</i>, not clobbered.'
          ]
        },
        dataTypes: [
          { name: 'state', type: 'json', shape: 'object', example: '{messages, step}' },
          { name: 'messages channel', type: 'json', shape: 'list[message]', example: '3 messages' },
          { name: 'step counter', type: 'integer', shape: 'scalar', example: '2' },
          { name: 'node update', type: 'json', shape: 'object', example: '{"messages":[AI]}' },
          { name: 'reducer', type: 'function', shape: '(old, new) → merged', example: 'add_messages' }
        ],
        levels: [
          { name: 'Channels', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'n', label: 'node returns update', sub: '{"messages":[AI]}', type: 'json' },
              { id: 'r', label: 'reducer per channel', sub: 'add_messages / overwrite', type: 'function' },
              { id: 's', label: 'new state', sub: 'messages: old + new', type: 'json' }],
              caption: 'Each node reads the state, returns a partial update, and the reducer merges it.' });
            api.callout(el, 'Mental model: state is a <b>shared whiteboard</b>; nodes never edit it directly, they hand in updates.', 'tip');
          } },
          { name: 'Try updates', render: function (el, api) {
            var useReducer = true, msgs = [H], step = 0, lastNew = 0, lastUpd = '(none yet)';
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            var A1b = { type: 'ai', id: 'm2', content: 'edited: also check metro strikes', tool_calls: [] };
            function apply(upd, label) {
              lastUpd = label;
              if (upd.messages) {
                if (useReducer) {
                  var m = msgs.slice();
                  upd.messages.forEach(function (n) {
                    var idx = -1; m.forEach(function (x, i) { if (x.id === n.id) idx = i; });
                    if (idx >= 0) m[idx] = n; else m.push(n);
                  });
                  lastNew = msgs.length; msgs = m;
                } else { msgs = upd.messages.slice(); lastNew = 0; }
              }
              if (upd.step != null) step = upd.step;
              draw();
            }
            function draw() {
              out.innerHTML = '';
              box(out, '<div class="langgraph-route">Last update: <code>' + esc(lastUpd) + '</code><br>Reducer on <code>messages</code>: <b>' + (useReducer ? 'add_messages (append / replace-by-id)' : 'none (overwrite)') + '</b>; on <code>step</code>: overwrite</div>');
              box(out, '<b>messages</b> ' + api.badge('json', '[' + msgs.length + ' messages]') + ' &nbsp; <b>step</b> ' + api.badge('integer', 'scalar') + ' = ' + step);
              msgList(out, msgs, lastNew);
            }
            api.toggle(ctl, { label: 'Use add_messages reducer', value: true, onChange: function (v) { useReducer = v; msgs = [H]; step = 0; lastNew = 0; lastUpd = '(reset)'; draw(); } });
            api.button(ctl, 'agent → +AI tool call', function () { apply({ messages: [A1] }, '{"messages": [AIMessage(tool_calls=[get_weather, search_restaurants])]}'); }, 'primary');
            api.button(ctl, 'tools → +2 results', function () { apply({ messages: [T1, T2] }, '{"messages": [ToolMessage, ToolMessage]}'); });
            api.button(ctl, 'same id as AI msg → edit', function () { apply({ messages: [A1b] }, '{"messages": [AIMessage(id="m2", content="edited ...")]}'); });
            api.button(ctl, 'counter → step + 1', function () { apply({ step: step + 1 }, '{"step": ' + (step + 1) + '}'); });
            api.button(ctl, 'Reset', function () { msgs = [H]; step = 0; lastNew = 0; lastUpd = '(reset)'; draw(); });
            draw();
            api.callout(el, 'Switch the reducer off and send "+2 results": the whole conversation is replaced by two tool messages. That is the classic bug when a custom state forgets its reducer.', 'warn');
          } },
          { name: 'Code', render: function (el, api) {
            codeBlock(el, api, `from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState

class State(TypedDict):
    messages: Annotated[list, add_messages]   # reducer: append / replace by id
    step: int                                 # no reducer: last write wins

# MessagesState is the same thing, pre-made:
class MyState(MessagesState):
    step: int

def count(state: State):
    return {"step": state["step"] + 1}        # partial update: only this key

# a custom reducer is just (old, new) -> merged
import operator
class S2(TypedDict):
    visited: Annotated[list[str], operator.add]`);
            api.docsNote(el);
          } }
        ]
      },

      // ================================================================= 3. ReAct graph executor
      {
        id: 'react-graph',
        title: 'A ReAct agent as a graph (interactive executor)',
        critical: {
          understand: [
            'The agent loop becomes two nodes and a conditional edge: <b>agent</b> (model call) and <b>tools</b> (run the requested tools).',
            'Each <b>super-step</b> runs the active node, merges its update into the state, then follows an edge. The router checks <code>tool_calls</code> on the last AI message.',
            'Under the hood each <code>agent</code> call is still one next-token-prediction run over the <b>whole serialized message list</b>; the graph is the harness around it.'
          ],
          details: [
            'Both tool calls in the first AI message run in the same <code>tools</code> super-step (ToolNode can run them in parallel).',
            'The model is simulated here; the node order, routing booleans and state updates mirror what the real graph does.'
          ]
        },
        dataTypes: [
          { name: 'state.messages', type: 'json', shape: 'list[message]', example: '1 → 7 messages' },
          { name: 'agent / tools', type: 'function', shape: 'state → update', example: 'nodes' },
          { name: 'tool_calls', type: 'json', shape: 'list[{name,args}]', example: '[book_table]' },
          { name: 'has tool_calls?', type: 'boolean', shape: 'scalar', example: 'true / false' },
          { name: 'user request', type: 'text', shape: 'string', example: 'Book me a table…' }
        ],
        levels: [
          { name: 'Graph', render: function (el, api) {
            var d = box(el, svgGraph(REACT, 'lgr1'));
            setActive(d, 'lgr1', [], []);
            var tags = api.h('div', { class: 'row' });
            tags.innerHTML = api.badge('function', 'node') + ' agent, tools &nbsp; ' + api.badge('boolean', 'router') + ' tool_calls? &nbsp; ' + api.badge('json', 'state') + ' messages';
            el.appendChild(tags);
            api.callout(el, 'Request: <i>' + api.esc(api.ex.agent.request) + '</i>. Dive in to run it step by step.', 'info');
          } },
          { name: 'Executor', render: function (el, api) {
            var steps = [
              { node: 'START', edges: [], add: [H], route: null, text: 'Input written to state: messages = [HumanMessage]. Edge START → agent.', next: ['start'] },
              { node: 'agent', edges: ['start'], add: [A1], route: true, text: 'agent ran the LLM over all messages. It asked for get_weather and search_restaurants.' },
              { node: 'tools', edges: ['toTools'], add: [T1, T2], route: null, text: 'ToolNode executed both calls and appended two ToolMessages.' },
              { node: 'agent', edges: ['back'], add: [A2], route: true, text: 'agent sees weather + restaurants and requests book_table.' },
              { node: 'tools', edges: ['toTools'], add: [T3], route: null, text: 'book_table ran and returned a confirmation.' },
              { node: 'agent', edges: ['back'], add: [A3], route: false, text: 'agent produced a final answer with no tool_calls.' },
              { node: 'END', edges: ['toEnd'], add: [], route: null, text: 'Routing sent the run to END. Final state returned.' }
            ];
            var idx = -1, msgs = [], newFrom = 0, done = [];
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var left = api.h('div'), right = api.h('div'); el.appendChild(left); el.appendChild(right);
            var gd = box(left, svgGraph(REACT, 'lgr2'));
            var info = api.h('div'); left.appendChild(info);
            var stateBox = api.h('div'); right.appendChild(stateBox);
            var timer = null;
            function draw() {
              var s = steps[idx];
              if (!s) {
                setActive(gd, 'lgr2', [], [], []);
                info.innerHTML = '<div class="langgraph-route">Press <b>Step</b> to run one super-step, or <b>Run</b> to play all.</div>';
                stateBox.innerHTML = '<b>State</b> ' + api.badge('json', 'object') + '<div class="muted">messages = [ ]</div>';
                return;
              }
              setActive(gd, 'lgr2', [s.node], s.edges, done);
              var rt = '';
              if (s.route === true) rt = 'Routing after agent: <b>has tool_calls?</b> ' + api.type('boolean', 'true') + ' → <code>"tools"</code>';
              else if (s.route === false) rt = 'Routing after agent: <b>has tool_calls?</b> ' + api.type('boolean', 'false') + ' → <code>END</code>';
              else rt = 'Edge is unconditional.';
              info.innerHTML = '<div class="langgraph-route"><b>Super-step ' + idx + ': ' + s.node + '</b> ' + api.type('function', s.node === 'START' || s.node === 'END' ? 'marker' : 'node') + '<br>' + esc(s.text) + '<br>' + rt + '</div>';
              stateBox.innerHTML = '<b>State after super-step ' + idx + '</b> ' + api.badge('json', '[' + msgs.length + ' messages]');
              msgList(stateBox, msgs, newFrom);
              var pre = api.h('pre', { class: 'langgraph-json', text: stateJson(msgs) });
              stateBox.appendChild(pre);
            }
            function step() {
              if (idx >= steps.length - 1) return false;
              idx++;
              var s = steps[idx];
              newFrom = msgs.length; msgs = msgs.concat(s.add);
              if (done.indexOf(s.node) < 0) done.push(s.node);
              draw(); return idx < steps.length - 1;
            }
            function reset() { if (timer) { timer.stop && timer.stop(); } idx = -1; msgs = []; newFrom = 0; done = []; draw(); }
            var running = false;
            api.button(ctl, 'Step ▶', function () { step(); }, 'primary');
            api.button(ctl, 'Run ⏵⏵', function () {
              if (running) return; if (idx >= steps.length - 1) reset(); running = true; step();
              api.every(1200, function () { if (!running) return; if (!step()) running = false; });
            });
            api.button(ctl, 'Reset', function () { running = false; reset(); });
            draw();
          } },
          { name: 'Code', render: function (el, api) {
            codeBlock(el, api, REACT_CODE);
            api.callout(el, '<code>tools_condition</code> is the prebuilt router from the diagram: it returns <code>"tools"</code> if the last AI message has <code>tool_calls</code>, else <code>END</code>. The prebuilt <code>create_react_agent</code> builds this same graph for you.', 'info');
            api.docsNote(el, 'Newer releases may expose the prebuilt agent under a different import path.');
          } }
        ]
      },

      // ================================================================= 4. checkpoint + interrupt
      {
        id: 'checkpoint-interrupt',
        title: 'Checkpointing and interrupt(): human approval gate',
        critical: {
          understand: [
            'With a <b>checkpointer</b> the graph saves a snapshot of the state after every super-step, keyed by a <code>thread_id</code>. That is the agent\'s short-term memory and crash recovery.',
            '<code>interrupt()</code> pauses the graph inside a node and surfaces a payload to a human. Resuming with <code>Command(resume=...)</code> continues from the saved checkpoint.',
            'Use it for risky tools such as <code>book_table</code>: the approval decision is a <b>boolean</b> that lives in ordinary code, not in the model.'
          ],
          details: [
            'Saved checkpoints can be listed and re-run: "time travel" lets you fork from an earlier state and try a different branch.',
            'Resuming re-runs the interrupted node from its start, so side effects before <code>interrupt()</code> should be idempotent.',
            'The in-memory saver is for demos; production uses a database-backed checkpointer.'
          ]
        },
        dataTypes: [
          { name: 'thread_id', type: 'text', shape: 'string', example: '"booking-42"' },
          { name: 'checkpoint', type: 'json', shape: 'object', example: '{values, next}' },
          { name: 'interrupt payload', type: 'json', shape: 'object', example: '{action: book_table}' },
          { name: 'approved', type: 'boolean', shape: 'scalar', example: 'true' },
          { name: 'Command(resume)', type: 'json', shape: 'object', example: '{"approved": true}' }
        ],
        levels: [
          { name: 'Idea', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'a', label: 'agent', type: 'function', sub: 'plans book_table' },
              { id: 'g', label: 'interrupt()', type: 'boolean', sub: 'pause + save', edge: 'needs approval' },
              { id: 'h', label: 'human', type: 'text', sub: 'approve / reject', edge: 'payload' },
              { id: 'r', label: 'Command(resume=…)', type: 'json', sub: 'continue', edge: 'decision' }],
              caption: 'The checkpointer stores state at every arrow; the run can sleep for days between "pause" and "resume".' });
            api.callout(el, 'Autonomy levels and where to put human gates: ' + api.link('safety-eval/autonomy-levels', 'autonomy levels') + '.', 'info');
          } },
          { name: 'Try the gate', render: function (el, api) {
            var cps = [], decision = null, sel = -1;
            function mk(i, dec) {
              var defs = [
                { node: '__start__', msgs: [H], next: 'agent' },
                { node: 'agent', msgs: [H, A1], next: 'tools' },
                { node: 'tools', msgs: [H, A1, T1, T2], next: 'agent' },
                { node: 'agent', msgs: [H, A1, T1, T2, A2], next: 'tools (interrupt!)' }];
              if (i < 4) return defs[i];
              var tm = dec ? T3 : { type: 'tool', id: 'm6', name: 'book_table', tool_call_id: 'c3', content: 'REJECTED by user: not executed' };
              var fin = dec ? A3 : { type: 'ai', id: 'm7', content: 'I did not book anything because you declined. Want other options?', tool_calls: [] };
              if (i === 4) return { node: 'tools', msgs: [H, A1, T1, T2, A2, tm], next: 'agent' };
              return { node: 'agent', msgs: [H, A1, T1, T2, A2, tm, fin], next: '— (END)' };
            }
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var g = api.grid(el, 300);
            var left = api.h('div'), right = api.h('div'); g.appendChild(left); g.appendChild(right);
            function paused() { return cps.length === 4 && decision === null; }
            function finished() { return cps.length === 6; }
            function advance() {
              if (paused() || finished()) return;
              var i = cps.length; var c = mk(i, decision);
              cps.push({ i: i, node: c.node, msgs: c.msgs, next: c.next, id: 'cp-' + (i + 1) });
              sel = cps.length - 1; draw();
            }
            function draw() {
              left.innerHTML = ''; right.innerHTML = '';
              box(left, '<b>Checkpoints</b> for <code>thread_id="booking-42"</code> ' + api.badge('json', 'list[checkpoint]'));
              cps.forEach(function (c, k) {
                var r = api.h('div', { class: 'langgraph-cp' + (k === sel ? ' sel' : ''), onclick: function () { sel = k; draw(); } });
                r.innerHTML = '<code>' + c.id + '</code> after <b>' + c.node + '</b> · next: <code>' + esc(c.next) + '</code> · ' + c.msgs.length + ' msgs';
                left.appendChild(r);
              });
              if (!cps.length) box(left, '<div class="muted">No checkpoints yet. Press Step.</div>');
              if (paused()) {
                var gate = api.h('div', { class: 'langgraph-gate' });
                gate.innerHTML = '<b>⏸ Graph interrupted</b> inside <code>tools</code><br><code>interrupt({"action":"book_table","args":{"restaurant":"Le Petit Zinc","time":"20:00","party":2}})</code><br>' + api.type('boolean', 'needs_approval') + ' → waiting for a human.';
                var b = api.h('div', { class: 'row' }); gate.appendChild(b);
                api.button(b, 'Approve ✓', function () { decision = true; draw(); }, 'primary');
                api.button(b, 'Reject ✗', function () { decision = false; draw(); });
                left.appendChild(gate);
              }
              if (decision !== null && cps.length >= 4) box(left, '<div class="langgraph-verdict">Resumed with <code>Command(resume={"approved": ' + (decision ? 'True' : 'False') + '})</code>' + (decision ? '' : ' · the tool returns a rejection message instead of booking') + '</div>');
              if (finished()) box(left, '<div class="langgraph-verdict">Run complete.</div>');
              if (sel >= 0 && cps[sel]) {
                var c = cps[sel];
                box(right, '<b>Selected: ' + c.id + '</b> ' + api.badge('json', 'values + next'));
                msgList(right, c.msgs, c.msgs.length - 1);
                right.appendChild(api.h('pre', { class: 'langgraph-json', text: stateJson(c.msgs) }));
                if (sel < cps.length - 1 || finished()) {
                  var fb = api.h('div', { class: 'row' });
                  api.button(fb, 'Time travel: fork from ' + c.id, function () {
                    cps = cps.slice(0, sel + 1); if (cps.length <= 4) decision = null; sel = cps.length - 1; draw();
                  });
                  right.appendChild(fb);
                }
              }
            }
            api.button(ctl, 'Step ▶', advance, 'primary');
            api.button(ctl, 'Run until gate ⏵⏵', function () { while (!paused() && !finished()) advance(); });
            api.button(ctl, 'Reset', function () { cps = []; decision = null; sel = -1; draw(); });
            draw();
            api.illustrative(el, 'Model outputs are scripted; checkpoint structure is simplified.');
          } },
          { name: 'Code', render: function (el, api) {
            codeBlock(el, api, `from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

@tool
def book_table(restaurant: str, time: str, party: int) -> str:
    """Book a table."""
    decision = interrupt({"action": "book_table",
                          "args": {"restaurant": restaurant, "time": time, "party": party}})
    if not decision["approved"]:
        return "REJECTED by user: not executed"
    return reserve(restaurant, time, party)       # real side effect

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "booking-42"}}

graph.invoke({"messages": [("user", "Book me a table ...")]}, cfg)
# -> returns with "__interrupt__" in the result; state is saved

graph.invoke(Command(resume={"approved": True}), cfg)    # resume the same thread

# time travel
history = list(graph.get_state_history(cfg))      # newest first
past = history[3]
graph.invoke(None, past.config)                   # re-run from that checkpoint`);
            api.docsNote(el, 'Interrupt/resume details (and how results are surfaced) have changed between versions.');
          } }
        ]
      },

      // ================================================================= 5. multi-agent patterns
      {
        id: 'multi-agent-patterns',
        title: 'Multi-agent patterns: supervisor, handoffs, Send fan-out',
        critical: {
          understand: [
            'Multi-agent graphs reuse the same primitives: nodes, edges, state. Each "agent" is a node (or a whole subgraph).',
            '<b>Supervisor</b>: a router node picks which worker runs next. <b>Handoff</b>: an agent returns <code>Command(goto=...)</code> to pass control. <b>Send</b>: fan out one task per item and reduce the results (map-reduce).',
            'More agents means more LLM calls and more places to fail. See ' + '<span class="xlink" data-goto="planning-multiagent/multi-agent">multi-agent cost</span>.'
          ],
          details: [
            'Fan-out writes land in one channel concurrently, so that channel needs a reducer such as <code>operator.add</code>.',
            'Start with one agent and a few tools; split only when context or responsibilities genuinely conflict.'
          ]
        },
        dataTypes: [
          { name: 'next worker', type: 'text', shape: 'string', example: '"researcher"' },
          { name: 'Command', type: 'json', shape: 'object', example: '{goto, update}' },
          { name: 'Send(node, payload)', type: 'json', shape: 'object', example: 'Send("summarize", {doc})' },
          { name: 'items N', type: 'integer', shape: 'scalar', example: '3' },
          { name: 'worker node', type: 'function', shape: 'state → update', example: 'researcher' }
        ],
        levels: [
          { name: 'Patterns', render: function (el, api) {
            var DIAG = {
              supervisor: { w: 520, h: 230, aria: 'supervisor pattern',
                nodes: [{ id: 'sup', x: 200, y: 10, w: 120, h: 46, label: 'supervisor', sub: 'router (LLM)' },
                        { id: 'w1', x: 20, y: 140, w: 130, h: 46, label: 'researcher' }, { id: 'w2', x: 195, y: 140, w: 130, h: 46, label: 'coder' }, { id: 'w3', x: 370, y: 140, w: 130, h: 46, label: 'writer' }],
                edges: [{ id: 'a', d: 'M215,56 L100,138', cls: 'cond' }, { id: 'b', d: 'M260,56 L260,138', cls: 'cond' }, { id: 'c', d: 'M305,56 L420,138', cls: 'cond' },
                        { id: 'd', d: 'M20,163 C-10,100 40,40 198,30' , label: 'back to supervisor', lx: 90, ly: 210 }],
                notes: [{ x: 330, y: 30, t: 'picks next worker or FINISH' }] },
              handoff: { w: 520, h: 200, aria: 'handoff pattern',
                nodes: [{ id: 'a', x: 20, y: 60, w: 120, h: 46, label: 'triage' }, { id: 'b', x: 200, y: 60, w: 120, h: 46, label: 'billing' }, { id: 'c', x: 380, y: 60, w: 120, h: 46, label: 'tech support' }],
                edges: [{ id: 'ab', d: 'M140,76 L198,76', cls: 'cond', label: 'goto="billing"', lx: 170, ly: 62 }, { id: 'bc', d: 'M320,76 L378,76', cls: 'cond', label: 'goto', lx: 350, ly: 62 },
                        { id: 'ca', d: 'M440,106 C440,170 80,170 80,108', cls: 'cond', label: 'goto="triage"', lx: 260, ly: 168 }],
                notes: [{ x: 20, y: 24, t: 'no central router: each node returns Command(goto=..., update=...)' }] },
              send: { w: 520, h: 230, aria: 'Send fan-out map-reduce',
                nodes: [{ id: 'split', x: 10, y: 92, w: 100, h: 46, label: 'split', sub: 'N items' },
                        { id: 's1', x: 200, y: 10, w: 120, h: 40, label: 'summarize(doc 1)' }, { id: 's2', x: 200, y: 92, w: 120, h: 46, label: 'summarize(doc 2)' }, { id: 's3', x: 200, y: 180, w: 120, h: 40, label: 'summarize(doc N)' },
                        { id: 'red', x: 400, y: 92, w: 110, h: 46, label: 'reduce', sub: 'operator.add' }],
                edges: [{ id: 'a', d: 'M110,105 L198,30', cls: 'cond' }, { id: 'b', d: 'M110,115 L198,115', cls: 'cond', label: 'Send', lx: 155, ly: 108 }, { id: 'c', d: 'M110,125 L198,200', cls: 'cond' },
                        { id: 'd', d: 'M320,30 L398,105' }, { id: 'e', d: 'M320,115 L398,115' }, { id: 'f', d: 'M320,200 L398,125' }] }
            };
            var info = {
              supervisor: ['Supervisor', 'One LLM-based router node chooses the next worker, workers report back, the supervisor decides when to finish. Easy to reason about, but every step passes through the router (extra calls).'],
              handoff: ['Handoffs', 'Peers pass control directly using <code>Command(goto=...)</code>. Good for triage/specialist flows. Less central control, so loops are easier to create.'],
              send: ['Send fan-out', 'A conditional edge returns one <code>Send("node", payload)</code> per item. Branches run in parallel in the same super-step; a reducer on the shared channel combines results.']
            };
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            function show(v) {
              out.innerHTML = '';
              box(out, '<h4>' + info[v][0] + '</h4><p>' + info[v][1] + '</p>');
              box(out, svgGraph(DIAG[v], 'lgm' + v));
            }
            api.select(ctl, { label: 'Pattern', options: [{ value: 'supervisor', label: 'Supervisor' }, { value: 'handoff', label: 'Handoffs' }, { value: 'send', label: 'Send fan-out' }], value: 'supervisor', onChange: show, init: true });
          } },
          { name: 'Fan-out demo', render: function (el, api) {
            var n = 3, ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            function draw() {
              out.innerHTML = '';
              var rows = [];
              for (var i = 1; i <= n; i++) rows.push(['Send("summarize", {"doc": ' + i + '})', 'summary_' + i, '1 LLM call']);
              api.table(out, { headers: ['Branch ' + api.type('json'), 'Returns', 'Cost'], rows: rows });
              box(out, '<div class="langgraph-route"><code>summaries</code> ' + api.badge('json', 'list[str]') + ' after reduce: [' + rows.map(function (r) { return r[1]; }).join(', ') + ']<br>Wall time ≈ 1 branch (parallel); total LLM calls = N + 1 = <b>' + (n + 1) + '</b> (N summaries + 1 combine).</div>');
            }
            api.slider(ctl, { label: 'Items N', min: 1, max: 8, step: 1, value: 3, format: function (v) { return String(v); }, onInput: function (v) { n = v; draw(); } });
          } },
          { name: 'Code', render: function (el, api) {
            codeBlock(el, api, `from langgraph.types import Command, Send
import operator

# Handoff: node decides where to go
def triage(state) -> Command[Literal["billing", "tech"]]:
    target = classify(state["messages"])
    return Command(goto=target, update={"messages": [AIMessage(f"Handing to {target}")]})

# Supervisor: router returns next worker name
def supervisor(state) -> Command[Literal["researcher", "coder", "__end__"]]:
    nxt = llm_pick_worker(state)                 # structured output
    return Command(goto=nxt)

# Send fan-out (map-reduce)
class S(TypedDict):
    docs: list[str]
    summaries: Annotated[list[str], operator.add]   # reducer joins parallel writes

def fan_out(state: S):
    return [Send("summarize", {"doc": d}) for d in state["docs"]]

builder.add_conditional_edges(START, fan_out, ["summarize"])
builder.add_edge("summarize", "reduce")`);
            api.docsNote(el, 'Prebuilt supervisor/swarm helpers exist in companion packages and their names change.');
          } }
        ]
      },

      // ================================================================= 6. when to use
      {
        id: 'when-to-use',
        title: 'LangChain vs LangGraph vs plain code',
        critical: {
          understand: [
            'A plain <code>while</code> loop around the chat API is a complete agent. Frameworks add structure, not magic.',
            'Reach for <b>LangGraph</b> when you need durable state, human-in-the-loop pauses, branching/parallelism or resumable long runs.',
            '<b>LangChain</b> components (models, tools, retrievers, LCEL) are building blocks usable with either.'
          ],
          details: [
            'Costs of frameworks: another abstraction to debug, API churn, and hidden prompts. Costs of plain code: you build persistence and approval flows yourself.',
            'See ' + '<span class="xlink" data-goto="langchain/agents">LangChain agents</span> and ' + '<span class="xlink" data-goto="llm-to-agent/agent-loop">the raw agent loop</span>.'
          ]
        },
        dataTypes: [
          { name: 'requirement flags', type: 'boolean', shape: '5 flags', example: 'needs pause?' },
          { name: 'recommendation', type: 'text', shape: 'string', example: '"LangGraph"' },
          { name: 'agent loop', type: 'function', shape: 'messages → messages', example: 'while True: …' }
        ],
        levels: [
          { name: 'Comparison', render: function (el, api) {
            api.table(el, { headers: ['', 'Plain code', 'LangChain (LCEL / agents)', 'LangGraph'],
              rows: [
                ['Core idea', 'while loop + API client', 'Composable components, pipelines', 'Stateful graph: nodes, edges, state'],
                ['Control flow', 'Any Python', 'Mostly linear; prebuilt agent', 'Explicit cycles, branches, parallel'],
                ['Persistence / resume', 'DIY', 'Limited', 'Built-in checkpointers'],
                ['Human approval', 'DIY (input())', 'Callbacks', '<code>interrupt()</code> + resume'],
                ['Learning curve', 'Lowest', 'Medium', 'Highest'],
                ['Best for', 'Simple tool loops, prototypes', 'RAG pipelines, structured output', 'Long-running, multi-step, supervised agents']
              ] });
            api.docsNote(el);
          } },
          { name: 'Decide', render: function (el, api) {
            var flags = { pause: false, resume: false, parallel: false, multi: false, simple: true };
            var ctl = api.h('div', { class: 'controls' }); el.appendChild(ctl);
            var out = api.h('div'); el.appendChild(out);
            function draw() {
              var score = (flags.pause ? 1 : 0) + (flags.resume ? 1 : 0) + (flags.parallel ? 1 : 0) + (flags.multi ? 1 : 0);
              var r, why;
              if (score === 0) { r = 'Plain while-loop (or a prebuilt agent)'; why = 'No state-heavy requirement: a loop with a step limit is enough.'; }
              else if (score <= 1) { r = 'Plain code first; LangGraph if the need grows'; why = 'One requirement is manageable by hand; revisit if it spreads.'; }
              else { r = 'LangGraph'; why = score + ' of 4 requirements (pause, resume, parallelism, multi-agent) are what checkpointers and graphs provide.'; }
              out.innerHTML = '<div class="langgraph-verdict">' + api.type('boolean', 'flags: ' + score + '/4') + ' → <b>' + r + '</b><br>' + why + '</div>';
            }
            [['pause', 'Needs human approval / pauses'], ['resume', 'Must survive restarts, resume later'], ['parallel', 'Parallel branches / fan-out'], ['multi', 'Several cooperating agents']].forEach(function (f) {
              api.toggle(ctl, { label: f[1], value: false, onChange: function (v) { flags[f[0]] = v; draw(); } });
            });
            draw();
            api.illustrative(el, 'Rule of thumb, not a law.');
          } },
          { name: 'Plain loop', render: function (el, api) {
            codeBlock(el, api, `# The whole agent, no framework (provider-neutral pseudo-code)
messages = [{"role": "user", "content": request}]
for step in range(MAX_STEPS):                 # budget = recursion_limit
    reply = chat(model, messages, tools=TOOLS)
    messages.append(reply)
    if not reply.tool_calls:                  # routing boolean
        break
    for call in reply.tool_calls:
        result = run_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})`);
            api.callout(el, 'Compare with the graph: <code>messages</code> is the state, the <code>if</code> is the conditional edge, <code>chat</code> and <code>run_tool</code> are the nodes. LangGraph adds checkpoints, interrupts and parallelism on top of exactly this loop.', 'key');
          } }
        ]
      }
    ]
  });
})();
