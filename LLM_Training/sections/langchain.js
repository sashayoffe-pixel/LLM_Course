(function () {
  'use strict';

  function stage(api, parent, title, type, shape) {
    var d = api.h('div', { class: 'langchain-stage' });
    d.appendChild(api.h('div', { html: '<b>' + title + '</b><br>' + api.badge(type, shape) }));
    var data = api.h('div', { class: 'langchain-data' }); d.appendChild(data);
    parent.appendChild(d);
    return { box: d, data: data };
  }
  function cw(api, el) { var d = api.h('div', { class: 'langchain-split' }); el.appendChild(d); var i = api.h('div'); d.appendChild(i); return i; }
  function pipeRow(api, el, defs) {
    var row = api.h('div', { class: 'langchain-pipe' }); el.appendChild(row);
    var out = [];
    defs.forEach(function (d, i) {
      if (i) row.appendChild(api.h('span', { class: 'langchain-pipe-sym', text: i % 2 ? '→' : '→' }));
      out.push(stage(api, row, d[0], d[1], d[2]));
    });
    return out;
  }

  Course.register({
    id: 'langchain',
    part: 'II',
    order: 125,
    title: 'LangChain',
    slides: [
      // ------------------------------------------------------------ where-frameworks-fit
      {
        id: 'where-frameworks-fit',
        title: 'Where frameworks fit: your code ↔ framework ↔ model API',
        critical: {
          understand: [
            'A framework like LangChain sits <b>between your code and the chat API</b>: it standardises prompts, model calls, parsing, tools and retrieval.',
            'It does not change what the model does — underneath it is still <b>messages in, tokens out</b> (see ' + '<span data-goto="chat-api/messages-api" class="xlink">the messages API</span>).',
            'The value is interchangeable parts (swap providers, vector stores) and ready-made patterns; the cost is another abstraction to learn and debug.'
          ],
          details: [
            'LangChain ≥ 1.0 splits into packages: <code>langchain_core</code> (interfaces), provider packages (e.g. <code>langchain_openai</code>, <code>langchain_anthropic</code>), <code>langchain_community</code>, and <code>langgraph</code> for stateful agents.',
            'You can always drop down to the raw API for one step and keep the framework for the rest.'
          ]
        },
        dataTypes: [
          { name: 'your input', type: 'text', shape: 'string | dict', example: '{"question": "…"}' },
          { name: 'provider request', type: 'json', shape: 'object', example: '{model, messages, …}' },
          { name: 'provider response', type: 'json', shape: 'object', example: '{content, usage, …}' },
          { name: 'chain', type: 'function', shape: 'Runnable', example: 'prompt | model | parser' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'you', label: 'Your code', sub: 'app logic, UI, tests', type: 'function' },
              { id: 'fw', label: 'Framework', sub: 'prompts, parsers, tools, retrievers', type: 'function', edge: 'Runnable.invoke(x)', zoom: true },
              { id: 'api', label: 'Model API', sub: 'messages → response', type: 'json', edge: 'provider JSON', goto: 'chat-api/messages-api' },
              { id: 'llm', label: 'LLM', sub: 'one token sequence', type: 'token', edge: 'chat template', goto: 'chat-api/chat-template' }
            ], caption: 'Every layer just transforms data for the next one.' });
          } },
          { name: 'Layers', render: function (el, api) {
            var st = api.h('div', { class: 'langchain-stack' }); el.appendChild(st);
            [
              ['you', '<b>Your code</b> — decides <i>what</i> to ask, validates results, handles users.'],
              ['fw', '<b>Framework</b> — prompt templates, output parsers, tool binding, retrievers, memory/graph runtime, tracing hooks.'],
              ['api', '<b>Provider SDK / HTTP API</b> — auth, retries, JSON ' + api.badge('json', 'object') + '.'],
              ['llm', '<b>Model</b> — templated token sequence in, next tokens out.']
            ].forEach(function (l) { st.appendChild(api.h('div', { class: 'langchain-layer ' + l[0], html: l[1] })); });
            api.callout(el, '<b>Rule of thumb.</b> Frameworks pay off when you swap components often or want standard patterns (RAG, tool agents). For a single prompt → single call, plain SDK code is often simpler. Judge by your own project.', 'hedge');
          } },
          { name: 'Code compare', render: function (el, api) {
            var g = api.h('div', { class: 'langchain-split' }); el.appendChild(g);
            var a = api.h('div'), b = api.h('div'); g.appendChild(a); g.appendChild(b);
            a.appendChild(api.h('div', { class: 'muted', text: 'Direct (provider-neutral pseudo-code)' }));
            api.code(a, 'resp = client.chat(\n    model="some-model",\n    system="Answer briefly.",\n    messages=[{"role": "user",\n               "content": question}],\n    max_tokens=200,\n)\nanswer = resp["content"][0]["text"]', 'python');
            b.appendChild(api.h('div', { class: 'muted', text: 'LangChain (≥ 1.0 style)' }));
            api.code(b, 'from langchain_core.prompts import ChatPromptTemplate\nfrom langchain_core.output_parsers import StrOutputParser\nfrom langchain.chat_models import init_chat_model\n\nllm = init_chat_model("provider:some-model")\nprompt = ChatPromptTemplate.from_messages([\n    ("system", "Answer briefly."),\n    ("human", "{question}"),\n])\nchain = prompt | llm | StrOutputParser()\nanswer = chain.invoke({"question": question})', 'python');
            api.docsNote(el, 'Model-initialisation helpers and package layout have changed across LangChain versions; verify imports against the docs for your installed version.');
          } }
        ]
      },

      // ------------------------------------------------------------ lcel
      {
        id: 'lcel',
        title: 'Components & LCEL: prompt | model | parser',
        critical: {
          understand: [
            'Every component is a <b>Runnable</b> with the same interface: <code>invoke</code> (one input), <code>stream</code> (chunks as they arrive), <code>batch</code> (many inputs).',
            'The <code>|</code> operator (LCEL) pipes the output of one Runnable into the next, so types must line up: <code>dict → PromptValue → AIMessage → str</code>.',
            'Piping composes a new Runnable, so chains can themselves be piped into bigger chains.'
          ],
          details: [
            'The prompt template turns a dict into messages; the model turns messages into an <code>AIMessage</code> (with content, tool calls, usage metadata); the parser extracts what you want.',
            'Streaming works through the chain if each stage supports it.'
          ]
        },
        dataTypes: [
          { name: 'input', type: 'json', shape: 'dict', example: '{"topic": "cats"}' },
          { name: 'PromptValue / messages', type: 'json', shape: 'list[Message]', example: '[System, Human]' },
          { name: 'AIMessage', type: 'json', shape: 'object', example: '{content, tool_calls, usage_metadata}' },
          { name: 'parsed output', type: 'text', shape: 'string', example: '"Cats sleep a lot."' },
          { name: 'chain', type: 'function', shape: 'Runnable', example: 'prompt | llm | parser' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 'in', label: 'input', type: 'json', shape: 'dict' },
              { id: 'p', label: 'prompt', sub: 'ChatPromptTemplate', type: 'function', edge: '|', zoom: true },
              { id: 'm', label: 'model', sub: 'chat model', type: 'function', edge: '| PromptValue', goto: 'chat-api/chat-template' },
              { id: 'o', label: 'parser', sub: 'StrOutputParser', type: 'function', edge: '| AIMessage' },
              { id: 'out', label: 'output', type: 'text', shape: 'string', edge: 'str' }
            ], caption: 'chain = prompt | model | parser — one Runnable made of three.' });
          } },
          { name: 'Data flow', render: function (el, api) {
            var st = pipeRow(api, el, [
              ['prompt', 'json', 'dict → PromptValue'], ['model', 'json', 'messages → AIMessage'], ['parser', 'text', 'AIMessage → str']
            ]);
            var inp = { topic: 'cats' };
            var datas = [
              'in:  ' + JSON.stringify(inp) + '\nout: [\n  SystemMessage("Answer in one sentence."),\n  HumanMessage("Tell me about cats")\n]',
              'in:  [SystemMessage, HumanMessage]\n(chat template → tokens → generate)\nout: AIMessage(\n  content="Cats sleep 12–16 h a day.",\n  usage_metadata={input_tokens: 24,\n                  output_tokens: 11})',
              'in:  AIMessage(...)\nout: "Cats sleep 12–16 h a day."'
            ];
            api.stepper(el, { steps: [{ title: 'invoke({"topic": "cats"})' }, { title: 'prompt runs' }, { title: 'model runs' }, { title: 'parser runs' }], interval: 2200, onStep: function (i) {
              st.forEach(function (s, k) { s.box.classList.toggle('on', k === i - 1); s.data.textContent = k < i ? datas[k] : ''; });
            } });
            api.illustrative(el, 'The model output and token counts are made up; the type flow is the real point.');
          } },
          { name: 'Runnable API', render: function (el, api) {
            var words = 'Cats sleep twelve to sixteen hours a day.'.split(' ');
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var out = api.h('div', { class: 'langchain-stream' });
            var timerOff = null;
            function stop() { if (timerOff) { timerOff(); timerOff = null; } }
            api.button(c, 'invoke()', function () { stop(); out.textContent = '"' + words.join(' ') + '"   ← one complete result'; }, 'btn');
            api.button(c, 'stream()', function () {
              stop(); out.textContent = ''; var i = 0;
              var h = api.every(250, function () { if (i >= words.length) { stop(); return; } out.textContent += (i ? ' ' : '') + words[i++]; });
              timerOff = function () { clearInterval(h); };
            }, 'btn primary');
            api.button(c, 'batch([3 inputs])', function () { stop(); out.textContent = '["Cats sleep…", "Dogs bark…", "Owls hunt…"]   ← list, same order, run concurrently'; }, 'btn');
            el.appendChild(out);
            api.code(cw(api, el), 'chain.invoke({"topic": "cats"})                       # -> str\nfor chunk in chain.stream({"topic": "cats"}):         # -> chunks\n    print(chunk, end="")\nchain.batch([{"topic": "cats"}, {"topic": "dogs"}])  # -> list[str]\n# async twins: ainvoke / astream / abatch', 'python');
            api.docsNote(el, 'Method names have been stable, but check the current docs for config options (callbacks, concurrency limits).');
          } }
        ]
      },

      // ------------------------------------------------------------ tools-structured
      {
        id: 'tools-structured',
        title: 'bind_tools & with_structured_output',
        critical: {
          understand: [
            '<code>bind_tools([...])</code> converts your functions/schemas into the JSON tool definitions the API expects and attaches them to every call.',
            'The model answers with a <b>tool call</b> (name + JSON arguments) inside the <code>AIMessage</code>; <b>you or an agent runtime</b> executes it.',
            '<code>with_structured_output(Schema)</code> uses the same machinery to force JSON matching a schema, then parses it into an object.'
          ],
          details: [
            'Schemas are usually Pydantic classes, TypedDicts or JSON Schema; validation errors are possible — handle them.',
            'Under the hood this is the tool-calling flow from ' + '<span data-goto="tool-calling/round-trip" class="xlink">round trip</span>, and the schema is serialised into the prompt tokens.'
          ]
        },
        dataTypes: [
          { name: 'schema', type: 'json', shape: 'JSON Schema', example: '{name, parameters}' },
          { name: 'tool call', type: 'json', shape: '{name, args, id}', example: '{"name":"get_weather",…}' },
          { name: 'parsed object', type: 'json', shape: 'Pydantic / dict', example: 'Weather(city="Paris")' },
          { name: 'tool function', type: 'function', shape: 'callable', example: 'get_weather(city)' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { blocks: [
              { id: 's', label: 'Python schema', sub: 'function / Pydantic', type: 'function' },
              { id: 'b', label: 'bind_tools', sub: 'schema → JSON', type: 'json', edge: 'convert', zoom: true },
              { id: 'm', label: 'model', sub: 'emits tokens', type: 'function', edge: 'tools in request' },
              { id: 'c', label: 'tool call JSON', type: 'json', shape: '{name, args}', edge: 'AIMessage.tool_calls' },
              { id: 'p', label: 'parsed / executed', type: 'function', goto: 'tool-calling/round-trip' }
            ], caption: 'The model only ever writes the JSON; code outside the model runs the function.' });
          } },
          { name: 'Step by step', render: function (el, api) {
            var city = { value: 'Paris' };
            var c = api.h('div', { class: 'controls' }); el.appendChild(c);
            var body = api.h('div'); el.appendChild(body);
            var schema = { name: 'get_weather', description: 'Current weather for a city', parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } };
            function steps() {
              var call = { name: 'get_weather', args: { city: city.value }, id: 'call_1' };
              return [
                { title: '1. Schema', html: 'Your function becomes a ' + api.badge('json', 'JSON Schema') + ':<pre class="muted" style="white-space:pre-wrap">' + api.esc(JSON.stringify(schema, null, 2)) + '</pre>' },
                { title: '2. Model emits call', html: 'The model generates tokens that form a ' + api.badge('json', '{name, args}') + ':<pre class="muted" style="white-space:pre-wrap">' + api.esc(JSON.stringify(call, null, 2)) + '</pre>' },
                { title: '3. Parsed / executed', html: 'LangChain exposes it as <code>ai_message.tool_calls[0]</code>. Your code (or an agent) calls <code>get_weather("' + api.esc(city.value) + '")</code>. With <code>with_structured_output</code>, the args are validated into <code>Weather(city="' + api.esc(city.value) + '")</code> instead.' }
              ];
            }
            var sp = null;
            function build() { body.innerHTML = ''; sp = api.stepper(body, { steps: steps(), interval: 2500 }); }
            api.input(c, { label: 'city', value: 'Paris', onInput: function (v) { city.value = v || 'Paris'; build(); }, init: false });
            build();
            api.docsNote(el, 'The JSON shape of tool definitions/calls differs by provider; LangChain normalises it to <code>tool_calls=[{"name","args","id"}]</code> — verify in current docs.');
          } },
          { name: 'Code', render: function (el, api) {
            api.code(cw(api, el), 'from pydantic import BaseModel, Field\nfrom langchain_core.tools import tool\n\n@tool\ndef get_weather(city: str) -> str:\n    """Current weather for a city."""\n    return f"Sunny in {city}"\n\nllm_tools = llm.bind_tools([get_weather])\nmsg = llm_tools.invoke("Do I need an umbrella in Paris?")\nprint(msg.tool_calls)\n# [{"name": "get_weather", "args": {"city": "Paris"}, "id": "..."}]\n\nclass Trip(BaseModel):\n    city: str = Field(description="Destination")\n    nights: int\n\nstructured = llm.with_structured_output(Trip)\ntrip = structured.invoke("Two nights in Rome")   # Trip(city="Rome", nights=2)', 'python');
            api.docsNote(el, 'Import paths such as <code>langchain_core.tools.tool</code> reflect LangChain ≥ 1.0; provider support for structured output varies.');
          } }
        ]
      },

      // ------------------------------------------------------------ agents
      {
        id: 'agents',
        title: 'Agents: built on LangGraph — framework vs direct API',
        critical: {
          understand: [
            'A LangChain <b>agent</b> is the agent loop from ' + '<span data-goto="llm-to-agent/agent-loop" class="xlink">the agent loop</span> packaged up: model call → tool calls → run tools → append results → repeat until no more tool calls.',
            'In LangChain ≥ 1.0 agents are built on <b>LangGraph</b> (a graph runtime with state and checkpoints); <code>create_agent</code> is the high-level entry point.',
            'Trade-off: less boilerplate and built-in persistence/streaming vs. less visibility and another dependency.'
          ],
          details: [
            'For custom control flow (branches, approvals, multi-agent), use LangGraph directly: ' + '<span data-goto="langgraph" class="xlink">LangGraph section</span>.',
            'Whatever the framework, the loop is still repeated next-token prediction over one growing token sequence.'
          ]
        },
        dataTypes: [
          { name: 'agent', type: 'function', shape: 'compiled graph', example: 'create_agent(model, tools)' },
          { name: 'state messages', type: 'json', shape: 'list[Message]', example: 'grows each loop' },
          { name: 'tool result', type: 'text', shape: 'string', example: '"Sunny, 24°C"' },
          { name: 'tool call', type: 'json', shape: '{name, args, id}', example: 'get_weather' }
        ],
        levels: [
          { name: 'Overview', render: function (el, api) {
            api.blocks(el, { rows: [[
              { id: 'm', label: 'model node', sub: 'LLM call', type: 'function', edge: 'AIMessage', goto: 'llm-to-agent/agent-loop' },
              { id: 'd', label: 'tool calls?', type: 'boolean', shape: 'scalar', edge: 'yes' },
              { id: 't', label: 'tools node', sub: 'run functions', type: 'function', edge: 'ToolMessages' },
              { id: 'back', label: 'append to state', type: 'json', shape: 'list[Message]', edge: 'loop back to model' }
            ]], caption: 'No tool calls → the loop ends and the last AIMessage is the answer.' });
            api.callout(el, 'Under the hood this is a LangGraph with a model node and a tools node: ' + api.link('langgraph/react-graph', 'ReAct graph'), 'key');
          } },
          { name: 'Simulate', render: function (el, api) {
            var trace = [
              ['Human', 'Do I need an umbrella in Paris tomorrow?'],
              ['AIMessage', 'tool_calls: get_weather(city="Paris")'],
              ['ToolMessage', '"Rain likely, 14°C"'],
              ['AIMessage', 'Yes, bring an umbrella — rain is likely.']
            ];
            var box = api.h('div'); el.appendChild(box);
            api.stepper(el, { steps: trace.map(function (t, i) { return { title: i === 0 ? 'invoke' : i === 1 ? 'model' : i === 2 ? 'tools' : 'model (done)' }; }), interval: 1800, onStep: function (i) {
              box.innerHTML = '';
              trace.slice(0, i + 1).forEach(function (t, k) {
                box.appendChild(api.h('div', { class: 'langchain-layer ' + (k === i ? 'fw' : ''), html: '<b>' + t[0] + '</b> &nbsp;' + api.esc(t[1]) }));
              });
              box.appendChild(api.h('div', { class: 'muted', html: 'state.messages length: <b>' + (i + 1) + '</b>' }));
            } });
            api.illustrative(el, 'Scripted trace for illustration; a real agent would call a real tool and a real model.');
          } },
          { name: 'Code & trade-offs', render: function (el, api) {
            api.code(cw(api, el), 'from langchain.agents import create_agent\nfrom langchain_core.tools import tool\n\n@tool\ndef get_weather(city: str) -> str:\n    """Current weather for a city."""\n    return "Rain likely, 14°C"\n\nagent = create_agent(model="provider:some-model", tools=[get_weather],\n                     system_prompt="Be concise.")\nresult = agent.invoke({"messages": [{"role": "user",\n                       "content": "Umbrella in Paris tomorrow?"}]})\nprint(result["messages"][-1].content)', 'python');
            api.docsNote(el, '<code>create_agent</code> and its arguments are LangChain 1.x; earlier versions used other helpers. Check the docs for your version.');
            api.table(cw(api, el), { headers: ['', 'Framework agent', 'Direct API loop'], rows: [
              ['Boilerplate', 'little', 'you write the loop (~30 lines)'],
              ['Provider swap', 'easy-ish', 'rewrite adapters'],
              ['Persistence / streaming', 'built in (LangGraph)', 'build yourself'],
              ['Debuggability', 'more layers; tracing helps', 'full visibility'],
              ['Custom control flow', 'use LangGraph directly', 'plain code']
            ] });
          } }
        ]
      }
    ]
  });
})();
