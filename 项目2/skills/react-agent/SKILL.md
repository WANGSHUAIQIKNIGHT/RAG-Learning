---
name: react-agent
description: "Use this skill when building, debugging, or explaining ReAct (Reason+Act) agents. Covers the full spectrum from hand-written ReAct loops to LangGraph multi-agent collaboration with Supervisor quality control. Includes tool definition standards, defensive tool calling patterns, and reusable agent factory patterns. Trigger when the user mentions 'agent', 'ReAct', 'tool call', 'multi-agent', 'LangGraph', 'Agent workflow', or asks to build an AI agent that uses tools."
---

# ReAct Agent Skill

## Quick Reference

| Level | Pattern | When to Use |
|-------|---------|-------------|
| **L0** | Hand-written ReAct loop | Learn the core idea, interview prep |
| **L1** | LLM + tools + while loop | Simple single-agent tasks |
| **L2** | Reusable `build_react_agent()` factory | Multiple agents sharing same logic |
| **L3** | LangGraph StateGraph | Complex multi-step workflows |
| **L4** | Multi-agent + Supervisor | Production-grade quality control |

---

## Core Concept

ReAct = **Rea**son + **Act**

The loop at a glance:

```
LLM: Thought → Action (tool call)
           ↓
     Execute tool → Observation
           ↓
LLM: Thought → Action (or Final Answer)
```

**Key insight**: The LLM decides *what* to do at every step. You (the code) only execute tools and feed results back. No magic, just a `while` loop + message list management.

---

## L0: Hand-written ReAct (Pure Python, No Framework)

**Purpose**: Teach the core idea. No LLM involved, just rule-based simulation.

```python
def react_agent(question: str, max_steps: int = 5) -> str:
    history = []
    for step in range(max_steps):
        # Step A: Thought → Action
        response = llm_reason(question, history)

        if response.startswith("Final Answer:"):
            return response.replace("Final Answer:", "").strip()

        # Step B: Parse Action
        action = re.search(r"Action:\s*(\w+)", response).group(1)
        action_input = re.search(r"Action Input:\s*(.+)", response).group(1)

        # Step C: Execute → Observation
        observation = TOOLS[action](action_input)
        history.append(f"Observation: {observation}")
```

**What this teaches**:
- The loop structure never changes, no matter how sophisticated the framework
- Tool parsing, execution, observation feeding — that's the entire pattern

---

## L1: Real LLM + Tool Calls

**Purpose**: Replace simulated reasoning with a real LLM (`ChatTongyi` here).

```python
def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [
        SystemMessage(content="你是一个有用的助手。需要工具时调用工具，不需要就直接回答。"),
        HumanMessage(content=question),
    ]

    for step in range(max_steps):
        response = llm.invoke(messages, tools=TOOLS)

        # LLM thinks no more tools needed → done
        if not response.tool_calls:
            return response.content

        # Execute each tool call
        messages.append(response)
        for tc in response.tool_calls:
            result = TOOL_MAP[tc["name"]](**tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
```

**Key differences from L0**:
- LLM outputs structured `tool_calls` natively → no regex parsing needed
- `ToolMessage` pairs tool results to the call that triggered them

---

## L2: Reusable Agent Factory

**Purpose**: Extract the loop into a factory so you can spin up agents with different tools.

```python
def build_react_agent(llm, tools_config: list[dict], tool_map: dict, max_steps: int = 8):
    """Returns a callable agent function."""
    def run(question: str, system_prompt: str = "你是一个有用的助手。") -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        for step in range(max_steps):
            response = llm.invoke(messages, tools=tools_config)
            if not response.tool_calls:
                return response.content
            messages.append(response)
            for tc in response.tool_calls:
                result = tool_map[tc["name"]](**tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return "已达到最大步数，未完成任务。"
    return run
```

**Usage**:
```python
agent = build_react_agent(llm, tools_config, tool_map)
result = agent("北京今天多少度？")
```

---

## L3: LangGraph StateGraph

**Purpose**: When the workflow has multiple stages (not just one loop).

```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    research_result: str
    final_output: str

workflow = StateGraph(AgentState)
workflow.add_node("research", research_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("research")
workflow.add_conditional_edges("research", router, {...})
graph = workflow.compile()
```

---

## L4: Multi-Agent + Supervisor (Production)

**Purpose**: Add a quality gate between agents.

```
research → supervisor → writer → supervisor → END
                          ↑            |
                          └── FAIL ─────┘  (max 3 retries)
```

### Agent Roles

| Agent | Responsibility |
|-------|---------------|
| **Research Agent** | Collects information using tools (search, read, RAG) |
| **Supervisor Agent** | Checks quality, returns PASS or FAIL with feedback |
| **Writer Agent** | Produces final output based on research |

### Supervisor Logic

```python
def supervisor_agent(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0)

    if is_pass:
        return {"next_agent": "writer" if reviewing_research else "finish"}
    else:
        if retry_count >= max_retries:
            return {"next_agent": "writer" if reviewing_research else "finish"}  # force pass
        return {"next_agent": "research" if reviewing_research else "writer",
                "retry_count": retry_count + 1}
```

---

## Tool Definition Standard

Use OpenAI-compatible JSON schema (works with LangChain, direct API, most LLMs):

```python
TOOL_SCHEMA = {
    "name": "tool_name",
    "description": "清晰描述工具功能，LLM 据此判断何时调用",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数说明，包括可选值和格式要求",
            },
        },
        "required": ["param1"],
    },
}
```

**Best practices**:
- Description 要写清楚**什么情况下调用**，LLM 靠这个做决策
- required 只放真正的必填项
- 参数类型准确（string / integer / array）

### Tool Implementation + Mapping

```python
def get_weather(city: str) -> str:  # signature matches schema params
    return data.get(city, "默认返回")

TOOL_MAP = {
    "get_weather": get_weather,      # name → function
    "calculator": calculator,
}
```

---

## Defensive Tool Calling Patterns

### 1. Parameter Name Fuzzy Matching

When LLMs miss parameter names:

```python
import difflib

def correct_param_name(param: str, valid_params: list[str]) -> str:
    matches = difflib.get_close_matches(param, valid_params, n=1, cutoff=0.6)
    return matches[0] if matches else param
```

### 2. Filter Extraneous Parameters

LLMs sometimes pass extra params the tool doesn't need:

```python
import inspect

def safe_call_tool(func, args: dict) -> str:
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    return func(**valid_args)
```

### 3. Loop Safety

```python
MAX_STEPS = 8  # prevent infinite loops
```

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| No max_steps | Infinite loop | Always set `max_steps=8` |
| Forgot to append LLM response | Tool result has no context | `messages.append(response)` before ToolMessage |
| Wrong tool_call_id | ToolMessage ignored | Extract `tc["id"]` from the tool call |
| Tool description too vague | LLM never calls it | Write "调用此工具当..." not just "搜索工具" |
| Reusing same `tools=TOOLS` | Stale tool list | Pass fresh on each `llm.invoke()` |
| No Supervisor | Silent quality issues | Always gate critical outputs |

---

## Agent Prompt Design Patterns

### Research Agent

```
你是 Research Agent（研究助理）。
职责：收集信息、搜索资料、读取文件，汇总研究发现。

规则：
- 优先使用知识库搜索工具
- 信息不足时用 web_search
- 不要编造，基于检索结果
- 输出：中文条理清晰地列出发现
```

### Writer Agent

```
你是 Writer Agent（写作助理）。
职责：根据 Research 结果撰写结构化文档并保存。

规则：
- 用 write_file 保存最终文档
- 文档包含标题、摘要、正文、结论
- 引用来源时标注编号
```

### Supervisor Agent

```
你是 Supervisor Agent（质量监管）。
检查 Research / Writer 的工作成果。

检查标准：
1. 内容是否与问题相关
2. 信息是否有依据
3. 结构是否完整

输出格式：
通过：PASS
意见：<具体反馈>
```

---

## Dependencies

| Level | Required |
|-------|----------|
| L0 (hand-written) | `re`, `json` (stdlib) |
| L1 (real LLM) | `langchain-community`, `langchain-core`, LLM SDK |
| L2 (factory) | Same as L1 |
| L3 (LangGraph) | `langgraph`, `langchain-core`, LLM SDK |
| L4 (multi-agent) | Same as L3 |

```bash
pip install langchain-community langchain-core langgraph
```

---

## QA Checklist

- [ ] Agent stops within `max_steps` (no infinite loop)
- [ ] Tool results are correctly threaded via `ToolMessage`
- [ ] LLM response is appended before tool call results
- [ ] Tool descriptions accurately trigger correct tool selection
- [ ] Supervisor actually catches errors (test with intentionally bad output)
- [ ] Retry mechanism terminates (doesn't loop forever)
- [ ] All tool call IDs match correctly
