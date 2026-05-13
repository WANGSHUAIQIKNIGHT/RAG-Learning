# -*- coding: utf-8 -*-
"""
第5步：多 Agent 协作 — LangGraph 状态图，多个 Agent 分工协作
核心：用 StateGraph 构建多步工作流，Research Agent + Writer Agent 协同完成任务
"""

import operator
import sys
from typing import TypedDict, Annotated, Sequence, Literal

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END

# 解决 Windows 终端 emoji 打印问题
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)

llm = ChatTongyi(model="qwen-plus", temperature=0.0)


# ============================================================
# 1. 工具定义（复用 A3/A4 的部分工具 + 新增 research 工具）
# ============================================================

TOOLS = [
    {
        "name": "web_search",
        "description": "搜索指定主题的最新信息，返回摘要结果",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "读取本地文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "写入内容到本地文件",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "summarize_text",
        "description": "对一段长文本进行摘要浓缩",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "需要摘要的原文"},
                "max_words": {"type": "integer", "description": "摘要最大字数"},
            },
            "required": ["text", "max_words"],
        },
    },
]

# ===== 工具实现 =====

def web_search(query: str) -> str:
    """模拟搜索 — 实际项目中替换为真实搜索引擎 API"""
    db = {
        "langgraph": (
            "LangGraph 是 LangChain 推出的多 Agent 编排框架，基于 StateGraph 构建有状态的工作流。"
            "支持条件路由、人机交互、持久化 checkpoint。v1.0 后 API 更稳定，广泛用于生产级 Agent 系统。"
        ),
        "rag 技术": (
            "RAG (Retrieval-Augmented Generation) 通过检索外部知识库增强 LLM 生成质量。"
            "核心流程：文档解析→分块→Embedding→向量检索→LLM 生成。2024 年后 RAG 与 Agent 融合趋势明显。"
        ),
        "agent 框架": (
            "主流 Agent 框架包括 LangChain Agent、LangGraph、AutoGen、CrewAI。"
            "LangGraph 在复杂工作流场景表现突出，支持图状结构和条件路由。"
        ),
    }
    # 模糊匹配
    for key, val in db.items():
        if key in query.lower() or query.lower() in key:
            return val
    return f"未找到「{query}」的相关信息。"

def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败：{e}"

def write_file(file_path: str, content: str) -> str:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {len(content)} 字符到 {file_path}"
    except Exception as e:
        return f"写入失败：{e}"

def summarize_text(text: str, max_words: int = 100) -> str:
    """用 LLM 自己来摘要（直接调 LLM 更准确）"""
    resp = llm.invoke([
        SystemMessage(content=f"请将以下文本摘要为 {max_words} 字以内，保留核心信息："),
        HumanMessage(content=text),
    ])
    return resp.content[:500]

TOOL_MAP = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "summarize_text": summarize_text,
}


# ============================================================
# 2. Agent 定义（Research Agent / Writer Agent）
# ============================================================

RESEARCH_SYSTEM_PROMPT = """你是 Research Agent（研究助理）。
你的职责是收集信息、搜索资料、读取文件，然后把研究发现汇总交给下一个 Agent。

规则：
- 需要搜索时用 web_search 工具
- 需要读取文件时用 read_file
- 总结收集到的信息，不要编造
- 输出格式：用中文条理清晰地列出发现"""

WRITER_SYSTEM_PROMPT = """你是 Writer Agent（写作助理）。
你的职责是根据 Research Agent 提供的研究结果，撰写最终文档并保存。

规则：
- 需要长文本时先用 summarize_text 摘要
- 用 write_file 保存最终文档
- 文档结构清晰，包含标题和分点
- 输出格式：说明你写了什么、保存在哪里"""


# ============================================================
# 3. LangGraph 状态图
# ============================================================

class AgentState(TypedDict):
    """整个工作流的状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str          # 当前应该哪个 Agent 工作
    research_result: str     # Research Agent 的输出
    final_output: str        # 最终输出


def research_agent(state: AgentState) -> dict:
    """Research Agent 节点"""
    print(f"{'='*50}")
    print(f"  [Research Agent] 开始调研...")
    print(f"{'='*50}")

    # 从用户消息里提取研究问题
    user_msg = state["messages"][-1].content

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"请调研以下主题，收集相关信息：\n{user_msg}"),
    ]

    # 循环调用，让 Agent 自主决定搜索什么
    for step in range(4):
        response = llm.invoke(messages, tools=TOOLS)
        messages.append(response)

        if not response.tool_calls:
            break  # Agent 认为信息足够，不再调用工具

        for tc in response.tool_calls:
            func_name = tc["name"]
            args = tc["args"]
            print(f"  >> Research 调用工具: {func_name}({args})")
            result = TOOL_MAP[func_name](**args)
            print(f"  << 返回: {result[:100]}...")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # 提取最终的研究汇总
    research_text = messages[-1].content if messages[-1].content else ""
    print(f"\n  [Research Agent] 调研完成。\n")

    return {
        "research_result": research_text,
        "next_agent": "writer",
    }


def writer_agent(state: AgentState) -> dict:
    """Writer Agent 节点"""
    print(f"{'='*50}")
    print(f"  [Writer Agent] 开始撰写...")
    print(f"{'='*50}")

    research = state.get("research_result", "")

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"以下是 Research Agent 收集到的信息：\n\n{research}\n\n"
                    f"请根据这些信息撰写一份文档，并用 write_file 保存到 'output.md'。"
        ),
    ]

    for step in range(4):
        response = llm.invoke(messages, tools=TOOLS)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            func_name = tc["name"]
            args = tc["args"]
            print(f"  >> Writer 调用工具: {func_name}({args})")
            result = TOOL_MAP[func_name](**args)
            print(f"  << 返回: {result[:150]}...")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    output = messages[-1].content if messages[-1].content else "文档撰写完成。"
    print(f"\n  [Writer Agent] 撰写完成。\n")

    return {
        "final_output": output,
        "next_agent": "finish",
    }


def router(state: AgentState) -> Literal["research", "writer", "__end__"]:
    """路由逻辑：决定下一步执行哪个 Agent"""
    next_agent = state.get("next_agent", "research")
    if next_agent == "finish":
        return "__end__"
    return next_agent


# ============================================================
# 4. 构建图
# ============================================================

def build_multi_agent_graph():
    """构建 Research → Writer → End 的多 Agent 流程图"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("research", research_agent)
    workflow.add_node("writer", writer_agent)

    # 设置入口
    workflow.set_entry_point("research")

    # 添加边：research -> writer -> end
    workflow.add_conditional_edges("research", router, {
        "writer": "writer",
        "__end__": END,
    })
    workflow.add_conditional_edges("writer", router, {
        "research": "research",
        "__end__": END,
    })

    return workflow.compile()


# ============================================================
# 5. 运行
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  多 Agent 协作演示 — Research Agent + Writer Agent")
    print("  主题：调研 LangGraph 和 RAG 技术")
    print("=" * 55, end="\n\n")

    graph = build_multi_agent_graph()

    result = graph.invoke({
        "messages": [HumanMessage(content="LangGraph 是什么？以及 RAG 和 Agent 框架的最新发展")],
        "next_agent": "research",
        "research_result": "",
        "final_output": "",
    })

    print("\n" + "=" * 55)
    print("  最终结果")
    print("=" * 55)
    print(f"调研结果：{result.get('research_result', '无')[:200]}...")
    print(f"\n文档输出：{result.get('final_output', '无')}")
