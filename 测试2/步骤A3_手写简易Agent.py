# -*- coding: utf-8 -*-
"""
第3步：手写简易 Agent — 用 ChatTongyi + 手写 ReAct 循环
核心：Agent = LLM + Tools + while 循环，没有魔法
"""

import re

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# ===== 1. LLM =====

llm = ChatTongyi(model="qwen-plus", temperature=0.0)


# ===== 2. 工具定义 (dict 格式, 自己管理) =====

TOOLS = [
    {
        "name": "get_weather","description": "查询城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京、上海"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculator","description": "数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "数学表达式，如 3+(5*2)"},
            },
            "required": ["expr"],
        },
    },
]


# ===== 3. 工具实现 =====

def get_weather(city: str) -> str:
    data = {
        "北京": "25°C，晴",
        "上海": "28°C，多云",
        "深圳": "30°C，小雨",
        "哈尔滨": "15°C，大风",
    }
    return data.get(city, f"没有{city}的天气数据")


def calculator(expr: str) -> str:
    try:
        if not re.match(r'^[\d+\-*/().\s]+$', expr):
            return "表达式不合法"
        return str(eval(expr))
    except Exception as e:
        return f"计算错误：{e}"


TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
}


# ===== 4. 手写 ReAct 循环 =====

def run_agent(question: str, max_steps: int = 5) -> str:
    print("=" * 55)
    print(f"  用户问题: {question}")
    print("=" * 55, end="\n\n")

    messages = [
        SystemMessage(content="你是一个有用的助手。需要工具时调用工具，不需要就直接回答。"),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        print(f">> Step {step} -- 调用 LLM ...")

        # Step A: LLM 思考
        response = llm.invoke(messages, tools=TOOLS)

        # 打印 LLM 的文字回答（如果有）
        if response.content:
            print(f"  [LLM]: {response.content[:200]}")

        # Step B: 判断是否有工具调用
        if not response.tool_calls:
            print("\n" + "=" * 55)
            print(f"  最终答案: {response.content}")
            print("=" * 55)
            return response.content

        # Step C: 执行工具
        messages.append(response)

        for tc in response.tool_calls:
            func_name = tc["name"]
            args = tc["args"]
            tool_call_id = tc["id"]
            print(f"  => 调用工具: {func_name}({args})")

            result = TOOL_MAP[func_name](**args)
            print(f"  => 工具返回: {result}")

            messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

    return "超出最大步数，未得出最终答案。"


# ===== 5. 运行测试 =====

if __name__ == "__main__":
    run_agent("北京今天多少度？")
    print("\n\n")
    run_agent("计算 24 + (13 * 2) 等于多少？")
