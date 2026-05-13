# -*- coding: utf-8 -*-
"""
ReAct 框架手写 Demo — 纯 Python，不用 LangChain
核心循环：Thought → Action → Observation → ... → Final Answer
"""

import json
import re


# ===== 1. 定义工具 =====

def get_weather(city: str) -> str:
    """模拟查天气"""
    data = {
        "北京": "25°C，晴",
        "上海": "28°C，多云",
        "深圳": "30°C，小雨",
        "哈尔滨": "15°C，大风",
    }
    result = data.get(city, f"没有{city}的天气数据")
    return result


def calculator(expr: str) -> str:
    """简单计算器"""
    try:
        # 只允许数字和运算符，防注入
        if not re.match(r'^[\d+\-*/().\s]+$', expr):
            return "表达式不合法"
        return str(eval(expr))
    except Exception as e:
        return f"计算错误：{e}"


TOOLS = {
    "get_weather": {"func": get_weather, "desc": "查询城市天气，参数：城市名"},
    "calculator": {"func": calculator, "desc": "数学计算，参数：表达式如 3+(5*2)"},
}


# ===== 2. 模拟 LLM 的"思考" =====

def llm_reason(question: str, history: list) -> str:
    """
    模拟 LLM 的推理输出。
    真实场景调用 OpenAI/Claude API，这里用规则模拟。
    """
    conv = "\n".join(history[-6:])

    # 检查是否已有 Observation → 直接给 Final Answer
    for h in reversed(history):
        if h.startswith("Observation"):
            return f"Final Answer: {h.replace('Observation: ', '')}"

    # 根据问题决定行动
    if any(kw in question for kw in ["天气", "温度", "度", "下雨", "晴", "刮风"]):
        # 提取城市名
        cities = ["北京", "上海", "深圳", "哈尔滨"]
        mentioned_city = None
        for c in cities:
            if c in question:
                mentioned_city = c
                break
        if mentioned_city:
            return f"Thought: 用户想查{mentioned_city}的天气，我需要调用 get_weather\nAction: get_weather\nAction Input: {mentioned_city}"
        else:
            return "Final Answer: 我只支持查询北京、上海、深圳、哈尔滨的天气"

    if "计算" in question or any(op in question for op in ["+", "-", "*", "/"]):
        # 提取数字和运算符
        expr = "".join(c for c in question if c.isdigit() or c in "+-*/(). ")
        return f"Thought: 用户想计算，我调用 calculator\nAction: calculator\nAction Input: {expr}"

    return "Final Answer: 我只支持查天气和计算。" if not history else "Final Answer: 已处理完毕。"


# ===== 3. ReAct 主循环 =====

def react_agent(question: str, max_steps: int = 5) -> str:
    print(f"{'='*50}")
    print(f"用户问题: {question}")
    print(f"{'='*50}\n")

    history = []
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")

        # Step A: Thought → Action
        response = llm_reason(question, history)
        print(f"[LLM 输出]\n{response}\n")

        # 检查是否 Final Answer
        if response.startswith("Final Answer:"):
            answer = response.replace("Final Answer:", "").strip()
            print(f"{'='*50}")
            print(f"最终答案: {answer}")
            print(f"{'='*50}")
            return answer

        # Step B: 解析 Action
        thought_match = re.search(r"Thought:\s*(.+)", response)
        action_match = re.search(r"Action:\s*(\w+)", response)

        input_match = re.search(r"Action Input:\s*(.+)", response)

        if not action_match or not input_match:
            history.append("Error: 无法解析 LLM 输出")
            continue

        action = action_match.group(1)
        action_input = input_match.group(1).strip().strip('"\'')
        thought = thought_match.group(1) if thought_match else ""

        print(f"  -> 解析: 思考[{thought}] -> 调用 {action}({action_input})")

        # Step C: 执行 Action -> Observation
        tool = TOOLS.get(action)
        if not tool:
            observation = f"错误：没有叫 {action} 的工具"
        else:
            observation = tool["func"](action_input)

        print(f"  -> 观察: 工具返回 -> {observation}")

        history.append(f"Thought: {thought}")
        history.append(f"Action: {action}({action_input})")
        history.append(f"Observation: {observation}")

    return "达到最大步数，未得出最终答案。"


# ===== 4. 运行测试 =====

if __name__ == "__main__":
    react_agent("北京今天多少度？")
    print("\n\n")
    react_agent("计算 24 + (13 * 2) 等于多少？")
