"""
第2步：LangChain Agent 基础 — 极简版
"""

from langchain_classic.tools import tool
from langchain_community.chat_models import ChatTongyi


@tool
def get_weather(city: str) -> str:
    """查询城市的天气，参数：城市名（如 北京、上海）"""
    data = {"北京": "25°C，晴", "上海": "28°C，多云", "深圳": "30°C，小雨"}
    return data.get(city, f"没有{city}的天气数据")


@tool
def calculator(expr: str) -> str:
    """数学计算，参数：数学表达式（如 3+(5*2)）"""
    import re
    if not re.match(r'^[\d+\-*/().\s]+$', expr):
        return "表达式不合法"
    return str(eval(expr))


llm = ChatTongyi(model="qwen-plus", temperature=0.0)
tools = {"get_weather": get_weather, "calculator": calculator}
agent = llm.bind_tools(tools.values())

SYSTEM_PROMPT = "你是一个有用的助手。需要使用工具时就调用工具，不需要就直接回答。"


def ask(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for i in range(5):
        response = agent.invoke(messages)
        if not response.tool_calls:
            return response.content
        messages.append(response)
        for tc in response.tool_calls:
            result = tools[tc["name"]].invoke(tc["args"])
            messages.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
    return "超出最大步数"


if __name__ == "__main__":
    print(ask("北京今天多少度？"))
    print(ask("计算 24 + (13 * 2) 等于多少？"))
