# -*- coding: utf-8 -*-
"""
第4步：工具调用实战 — 接入真实工具（文件读写、网页抓取、Python 执行）
"""

import re
import os
import subprocess
import tempfile

import requests
from bs4 import BeautifulSoup

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

llm = ChatTongyi(model="qwen-plus", temperature=0.0)


# ===== 2. 工具定义 =====

TOOLS = [
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
        "description": "写入内容到本地文件（会覆盖已有内容）",
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
        "name": "fetch_url",
        "description": "读取指定 URL 的网页内容并返回纯文本，参数 url 必须是完整的网页地址",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "网页 URL"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "execute_python",
        "description": "执行 Python 代码并返回结果（可用于数据分析、计算等）",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
            },
            "required": ["code"],
        },
    },
]


# ===== 3. 工具实现 =====

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


def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Agent-Learning/1.0)"
        })
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        # 去掉 script/style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # 取前 2000 字符，避免太长
        return text[:2000] + ("..." if len(text) > 2000 else "")
    except Exception as e:
        return f"抓取失败：{e}"


def execute_python(code: str) -> str:
    try:
        # 写入临时文件执行，避免直接 eval 的安全风险
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(
            ["python", tmp],
            capture_output=True, text=True, timeout=15,
        )
        os.unlink(tmp)
        output = ""
        if result.stdout:
            output += f"标准输出：\n{result.stdout.strip()}\n"
        if result.stderr:
            output += f"标准错误：\n{result.stderr.strip()}\n"
        if result.returncode != 0:
            output = f"进程退出码：{result.returncode}\n{output}"
        return output.strip() or "执行完成，无输出。"
    except subprocess.TimeoutExpired:
        return "执行超时（>15秒）"
    except Exception as e:
        return f"执行失败：{e}"


TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "fetch_url": fetch_url,
    "execute_python": execute_python,
}


# ===== 4. ReAct 循环 (复用 Step 3 的逻辑) =====

def run_agent(question: str, max_steps: int = 8) -> str:
    print("=" * 55)
    print(f"  用户: {question}")
    print("=" * 55, end="\n\n")

    messages = [
        SystemMessage(content="你是一个有用的助手。需要工具时调用工具，不需要就直接回答。"),
        HumanMessage(content=question),
    ]

    for step in range(1, max_steps + 1):
        print(f">> Step {step}")
        response = llm.invoke(messages, tools=TOOLS)

        if response.content:
            print(f"  [LLM]: {response.content[:200]}")

        if not response.tool_calls:
            print("\n" + "=" * 55)
            print(f"  最终答案: {response.content}")
            print("=" * 55)
            return response.content

        messages.append(response)
        for tc in response.tool_calls:
            func_name = tc["name"]
            args = tc["args"]
            print(f"  => {func_name}({args})")
            result = TOOL_MAP[func_name](**args)
            print(f"  => 返回: {result[:120]}...")
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return "超出最大步数，未得出最终答案。"


# ===== 5. 测试 =====

if __name__ == "__main__":
    # 测试 1: 写文件 → 读文件
    run_agent("写一个文件叫 test.txt，内容为 'Hello Agent!'，然后读出来确认")

    print("\n" + "=" * 60 + "\n")

    # 测试 2: 执行 Python（让 LLM 用工具跑，而不是自己算）
    run_agent("用 execute_python 工具计算 1 到 100 的和并打印结果，然后把结果写入 sum.txt")

    print("\n" + "=" * 60 + "\n")

    # 测试 3: 抓取网页
    run_agent("抓取 https://www.example.com 的内容，总结这个页面说了什么")
