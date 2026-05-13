# RAG-Learning

RAG（检索增强生成）与 Agent（智能体）学习项目，基于 LangChain + 通义千问，从零到一落地完整管线。

## 目录结构

所有代码在 `测试2/` 目录下，分两大模块：

### RAG 基础管线

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | 步骤1_文档加载器.py | 多种文档格式加载（PDF、TXT、Markdown） |
| 2 | 步骤2_文本切片.py | 文本分块策略（按段落、指定大小） |
| 3 | 步骤3_Embedding+向量库.py | 文本向量化 + FAISS 向量存储 |
| 4 | 步骤4_检索与重排序.py | 语义检索 + 重排序（Rerank） |
| 5 | 步骤5_生成+对话记忆.py | LLM 生成回答 + 对话历史记忆 |
| 6 | 步骤6_完整管线_评估.py | 全流程整合 + 检索质量评估 |
| 7 | 步骤7_Streamlit界面.py | Web 交互界面（Streamlit） |

### Agent 学习系列

| 步骤 | 文件 | 内容 |
|------|------|------|
| A1 | 步骤A1_ReAct手写demo.py | ReAct 推理循环核心原理 |
| A2 | 步骤A2_LangChain_Agent基础.py | LangChain Agent 框架使用 |
| A3 | 步骤A3_手写简易Agent.py | 不依赖框架，自己实现 ReAct 循环 |
| A4 | 步骤A4_工具调用实战.py | 接入真实工具（文件读写、网页抓取、Python 执行） |
| A5 | 步骤A5_多Agent协作.py | LangGraph 状态图，Research Agent + Writer Agent 多步协作 |

## 运行要求

```bash
pip install langchain-community langchain-core langgraph requests beautifulsoup4
```

需要在环境变量中配置 `DASHSCOPE_API_KEY`（通义千问 API Key）。

## 学习路线

1. 先跑通 **RAG 步骤 1~7**，理解检索增强生成全流程
2. 进入 **Agent 步骤 A1~A5**，从原理到手写到工具调用再到多 Agent 协作
3. 持续规划中...