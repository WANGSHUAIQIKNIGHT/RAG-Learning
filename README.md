<p align="center">
  <h1 align="center">🦾 RAG-Learning</h1>
  <p align="center">
    RAG 智能问答系统 + 自主 Agent 智能体框架
    <br />
    从零到一落地完整管线 · 全部开源 · 在线可体验
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12" />
    </a>
    <a href="https://huggingface.co/spaces/wangshuaiqi/agent-collab">
      <img src="https://img.shields.io/badge/demo-HF%20Space-yellow.svg" alt="HF Space Demo" />
    </a>
    <a href="https://github.com/WANGSHUAIQIKNIGHT/RAG-Learning/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" />
    </a>
  </p>
</p>

<br />

> 🚀 **在线体验**：👉 [HuggingFace Space — 多 Agent 协作系统](https://huggingface.co/spaces/wangshuaiqi/agent-collab)
>
> 输入一个研究主题，三个 AI Agent 自动调研、审核、撰写，生成完整文档。

---

## 📋 项目概览

从零搭建企业级 RAG 管线 + 多 Agent 协作框架，覆盖 **文档解析 → 向量检索 → 评估体系 → 手写 ReAct → 工具调用 → LangGraph 多 Agent 编排** 全流程。

### 两大核心模块

| 模块 | 说明 | 亮点 |
|------|------|------|
| 🔍 **RAG 智能问答** | 文档加载→分块→Embedding→多策略检索→LLM生成 | 手写 BM25 · 3 种检索策略对比 · Hit Rate/MRR 评估 |
| 🤖 **Agent 智能体框架** | 手写 ReAct → 真实工具 → LangGraph 多 Agent 协作 | 零框架依赖手写实现 · Supervisor 质量管控 · 安全沙箱 |

---

## 🔥 核心亮点

### 1. 手写 BM25 检索引擎
不依赖任何第三方库，从零实现完整 BM25 算法：分词、IDF 计算、TF 归一化。与 FAISS 向量检索、MMR 多样化检索自由切换，展示对检索原理的底层理解。

### 2. 检索策略量化评估
构建 **8 条 QA 测试集 × 3 类场景**（短查询、模糊查询、多跳推理），以 **Hit Rate 和 MRR 双指标** 自动化评估三种策略。实验证明混合检索（向量 + BM25 + RRF 融合）综合效果最优。

### 3. 手写 ReAct 推理引擎
从零实现 Thought→Action→Observation 循环（最多 8 步），不依赖 LangChain Agent 框架。对比测试表明：**简单任务上手写 ReAct 比 LangChain Agent 快约 30%**，多步复杂任务框架更可靠——具备框架选型判断力。

### 4. 多 Agent 协作（Research → Supervisor → Writer）
基于 **LangGraph StateGraph** 构建生产级多 Agent 工作流：
- **Research Agent**：搜索调研、收集信息
- **Supervisor Agent**：LLM-as-Judge 质量审核，不通过打回重做（最多 3 轮）
- **Writer Agent**：根据调研结果撰写文档并保存

### 5. 工程化安全沙箱
结合安全工程背景，覆盖三类风险防护：
- 🔒 死循环防护 · 内存泄漏监控 · 文件逃逸隔离
- ⏱ Python 代码 `tempfile`+`subprocess` 隔离执行，15s 超时
- 🌐 网页抓取编码自动检测（gbk/utf-8），Selenium 动态渲染降级兜底

---

## 🧱 项目结构

```
RAG 管线（7 步渐进）
├── 步骤1_文档加载器.py          # 多格式文档解析
├── 步骤2_文本切片.py            # RecursiveCharacter 分块
├── 步骤3_Embedding+向量库.py    # all-MiniLM-L6-v2 → FAISS
├── 步骤4_检索与重排序.py        # 向量/BM25/MMR + RRF 融合
├── 步骤5_生成+对话记忆.py       # LLM 生成 + 滑动窗口记忆
├── 步骤6_完整管线_评估.py       # 全流程自动化 + 评估
└── 步骤7_Streamlit界面.py      # Web 交互界面

Agent 框架（5 步渐进）
├── 步骤A1_ReAct手写demo.py      # 理解 Thought→Action→Observation
├── 步骤A2_LangChain_Agent基础.py # 框架使用入门
├── 步骤A3_手写简易Agent.py      # 零框架依赖手写实现
├── 步骤A4_工具调用实战.py       # 文件/网页/Python 沙箱
└── 步骤A5_多Agent协作.py       # LangGraph StateGraph 编排
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（通义千问 API Key）
# Windows:
set DASHSCOPE_API_KEY=sk-your-api-key
# Linux/Mac:
export DASHSCOPE_API_KEY=sk-your-api-key

# 3. 运行 RAG 全流程评估
python "测试2/步骤6_完整管线_评估.py"

# 4. 运行 Agent 多步演示
python "测试2/步骤A5_多Agent协作.py"

# 5. 启动 Web 界面
streamlit run "测试2/步骤A5_Streamlit界面.py"
```

> 💡 **在线体验**：不想本地配置？👉 [HF Space 在线 Demo](https://huggingface.co/spaces/wangshuaiqi/agent-collab)

---

## 🧪 评估结果

| 检索策略 | Hit Rate | MRR | 适用场景 |
|----------|:--------:|:---:|----------|
| 纯向量检索 (FAISS IVF_FLAT) | — | — | 语义相似度匹配 |
| BM25 关键词检索 | — | — | 专有名词/缩写匹配 |
| 混合检索 (向量+BM25+RRF) | ✅ 最优 | ✅ 最优 | 通用场景 |

> 基于 8 条 QA 测试集（短查询/模糊查询/多跳推理）评估。具体数据请运行 `步骤6_完整管线_评估.py` 查看。

---

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| **LLM** | Qwen-turbo / Qwen-plus（通义千问） |
| **框架** | LangChain · LangGraph · ReAct |
| **向量库** | FAISS (IVF_FLAT) |
| **Embedding** | all-MiniLM-L6-v2 (384维) |
| **检索增强** | BM25 · RRF 融合排序 · MMR |
| **安全沙箱** | subprocess · tempfile · 超时控制 |
| **部署** | Docker · HuggingFace Spaces |
| **前端** | Streamlit |
| **数据抓取** | Requests · BeautifulSoup4 · Selenium |

---

## 📚 学习路线

```
RAG 基础 → 步骤1 → 步骤2 → 步骤3 → 步骤4 → 步骤5 → 步骤6 → 步骤7
                                      ↓
Agent 核心 → 步骤A1 → 步骤A2 → 步骤A3
                                      ↓
工具与安全 → 步骤A4
                                      ↓
多 Agent 协作 → 步骤A5
```

每一份代码都可独立运行，从原理到实战层层递进。

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🤝 贡献

欢迎 Issue 和 PR！如果有问题或建议，直接开 Issue 讨论。

---

<p align="center">
  <a href="https://github.com/WANGSHUAIQIKNIGHT">
    <img src="https://img.shields.io/badge/GitHub-WANGSHUAIQIKNIGHT-181717?logo=github" alt="GitHub" />
  </a>
  <a href="https://huggingface.co/spaces/wangshuaiqi/agent-collab">
    <img src="https://img.shields.io/badge/HF%20Space-在线体验-yellow" alt="HF Space" />
  </a>
</p>
