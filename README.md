# RAG-Learning

RAG 智能问答系统 + 自主 Agent 智能体框架 — 从零到一落地完整管线，全部开源。

基于 LangChain + 通义千问，涵盖文档处理、向量检索、评估体系、手写 ReAct、工具调用、LangGraph 多 Agent 协作等核心能力。

---

## 项目亮点

- **完整 RAG 管线**：文档加载 → 文本分块 → Embedding → 向量检索 → 混合检索(BM25+向量) → RRF 融合 → LLM 生成 → 质量评估
- **自主 Agent 框架**：手写 ReAct 循环(LangChain 零依赖) → 真实工具调用(文件/网页/Python沙箱) → LangGraph 多 Agent 协作(Research + Writer + Supervisor)
- **工程落地**：subprocess 安全沙箱、超时控制、编码自动检测、滑动窗口记忆、检索评估体系(Hit Rate / MRR)
- **5 步渐进式学习**：每步可独立运行，从原理到实战层层递进

---

## RAG 智能问答系统

### 管线架构

```
文档 → 切片(RecursiveCharacter, chunk_size=200, overlap=40)
     → Embedding(all-MiniLM-L6-v2, 384维)
     → FAISS 向量库
     → 检索(向量/MMR/混合) → RRF 融合 → LLM(Qwen-turbo)生成
```

### 三种检索策略对比

| 策略 | 说明 |
|------|------|
| 向量检索 | 原生余弦相似度，快速精确 |
| MMR 多样化检索 | lambda_mult=0.7, fetch_k=9，提升结果多样性 |
| 混合检索 | 向量 + BM25 关键词，RRF 融合排序 |

### 评估指标

基于 8 条 QA 测试集，自动化评估检索质量（Hit Rate / MRR）与生成质量。

### 交互界面

Streamlit Web 界面，支持策略切换、Top-K 调节、一键运行评估、实时指标展示。

---

## 自主 Agent 智能体框架

### 5 步渐进路线

| 步骤 | 文件 | 核心内容 |
|------|------|----------|
| A1 | `步骤A1_ReAct手写demo.py` | ReAct 推理循环(Thought→Action→Observation) |
| A2 | `步骤A2_LangChain_Agent基础.py` | LangChain Agent 框架使用 |
| A3 | `步骤A3_手写简易Agent.py` | 零框架依赖，手写 ReAct + 工具调度 |
| A4 | `步骤A4_工具调用实战.py` | 真实工具：文件读写 / 网页抓取 / Python 沙箱执行 |
| A5 | `步骤A5_多Agent协作.py` | LangGraph StateGraph：Research→Supervisor→Writer→Supervisor→End |

### Agent 架构

```
用户问题 → Research Agent(搜索/调研) → Supervisor(质量审核)
         → Writer Agent(撰写/保存) → Supervisor(终审) → 输出
```

Supervisor 审核不通过时自动返工重做，最多重试 3 次。

### 安全沙箱

- Python 代码隔离执行：`tempfile` + `subprocess`，15 秒超时
- 死循环 / 内存泄漏 / 文件逃逸三重防护
- 网页抓取编码自动检测(gbk/utf-8)，Selenium 降级兜底

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
# Windows:
set DASHSCOPE_API_KEY=your_api_key_here
# / Linux:
export DASHSCOPE_API_KEY=your_api_key_here

# 3. 运行 RAG 全流程
python "测试2/步骤6_完整管线_评估.py"

# 4. 运行 Agent 演示
python "测试2/步骤A3_手写简易Agent.py"
python "测试2/步骤A4_工具调用实战.py"
python "测试2/步骤A5_多Agent协作.py"

# 5. 启动 Web 界面
streamlit run "测试2/步骤7_Streamlit界面.py"
```

### 学习路线

1. **RAG 基础**：步骤 1→2→3→4→5→6→7，理解检索增强生成全流程
2. **Agent 核心**：步骤 A1→A2→A3，掌握 ReAct 原理与手写实现
3. **工具与安全**：步骤 A4，接入真实工具 + 安全沙箱
4. **多 Agent 协作**：步骤 A5，LangGraph 状态图 + Supervisor 质量管控

---

## 技术栈

| 类别 | 工具 |
|------|------|
| LLM | Qwen-turbo / Qwen-plus (通义千问) |
| 框架 | LangChain, LangGraph |
| 向量库 | FAISS |
| Embedding | all-MiniLM-L6-v2 (384维) |
| 检索增强 | BM25, RRF 融合排序, MMR |
| 安全沙箱 | subprocess, tempfile |
| 前端 | Streamlit |
| 数据抓取 | Requests, BeautifulSoup4, Selenium |

---

## 设计决策

- **为什么手写 ReAct 而不是直接用 LangChain Agent？** 简单任务手写效率高约 30%，且方便自定义工具调度逻辑和调试。多步复杂任务再用框架。
- **为什么用 RRF 融合而非纯向量检索？** 纯向量检索对专有名词、缩写匹配不佳，BM25 关键词检索互补，RRF 融合后整体 Hit Rate 更高。
- **为什么用 subprocess 而非 `exec()`？** 安全隔离，防止恶意代码影响主进程，配合超时控制避免资源耗尽。

---

## 在线演示

- GitHub: [github.com/WANGSHUAIQIKNIGHT](https://github.com/WANGSHUAIQIKNIGHT)
- 项目源码: [github.com/WANGSHUAIQIKNIGHT/RAG-Learning](https://github.com/WANGSHUAIQIKNIGHT/RAG-Learning)
- RAG 在线体验: [RAG 演示站](https://github.com/WANGSHUAIQIKNIGHT/RAG-Learning) (Streamlit)
