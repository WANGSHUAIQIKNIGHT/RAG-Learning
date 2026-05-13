"""
第 7 步：Streamlit 界面 — 把 RAG 做成网页应用
"""
import os
import sys
from pathlib import Path

import streamlit as st

# 确保能找到步骤6的模块
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from 步骤6_完整管线_评估 import RAGPipeline, build_test_set, evaluate_retrieval

st.set_page_config(page_title="RAG Playground", page_icon="🤖", layout="wide")

# ============================================================
# Session State 初始化
# ============================================================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(retrieve_type="similarity", k=3)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


def rebuild_pipeline(retrieve_type: str, k: int):
    st.session_state.pipeline = RAGPipeline(retrieve_type=retrieve_type, k=k)
    st.session_state.messages = []
    st.session_state.eval_results = None


# ============================================================
# 侧边栏 — 配置
# ============================================================
with st.sidebar:
    st.title("RAG 配置")

    strategy = st.selectbox(
        "检索策略",
        options=["similarity", "mmr", "hybrid"],
        format_func=lambda x: {"similarity": "向量相似度", "mmr": "MMR 多样化", "hybrid": "混合检索（向量+BM25）"}[x],
        index=0,
        help="similarity: 只看相关度 | mmr: 兼顾多样性 | hybrid: 向量+关键词融合",
    )
    top_k = st.slider("检索数量 (top-k)", min_value=1, max_value=5, value=3)

    if st.button("应用配置并清空对话"):
        rebuild_pipeline(strategy, top_k)
        st.rerun()

    st.divider()
    st.caption("文档: sample.txt")

    if st.button("运行评估"):
        with st.spinner("正在评估..."):
            pipe = RAGPipeline(retrieve_type=strategy, k=top_k)
            test_set = build_test_set()
            metrics = evaluate_retrieval(pipe, test_set, top_k=top_k)
            st.session_state.eval_results = {
                "strategy": strategy,
                "hit_rate": metrics["hit_rate"],
                "mrr": metrics["mrr"],
            }
        st.rerun()

    if st.session_state.eval_results:
        r = st.session_state.eval_results
        st.divider()
        st.subheader("评估结果")
        st.metric("检索策略", r["strategy"])
        st.metric("Hit Rate", f"{r['hit_rate']:.1%}")
        st.metric("MRR", f"{r['mrr']:.4f}")

# ============================================================
# 主区域 — 对话界面
# ============================================================
st.title("RAG 问答系统")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "docs" in msg:
            with st.expander(f"检索到的 {len(msg['docs'])} 条文档"):
                for i, doc in enumerate(msg["docs"], 1):
                    st.text(f"[{i}] {doc.page_content}")

# 输入框
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 检索 + 生成
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            pipe = st.session_state.pipeline
            docs = pipe.retrieve(prompt)
            answer = pipe.answer(prompt)
        st.markdown(answer)
        with st.expander(f"检索到的 {len(docs)} 条文档"):
            for i, doc in enumerate(docs, 1):
                st.text_area(
                    label=f"文档 {i}",
                    value=doc.page_content,
                    height=80,
                    key=f"doc_{len(st.session_state.messages)}_{i}",
                )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "docs": docs,
    })
