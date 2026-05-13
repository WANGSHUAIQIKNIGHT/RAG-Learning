"""
第 6 步：完整管线 + 评估
目标：封装 RAG 管线，用指标量化检索和生成质量
"""
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import math
from collections import Counter
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


# ====================================================================
# 手写 BM25（无需安装 rank_bm25）
# ====================================================================
class SimpleBM25Retriever(BaseRetriever):
    """基于 BM25 算法的检索器，不依赖外部包"""

    documents: list[Document]
    k: int = 3
    k1: float = 1.5
    b: float = 0.75

    def _initialize(self):
        if hasattr(self, '_initialized'):
            return
        self._corpus = [doc.page_content for doc in self.documents]
        self._avgdl = sum(len(d.split()) for d in self._corpus) / max(len(self._corpus), 1)
        self._doc_freq = Counter()
        for text in self._corpus:
            for word in set(text.split()):
                self._doc_freq[word] += 1
        self._nd = len(self._corpus)
        self._initialized = True

    def _score(self, query: str, doc: str) -> float:
        self._initialize()
        score = 0.0
        doc_len = len(doc.split())
        for q_word in query.split():
            if q_word not in self._doc_freq:
                continue
            idf = math.log((self._nd - self._doc_freq[q_word] + 0.5) / (self._doc_freq[q_word] + 0.5) + 1)
            tf = doc.split().count(q_word)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl))
        return score

    def _get_relevant_documents(self, query: str) -> list[Document]:
        self._initialize()
        scored = [(self._score(query, d.page_content), d) for d in self.documents]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.k]]


class EnsembleRetrieverSimple(BaseRetriever):
    """简单 RRF 融合检索器"""
    retrievers: list
    weights: list[float] = None
    k: int = 3

    def _get_relevant_documents(self, query: str) -> list[Document]:
        if self.weights is None:
            self.weights = [1.0 / len(self.retrievers)] * len(self.retrievers)
        # 收集所有检索结果
        all_docs = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.invoke(query)
            for rank, doc in enumerate(results):
                # RRF 分数：weight * 1/(60 + rank)
                score = weight * 1.0 / (60 + rank)
                doc_id = doc.page_content[:50]
                if doc_id in all_docs:
                    all_docs[doc_id][1] += score
                else:
                    all_docs[doc_id] = [doc, score]
        # 按总分排序
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:self.k]]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(BASE_DIR, "resourse", "sample.txt")


# ====================================================================
# 1. RAGPipeline — 封装完整管线
# ====================================================================
class RAGPipeline:
    """封装的 RAG 管线：加载 → 切片 → 向量化 → 检索 → 生成"""

    def __init__(
        self,
        txt_path: str = TXT_PATH,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        k: int = 3,
        retrieve_type: str = "similarity",
        llm_model: str = "qwen-turbo",
    ):
        self.k = k
        self.retrieve_type = retrieve_type
        self._build_pipeline(txt_path, chunk_size, chunk_overlap, llm_model)

    def _build_pipeline(self, txt_path: str, chunk_size: int, chunk_overlap: int, llm_model: str):
        """构建完整的 RAG 管线"""
        # 1) 加载
        loader = TextLoader(txt_path, encoding="utf-8")
        self.docs = loader.load()

        # 2) 切片
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        self.chunks = splitter.split_documents(self.docs)

        # 3) 向量化
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(self.chunks, self.embedding_model)
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

        # 4) BM25 检索器（用于混合检索，手写实现无需安装）
        self.bm25_retriever = SimpleBM25Retriever(documents=self.chunks, k=self.k)

        # 5) LLM
        self.llm = ChatTongyi(model=llm_model, temperature=0)

        # 6) Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个AI助手。根据检索到的上下文回答问题。"
             "如果上下文里没有相关信息，就说不知道。\n\n上下文：\n{context}"),
            ("human", "{input}"),
        ])

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_retriever(self):
        """根据指定的检索类型返回检索器"""
        if self.retrieve_type == "similarity":
            r = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k}
            )
        elif self.retrieve_type == "mmr":
            r = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.k, "fetch_k": self.k * 3, "lambda_mult": 0.7},
            )
        elif self.retrieve_type == "hybrid":
            # 混合检索：向量 + BM25，用 RRF 融合
            vector_r = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k}
            )
            bm25_r = SimpleBM25Retriever(documents=self.chunks, k=self.k)
            r = EnsembleRetrieverSimple(
                retrievers=[vector_r, bm25_r],
                weights=[0.5, 0.5],
                k=self.k,
            )
        else:
            raise ValueError(f"未知检索类型: {self.retrieve_type}")
        return r

    def retrieve(self, query: str) -> list:
        """仅检索，不生成"""
        retriever = self._get_retriever()
        docs = retriever.invoke(query)
        return docs

    def answer(self, query: str) -> str:
        """检索 + 生成，返回回答"""
        retriever = self._get_retriever()
        chain = (
            {"context": retriever | self.format_docs, "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def batch_answer(self, queries: list[str]) -> list[str]:
        return [self.answer(q) for q in queries]


# ====================================================================
# 2. 评估函数
# ====================================================================
def build_test_set() -> list[dict]:
    """基于 sample.txt 内容构造 Q&A 测试集"""
    return [
        {"question": "AI是什么时候诞生的？", "ground_truth": "20世纪50年代"},
        {"question": "深度学习是什么？", "ground_truth": "机器学习的一个子集，使用多层神经网络学习数据表示"},
        {"question": "大语言模型（LLM）有哪些典型例子？", "ground_truth": "GPT系列、Claude、Qwen（通义千问）"},
        {"question": "RAG能解决什么问题？", "ground_truth": "大模型知识更新不及时和幻觉问题"},
        {"question": "Agent通过什么框架执行任务？", "ground_truth": "ReAct（思考+行动）框架"},
        {"question": "LangChain是什么？", "ground_truth": "构建大语言模型应用的开发框架"},
        {"question": "向量数据库有什么用？", "ground_truth": "将文本转换为向量存储，支持高效相似性搜索"},
        {"question": "通常企业怎么结合RAG和Agent？", "ground_truth": "RAG处理知识密集型任务，Agent处理多步推理和工具调用"},
    ]


def evaluate_retrieval(
    pipeline: RAGPipeline,
    test_set: list[dict],
    top_k: int = 3,
) -> dict[str, float]:
    """
    检索评估：Hit Rate（命中率）和 MRR（平均倒数排名）
      - Hit Rate：正确答案出现在 top-k 结果中的比例
      - MRR：第一个正确答案的倒数排名的平均值
    """
    hits = 0
    reciprocal_ranks = []

    print(f"\n{'='*60}")
    print(f"检索评估 (retrieve_type={pipeline.retrieve_type}, top_k={top_k})")
    print(f"{'='*60}")

    for i, item in enumerate(test_set, 1):
        query = item["question"]
        gt = item["ground_truth"]

        docs = pipeline.retrieve(query)
        retrieved_texts = [d.page_content for d in docs[:top_k]]

        # 检查 ground_truth 是否出现在某条检索结果中
        found = False
        rank = 0
        for j, text in enumerate(retrieved_texts):
            if gt[:8] in text:  # 用前8个字判断是否命中
                found = True
                rank = j + 1
                break

        if found:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        status = "OK" if found else "XX"
        print(f"  Q{i}: {query[:20]}... -> {status} (rank={rank})")

    n = len(test_set)
    hit_rate = hits / n
    mrr = sum(reciprocal_ranks) / n
    print(f"  Hit Rate@{top_k}: {hit_rate:.2%} ({hits}/{n})")
    print(f"  MRR@{top_k}:      {mrr:.4f}")
    return {"hit_rate": hit_rate, "mrr": mrr}


def evaluate_generation(
    pipeline: RAGPipeline,
    test_set: list[dict],
) -> list[dict]:
    """
    生成评估：让 LLM 给自己的回答打分（或者回答后人工判断）
    这里用两种方式：
      1. 关键词覆盖度（简单自动评估）
      2. LLM 自评分数（0-10）
    """
    print(f"\n{'='*60}")
    print(f"生成评估 (retrieve_type={pipeline.retrieve_type})")
    print(f"{'='*60}")

    scoring_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个严格的评分员。根据标准答案给AI回答打分（0-10分）。"
         "标准答案中的关键信息必须出现在AI回答中。只返回数字。"),
        ("human", "标准答案：{ground_truth}\n\nAI回答：{answer}\n\n分数："),
    ])
    scorer = scoring_prompt | pipeline.llm | StrOutputParser()

    results = []
    for i, item in enumerate(test_set, 1):
        query = item["question"]
        gt = item["ground_truth"]

        answer = pipeline.answer(query)

        # 关键词覆盖度（简单匹配）：ground_truth 中连续 5 个字以上出现在回答中
        def keyword_coverage(gt_text: str, answer_text: str) -> float:
            hits = 0
            for length in range(8, 0, -2):
                count = 0
                for start in range(0, len(gt_text) - length + 1):
                    seg = gt_text[start:start + length]
                    if seg in answer_text:
                        count += 1
                if count > 0:
                    hits += count
            return min(hits / 3, 1.0)

        coverage = keyword_coverage(gt, answer)

        # LLM 打分
        try:
            score_text = scorer.invoke({"ground_truth": gt, "answer": answer})
            score = float(score_text.strip())
            score = max(0, min(10, score))
        except Exception:
            score = 0.0

        results.append({
            "question": query,
            "answer": answer,
            "ground_truth": gt,
            "coverage": coverage,
            "score": score,
        })

        print(f"  Q{i}: {query[:20]}...")
        print(f"    回答: {answer[:60]}...")
        print(f"    关键词覆盖: {coverage:.0%}  LLM评分: {score:.1f}/10")
    return results


# ====================================================================
# 3. 对比不同检索策略
# ====================================================================
def compare_retrieval_strategies(test_set: list[dict]):
    print("\n" + "=" * 60)
    print("策略对比：similarity vs mmr vs hybrid")
    print("=" * 60)

    strategies = ["similarity", "mmr", "hybrid"]
    results = {}
    for strategy in strategies:
        pipe = RAGPipeline(retrieve_type=strategy)
        metrics = evaluate_retrieval(pipe, test_set, top_k=3)
        results[strategy] = metrics

    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")
    print(f"{'策略':<15} {'Hit Rate':<12} {'MRR':<10}")
    print("-" * 37)
    for name, m in results.items():
        print(f"{name:<15} {m['hit_rate']:.2%}          {m['mrr']:.4f}")
    return results


# ====================================================================
# 4. 完整管线演示
# ====================================================================
def full_pipeline_demo():
    print("=" * 60)
    print("完整 RAG 管线演示")
    print("=" * 60)

    # 默认使用混合检索
    pipe = RAGPipeline(retrieve_type="hybrid", k=3)

    print(f"\n文档: {os.path.basename(TXT_PATH)}")
    print(f"切片数: {len(pipe.chunks)}")
    print(f"向量数: {pipe.vector_store.index.ntotal}")
    print(f"检索策略: hybrid (向量+BM25)")
    print(f"LLM: qwen-turbo")

    queries = [
        "RAG是什么技术？",
        "LangChain和LangGraph有什么关系？",
        "向量数据库在RAG中起什么作用？",
    ]

    for q in queries:
        print(f"\n  用户: {q}")
        docs = pipe.retrieve(q)
        print(f"  检索到 {len(docs)} 条文档:")
        for d in docs:
            print(f"    - {d.page_content[:60]}...")
        ans = pipe.answer(q)
        print(f"  回答: {ans}")


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    start = time.time()

    # 1) 完整管线演示
    full_pipeline_demo()

    # 2) 构建测试集
    test_set = build_test_set()
    print(f"\n测试集: {len(test_set)} 条 Q&A 对")

    # 3) 默认策略评估
    pipe = RAGPipeline(retrieve_type="hybrid")
    eval_retrieval = evaluate_retrieval(pipe, test_set, top_k=3)
    eval_gen = evaluate_generation(pipe, test_set)

    # 4) 策略对比
    compare_retrieval_strategies(test_set)

    elapsed = time.time() - start
    print(f"\n[完成] 第 6 步执行完毕，耗时 {elapsed:.1f}s")
    print("   下一关: Streamlit 界面 — 把 RAG 做成网页应用")
