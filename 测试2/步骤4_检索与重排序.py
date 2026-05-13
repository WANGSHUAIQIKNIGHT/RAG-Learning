"""
第 4 步：检索与重排序
目标：理解不同检索方式，学会用重排序提升结果质量
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(BASE_DIR, "resourse", "sample.txt")
index_path = r"D:\py_rag_faiss_index"

# 加载模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ============================================================
# 准备数据：加载 → 切片 → 向量化
# ============================================================
print("=" * 60)
print("准备数据：加载 → 切片 → 向量化")
print("=" * 60)

loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"   切分成 {len(chunks)} 块")

vector_store = FAISS.from_documents(chunks, embedding_model)
print(f"   已存入 {vector_store.index.ntotal} 个向量\n")

# ============================================================
# 1. 基础检索 — similarity_search
# ============================================================
print("=" * 60)
print("1. 基础检索 similarity_search")
print("=" * 60)

query = "RAG能解决什么问题？"
results = vector_store.similarity_search(query, k=3)
print(f"   问题: {query}")
for i, doc in enumerate(results):
    print(f"   第 {i+1} 名: {doc.page_content}")
print()

# ============================================================
# 2. MMR 检索 — 兼顾相关性和多样性
# ============================================================
print("=" * 60)
print("2. MMR 检索 — 避免结果太重复")
print("=" * 60)

# MMR 的原理：
# 普通检索：找出和问题最像的 k 块 → 可能 3 块都在讲同一个意思
# MMR 检索：选和问题相关的，同时保证这 k 块之间不重复

mmr_results = vector_store.max_marginal_relevance_search(
    query,
    k=3,
    fetch_k=10,     # 先拉 10 个候选
    lambda_mult=0.5 # 0.5 = 相关性和多样性各占一半
)
print(f"   问题: {query}")
for i, doc in enumerate(mmr_results):
    print(f"   第 {i+1} 名: {doc.page_content}")
print()

# ============================================================
# 3. 普通检索 vs MMR 对比
# ============================================================
print("=" * 60)
print("3. 对比：普通检索 vs MMR")
print("=" * 60)

# 选一个能看出差异的问题
q2 = "深度学习"
sim_results = vector_store.similarity_search(q2, k=3)
mmr_results = vector_store.max_marginal_relevance_search(q2, k=3, fetch_k=10)

print(f"   问题: {q2}")
print(f"\n   普通检索 前 3:")
for i, doc in enumerate(sim_results):
    print(f"     第 {i+1}: {doc.page_content[:50]}...")

print(f"\n   MMR 检索 前 3:")
for i, doc in enumerate(mmr_results):
    print(f"     第 {i+1}: {doc.page_content[:50]}...")
print()

# ============================================================
# 4. 什么是重排序（Re-ranking）
# ============================================================
print("=" * 60)
print("4. 重排序（Re-ranking）的概念")
print("=" * 60)
print("""
   普通流程： 问题 → 向量检索 top10
   重排序流程：问题 → 向量检索 top30 → 重排序模型 → top10

   为什么要重排序？
   向量检索快但粗，可能漏掉好的或混入差的。
   重排序模型慢但准，用更精细的算法重新打分排序。

   重排序在 RAG 中是一个常用技巧：
   先快速捞出候选（向量检索），再精细筛选（重排序）。
""")

# ============================================================
# 5. 模拟重排序效果 — 用相似度分数重新排序
# ============================================================
print("=" * 60)
print("5. 模拟重排序：先多捞再精选")
print("=" * 60)

query3 = "什么是Agent？"

# 第一步：向量检索多捞一些（捞 5 个）
initial_results = vector_store.similarity_search_with_score(query3, k=5)
print(f"   问题: {query3}")
print(f"   第一步：向量检索捞了 {len(initial_results)} 个候选")
for i, (doc, score) in enumerate(initial_results):
    print(f"      候选 {i+1} (距离 {score:.4f}): {doc.page_content[:40]}...")

# 第二步：按距离排序（越小越相关）
ranked = sorted(initial_results, key=lambda x: x[1])

print(f"\n   第二步：重排序后（按距离升序）")
for i, (doc, score) in enumerate(ranked):
    print(f"      第 {i+1} 名 (距离 {score:.4f}): {doc.page_content[:40]}...")

# 第三步：只取前 2 个给 LLM
final = ranked[:2]
print(f"\n   第三步：只取前 2 名送入 LLM")
for i, (doc, _) in enumerate(final):
    print(f"      给 LLM 的第 {i+1} 块: {doc.page_content}")
print()

# ============================================================
# 6. 检索方式总结：什么时候用哪种？
# ============================================================
print("=" * 60)
print("6. 总结：检索方式怎么选？")
print("=" * 60)
print("""
   similarity_search       → 多数情况首选，又快又准
   MMR                     → 文档内容相似度高时用，避免重复
   similarity_search + 重排序 → 精度要求高时用，先多捞再精筛

   实际 RAG 管线通常是：
   问题 → 向量检索 top20 → 重排序 top5 → LLM 生成回答
""")

print("✅ 第 4 步完成！你理解了检索方式和重排序策略。")
print("   下一关: 生成 + 对话记忆 — 把检索结果交给 LLM 生成回答")
