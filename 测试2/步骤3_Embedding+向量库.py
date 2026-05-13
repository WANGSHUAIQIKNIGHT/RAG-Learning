"""
第 3 步：Embedding + 向量库
目标：理解文本转向量，掌握 FAISS 存储和相似度检索
"""

import os
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(BASE_DIR, "resourse", "sample.txt")

# ============================================================
# 1. 什么是 Embedding？
# ============================================================
print("=" * 60)
print("1. 什么是 Embedding（嵌入）？")
print("=" * 60)
print("""
   "苹果" → [0.12, -0.45, 0.78, ...]   ← 384 个数字
   "香蕉" → [0.15, -0.40, 0.72, ...]   ← 语义相近 → 向量距离近
   "编程" → [-0.30, 0.55, -0.12, ...]  ← 语义不同 → 向量距离远

   Embedding 就是把文字变成一串数字（向量），
   语义越像，数字越接近。
""")

# ============================================================
# 2. 加载 Embedding 模型
# ============================================================
print("=" * 60)
print("2. 加载 Embedding 模型")
print("=" * 60)

# 使用轻量本地模型，中文英文都支持
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
print("   模型: all-MiniLM-L6-v2")
print("   向量维度: 384")
print()

# ============================================================
# 3. 试试把几句话变成向量
# ============================================================
print("=" * 60)
print("3. 生成向量 — 看看长什么样")
print("=" * 60)

texts = [
    "人工智能是计算机科学的重要分支",
    "深度学习是机器学习的一个子集",
    "今天天气真好，适合出去散步",
]

vectors = embedding_model.embed_documents(texts)

for i, (text, vec) in enumerate(zip(texts, vectors)):
    print(f"   文本 {i+1}: {text}")
    print(f"   向量维度: {len(vec)}")
    print(f"   前 8 个数字: {[round(v, 4) for v in vec[:8]]}")
    print()

# ============================================================
# 4. 计算相似度（余弦相似度）
# ============================================================
print("=" * 60)
print("4. 语义相似度计算")
print("=" * 60)

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度（-1 ~ 1，越大越相似）"""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    return dot / (n1 * n2)

sim_01 = cosine_similarity(vectors[0], vectors[1])  # AI vs 深度学习
sim_02 = cosine_similarity(vectors[0], vectors[2])  # AI vs 天气

print(f"   「人工智能」->「深度学习」: {sim_01:.4f}  ← 语义接近，分数高")
print(f"   「人工智能」->「今天天气」: {sim_02:.4f}  ← 语义不同，分数低")
print()

# ============================================================
# 5. 完整 RAG 流程：加载 → 切片 → 向量化 → 存库
# ============================================================
print("=" * 60)
print("5. 完整流程：加载 → 切片 → 向量化 → 存库")
print("=" * 60)

# 5.1 加载
loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()
print(f"   ① 加载文档: {len(docs[0].page_content)} 字符")

# 5.2 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 为了演示效果，切小一点
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"   ② 切分成: {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"      块 {i+1}: {chunk.page_content}")

# 5.3 向量化 + 存 FAISS（一步完成）
print(f"\n   ③ 向量化并存入 FAISS 向量库...")
vector_store = FAISS.from_documents(chunks, embedding_model)
print(f"      已存入 {vector_store.index.ntotal} 个向量")

# 保存到本地 -- FAISS Windows 不支持中文路径
index_path = r"D:\py_rag_faiss_index"
os.makedirs(index_path, exist_ok=True)
vector_store.save_local(index_path)
print(f"   ④ 索引已保存到: {index_path}/")
print()

# ============================================================
# 6. 检索测试 — 用问题搜最相关的块
# ============================================================
print("=" * 60)
print("6. 检索测试 — 搜什么？")
print("=" * 60)

questions = [
    "什么是RAG？",
    "向量数据库有什么用？",
    "LangChain是什么？",
    "今天中午吃什么？",  # 故意问个无关的
]

for q in questions:
    # similarity_search 返回最相似的 k 个块
    results = vector_store.similarity_search(q, k=1)
    best = results[0]
    print(f"\n   问题: {q}")
    print(f"   匹配: {best.page_content}")
    print(f"   └─ 相似度排名第 1 的块")

print()

# ============================================================
# 7. 带分数检索 — 看看具体有多相似
# ============================================================
print("=" * 60)
print("7. 带相似度分数的检索")
print("=" * 60)

results = vector_store.similarity_search_with_score("RAG是什么？", k=3)
print(f"   问题: RAG是什么？")
for i, (doc, score) in enumerate(results):
    # FAISS 的分数是 L2 距离，越小越相似
    print(f"\n   第 {i+1} 名 (距离: {score:.4f}): {doc.page_content}")
print("   （L2 距离越小表示越相似）")
print()

# ============================================================
# 8. 重新加载已保存的索引（模拟下次启动）
# ============================================================
print("=" * 60)
print("8. 重新加载 FAISS 索引（持久化验证）")
print("=" * 60)

loaded_store = FAISS.load_local(
    index_path,
    embedding_model,
    allow_dangerous_deserialization=True,
)
test_q = "什么是Agent？"
results = loaded_store.similarity_search(test_q, k=1)
print(f"   问题: {test_q}")
print(f"   匹配: {results[0].page_content}")

print()
print("✅ 第 3 步完成！你掌握了文本转向量和向量检索。")
print("   下一关: 检索与重排序 — 让检索结果更精准")
