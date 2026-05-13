"""
第 2 步：文本切片策略
目标：理解为什么要切、怎么切、切多大最合适
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(BASE_DIR, "resourse", "sample.txt")

# 先加载文档
loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()
full_text = docs[0].page_content

print("原始文档长度:", len(full_text), "字符")
print()

# ============================================================
# 1. 最简单的切分 — 只看 chunk_size
# ============================================================
print("=" * 60)
print("1. 基础切分: chunk_size=200, chunk_overlap=0")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块最多 200 字符
    chunk_overlap=0,      # 块之间不重叠
)
chunks = splitter.split_documents(docs)

print(f"   切成了 {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"   块 {i+1}: ({len(chunk.page_content)} 字) {chunk.page_content[:50]}...")

print()

# ============================================================
# 2. 加 overlap — 块之间重叠
# ============================================================
print("=" * 60)
print("2. 加入 overlap: chunk_size=200, chunk_overlap=50")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,     # 每块包含上一块末尾 50 字
)
chunks = splitter.split_documents(docs)

print(f"   切成了 {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"   块 {i+1}: ({len(chunk.page_content)} 字) {chunk.page_content[:50]}...")

print()

# ============================================================
# 3. overlap 的作用演示 — 看看连接处
# ============================================================
print("=" * 60)
print("3. overlap 的作用 — 不丢上下文")
print("=" * 60)

splitter_no = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
splitter_yes = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

chunks_no = splitter_no.split_documents(docs)
chunks_yes = splitter_yes.split_documents(docs)

print("   chunk_overlap=0 的两块连接处:")
print(f"   块 1 末尾: ...{chunks_no[0].page_content[-30:]}")
print(f"   块 2 开头: {chunks_no[1].page_content[:30]}...")
print()
print("   chunk_overlap=50 的两块连接处:")
print(f"   块 1 末尾: ...{chunks_yes[0].page_content[-30:]}")
print(f"   块 2 开头: {chunks_yes[1].page_content[:30]}...")
print("   → 有重叠时，块 2 开头保留了块 1 末尾的内容")
print()

# ============================================================
# 4. 不同分隔符对
# ============================================================
print("=" * 60)
print("4. 分隔符顺序对中文的影响")
print("=" * 60)

# RecursiveCharacterTextSplitter 的 separators 参数决定切分优先级
# 默认顺序: ["\n\n", "\n", " ", ""]
# 对于中文，句号/感叹号/问号也是天然分隔符

splitter_cn = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 加入了中文标点
)
chunks_cn = splitter_cn.split_documents(docs)

print(f"   含中文标点: 切成了 {len(chunks_cn)} 块")
print(f"   不含中文标点: 切成了 {len(chunks_no)} 块（上一个例子）")
print()
print("   含中文标点的切分结果:")
for i, chunk in enumerate(chunks_cn):
    print(f"   块 {i+1}: {chunk.page_content}")

print()

# ============================================================
# 5. 实际 RAG 中 chunk_size 怎么选？
# ============================================================
print("=" * 60)
print("5. 面试常问: chunk_size 和 overlap 怎么定？")
print("=" * 60)
print("""
   一般经验:
   ├── chunk_size = 500  ← 最常用，适合大部分场景
   ├── chunk_size = 300  ← 答案比较精准，但可能丢上下文
   └── chunk_size = 1000 ← 上下文更丰富，但可能混入噪声

   chunk_overlap 一般是 chunk_size 的 10%-20%

   到底怎么选？看实际效果调，没有绝对标准。
""")

print("✅ 第 2 步完成！你理解了切片原理和参数含义。")
print("   下一关: Embedding + 向量库 — 把文字变成向量存起来")
