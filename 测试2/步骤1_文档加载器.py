"""
第 1 步：文档加载器
目标：掌握 3 种常用文档加载方式，理解 Document 对象的结构
"""

import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
)

# 获取项目根目录 (当前文件在 测试2/ 下)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "resourse")

# =====================================================
# 1. TextLoader — 读取纯文本文件（最简单，最常用）
# =====================================================
print("=" * 55)
print("1. TextLoader — 读取 TXT 文件")
print("=" * 55)

txt_path = os.path.join(RES_DIR, "sample.txt")
loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()  # 返回一个列表，每个元素是一个 Document 对象

doc = docs[0]
print(f"   文档数量: {len(docs)}")
print(f"   数据来源: {doc.metadata}")
print(f"   内容长度: {len(doc.page_content)} 字符")
print(f"   前 100 字: {doc.page_content[:100]}")
print()

# =====================================================
# 2. PyPDFLoader — 读取 PDF 文件（面试常问）
# =====================================================
print("=" * 55)
print("2. PyPDFLoader — 读取 PDF 文件")
print("=" * 55)

pdf_path = os.path.join(RES_DIR, "sample.pdf")
if not os.path.exists(pdf_path):
    print("   未找到 sample.pdf，正在生成...")
    from fpdf import FPDF
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("NotoSans", "", "C:/Windows/Fonts/msyh.ttc", uni=True)
    pdf.set_font("NotoSans", size=12)
    for para in content.split("\n\n"):
        pdf.multi_cell(0, 8, para)
        pdf.ln(4)
    pdf.output(pdf_path)
    print("   已生成 sample.pdf")

loader = PyPDFLoader(pdf_path)
docs = loader.load()  # 每一页是一个 Document

print(f"   总页数: {len(docs)} 页")
for i, doc in enumerate(docs):
    print(f"   第 {i+1} 页 → 长度: {len(doc.page_content)} 字")
print()

# =====================================================
# 3. WebBaseLoader — 读取网页内容（拓展能力）
# =====================================================
print("=" * 55)
print("3. WebBaseLoader — 读取网页内容")
print("=" * 55)

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
url = "https://www.langchain.com/"
loader = WebBaseLoader(url)

try:
    docs = loader.load()
    doc = docs[0]
    print(f"   目标网址: {url}")
    print(f"   来源: {doc.metadata.get('source', '?')}")
    print(f"   标题: {doc.metadata.get('title', '?')}")
    print(f"   内容长度: {len(doc.page_content)} 字符")
    print(f"   前 150 字: {doc.page_content[:150]}")
except Exception as e:
    print(f"   加载失败: {e}")
    print("   （网络问题或反爬，面试时提一下这个局限性即可）")
print()

# =====================================================
# 4. Document 对象结构 — 核心理解
# =====================================================
print("=" * 55)
print("4. Document 对象长什么样？")
print("=" * 55)

# 重新加载 txt 作为演示
d = TextLoader(txt_path, encoding="utf-8").load()[0]

print(f"""
Document 只有 2 个属性:
┌─────────────────────────────────────────────────────┐
│  page_content : str   ← 实际的文本内容               │
│                                                      │
│  metadata     : dict  ← 来源信息（谁生的它）          │
│                  TextLoader → {{'source': '文件路径'}} │
│                  PyPDFLoader → {{'source', 'page'}}   │
│                  WebBaseLoader → {{'source', 'title'}}│
└─────────────────────────────────────────────────────┘

当前文档:
  page_content (前 50 字): {d.page_content[:50]}
  metadata: {d.metadata}
""")

print("✅ 第 1 步完成！你已经掌握了 3 种加载器和 Document 结构。")
print("   下一关: 文本切片策略 — 把长文档切成合适的块")
