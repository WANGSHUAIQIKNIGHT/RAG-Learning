"""
第 5 步：生成 + 对话记忆
目标：把检索结果交给 LLM 生成回答，并让多轮对话有记忆
"""

import os

# 强制离线模式，HuggingFace 模型已在本地缓存
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatTongyi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(BASE_DIR, "resourse", "sample.txt")

# ============================================================
# 准备数据：加载 → 切片 → 向量化
# ============================================================
print("=" * 60)
print("准备数据")
print("=" * 60)
loader = TextLoader(txt_path, encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
print(f"   准备就绪：{len(chunks)} 块，{vector_store.index.ntotal} 个向量\n")

# ============================================================
# 1. 不带记忆的单轮 RAG
# ============================================================
print("=" * 60)
print("1. 单轮 RAG — 检索 + LLM 生成")
print("=" * 60)

llm = ChatTongyi(model="qwen-turbo", temperature=0)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手。根据检索到的上下文回答问题。如果上下文里没有相关信息，就说不知道。\n\n上下文：\n{context}"),
    ("human", "{input}"),
])

def format_docs(docs):
    """把检索到的多块文档合并成一段文本"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "RAG能解决什么问题？"
print(f"   问题: {question}")
answer = rag_chain.invoke(question)
print(f"   回答: {answer}\n")

# ============================================================
# 2. 加上对话记忆 — 多轮 RAG
# ============================================================
print("=" * 60)
print("2. 多轮 RAG — 带对话记忆")
print("=" * 60)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
)

# 第一轮
r1 = conversation.invoke({"question": "什么是RAG？"})
print(f"   问: 什么是RAG？")
print(f"   答: {r1['answer']}\n")

# 第二轮（依赖上一轮的上下文）
r2 = conversation.invoke({"question": "它有什么作用？"})
print(f"   问: 它有什么作用？")
print(f"   答: {r2['answer']}\n")

# 第三轮
r3 = conversation.invoke({"question": "Agent和RAG是什么关系？"})
print(f"   问: Agent和RAG是什么关系？")
print(f"   答: {r3['answer']}\n")

# ============================================================
# 3. 看看记忆里存了什么
# ============================================================
print("=" * 60)
print("3. 查看对话记忆内容")
print("=" * 60)
print(f"记忆中的历史消息数: {len(memory.chat_memory.messages)}")
for msg in memory.chat_memory.messages:
    print(f"   [{msg.type}] {msg.content[:80]}...")

print()
print("[完成] 第 5 步完成！你实现了完整 RAG 对话。")
print("   下一关: 完整管线 + 评估 — 把各步骤整合并评价效果")
