# 📚 End-to-End RAG Project (Indexing ➤ Retrieval ➤ Generation)
# ----------------------------------------------------------
# 💡 Goal: Build a simple LLM app using LangChain that indexes a text file, retrieves relevant context, and answers questions based on it.

# Required Installs (in terminal):
# pip install -U langchain langchain-community langchain-openai openai chromadb sentence-transformers

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ----------------------------------------------------------
# 1️⃣ INDEXING: Load, Chunk & Embed documents into Vector DB
# ----------------------------------------------------------

# 📁 Load a plain text file
loader = TextLoader("Stages.txt")  # Place a big document in same folder
raw_docs = loader.load()

# ✂️ Split into manageable chunks with overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
doc_chunks = splitter.split_documents(raw_docs)
print(f"✅ Total chunks created: {len(doc_chunks)}")

# 🔤 Create embeddings using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 💾 Store into Chroma vector database
vectorstore = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embedding_model,
    persist_directory="chroma_rag_db"
)
vectorstore.persist()  # Saves to disk

# ----------------------------------------------------------
# 2️⃣ RETRIEVAL: Search relevant chunks using similarity search
# ----------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 relevant chunks

# ----------------------------------------------------------
# 3️⃣ GENERATION: Ask questions using LLM (via OpenRouter or OpenAI)
# ----------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"  # Replace with real key
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# 🔄 Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ----------------------------------------------------------
# ❓ INTERACTIVE Q&A LOOP
# ----------------------------------------------------------
print("\n🤖 Ask questions based on the document (type 'exit' to quit)")
while True:
    query = input("\n🧠 Your Question: ")
    if query.lower() == "exit":
        break
    result = qa.invoke({"query": query})
    print("\n💬 Answer:", result['result'])
    print("\n📚 Source Chunks:")
    for i, doc in enumerate(result['source_documents']):
        print(f"   [{i+1}]:", doc.page_content[:200], "...")

# 📁 Files to keep:
# - knowledge.txt             ➤ Input document
# - chroma_rag_db/           ➤ Auto-created vector DB
# - rag_main.py              ➤ This script

# 💡 Bonus Tips:
# - Replace `TextLoader` with `PyPDFLoader` for PDF
# - Use better embedding models for semantic understanding
# - Add LangSmith callbacks to trace query execution
