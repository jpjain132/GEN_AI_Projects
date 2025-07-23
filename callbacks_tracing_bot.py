# ✅ End-to-End LangChain Callbacks + Tracing Project
# ------------------------------------------------------------
# Goal: Add logging, tracing, and performance tracking to a QA bot
# using LangChain's Callback system and LangSmith for visualization.

# 🔧 Install Requirements:
# pip install langchain langsmith openai chromadb sentence-transformers

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ------------------------------------------------------------
# 1. Setup LangSmith environment for trace logging (OPTIONAL)
# ------------------------------------------------------------
# You must create an account at https://smith.langchain.com
# Get your API Key and project name from there.

os.environ["LANGCHAIN_API_KEY"] = "put_your_api_key_here"
os.environ["LANGCHAIN_PROJECT"] = "put_your_project_id_here"

# LANGSMITH_TRACING="true"
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY="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"
# LANGSMITH_PROJECT="pr-plaintive-increase-69"

# LangSmith tracer for visualization
tracer = LangChainTracer()
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler(), tracer])

# ------------------------------------------------------------
# 2. Load and split document
# ------------------------------------------------------------
loader = TextLoader("stages.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Total Chunks: {len(chunks)}")

# ------------------------------------------------------------
# 3. Embedding and vector store (Chroma)
# ------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="trace_chroma")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------------------------------------------
# 4. Language Model (OpenAI or OpenRouter)
# ------------------------------------------------------------
# 🔐 Set your OpenAI API Key (or use OpenRouter)
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    temperature=0.3,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    streaming=True,
    callback_manager=callback_manager,  # Attach tracing + console logging
    verbose=True
)

# ------------------------------------------------------------
# 5. RetrievalQA Chain with Callback Manager
# ------------------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    callback_manager=callback_manager,
    verbose=True
)

# ------------------------------------------------------------
# 6. Ask questions interactively
# ------------------------------------------------------------
print("\n💬 Ask questions based on the document (type 'exit' to quit):")
while True:
    query = input("\n🧠 Your Question: ")
    if query.lower() == "exit":
        break
    result = qa.invoke({"query": query})
    print("\n🤖 Answer:", result['result'])
    print("\n📚 Sources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"   [{i+1}]", doc.page_content[:200], "...")

# ------------------------------------------------------------
# 🔍 What This Project Shows:
# ------------------------------------------------------------
# - Console output of LLM and retriever steps via StreamingStdOutHandler
# - Tracing of chain events inside your LangSmith dashboard
# - Error tracking, prompt latency, and performance in LangSmith UI
#
# 📍 To view traces, go to https://smith.langchain.com -> Your Project
# 📁 Files:
# - stages.txt : Your input document
# - trace_chroma/ : Local Chroma DB
# - callback_tracing_bot.py : This script
