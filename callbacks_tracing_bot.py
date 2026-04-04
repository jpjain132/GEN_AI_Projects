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
# LANGSMITH_API_KEY="*****************************400d"
# LANGSMITH_PROJECT="********************ease-69"

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









# # ============================================================
# # 🧠 END-TO-END LANGCHAIN CALLBACKS + TRACING PROJECT
# # ============================================================
# #
# # 🎯 GOAL (WHAT THIS PROJECT DOES):
# # Build a QA system over documents (RAG) + add tracing + logging
# #
# # 👉 PIPELINE:
# # Raw Document → Split → Embeddings → Vector DB → Retriever
# # → LLM → RetrievalQA Chain → Callback (logging + tracing)
# # → Answer + Sources
# #
# # ============================================================



# # ============================================================
# # 📦 0. INSTALL LIBRARIES
# # ============================================================

# # langchain → orchestration framework
# # langsmith → tracing + monitoring platform
# # openai → LLM API
# # chromadb → vector database
# # sentence-transformers → embedding models

# # pip install langchain langsmith openai chromadb sentence-transformers



# # ============================================================
# # 📥 1. IMPORTS (ARCHITECTURE COMPONENTS)
# # ============================================================

# import os  # OS module → used to store environment variables (API keys)

# # Document loader → loads text files into LangChain format
# from langchain_community.document_loaders import TextLoader  

# # Splits long text into smaller chunks
# from langchain.text_splitter import RecursiveCharacterTextSplitter  

# # Embedding model → converts text → vectors
# from langchain_community.embeddings import HuggingFaceEmbeddings  

# # Vector database → stores embeddings
# from langchain_community.vectorstores import Chroma  

# # LLM wrapper → connects to model API
# from langchain_community.chat_models import ChatOpenAI  

# # Chain → combines retriever + LLM
# from langchain.chains import RetrievalQA  

# # Tracer → sends logs to LangSmith dashboard
# from langchain.callbacks.tracers.langchain import LangChainTracer  

# # Callback manager → manages logging/tracing handlers
# from langchain.callbacks.base import BaseCallbackManager  

# # Streaming handler → prints LLM output token-by-token
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  



# # ============================================================
# # 🔐 2. LANGSMITH SETUP (TRACING SYSTEM)
# # ============================================================

# # API key for LangSmith (monitoring platform)
# os.environ["LANGCHAIN_API_KEY"] = "put_your_api_key_here"

# # Project name in LangSmith dashboard
# os.environ["LANGCHAIN_PROJECT"] = "put_your_project_id_here"

# # 👉 These environment variables allow:
# # - tracing
# # - logging
# # - debugging
# # - performance monitoring

# # Create tracer object → collects execution data
# tracer = LangChainTracer()

# # Callback manager → combines multiple handlers
# callback_manager = BaseCallbackManager([
#     StreamingStdOutCallbackHandler(),  # prints live output
#     tracer                             # sends trace to LangSmith
# ])



# # ============================================================
# # 📄 3. DOCUMENT LOADING + SPLITTING
# # ============================================================

# # Load file "stages.txt"
# loader = TextLoader("stages.txt")

# # Convert file → LangChain Document objects
# docs = loader.load()

# # Split large text into chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,      # max characters per chunk
#     chunk_overlap=100    # overlap to preserve context (Each new chunk shares LAST 100 characters of previous chunk)
# )

# # Apply splitting
# chunks = splitter.split_documents(docs)

# print(f"✅ Total Chunks: {len(chunks)}")

# # 🧠 CONCEPT:
# # LLM cannot process large documents directly
# # → so we split into smaller pieces



# # ============================================================
# # 🔢 4. EMBEDDINGS + VECTOR DATABASE (CORE OF RAG)
# # ============================================================

# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

# # Convert chunks → vectors + store in Chroma DB
# vectorstore = Chroma.from_documents(
#     chunks,                  # input documents
#     embedding_model,         # embedding model
#     persist_directory="trace_chroma"  # save DB locally
# )

# # Create retriever (search engine)
# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 4}  # return top 4 relevant chunks
# )

# # 🧠 CONCEPT:
# # Embeddings = semantic meaning of text
# # Vector DB = fast similarity search
# # Retriever = fetch relevant context



# # ============================================================
# # 🤖 5. LLM SETUP (MODEL CONNECTION)
# # ============================================================

# # Set API key for LLM
# os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"

# # OpenRouter endpoint (acts as proxy to models)
# os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# # Initialize LLM
# llm = ChatOpenAI(
#     model_name="deepseek/deepseek-r1:free",  # model used
#     temperature=0.3,                        # randomness
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_api_base=os.environ["OPENAI_API_BASE"],
    
#     streaming=True,                         # stream tokens live
#     callback_manager=callback_manager,     # attach logging + tracing
#     verbose=True                           # print debug info
# )

# # 🧠 CONCEPT:
# # LLM = reasoning engine
# # Wrapper = simplifies API calls



# # ============================================================
# # 🔗 6. RETRIEVAL QA CHAIN (RAG PIPELINE)
# # ============================================================

# qa = RetrievalQA.from_chain_type(
#     llm=llm,                        # language model
#     retriever=retriever,            # retrieval system
#     return_source_documents=True,   # return sources
#     callback_manager=callback_manager,  # attach tracing
#     verbose=True
# )

# # 🧠 INTERNAL FLOW:
# """
# User Query
# ↓
# Retriever → fetch relevant chunks
# ↓
# Context + Query → LLM
# ↓
# LLM generates answer
# ↓
# Return answer + sources
# """



# # ============================================================
# # 🔄 7. INTERACTIVE LOOP (USER INTERFACE)
# # ============================================================

# print("\n💬 Ask questions based on the document (type 'exit' to quit):")

# while True:
#     query = input("\n🧠 Your Question: ")
    
#     if query.lower() == "exit":
#         break
    
#     # Invoke chain (execute pipeline)
#     result = qa.invoke({"query": query})
    
#     # Print final answer
#     print("\n🤖 Answer:", result['result'])
    
#     # Print retrieved sources
#     print("\n📚 Sources:")
#     for i, doc in enumerate(result['source_documents']):
#         print(f"   [{i+1}]", doc.page_content[:200], "...")



# # ============================================================
# # 🧠 CORE CONCEPTS USED (VERY IMPORTANT 🔥)
# # ============================================================

# """
# 1. RAG (Retrieval Augmented Generation)
#    → Combine retrieval + LLM

# 2. Embeddings
#    → Convert text → vectors (semantic meaning)

# 3. Vector Database (Chroma)
#    → Store and search embeddings

# 4. Retriever
#    → Fetch relevant chunks

# 5. LLM Wrapper
#    → Simplifies API interaction

# 6. Chains
#    → Combine multiple steps into pipeline

# 7. Callbacks
#    → Monitor execution

# 8. Tracing (LangSmith)
#    → Visualize pipeline execution

# 9. Streaming
#    → Token-by-token output

# 10. Chunking
#    → Split large text into manageable pieces
# """



# # ============================================================
# # 📊 ENTREPRENEURIAL LEVEL UNDERSTANDING
# # ============================================================

# """
# This system is a production pattern used in:

# - ChatGPT plugins
# - Customer support bots
# - Enterprise knowledge assistants
# - Legal / finance AI systems

# Stack:
# User → API → Retriever → LLM → Response → Monitoring

# LangSmith:
# - Debug errors
# - Measure latency
# - Track usage
# - Optimize prompts

# This is:
# 👉 Observability layer for AI systems
# 👉 Like logs + metrics in DevOps
# """



# # ============================================================
# # 🎯 INTERVIEW KILLER ANSWER
# # ============================================================

# """
# "I built a Retrieval-Augmented Generation pipeline using LangChain, 
# integrated with Chroma vector DB and HuggingFace embeddings, and 
# added observability using LangChain callbacks and LangSmith tracing 
# to monitor performance, latency, and execution flow."
# """








# Interview qs:-

# Q1. How are chunk size and chunk overlap determined? 
#     Provide different embedding models with their pros, cons, and rankings. 
#     Explain persist directory options, meaning of kwargs, and ideal values. 
#     Also explain API base vs API key, types of LLMs, retrievers, and vector stores 
#     along with their pros, cons, and rankings. 
#     Where can performance, latency, and execution flow be monitored?
# # ============================================================
# # 🧠 RAG + LLM SYSTEM DESIGN (Basic → Entrepreneurial Level)
# # ============================================================
# # Covers:
# # - Chunk size & overlap selection
# # - Embedding models (pros/cons + ranking)
# # - Vectorstore + retriever choices
# # - persist_directory
# # - kwargs meaning
# # - API base vs API key
# # - LLM types
# # - Monitoring (latency, performance, tracing)
# # ============================================================


# # ============================================================
# # 🧩 1. CHUNK SIZE & OVERLAP (HOW TO DECIDE)
# # ============================================================

# """
# 🟢 BASIC:
# - chunk_size = 300–800 chars
# - chunk_overlap = 50–150 chars
# - Goal: fit into model context + keep sentences intact

# 🟡 INTERMEDIATE:
# - Choose based on:
#    1. LLM context window (e.g., 4k, 8k, 32k tokens)
#    2. Document type:
#         - FAQs → small chunks (300–500)
#         - Articles → medium (500–1000)
#         - Legal/docs → larger (1000–1500)

# - Overlap rule:
#    overlap ≈ 10–20% of chunk_size

# 🔴 ENTREPRENEURIAL:
# - Dynamic chunking:
#    - semantic splitting (sentence/paragraph aware)
#    - adaptive chunk size per document type
# - Optimize using evaluation:
#    - retrieval accuracy
#    - answer relevance

# 🔥 RULE:
# Too small → lose context ❌
# Too large → irrelevant retrieval ❌
# Balanced → best RAG performance ✅
# """


# # ============================================================
# # 🔢 2. EMBEDDING MODELS (RANKING + PROS/CONS)
# # ============================================================

# """
# 🏆 TOP EMBEDDING MODELS (2025 practical ranking)

# 1️⃣ text-embedding-3-large (OpenAI)
#    ✅ Best quality
#    ❌ Paid
#    💡 Production systems

# 2️⃣ bge-large-en (BAAI)
#    ✅ High accuracy
#    ❌ Heavy (needs GPU)
#    💡 Research / advanced RAG

# 3️⃣ all-MiniLM-L6-v2 ⭐
#    ✅ Fast, lightweight
#    ❌ Lower accuracy
#    💡 Beginners / local

# 4️⃣ e5-large
#    ✅ Good semantic understanding
#    ❌ Medium speed
#    💡 enterprise search

# 5️⃣ instructor-xl
#    ✅ Instruction-aware embeddings
#    ❌ Heavy model
#    💡 advanced applications


# 🧠 SUMMARY:
# Quality ranking:
# OpenAI > BGE > E5 > MiniLM

# Speed ranking:
# MiniLM > E5 > BGE > OpenAI (API latency)
# """


# # ============================================================
# # 💾 3. VECTORSTORE OPTIONS (RANKING + PROS/CONS)
# # ============================================================

# """
# 🏆 VECTOR DB RANKING

# 1️⃣ Pinecone
#    ✅ Fully managed
#    ✅ Scalable
#    ❌ Paid
#    💡 Production SaaS

# 2️⃣ Weaviate
#    ✅ Hybrid search (vector + keyword)
#    ❌ Setup complexity
#    💡 enterprise systems

# 3️⃣ Chroma ⭐
#    ✅ Easy local setup
#    ❌ Not highly scalable
#    💡 projects / prototypes

# 4️⃣ FAISS
#    ✅ Very fast
#    ❌ No persistence by default
#    💡 research / offline

# 5️⃣ Milvus
#    ✅ High scalability
#    ❌ Complex infra
#    💡 big data systems
# """


# # ============================================================
# # 🔍 4. RETRIEVER TYPES (RANKING + USE CASE)
# # ============================================================

# """
# 1️⃣ Vector Retriever (default)
#    ✅ Fast, semantic search
#    ❌ May miss keywords

# 2️⃣ BM25 Retriever
#    ✅ Keyword-based
#    ❌ No semantic meaning

# 3️⃣ Hybrid Retriever ⭐
#    ✅ Best accuracy (vector + keyword)
#    ❌ More complex

# 4️⃣ Multi-query Retriever
#    ✅ Expands query → better recall
#    ❌ More API cost

# 5️⃣ Self-query Retriever
#    ✅ LLM generates filters
#    ❌ Complex setup
# """


# # ============================================================
# # 🤖 5. TYPES OF LLMs (RANKING + USAGE)
# # ============================================================

# """
# 🏆 LLM TYPES

# 1️⃣ API-based (OpenAI, Claude, DeepSeek)
#    ✅ Easy
#    ✅ High quality
#    ❌ Cost
#    💡 Best for most users

# 2️⃣ Open-source (LLaMA, Mistral)
#    ✅ Free
#    ❌ Needs GPU
#    💡 advanced users

# 3️⃣ Fine-tuned models
#    ✅ Domain-specific accuracy
#    ❌ Training cost
#    💡 enterprise

# 4️⃣ Small models (TinyLlama, Phi-2)
#    ✅ Lightweight
#    ❌ Limited reasoning
#    💡 edge devices
# """


# # ============================================================
# # 💽 6. persist_directory (WHAT & OPTIONS)
# # ============================================================

# """
# persist_directory = where vector DB is stored

# Options:
# - Local folder → "./chroma_db"
# - Cloud storage (S3, GCS)
# - Database-backed systems

# 🟢 BASIC:
# - local folder

# 🟡 INTERMEDIATE:
# - shared storage / server

# 🔴 ENTREPRENEURIAL:
# - managed DB (Pinecone, Weaviate)

# 💡 WHY IMPORTANT:
# - avoids recomputing embeddings
# - enables fast reload
# """


# # ============================================================
# # ⚙️ 7. kwargs (WHAT IT MEANS)
# # ============================================================

# """
# kwargs = "keyword arguments"

# Example:
# search_kwargs={"k": 4}

# Meaning:
# - k = number of results to retrieve

# 🟢 BASIC:
# k = 3–5

# 🟡 INTERMEDIATE:
# k = 5–10

# 🔴 ENTREPRENEURIAL:
# dynamic k based on query

# 💡 Too low → miss info ❌
# 💡 Too high → noise ❌
# """


# # ============================================================
# # 🔐 8. API KEY vs API BASE
# # ============================================================

# """
# API KEY:
# - Authentication (who you are)
# - Example: OpenAI key

# API BASE:
# - Endpoint URL (where request goes)
# - Example: OpenRouter / custom server

# Example:
# API_KEY → password 🔑
# API_BASE → server address 🌐
# """


# # ============================================================
# # 📊 9. MONITORING (PERFORMANCE / LATENCY / FLOW)
# # ============================================================

# """
# 🟢 BASIC:
# - print logs
# - verbose=True

# 🟡 INTERMEDIATE:
# - LangChain callbacks
# - StreamingStdOutHandler

# 🔴 ENTREPRENEURIAL:
# - LangSmith ⭐
#    → trace execution
#    → latency tracking
#    → error debugging
#    → prompt analysis

# Other tools:
# - Weights & Biases
# - OpenTelemetry
# - Datadog

# 💡 WHAT TO MONITOR:
# - latency (response time)
# - token usage (cost)
# - retrieval quality
# - error rate
# """


# # ============================================================
# # 🧠 FINAL ARCHITECTURE (FULL PIPELINE)
# # ============================================================

# """
# User Query
# ↓
# Retriever (top-k chunks)
# ↓
# Context + Query
# ↓
# LLM
# ↓
# Answer
# ↓
# Callbacks (logging + tracing)
# ↓
# Monitoring Dashboard (LangSmith)
# """


# # ============================================================
# # 🎯 INTERVIEW KILLER ANSWER
# # ============================================================

# """
# "In RAG systems, chunk size and overlap are tuned based on context window 
# and document type, embeddings determine semantic quality, vector stores 
# handle retrieval efficiency, and performance is monitored using tools 
# like LangSmith for latency and execution tracing."
# """







# Q2. Provide rankings of best free options for vector databases, embedding models, 
#     retriever types, and LLMs. 
#     How can we implement dynamic k based on query? 
#     What are free cloud storage options for persist directory? 
#     How can Google Drive be used for this? 
#     Also explain pricing of managed databases like Pinecone.
# # ============================================================
# # 🏆 RANKINGS + ARCHITECTURE DECISIONS (FREE STACK → ENTERPRISE)
# # ============================================================

# # ============================================================
# # 🟢 1. BEST FREE OPTIONS (PRACTICAL RANKINGS)
# # ============================================================

# """
# 🔹 VECTOR DATABASE (FREE)
# 1️⃣ Chroma ⭐ (BEST FREE)
#    ✅ Very easy, local persistence
#    ❌ Not scalable

# 2️⃣ FAISS
#    ✅ Very fast
#    ❌ No built-in persistence (manual)

# 3️⃣ Weaviate (self-hosted)
#    ✅ Powerful hybrid search
#    ❌ Setup heavy

# 👉 Winner (student/project): Chroma
# 👉 Winner (advanced self-host): Weaviate


# 🔹 EMBEDDING MODELS (FREE)
# 1️⃣ bge-base / bge-small (BAAI) ⭐
#    ✅ Best free quality
#    ❌ Needs GPU for speed

# 2️⃣ all-MiniLM-L6-v2
#    ✅ Very fast, lightweight
#    ❌ Lower accuracy

# 3️⃣ e5-base
#    ✅ Balanced
#    ❌ Slightly slower

# 👉 Winner: BGE (quality), MiniLM (speed)


# 🔹 RETRIEVERS
# 1️⃣ Hybrid Retriever ⭐
#    ✅ Best accuracy (semantic + keyword)

# 2️⃣ Vector Retriever
#    ✅ Simple, fast

# 3️⃣ Multi-query Retriever
#    ✅ Better recall
#    ❌ More API cost

# 👉 Winner: Hybrid (production), Vector (simple)


# 🔹 LLMs (FREE)
# 1️⃣ DeepSeek (OpenRouter free) ⭐
# 2️⃣ LLaMA (local)
# 3️⃣ Mistral (open-source)

# 👉 Winner:
# - API free → DeepSeek
# - Local → Mistral / LLaMA
# """


# # ============================================================
# # ⚙️ 2. DYNAMIC K (ADVANCED RETRIEVAL CONTROL)
# # ============================================================

# """
# Idea:
# Adjust number of retrieved docs based on query complexity
# """

# def dynamic_k(query):
#     """
#     Simple heuristic:
#     - Short/simple query → small k
#     - Complex query → larger k
#     """
#     length = len(query.split())

#     if length < 5:
#         return 3   # simple query
#     elif length < 15:
#         return 5   # medium
#     else:
#         return 8   # complex


# # Advanced version:
# """
# Use LLM to classify query complexity:
# → "simple / medium / complex"
# → map to k dynamically
# """



# # ============================================================
# # 💾 3. FREE CLOUD STORAGE FOR persist_directory
# # ============================================================

# """
# Options:
# 1️⃣ Google Drive ⭐
# 2️⃣ Dropbox (limited)
# 3️⃣ AWS S3 (free tier)
# 4️⃣ HuggingFace Hub

# 👉 BEST FREE: Google Drive
# """


# # ============================================================
# # 📂 4. USING GOOGLE DRIVE FOR VECTOR DB
# # ============================================================

# """
# In Colab:
# """

# from google.colab import drive
# drive.mount('/content/drive')

# # Save vector DB
# persist_directory = "/content/drive/MyDrive/chroma_db"

# """
# Benefits:
# ✅ Persistent storage
# ✅ Free
# ❌ Slightly slower than local disk
# """



# # ============================================================
# # 💰 5. PRICING (MANAGED VECTOR DB)
# # ============================================================

# """
# 🔹 Pinecone:
# - Free tier: limited usage
# - Paid: ~$0.096 per hour (depends on index size)

# 🔹 Weaviate Cloud:
# - Free small tier
# - Paid scaling

# 🔹 Qdrant Cloud:
# - Free tier available
# - Paid for scale

# 👉 Reality:
# - Start free → scale when needed
# """



# # ============================================================
# # 🧠 6. REAL CASE: 50,000 PAGES COMPANY DATA
# # ============================================================

# """
# Assumption:
# - 1 page ≈ 500 words
# - 50,000 pages ≈ 25M words (~huge dataset)
# """


# # ------------------------------------------------------------
# # OPTION 1: RAG (RECOMMENDED ⭐)
# # ------------------------------------------------------------

# """
# Use:
# - Chunking + embeddings + vector DB
# - Retrieve relevant info

# Pros:
# ✅ No retraining needed
# ✅ Fast updates
# ✅ Lower cost

# Use when:
# ✔ Knowledge retrieval system
# ✔ Frequently changing data
# ✔ Multiple domains (CA + math + loan)

# 👉 BEST CHOICE for your case
# """


# # ------------------------------------------------------------
# # OPTION 2: FINE-TUNING
# # ------------------------------------------------------------

# """
# Pros:
# ✅ Better style / behavior

# Cons:
# ❌ Expensive
# ❌ Not good for large knowledge storage

# Use when:
# ✔ Need specific tone
# ✔ Need structured output
# ✔ Strict format required (100% consistency)
# ✔ Domain-specific tone (legal, CA, medical)
# ✔ Repeated task at scale
# ✔ Brand voice (company style)

# 👉 NOT ideal for 50k pages knowledge base
# """


# # ------------------------------------------------------------
# # OPTION 3: LOCAL GPU TRAINING
# # ------------------------------------------------------------

# """
# Pros:
# ✅ Full control
# ❌ Very expensive infra
# ❌ Maintenance heavy

# Use when:
# ✔ Data privacy critical
# ✔ Large company infra

# 👉 NOT for small company
# """

# # option 4: api based llm
# ''' 
# use when:
# ✔ Few tone variations needed
# ✔ Structured output (JSON, tables)
# ✔ General knowledge tasks
# ✔ Fast development
# '''

# # ------------------------------------------------------------
# # FINAL DECISION (IMPORTANT)
# # ------------------------------------------------------------

# """
# For your case:

# 👉 Use:
# RAG + API-based LLM ⭐

# Stack:
# - Chroma / Pinecone
# - BGE embeddings
# - OpenAI / DeepSeek API

# Optional:
# - Fine-tune for behavior only
# """


# # ============================================================
# # 🧠 7. WHEN TO USE WHAT (CLEAR RULES)
# # ============================================================

# """
# Use RAG:
# ✔ Large documents
# ✔ Knowledge retrieval

# Use Fine-tuning:
# ✔ Behavior/style control

# Use Local LLM:
# ✔ Privacy + budget available

# Use API:
# ✔ Fast development
# ✔ No infra headache
# """


# # ============================================================
# # 🔍 8. LANGSMITH (DO YOU NEED LOGIN?)
# # ============================================================

# """
# 👉 YES, you must login

# Steps:
# 1️⃣ Go to: https://smith.langchain.com
# 2️⃣ Sign up
# 3️⃣ Create project
# 4️⃣ Get API key
# 5️⃣ Add to code:
#    os.environ["LANGCHAIN_API_KEY"]

# 6️⃣ Enable tracing:
#    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 7️⃣ Run code
# 8️⃣ Open dashboard → see logs
# """


# # ============================================================
# # 📊 WHAT YOU SEE IN LANGSMITH
# # ============================================================

# """
# - Full execution flow
# - LLM calls
# - Retriever results
# - Latency
# - Errors
# - Token usage
# """


# # ============================================================
# # 🎯 FINAL INTERVIEW ANSWER
# # ============================================================

# """
# "For large-scale document systems, I prefer RAG over fine-tuning 
# because it scales better and allows dynamic updates. I use free 
# vector DBs like Chroma for prototyping and managed services like 
# Pinecone for production. I monitor performance using LangSmith 
# for tracing and latency analysis."
# """








# Q. 2.1     Hardware requirements for :-
# # - Embeddings model rankings
# # - Vector DB rankings
# # - Retriever
# # - LLM (API vs Local)
# # - RAG pipeline
# # - Fine-tuning
# # - Monitoring (LangSmith)

# # ============================================================
# # 🧠 1. EMBEDDING MODELS
# # ============================================================

# """
# 🟢 BASIC (MiniLM, small models)
# - CPU: i5 / Ryzen 5
# - RAM: 8 GB
# - GPU: Not required ❌
# - VRAM: 0

# 👉 Works locally easily

# 🟡 INTERMEDIATE (BGE-base, E5)
# - CPU: i7 / Ryzen 7
# - RAM: 16 GB
# - GPU: Optional (4–8 GB VRAM)
# 👉 Faster with GPU

# 🔴 ADVANCED (BGE-large, Instructor-xl)
# - CPU: high-end
# - RAM: 32 GB
# - GPU: Required ✅
# - VRAM: 12–24 GB
# 👉 Needed for large embeddings
# """



# # ============================================================
# # 💾 2. VECTOR DATABASE (Chroma / FAISS / Pinecone)
# # ============================================================

# """
# 🟢 BASIC (Chroma local)
# - CPU: any
# - RAM: 8–16 GB
# - Storage: 5–20 GB

# 🟡 INTERMEDIATE (FAISS / large dataset)
# - RAM: 16–32 GB
# - Storage: 50–200 GB

# 🔴 ADVANCED (Pinecone / Weaviate)
# - Managed cloud
# - Local hardware NOT needed
# 👉 scaling handled by service
# """



# # ============================================================
# # 🔍 3. RETRIEVER
# # ============================================================

# """
# Retriever is lightweight

# 🟢 BASIC:
# - CPU only
# - negligible RAM

# 🟡 / 🔴:
# - same (depends on vector DB size)

# 👉 No major hardware requirement
# """



# # ============================================================
# # 🤖 4. LLM (MOST IMPORTANT PART)
# # ============================================================

# """
# 🔹 OPTION 1: API-based LLM (BEST for most users)

# 🟢 BASIC:
# - CPU: any
# - RAM: 4–8 GB
# - GPU: NOT needed ❌

# 👉 computation done on cloud


# 🔹 OPTION 2: LOCAL LLM

# 🟢 SMALL MODELS (TinyLlama, Phi-2)
# - RAM: 8–16 GB
# - GPU: optional
# - VRAM: 4–8 GB

# 🟡 MEDIUM MODELS (7B)
# - RAM: 16–32 GB
# - GPU: REQUIRED
# - VRAM: 12–16 GB (QLoRA)

# 🔴 LARGE MODELS (13B–70B)
# - GPU: RTX 4090 / A100
# - VRAM: 24–80 GB 🔥
# - RAM: 64+ GB

# 👉 VERY EXPENSIVE
# """



# # ============================================================
# # 🔗 5. RAG PIPELINE (FULL SYSTEM)
# # ============================================================

# """
# 🟢 BASIC (student project)
# - CPU: i5
# - RAM: 8 GB
# - GPU: NOT required
# - Storage: 10–20 GB

# 👉 Use:
# - Chroma
# - MiniLM
# - API LLM

# 🟡 INTERMEDIATE (portfolio)
# - CPU: i7
# - RAM: 16–32 GB
# - GPU: optional
# - Storage: 50–100 GB

# 👉 Use:
# - BGE embeddings
# - larger datasets

# 🔴 ENTREPRENEURIAL (production)
# - API-based → minimal local hardware
# - OR cloud infra:
#    - GPU servers
#    - distributed DB

# 👉 Use:
# - Pinecone / Weaviate
# - OpenAI / Claude APIs
# """



# # ============================================================
# # 🧬 6. FINE-TUNING (HEAVY COMPUTE 🔥)
# # ============================================================

# """
# 🟢 BASIC (Tiny models)
# - GPU: 6–8 GB VRAM
# - RAM: 16 GB

# 🟡 INTERMEDIATE (QLoRA 7B)
# - GPU: 12–16 GB VRAM
# - RAM: 32 GB

# 🔴 ADVANCED (large LLMs)
# - GPU: A100 / H100
# - VRAM: 40–80 GB
# - RAM: 64–128 GB

# 👉 Not recommended for small users
# """



# # ============================================================
# # 📊 7. MONITORING (LangSmith, logs)
# # ============================================================

# """
# 🟢 BASIC:
# - CPU only
# - minimal RAM

# 🟡 / 🔴:
# - same

# 👉 Monitoring is lightweight
# 👉 runs via API/dashboard
# """



# # ============================================================
# # 💾 8. STORAGE REQUIREMENTS
# # ============================================================

# """
# Small project:
# - 5–20 GB

# Medium (10k–50k docs):
# - 50–200 GB

# Large enterprise:
# - TB scale

# 👉 Mostly used by:
# - vector DB
# - embeddings
# """



# # ============================================================
# # 🧠 FINAL SUMMARY (VERY IMPORTANT)
# # ============================================================

# """
# Component-wise heaviness:

# Lightweight:
# - Retriever
# - Monitoring
# - API LLM

# Medium:
# - Embeddings
# - Vector DB

# Heavy:
# - Local LLM
# - Fine-tuning
# """



# # ============================================================
# # 🎯 FOR YOUR SYSTEM (IMPORTANT)
# # ============================================================

# """
# Your setup:
# - 8 GB RAM
# - AMD GPU

# 👉 Use:
# - API LLM (OpenRouter / OpenAI)
# - Chroma (local)
# - MiniLM embeddings

# 👉 DO NOT:
# - train models locally
# - run large LLMs

# 👉 BEST:
# Colab for heavy tasks
# Local for integration
# """








# Q3. Will an LLM API answer only based on my document, or also use external knowledge? 
#     I want zero hallucination and answers strictly based on my documents. 
#     Will RAG fulfill this need? 
#     Compare RAG, fine-tuning, LangChain, LLM APIs, and local LLMs in terms of use cases, 
#     scalability, and flexibility (from beginner to entrepreneurial level). 
#     What should be used as document size increases? 
#     Also include hardware requirements.
# # ============================================================
# # 🧠 NO-HALLUCINATION + ARCHITECTURE DECISION GUIDE (BASIC → ENTREPRENEURIAL)
# # ============================================================

# # ============================================================
# # ❓ 1. WILL LLM API USE ONLY MY DOCUMENT?
# # ============================================================

# """
# ❌ By default: NO
# - LLM API uses its pretrained knowledge + your prompt
# - It may add outside info → hallucination risk

# ✅ To restrict:
# - Use RAG (Retrieval Augmented Generation)
# - Force grounding in retrieved documents
# """


# # ============================================================
# # 🧠 2. CAN RAG REMOVE HALLUCINATION?
# # ============================================================

# """
# ✅ RAG reduces hallucination significantly
# BUT ❌ does NOT guarantee 100% removal

# 👉 Why?
# - LLM still "generates" text
# - It may infer or guess beyond context

# 🔥 HOW TO MAKE RAG NEAR ZERO-HALLUCINATION:

# 1. Strict Prompt:
#    "Answer ONLY from provided context. If not found, say 'Not found'."

# 2. Context Injection:
#    Only pass retrieved chunks to LLM

# 3. Low temperature:
#    temperature = 0 (reduces creativity)

# 4. Answer validation:
#    - Check if answer exists in source
#    - Reject otherwise

# 5. Return sources (very important)

# 👉 RESULT:
# ~90–95% accurate grounding
# (Industry standard)
# """


# # ============================================================
# # ⚔️ 3. RAG vs FINE-TUNING vs LLM API vs LANGCHAIN vs LOCAL LLM
# # ============================================================

# """
# 🟢 BASIC UNDERSTANDING:

# LLM API:
# - General knowledge
# - Fast, easy
# - ❌ Not document-specific

# LangChain:
# - Framework (not model)
# - Connects components (LLM + DB + tools)

# RAG:
# - Adds document knowledge
# - Dynamic retrieval
# - ✅ Best for document QA

# Fine-tuning:
# - Changes model behavior
# - ❌ Not good for storing large knowledge

# Local LLM:
# - Runs on your hardware
# - ✅ privacy
# - ❌ expensive infra
# """


# # ============================================================
# # 🧠 4. DETAILED COMPARISON (USE CASE + SCALE)
# # ============================================================

# """
# 🔹 LLM API
# Use:
# ✔ General chatbot
# ✔ Quick apps

# Pros:
# + Easy
# + Scalable
# Cons:
# - Hallucination risk
# - No custom knowledge


# 🔹 RAG ⭐ (MOST IMPORTANT)
# Use:
# ✔ Document QA
# ✔ Knowledge base systems

# Pros:
# + Dynamic updates
# + No retraining
# + Scales with data
# Cons:
# - Needs tuning


# 🔹 Fine-tuning
# Use:
# ✔ Tone control
# ✔ Structured output

# Pros:
# + Consistency
# Cons:
# - Expensive
# - Not for large knowledge


# 🔹 LangChain
# Use:
# ✔ Orchestration

# Pros:
# + Modular
# + Connect everything
# Cons:
# - Adds complexity


# 🔹 Local LLM
# Use:
# ✔ Privacy-critical systems

# Pros:
# + No API cost
# Cons:
# - Heavy hardware
# - Maintenance
# """


# # ============================================================
# # 🧠 5. WHICH IS BEST FOR YOUR GOAL (NO HALLUCINATION)?
# # ============================================================

# """
# 👉 BEST STACK:

# RAG + API LLM + STRICT PROMPT ⭐

# Optional:
# + Output validation
# + Source citation

# 👉 NOT:
# - Fine-tuning alone ❌
# - Raw LLM API ❌
# """


# # ============================================================
# # 📈 6. SCALABILITY (WHEN DATA INCREASES)
# # ============================================================

# """
# Small (1k–10k docs):
# - Chroma / FAISS
# - MiniLM embeddings
# - API LLM

# Medium (10k–100k docs):
# - FAISS / Weaviate
# - BGE embeddings
# - API LLM

# Large (100k+ docs / millions):
# - Pinecone / Milvus
# - distributed DB
# - caching + indexing

# 👉 Key scaling factor:
# Vector DB, not LLM
# """


# # ============================================================
# # 💻 7. HARDWARE REQUIREMENTS (SCALING)
# # ============================================================

# """
# 🟢 BASIC:
# - RAM: 8 GB
# - CPU: normal
# - GPU: not needed

# 🟡 INTERMEDIATE:
# - RAM: 16–32 GB
# - Storage: 50–200 GB
# - GPU: optional

# 🔴 ENTREPRENEURIAL:
# - Cloud infra
# - Vector DB managed
# - API LLM
# - GPU only if self-hosting

# 👉 Most companies:
# NO local GPU ❌
# Use APIs + cloud ✅
# """


# # ============================================================
# # 🧠 8. FINAL DECISION TREE
# # ============================================================

# """
# IF goal = document QA:
# → Use RAG ⭐

# IF goal = tone/style:
# → Fine-tuning

# IF goal = privacy:
# → Local LLM

# IF goal = fast build:
# → LLM API

# IF goal = production:
# → RAG + API + Vector DB + Monitoring
# """


# # ============================================================
# # 🎯 INTERVIEW KILLER ANSWER
# # ============================================================

# """
# "To minimize hallucination, I use RAG with strict prompting and source validation, 
# ensuring the LLM answers only from retrieved documents. For scalability, I rely 
# on vector databases like Pinecone and API-based LLMs rather than fine-tuning, 
# which is not suitable for large knowledge bases."
# """
