# lcel_rag_bot.py

import os
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------
# 1. Load and chunk a text file
# --------------------------------------

loader = TextLoader("Stages.txt")  # Keep your file here
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Total Chunks: {len(chunks)}")

# --------------------------------------
# 2. Embedding and Chroma vector DB
# --------------------------------------

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_lcel_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --------------------------------------
# 3. Setup OpenRouter key and DeepSeek model
# --------------------------------------

os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",     
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# --------------------------------------
# 4. LangChain Expression Language (LCEL) Chain
# --------------------------------------

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")

# LCEL chain components
retrieval = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))
# retrieval
# 👉 variable name → this step will be used in LCEL pipeline

# RunnableLambda
# 👉 converts a Python function into a LangChain runnable (pipeline step)

# lambda x:
# 👉 anonymous function
# 👉 x = input dictionary (e.g., {"question": "What is GST?"})

# x["question"]
# 👉 extracts the "question" value from input

# retriever.get_relevant_documents(...)
# 👉 searches vector DB
# 👉 returns top-k relevant chunks (documents)

# FULL MEANING:
# 👉 Take input → extract question → retrieve relevant docs → return them

# INPUT:
# {"question": "What is GST?"}

# OUTPUT:
# [Document1, Document2, Document3

format_prompt = prompt
call_llm = llm
parse_output = StrOutputParser()

# LCEL PIPE CHAIN
rag_chain = (
    {"context": retrieval, "question": RunnablePassthrough()} |
    format_prompt |
    call_llm |
    parse_output
)

# --------------------------------------
# 5. Q&A Interactive Loop
# --------------------------------------

print("\n💬 Ask questions based on the document (type 'exit' to stop):")
while True:
    query = input("\n🧠 Your Question: ")
    if query.lower() == "exit":
        break
    response = rag_chain.invoke({"question": query})
    print("\n🤖 Answer:", response)










# ============================================================
# 🧠 FULL WORD-BY-WORD + PIPELINE + CONCEPT EXPLANATION (LCEL RAG)
# ============================================================

# This file: lcel_rag_bot.py
# 👉 Goal:
# Build a Retrieval-Augmented Generation (RAG) chatbot using LCEL (LangChain Expression Language)

# ============================================================
# 📦 IMPORTS (WHAT EACH LINE MEANS)
# ============================================================

import os  
# 👉 OS module → used to store API keys securely (environment variables)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough  
# 👉 RunnableLambda:
#    - Wraps custom Python function into LCEL pipeline
# 👉 RunnablePassthrough:
#    - Passes input unchanged (identity function)

from langchain_community.document_loaders import TextLoader  
# 👉 Loads text file → converts into LangChain "Document" objects

from langchain.text_splitter import RecursiveCharacterTextSplitter  
# 👉 Splits large text → smaller chunks (important for LLM limits)

from langchain_community.embeddings import HuggingFaceEmbeddings  
# 👉 Converts text → vectors (numerical representation)

from langchain_community.vectorstores import Chroma  
# 👉 Vector database → stores embeddings for similarity search

from langchain_community.chat_models import ChatOpenAI  
# 👉 LLM wrapper → connects to API (OpenAI / OpenRouter)

from langchain_core.prompts import ChatPromptTemplate  
# 👉 Template to structure prompt dynamically

from langchain_core.output_parsers import StrOutputParser  
# 👉 Converts LLM output → clean string



# ============================================================
# 📄 1. LOAD + CHUNK DOCUMENT
# ============================================================

loader = TextLoader("Stages.txt")  
# 👉 Reads file "Stages.txt"

docs = loader.load()  
# 👉 Converts file → list of Document objects

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # max characters per chunk
    chunk_overlap=100    # overlap between chunks
)

chunks = splitter.split_documents(docs)  
# 👉 Splits documents → smaller pieces

print(f"✅ Total Chunks: {len(chunks)}")

# 🧠 CONCEPT:
"""
Why chunking?
- LLM cannot process huge documents
- Splitting improves retrieval accuracy

Overlap:
- preserves context across chunks
"""



# ============================================================
# 🔢 2. EMBEDDINGS + VECTOR DB
# ============================================================

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 👉 Loads embedding model (fast + lightweight)

vectorstore = Chroma.from_documents(
    documents=chunks,                 # input data
    embedding=embedding_model,        # convert to vectors
    persist_directory="chroma_lcel_db" # save DB locally
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# 👉 Retriever = search engine
# 👉 k=3 → return top 3 relevant chunks

# 🧠 CONCEPT:
"""
Embeddings:
→ convert text → numbers (semantic meaning)

Vector DB:
→ stores embeddings

Retriever:
→ finds similar chunks based on query
"""



# ============================================================
# 🤖 3. LLM SETUP
# ============================================================

os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",     
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# 🧠 CONCEPT:
"""
LLM:
→ reasoning engine

temperature:
→ controls randomness
"""


# ============================================================
# 🧠 4. LCEL (LANGCHAIN EXPRESSION LANGUAGE)
# ============================================================

"""
LCEL = declarative way to build pipelines

Instead of:
step1 → step2 → step3

We write:
A | B | C

👉 Pipeline operator: |
"""

# ------------------------------------------------------------
# PROMPT TEMPLATE
# ------------------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")

# 👉 {context}, {question} are dynamic placeholders



# ------------------------------------------------------------
# LCEL COMPONENTS
# ------------------------------------------------------------

retrieval = RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"]))
# 👉 Takes input dict → extracts "question"
# 👉 Calls retriever → returns relevant docs

format_prompt = prompt  
# 👉 formats final prompt

call_llm = llm  
# 👉 sends prompt to LLM

parse_output = StrOutputParser()  
# 👉 converts output → plain string



# ------------------------------------------------------------
# 🔗 LCEL PIPELINE (CORE LOGIC)
# ------------------------------------------------------------

rag_chain = (
    {
        "context": retrieval,                 # get context
        "question": RunnablePassthrough()     # pass question unchanged
    }
    |
    format_prompt
    |
    call_llm
    |
    parse_output
)

# 🧠 FLOW:
"""
Input:
{"question": "What is GST?"}

Step 1:
context = retrieval(question)

Step 2:
format prompt with context + question

Step 3:
LLM generates answer

Step 4:
output parsed as string
"""



# ============================================================
# 🔄 5. INTERACTIVE LOOP
# ============================================================

print("\n💬 Ask questions based on the document (type 'exit' to stop):")

while True:
    query = input("\n🧠 Your Question: ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke({"question": query})
    # 👉 invoke() runs entire LCEL pipeline

    print("\n🤖 Answer:", response)



# ============================================================
# 🔄 FULL PIPELINE (IMPORTANT 🔥)
# ============================================================

"""
User Query
↓
Retriever (top 3 chunks)
↓
Context + Question
↓
Prompt Template
↓
LLM
↓
Parsed Output
↓
Final Answer
"""



# ============================================================
# 🧠 KEY LANGCHAIN + LCEL CONCEPTS
# ============================================================

"""
1. RAG (Retrieval Augmented Generation)
   → Combines retrieval + generation

2. Runnable
   → Basic unit in LCEL

3. RunnableLambda
   → custom function inside pipeline

4. RunnablePassthrough
   → passes input unchanged

5. Pipe Operator (|)
   → connects components

6. PromptTemplate
   → dynamic prompt building

7. OutputParser
   → formats LLM output

8. Vector DB
   → semantic search

9. Retriever
   → fetch relevant data
"""



# ============================================================
# 🚀 BASIC → ENTREPRENEURIAL UNDERSTANDING
# ============================================================

"""
🟢 BASIC:
- Build simple RAG
- Use Chroma + MiniLM + API

🟡 INTERMEDIATE:
- Add better embeddings
- Optimize chunking
- Add evaluation

🔴 ENTREPRENEURIAL:
- Use Pinecone / Weaviate
- Add caching
- Add monitoring (LangSmith)
- Use multi-agent (LangGraph)

👉 Real stack:
RAG + LCEL + Vector DB + API LLM
"""



# ============================================================
# 🎯 INTERVIEW KILLER ANSWER
# ============================================================

"""
"This system uses LCEL to build a declarative RAG pipeline where 
retrieval, prompt formatting, LLM inference, and output parsing 
are composed using runnables, enabling modular and scalable AI workflows."
"""
