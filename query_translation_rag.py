# query_translation_strategies.py

"""
📚 End-to-End RAG Query Translation Strategies using LangChain

Topics covered:
1. Multi-Query
2. RAG Fusion
3. Decomposition
4. Step-Back Querying
5. HyDE (Hypothetical Document Embeddings)

🔧 Install:
pip install -U langchain langchain-community langchain-openai langchain-huggingface sentence-transformers chromadb
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
# from langchain.retrievers.query_rewriting import LLMQueryRewriter
# from langchain_community.retrievers.hyde import HypotheticalDocumentEmbedder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.documents import Document
from langchain.schema import BaseRetriever

from pydantic import Field
from typing import List

# ----------------------------------------------------------
# STEP 1: Indexing - Load, Chunk, Embed into Chroma
# ----------------------------------------------------------
loader = TextLoader("Stages.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Total Chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="qts_rag_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------------------------------------
# STEP 2: Setup LLM
# ----------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    temperature=0.3,
    openai_api_key=os.environ["OPENAI_API_KEY"],

    openai_api_base=os.environ["OPENAI_API_BASE"],
)

# ----------------------------------------------------------
# STEP 3: Define All RAG Enhancements
# ----------------------------------------------------------

# 1️⃣ Multi-Query: Generate multiple rewordings of a query
multi_retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 2️⃣ RAG Fusion: Ensemble of retrievers (multi-query + vanilla)
rag_fusion = EnsembleRetriever(retrievers=[retriever, multi_retriever], weights=[0.5, 0.5])

# 3️⃣ Decomposition: Break complex questions into sub-queries
# 🧠 Custom prompt to break a question into sub-questions
decompose_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that decomposes complex questions into simpler sub-questions.

Original question: {question}

Sub-questions (numbered):
""")

# 🧩 LCEL Chain to rewrite the query into sub-questions
decompose_chain = (
    decompose_prompt |
    llm |
    StrOutputParser()
)

# 🔁 Custom retriever that handles multiple sub-queries
def decomposed_retriever(query: str):
    sub_questions = decompose_chain.invoke({"question": query}).split("\n")
    answers = []

    for sub_q in sub_questions:
        sub_q = sub_q.strip("-•1234567890. ")
        if sub_q:
            ans = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever).invoke({"question": sub_q})
            answers.append(f"Q: {sub_q}\nA: {ans['answer']}\nSources: {ans['sources']}\n")

    return {"answer": "\n---\n".join(answers), "sources": ""}


# 4️⃣ Step-back: Compress using query context
compressor = EmbeddingsFilter(embeddings=embeddings)
step_back = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# # 5️⃣ HyDE: Create hypothetical answers to get better matches
# hyde_embedder = HypotheticalDocumentEmbedder.from_llm(llm=llm, base_embeddings=embeddings)
# hyde_vectorstore = Chroma.from_documents(chunks, embedding=hyde_embedder, persist_directory="qts_hyde_db")
# hyde_retriever = hyde_vectorstore.as_retriever(search_kwargs={"k": 4})

# Simulate HyDE manually
def hyde_query_generator(llm, query):
    prompt = f"Write a hypothetical answer to the following question:\n\n{query}"
    return llm.invoke(prompt)

class ManualHyDERetriever(BaseRetriever):
    llm: ChatOpenAI
    vectorstore: Chroma
    embedder: HuggingFaceEmbeddings

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # 1. Generate hypothetical answer
        hyp_answer = self.llm.invoke(f"Answer this question hypothetically: {query}").content
        
        # 2. Embed hypothetical answer
        hyde_vector = self.embedder.embed_query(hyp_answer)
        
        # 3. Search vector DB
        return self.vectorstore.similarity_search_by_vector(hyde_vector, k=4)


    @property
    def lc_namespace(self) -> List[str]:
        return []

hyde_manual_retriever = ManualHyDERetriever(
    llm=llm,
    vectorstore=vectorstore,
    embedder=embeddings
)

def make_chain(r):
    return RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=r)

hyde_chain = make_chain(hyde_manual_retriever)

# ----------------------------------------------------------
# STEP 4: QA Chains for each strategy
# ----------------------------------------------------------

strategies = {
    "multi": make_chain(multi_retriever),
    "fusion": make_chain(rag_fusion),
    "stepback": make_chain(step_back),
    "hyde": make_chain(hyde_manual_retriever),
    "decompose": decomposed_retriever,
}

# ----------------------------------------------------------
# STEP 5: Ask Query and View All Variants
# ----------------------------------------------------------
print("\n🤖 Choose RAG strategy: multi, fusion, stepback, hyde")
print("(Type 'exit' to quit)")

while True:
    strategy = input("\n🧠 Strategy > ").lower()
    if strategy == "exit":
        break
    if strategy not in strategies:
        print("❌ Invalid. Try: multi, fusion, stepback, hyde, decompose")
        continue

    query = input("💬 Your Question > ")
    chain_or_func = strategies[strategy]
    
    result = (
        chain_or_func({"question": query})
        if callable(chain_or_func) else
        chain_or_func.invoke({"question": query})
    )
    
    print("\n✅ Answer:", result['answer'])
    print("📚 Sources:")
    for doc in result['sources'].split("\n"):
        print("  -", doc)








# # ============================================================
# # 🧠 FULL WORD-BY-WORD + PIPELINE + CONCEPT EXPLANATION
# # (ADVANCED RAG QUERY TRANSLATION STRATEGIES)
# # ============================================================

# # ============================================================
# # 🎯 OVERALL GOAL
# # ============================================================

# """
# This code improves RAG retrieval quality using advanced strategies:

# Normal RAG:
# Query → Retriever → LLM → Answer

# Advanced RAG:
# Query → Transform / Expand / Rewrite → Better Retrieval → Better Answer

# 👉 Goal:
# Reduce hallucination + improve retrieval accuracy
# """


# # ============================================================
# # 🧠 STEP 3: QUERY ENHANCEMENT STRATEGIES
# # ============================================================


# # ------------------------------------------------------------
# # 1️⃣ MULTI-QUERY RETRIEVER
# # ------------------------------------------------------------

# multi_retriever = MultiQueryRetriever.from_llm(
#     retriever=vectorstore.as_retriever(),
#     llm=llm
# )

# """
# WORD-BY-WORD:

# MultiQueryRetriever:
# 👉 Generates multiple versions of same query

# from_llm:
# 👉 Uses LLM to rewrite query

# retriever:
# 👉 base retriever (vector search)

# llm:
# 👉 model used to generate variations

# FLOW:
# User Query:
# "What is GST?"

# LLM generates:
# - "Explain GST"
# - "What is goods and services tax"
# - "GST definition India"

# Retriever runs ALL queries → merges results

# 🔥 BENEFIT:
# Better recall (find more relevant docs)
# """



# # ------------------------------------------------------------
# # 2️⃣ RAG FUSION (ENSEMBLE RETRIEVER)
# # ------------------------------------------------------------

# rag_fusion = EnsembleRetriever(
#     retrievers=[retriever, multi_retriever],
#     weights=[0.5, 0.5]
# )

# ------------------------------------------------------------
#  TYPES OF RAG FUSION (SHORT)
# ------------------------------------------------------------

"""
1️⃣ Simple Ensemble
→ Combine results from multiple retrievers

2️⃣ Weighted Fusion ⭐
→ Assign weights to retrievers (like your code)

3️⃣ Rank Fusion (RRF - Reciprocal Rank Fusion)
→ Combine rankings, not scores (most used in research)

4️⃣ Hybrid Fusion
→ Combine keyword (BM25) + vector search

🔥 Best:
Hybrid + RRF (production systems)
"""

# """
# WORD-BY-WORD:

# EnsembleRetriever:
# 👉 Combines multiple retrievers

# retrievers=[...]:
# 👉 list of retrievers

# weights=[0.5, 0.5]:
# 👉 importance of each retriever
# ------------------------------------------------------------
#  HOW WEIGHTS OF RETRIEVERS ARE DECIDED?
# ------------------------------------------------------------

"""
weights = [0.5, 0.5]

👉 Meaning:
- Each retriever contributes equally

How to decide:
1. Based on performance (validation set)
2. If one retriever is better → give higher weight
3. Tune using experiments

Example:
weights = [0.7, 0.3]
→ First retriever more important

🔥 Rule:
Start equal → adjust based on accuracy
"""




"""
HYBRID FUSION:
→ Combines keyword (BM25) + semantic (vector)
→ Uses weighted scoring

RRF:
→ Combines rankings (not scores)
→ Formula:
   score += 1 / (k + rank)

👉 Even lower ranked docs get some importance
👉 More robust than simple weighting



# FLOW:
# - Normal retriever → results
# - Multi-query retriever → results
# - Combine + rank

# 🔥 BENEFIT:
# More stable + accurate retrieval
# """



# # ------------------------------------------------------------
# # 3️⃣ DECOMPOSITION (BREAK QUESTION)
# # ------------------------------------------------------------

# decompose_prompt = ChatPromptTemplate.from_template("""
# You are an AI assistant that decomposes complex questions into simpler sub-questions.

# Original question: {question}

# Sub-questions (numbered):
# """)

# """
# 👉 Prompt template
# 👉 {question} = dynamic input
# """


# decompose_chain = (
#     decompose_prompt |
#     llm |
#     StrOutputParser()
# )

# """
# LCEL PIPE:

# Prompt → LLM → Parse output

# 👉 Converts:
# Complex query → multiple sub-queries
# """


# def decomposed_retriever(query: str):

#     sub_questions = decompose_chain.invoke({"question": query}).split("\n")

#     """
#     👉 Run decomposition
#     👉 split into list
#     """

#     answers = []

#     for sub_q in sub_questions:

#         sub_q = sub_q.strip("-•1234567890. ")

"""
👉 strip() removes characters from start and end

👉 Removes:
- bullet points (- •)
- numbers (1,2,3...)
- dots (.)

Example:
"1. What is GST?" → "What is GST?"

🔥 Purpose:
Clean sub-questions before processing
"""

#         if sub_q:
#             ans = RetrievalQAWithSourcesChain.from_chain_type(
#                 llm=llm,
#                 retriever=retriever
#             ).invoke({"question": sub_q})

#             """
#             👉 Run RAG for EACH sub-question
#             """

#             answers.append(
#                 f"Q: {sub_q}\nA: {ans['answer']}\nSources: {ans['sources']}\n"
#             )

#     return {
#         "answer": "\n---\n".join(answers),
#         "sources": ""
#     }

# """
# FLOW:
# Complex query → split → solve each → combine answers

# 🔥 BENEFIT:
# Handles multi-step reasoning questions
# """



# # ------------------------------------------------------------
# # 4️⃣ STEP-BACK RETRIEVER (CONTEXT COMPRESSION)
# # ------------------------------------------------------------

# compressor = EmbeddingsFilter(embeddings=embeddings)

# """
# EmbeddingsFilter:
# 👉 filters irrelevant docs using embeddings similarity
# """

# step_back = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever
# )

# """
# WORD-BY-WORD:

# ContextualCompressionRetriever:
# 👉 retrieves → compresses → returns smaller context

# base_retriever:
# 👉 original retriever

# base_compressor:
# 👉 filters content

# 🔥 BENEFIT:
# Less noise + faster LLM response
# """



# # ------------------------------------------------------------
# # 5️⃣ HyDE (HYPOTHETICAL DOCUMENT EMBEDDING)
# # ------------------------------------------------------------

# def hyde_query_generator(llm, query):
#     prompt = f"Write a hypothetical answer to the following question:\n\n{query}"
#     return llm.invoke(prompt)

# """
# 👉 Generates fake answer (hypothetical)
# """


# class ManualHyDERetriever(BaseRetriever):

#     llm: ChatOpenAI
#     vectorstore: Chroma
#     embedder: HuggingFaceEmbeddings

#     def _get_relevant_documents(self, query: str, *, run_manager=None):

#         # 1. Generate hypothetical answer
#         hyp_answer = self.llm.invoke(
#             f"Answer this question hypothetically: {query}"
#         ).content

#         """
#         👉 Step 1: fake answer
#         """

#         # 2. Embed hypothetical answer
#         hyde_vector = self.embedder.embed_query(hyp_answer)

#         """
#         👉 Step 2: convert to vector
#         """

#         # 3. Search vector DB
#         return self.vectorstore.similarity_search_by_vector(hyde_vector, k=4)

#         """
#         👉 Step 3: search using fake answer
#         """

# """
# 🔥 WHY HyDE?
# - Query may be weak
# - Hypothetical answer is richer → better embedding → better retrieval
# """



# hyde_manual_retriever = ManualHyDERetriever(
#     llm=llm,
#     vectorstore=vectorstore,
#     embedder=embeddings
# )



# # ------------------------------------------------------------
# # 🏗️ CHAIN CREATION
# # ------------------------------------------------------------

# def make_chain(r):
#     return RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         retriever=r
#     )

# """
# 👉 Wrap retriever into full RAG pipeline
# """

# hyde_chain = make_chain(hyde_manual_retriever)



# # ------------------------------------------------------------
# # 🧠 STRATEGY MAPPING
# # ------------------------------------------------------------

# strategies = {
#     "multi": make_chain(multi_retriever),
#     "fusion": make_chain(rag_fusion),
#     "stepback": make_chain(step_back),
#     "hyde": make_chain(hyde_manual_retriever),
#     "decompose": decomposed_retriever,
# }

# """
# 👉 Dictionary mapping:
# strategy name → execution logic
# """



# # ------------------------------------------------------------
# # ⚙️ EXECUTION LOGIC
# # ------------------------------------------------------------

# result = (
#     chain_or_func({"question": query})
#     if callable(chain_or_func)
#     else chain_or_func.invoke({"question": query})
# )

# """
# WORD-BY-WORD:

# callable(chain_or_func):
# 👉 checks if it's a function

# IF function:
# → call directly

# ELSE (chain):
# → use .invoke()

# 🔥 Reason:
# Some strategies are functions (decompose)
# Some are LangChain chains
# """



# # ============================================================
# # 🔄 FULL PIPELINE (IMPORTANT 🔥)
# # ============================================================

# """
# User Query
# ↓
# Strategy Selection
# ↓
# Query Transformation:
#    - Multi-query / Decompose / HyDE / etc.
# ↓
# Retriever
# ↓
# Context
# ↓
# LLM
# ↓
# Answer + Sources
# """



# # ============================================================
# # 🧠 IMPORTANT LANGCHAIN CONCEPTS
# # ============================================================

# """
# 1. Retriever:
# → fetch relevant documents

# 2. Chain:
# → pipeline of steps

# 3. RetrievalQAWithSourcesChain:
# → RAG with sources

# 4. LCEL:
# → pipeline using | operator

# 5. EnsembleRetriever:
# → combine retrievers

# 6. ContextualCompressionRetriever:
# → reduce noise

# 7. BaseRetriever:
# → custom retriever class

# 8. OutputParser:
# → clean output

# 9. PromptTemplate:
# → dynamic prompts
# """



# # ============================================================
# # 🚀 BASIC → ENTREPRENEURIAL LEVEL
# # ============================================================

# """
# 🟢 BASIC:
# - Simple retriever
# - single query

# 🟡 INTERMEDIATE:
# - Multi-query
# - better embeddings

# 🔴 ADVANCED:
# - Fusion
# - HyDE
# - Compression

# 🚀 ENTREPRENEURIAL:
# - Combine ALL strategies
# - Add evaluation
# - Add caching
# - Add monitoring

# 👉 Real systems:
# Hybrid RAG + Query Transformation + Multi-agent
# """



# # ============================================================
# # 🎯 FINAL INTERVIEW ANSWER
# # ============================================================

# """
# "This system enhances RAG by applying query transformation techniques 
# like MultiQuery, HyDE, and decomposition to improve retrieval quality, 
# reduce hallucination, and handle complex queries more effectively."
# """











# # sample code for rrf fusion and hybrid fusion:-
# HYBRID FUSION:
# → Combines keyword (BM25) + semantic (vector)
# → Uses weighted scoring

# RRF:
# → Combines rankings (not scores)
# → Formula:
#    score += 1 / (k + rank)

# 👉 Even lower ranked docs get some importance
# 👉 More robust than simple weighting

# # ------------------------------------------------------------
# # 📄 SAMPLE DOCUMENTS
# # ------------------------------------------------------------

# docs = [
#     Document(page_content="GST is a tax applied in India."),
#     Document(page_content="Machine learning is part of AI."),
#     Document(page_content="GST replaced VAT in India."),
#     Document(page_content="Deep learning is a subset of ML."),
# ]

# # ------------------------------------------------------------
# # 🔢 VECTOR RETRIEVER (SEMANTIC SEARCH)
# # ------------------------------------------------------------

# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(docs, embedding)
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # ------------------------------------------------------------
# # 🔍 BM25 RETRIEVER (KEYWORD SEARCH)
# # ------------------------------------------------------------

# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = 3
"""
BM25Retriever:
👉 Keyword-based retriever (NOT semantic)

BM25 formula (core idea):

Score(D, Q) = Σ [ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1*(1 - b + b*(|D|/avgdl))) ]

Where:
- qi = query term
- f(qi, D) = frequency of term qi in document D
- |D| = length of document
- avgdl = average document length
- k1, b = hyperparameters (usually k1=1.5, b=0.75)

👉 Intuition:
- More term matches → higher score
- Rare terms → more important (via IDF)

bm25_retriever.k = 3:
👉 return top 3 documents
"""

# # ============================================================
# # 🔥 1️⃣ HYBRID FUSION (BM25 + VECTOR)
# # ============================================================

# hybrid_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5]   # can tune later
# )

# query = "What is GST in India?"

# results = hybrid_retriever.get_relevant_documents(query)

# print("\n🔷 HYBRID FUSION RESULTS:")
# for doc in results:
#     print("-", doc.page_content)


# ------------------------------------------------------------
# 🔹 PART 2: RRF FUNCTION
# ------------------------------------------------------------
# def reciprocal_rank_fusion(results_list, k=60):
#     """
#     results_list = [list1, list2, ...]
#     Each list = ranked docs from one retriever
#     """

#     scores = {}

#     # --------------------------------------------------------
#     # LOOP OVER EACH RETRIEVER RESULT
#     # --------------------------------------------------------

#     for result in results_list:

#         # result = list of documents ranked by that retriever
#         # Example:
#         # [DocA, DocB, DocC]

#         for rank, doc in enumerate(result):

#             # rank = position in list (0-based)
#             # DocA → rank 0
#             # DocB → rank 1

#             key = doc.page_content

#             if key not in scores:
#                 scores[key] = 0

#             # ------------------------------------------------
#             # 🔥 CORE RRF FORMULA
#             # ------------------------------------------------

#             scores[key] += 1 / (k + rank + 1)

            """
            👉 RRF Formula:

            Score(D) += 1 / (k + rank + 1)

            Where:
            - D = document
            - rank = position in that retriever
            - k = constant (usually 60)

            👉 Example:
            rank = 0 → score = 1 / (60 + 0 + 1) = 1/61
            rank = 1 → score = 1 / (60 + 1 + 1) = 1/62

            👉 Lower rank → higher contribution
            👉 Higher rank → smaller contribution

            👉 Why +1?
            Avoid division by zero
            """

#     # Sort by final score
#     sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)    # 👉 Sort descending → best docs first

#     return [doc for doc, _ in sorted_docs]


# # Get results separately
# bm25_results = bm25_retriever.get_relevant_documents(query)
# vector_results = vector_retriever.get_relevant_documents(query)

# # Apply RRF
# rrf_results = reciprocal_rank_fusion([bm25_results, vector_results])

# print("\n🔶 RRF FUSION RESULTS:")
# for doc in rrf_results:
#     print("-", doc)


# ============================================================
# 🧠 FULL MATHEMATICAL UNDERSTANDING
# ============================================================

"""
👉 Suppose we have 2 retrievers:

Retriever 1:
[DocA, DocB, DocC]

Retriever 2:
[DocB, DocA, DocD]

------------------------------------------------------------
STEP 1: Assign ranks

Retriever 1:
DocA → rank 0
DocB → rank 1
DocC → rank 2

Retriever 2:
DocB → rank 0
DocA → rank 1
DocD → rank 2

------------------------------------------------------------
STEP 2: Apply RRF formula

Score(D) = Σ (1 / (k + rank + 1))

Let k = 60

DocA:
= 1/(61) + 1/(62)
≈ 0.01639 + 0.01613 = 0.03252

DocB:
= 1/(62) + 1/(61)
≈ same ≈ 0.03252

DocC:
= 1/(63) ≈ 0.01587

DocD:
= 1/(63) ≈ 0.01587

------------------------------------------------------------
STEP 3: SORT

DocA ≈ DocB > DocC ≈ DocD

👉 Final ranking:
[DocA, DocB, DocC, DocD]
"""

