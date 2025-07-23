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

