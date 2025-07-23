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
