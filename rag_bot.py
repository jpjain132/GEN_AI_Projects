# ✅ End-to-End RAG (Retrieval-Augmented Generation) Project
# -------------------------------------------------------------
# Goal: Build a QA chatbot that answers questions based on a large document using:
# 1. HuggingFace embeddings
# 2. FAISS for vector store
# 3. LangChain RAG pipeline
#
# 🔧 Steps:
# - Load a large `.txt` document
# - Split it into chunks
# - Embed those chunks using HuggingFace
# - Store them in FAISS (in-memory vector DB)
# - Use a retriever to fetch relevant chunks based on a user query
# - Feed retrieved chunks + query to an LLM (OpenRouter's DeepSeek)

# 🔽 Install dependencies (run in terminal):
# pip install langchain langchain-community langchainhub langchain-openai sentence-transformers faiss-cpu huggingface-hub openai

# 🔽 Imports
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 📁 1. Load your document (plain text)
loader = TextLoader("Stages.txt")  # Make sure the file is in the same folder
docs = loader.load()

# ✂️ 2. Split document into smaller overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,         # each chunk's max character length
    chunk_overlap=100       # overlap between adjacent chunks
)
chunks = splitter.split_documents(docs)
print(f"✅ Total Chunks Created: {len(chunks)}")

# 🔤 3. Create embeddings using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🧠 4. Store embeddings into FAISS Vector Store
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(
    search_type="similarity",  # You can also try 'mmr' (Max Marginal Relevance)
    search_kwargs={"k": 4}
)

# 🤖 5. Set OpenRouter API for DeepSeek
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"  # Replace with your actual key

llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",  # Correct model name for OpenRouter
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3,
)

# 🔄 6. Setup RetrievalQA chain with source docs
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ❓ 7. Interactive loop
print("\n🔍 Ask questions based on the document (type 'exit' to quit)")
while True:
    query = input("\n🧠 Your Question: ")
    if query.lower() == "exit":
        break

    result = qa.invoke({"query": query})  # 💡 KEY FIX: use 'query' here
    print("\n💬 Answer:", result.get('result', '[No Answer Returned]'))

    # 📚 Print source docs
    print("\n📚 Source Documents:")
    for i, doc in enumerate(result.get('source_documents', [])):
        print(f"   [{i+1}] {doc.page_content[:200]} ...")


        
# 📁 FILES:
# - my_large_doc.txt : your input document
# - this Python file (e.g. rag_bot.py)

# 💡 TIPS:
# - For PDFs: use PyPDFLoader instead of TextLoader
# - For large document sets: consider Chroma or Qdrant (disk-backed DBs)
# - Try different HuggingFace models for better performance or accuracy
