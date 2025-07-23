# ✅ End-to-End Context-Aware RAG Chatbot with Memory (LangChain + Chroma + DeepSeek)
# -----------------------------------------------------------------------------
# Goal: Build a chatbot that answers from your own knowledge base with memory.
# 📚 Vector DB: Chroma
# 🧠 Embeddings: HuggingFace (MiniLM)
# 🤖 LLM: DeepSeek Chat via OpenRouter
# 🧵 Memory: ConversationBufferMemory
# 📦 Install dependencies:
# pip install langchain langchain-community langchain-openai chromadb sentence-transformers openai tiktoken

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 🔐 Set your OpenRouter API Key
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"  # Replace this with your actual key

# 📁 1. Load your text file
loader = TextLoader("ol_dump_deletes_2025-06-30.txt")  # Add your custom document here
raw_docs = loader.load()

# ✂️ 2. Split into overlapping chunks (to preserve context)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(raw_docs)
print(f"✅ Total Chunks Created: {len(chunks)}")

# 🧠 3. Create vector embeddings using HuggingFace MiniLM
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 💾 4. Store vectors in persistent Chroma DB
persist_dir = "chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)
vectorstore.persist()

# 🔍 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Top 4 relevant chunks

# 💬 6. Memory: Stores chat history so bot remembers context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🤖 7. LLM via OpenRouter (DeepSeek)
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",  # You can also try "deepseek-coder"
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# 🔄 8. Build the RAG Conversational Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,                      # Context-aware memory
    return_source_documents=True       # So you know which chunks were used
)

# ❓ 9. Ask questions interactively
print("\n🧠 Ask your questions about the document (type 'exit' to quit):")
while True:
    query = input("\n💬 You: ")
    if query.strip().lower() == "exit":
        break
    result = qa_chain.invoke({"question": query})
    print("\n🤖 Bot:", result["answer"])
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print("   •", doc.metadata.get("source", "Local File"))

