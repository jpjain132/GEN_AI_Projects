# # 🔽 Import necessary modules
# from langchain.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI

# import os

# # 🔐 Set your API key (use OpenRouter below if you don’t have OpenAI)
# os.environ["OPENRUTER_API_KEY"] = "put_your_api_key_here"

# #step 1
# def load_txt(path):
#     """Load plain text file using TextLoader"""
#     return TextLoader(path).load()


# # 📝 Choose what to load
# docs = load_txt("stages.txt")  # Now loads your .txt file correctly

# # 📑 Step 2: Split the docs into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,       # Each chunk will be 500 characters
#     chunk_overlap=50      # Some overlap so we don't lose context
# )

# split_docs = text_splitter.split_documents(docs)
# print(f"✅ Total Chunks: {len(split_docs)}")

# # 🔍 Step 3: Embed and store in FAISS (in-memory vector DB)
# embedding = OpenAIEmbeddings()  # Uses OpenAI API to generate embeddings
# vectordb = FAISS.from_documents(split_docs, embedding)

# # 🔄 Step 4: Setup retriever
# retriever = vectordb.as_retriever()

# # 🧠 Step 5: Setup QA chain
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # You can change this to gpt-4 or OpenRouter version

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True  # Optional: returns where the answer came from
# )

# # ❓ Step 6: Ask a question!
# while True:
#     query = input("\nAsk your question (or type 'exit'): ")
#     if query.lower() == 'exit':
#         break
#     result = qa_chain(query)
#     print("\n💬 Answer:", result["result"])
    
#     # Optional: print source
#     print("📚 Sources:")
#     for doc in result["source_documents"]:
#         print(f" - {doc.metadata}")









# 🔽 Import necessary modules
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Using free embeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# 📄 Step 1: Load your text document
loader = PyPDFLoader("pkpadmin,+529-2711-1-CE_100MB.pdf")  # Replace with your file path
docs = loader.load()

# ✂️ Step 2: Split the document into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"✅ Total Chunks: {len(split_docs)}")

# 🧠 Step 3: Embed the chunks using free HuggingFace embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # lightweight, fast
vectordb = FAISS.from_documents(split_docs, embedding)

# 🔍 Step 4: Set up the retriever
retriever = vectordb.as_retriever()

# 💬 Step 5: Initialize DeepSeek LLM via OpenRouter
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",  # You can also try "deepseek-coder"
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="put_your_api_key_here",
    temperature=0.7,
)

# 🔗 Step 6: Connect retriever and LLM into a QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ❓ Step 7: Ask your questions
while True:
    query = input("\n📝 Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa(query)
    print("\n💬 Answer:", result['result'])

    # 🧾 Optional: Show source chunks
    print("📚 Source(s):")
    for doc in result["source_documents"]:
        print(f"  • {doc.metadata.get('source', 'Unknown Source')}")

