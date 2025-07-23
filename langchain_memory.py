# ✅ LangChain Memory Systems Project
# ----------------------------------------------------------------------------
# Goal: Understand and implement three memory types:
# - ConversationBufferMemory (stores raw chat history)
# - ConversationSummaryMemory (stores summarized history)
# - Token-aware memory (with custom logic)
#
# We'll use a simple chatbot powered by LangChain and OpenRouter (DeepSeek).
# Each memory mode will run independently to compare behaviors.

# 🔽 Required Installs:
# pip install langchain langchain-openai langchain-community openai

import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
# from langchain_community.memory import ConversationBufferMemory, ConversationSummaryMemory

# 🧠 Select LLM (we use DeepSeek via OpenRouter)
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3
)

# -----------------------------------------------------------------------------
# 🔹 1. ConversationBufferMemory (raw history)
buffer_memory = ConversationBufferMemory()
buffer_chain = ConversationChain(llm=llm, memory=buffer_memory, verbose=True)

# -----------------------------------------------------------------------------
# 🔹 2. ConversationSummaryMemory (auto-summarized history)
summary_memory = ConversationSummaryMemory(llm=llm)
summary_chain = ConversationChain(llm=llm, memory=summary_memory, verbose=True)

# -----------------------------------------------------------------------------
# # 🔹 3. ConversationTokenBufferMemory (token-aware memory)
token_buffer_memory = ConversationTokenBufferMemory(
    llm=llm, max_token_limit=200
)
token_buffer_chain = ConversationChain(llm=llm, memory=token_buffer_memory, verbose=True)

# -----------------------------------------------------------------------------
# 🔄 Function to run a conversation for each memory type
def run_conversation(chain, name):
    print(f"\n========== {name} ==========")
    inputs = [
        "Hello, I am building a memory system chatbot.",
        "Can you remember what I just said?",
        "Summarize our chat so far.",
        "Now let's talk about your favorite programming language."
    ]
    for question in inputs:
        response = chain.predict(input=question)
        print(f"\n🧠 You: {question}\n🤖 Bot: {response}")

# Run each type
run_conversation(buffer_chain, "ConversationBufferMemory")
run_conversation(summary_chain, "ConversationSummaryMemory")
run_conversation(token_buffer_chain, "TokenAwareMemory")

# 📁 FILES TO KEEP:
# - langchain_memory_demo.py  ← This file

# 💡 Tips:
# - BufferMemory stores everything, so responses can get lengthy.
# - SummaryMemory uses the LLM to summarize the chat history.
# - Token-aware memory limits context size to avoid LLM cutoff. (It’s the latest point in time up to which the model has been trained on data. It doesn't know about events or information after that date.)
# - You can inspect `.memory.buffer` or `.memory.chat_memory` to debug.

# ✅ You now understand LangChain memory systems!
