# ✅ End-to-End Project: Custom Tool Creation + Agent Orchestration (LangChain)
# ----------------------------------------------------------------------------------
# Goal: Create your own tools (functions), connect them via a custom agent that decides
#       which tool to use based on user input (question), and lets the LLM orchestrate logic.

# 🔧 Tools We'll Create:
# - Calculator Tool (math)
# - Weather Tool (mocked)
# - Joke Generator Tool

# 📦 Dependencies:
# pip install langchain langchain-core langchain-community openai

import os
from langchain_core.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI

# ----------------------------------------------------------------------
# 1. Define Custom Tools (as simple Python functions with docstrings)
# ----------------------------------------------------------------------

def calculator_tool(input_str: str) -> str:
    """Evaluates a math expression. Example: '2 + 2 * 3'"""
    try:
        return str(eval(input_str))
    except Exception as e:
        return f"Error: {str(e)}"

def weather_tool(city: str) -> str:
    """Returns fake weather report for any city (mocked)."""
    return f"Today's weather in {city.title()} is 28°C with mild clouds. ☁️"

def joke_tool(_: str = "") -> str:
    """Tells a random programming joke."""
    return "Why do programmers prefer dark mode? Because light attracts bugs."

# ----------------------------------------------------------------------
# 2. Wrap Python functions as LangChain Tools
# ----------------------------------------------------------------------

my_tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving math expressions. Input should be like: 3*5+2"
    ),
    Tool(
        name="Weather",
        func=weather_tool,
        description="Gives current weather info. Input should be a city name."
    ),
    Tool(
        name="Joke",
        func=joke_tool,
        description="Use this to make user laugh with a programming joke."
    )
]

# ----------------------------------------------------------------------
# 3. Setup LLM (DeepSeek via OpenRouter or any OpenAI model)
# ----------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = "your_api_key"      
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# ----------------------------------------------------------------------
# 4. Create Agent to decide which tool to use
# ----------------------------------------------------------------------

agent_executor = initialize_agent(
    tools=my_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ----------------------------------------------------------------------
# 5. Run the Agent in Loop: Ask Anything!
# ----------------------------------------------------------------------
print("\n💡 Ask anything (math, weather, joke). Type 'exit' to quit.")
while True:
    query = input("\n🧠 Your Question: ")
    if query.lower() == "exit":
        break
    answer = agent_executor.run(query)
    print("\n🤖 Answer:", answer)

# ----------------------------------------------------------------------
# 📁 FILES NEEDED:
# - custom_tool_agent.py      (this script)
# - No external DB/files needed

# ----------------------------------------------------------------------
# 💡 How It Works:
# - You ask something like "What's the weather in Jaipur?"








# further deep expl:-    
# ✅ Install libraries (LangChain ecosystem)
# langchain_core = core abstractions (tools, prompts)

# langchain = agents + orchestration logic; Orchestration (task management) = managing and coordinating multiple components (LLM + tools + steps) to complete a task automatically. 
# ( User asks:
# “Calculate 5×3 and tell weather in Jaipur”
# Orchestration does:
# 1. Understand query
# 2. Call Calculator tool → 15
# 3. Call Weather tool → Jaipur weather
# 4. Combine results
# 5. Return final answer )

# langchain_community = integrations (LLMs, APIs)
# openai = API client
# !pip install langchain langchain-core langchain-community openai


# -------------------------------------------------------------
# 🔑 IMPORTS (Understanding Architecture Layers)
# -------------------------------------------------------------

# import os  # used to store API keys securely

# # Tool abstraction = wraps normal Python functions into "LLM-callable tools"
# from langchain_core.tools import Tool  

# # Agent initialization logic (brain that decides which tool to use)
# from langchain.agents import initialize_agent  

# # Type of reasoning strategy used by agent: AgentType
# from langchain.agents.agent_types import AgentType  

# # Chat-based LLM wrapper (connects to OpenAI / OpenRouter models) # LLM wrapper = a layer of code that lets you easily interact with an LLM (API/model) using a simple interface (  eg. llm = ChatOpenAI(...)  )
# from langchain_community.chat_models import ChatOpenAI  


# # -------------------------------------------------------------
# # 🧠 1. DEFINE CUSTOM TOOLS (FUNCTIONS = ACTIONS)
# # -------------------------------------------------------------

# # 🔹 TOOL 1: Calculator
# def calculator_tool(input_str: str) -> str:
#     """
#     Docstring = VERY IMPORTANT. Docstring:👉 A docstring is a string written inside a function/class to explain what it does.👉 It helps humans and tools (like LLMs) understand the purpose and usage of the code.
#     👉 LLM reads this to understand when to use this tool
#     👉 This acts like "instruction manual" for agent
# Note: Comments are for developers and ignored by Python, while      Docstrings are used for documentation and can be accessed programmatically.
# Cases:-
# 1. 1st line in fun: docstring. 
# 2. anywhere else: just string, not docstring. 
# 3. # → real comment ✅
# 4. ''' ''' → string ❌ (not comment)
# 5. """ """ → docstring or string   

#     input_str: string input from user (math expression)
#     return: string output (result)
#     """
#     try:
#         # eval() executes math expression dynamically
#         # Example: "2+3*4" → 14
#         return str(eval(input_str))
#     except Exception as e:
#         # Handle errors safely
#         return f"Error: {str(e)}"
    

# # 🔹 TOOL 2: Weather (Mocked API)
# def weather_tool(city: str) -> str:
#     """
#     👉 This simulates an API call (like real weather API)
#     👉 In production: you'd call OpenWeatherMap API
    
#     city: input location
#     """
#     return f"Today's weather in {city.title()} is 28°C with mild clouds. ☁️"


# # 🔹 TOOL 3: Joke Generator
# def joke_tool(_: str = "") -> str:
#     """
#     👉 '_' means input not needed
#     👉 Tool ignores input but still follows interface
    
#     Use-case: entertainment / UX improvement
#     """
#     return "Why do programmers prefer dark mode? Because light attracts bugs."


# # -------------------------------------------------------------
# # 🧩 2. TOOL WRAPPING (CRITICAL LANGCHAIN CONCEPT)
# # -------------------------------------------------------------

# # Tool() converts normal functions → structured tool objects
# # This is required so LLM can "understand and call them"

# my_tools = [
#     Tool(
#         name="Calculator",  # tool name (LLM uses this internally)
#         func=calculator_tool,  # function to execute
#         description="Useful for solving math expressions. Input should be like: 3*5+2"
#         # 🔥 description is MOST IMPORTANT
#         # LLM uses this to decide:
#         # "Should I use this tool?"
#     ),
    
#     Tool(
#         name="Weather",
#         func=weather_tool,
#         description="Gives current weather info. Input should be a city name."
#     ),
    
#     Tool(
#         name="Joke",
#         func=joke_tool,
#         description="Use this to make user laugh with a programming joke."
#     )
# ]


# # -------------------------------------------------------------
# # 🤖 3. LLM SETUP (BRAIN OF AGENT)
# # -------------------------------------------------------------

# # Store API key securely (never hardcode in production)
# os.environ["OPENAI_API_KEY"] = "your_api_key"      

# # ChatOpenAI = wrapper (code for interaction) around LLM API
# llm = ChatOpenAI(
#     model_name="deepseek/deepseek-r1:free",  # model used
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter endpoint
    
#     temperature=0.3  
#     # temperature = randomness
#     # 0 → deterministic
#     # 1 → creative
# )


# # -------------------------------------------------------------
# # 🧠 4. AGENT CREATION (CORE MAGIC 🔥)
# # -------------------------------------------------------------

# agent_executor = initialize_agent(
#     tools=my_tools,  # list of available tools
    
#     llm=llm,  # brain (LLM)
    
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     # 🔥 VERY IMPORTANT CONCEPT:
#     # ZERO_SHOT = no training, uses reasoning
#     # REACT = Reason + Act loop   
#     # Internally:
#     # Thought → Action → Observation → Thought → Final Answer


# # 🤖 LANGCHAIN AGENT TYPES (Basic → Advanced → Entrepreneurial) :-
# # ------------------------------------------------------------
# # 🟢 BASIC LEVEL AGENTS
# # ------------------------------------------------------------

# AgentType.ZERO_SHOT_REACT_DESCRIPTION
# # 🔹 Most commonly used beginner agent
# # 🔹 ZERO_SHOT = no training needed
# # 🔹 REACT = Reason + Act loop
# # 🔹 Uses tool descriptions to decide actions
# # 🧠 Flow: Thought → Action → Observation → Final Answer
# # 💡 Use case: simple tool usage (calculator, weather, APIs)


# AgentType.REACT_DOCSTORE
# # 🔹 Similar to REACT but with document store
# # 🔹 Can search documents (like Wikipedia)
# # 🧠 Combines reasoning + retrieval
# # 💡 Use case: QA systems over documents


# AgentType.SELF_ASK_WITH_SEARCH
# # 🔹 Breaks question into sub-questions
# # 🔹 Uses search tool to answer each part
# # 🧠 Multi-step reasoning agent
# # 💡 Example:
# #   Q: Who is PM of India and their age?
# #   → Step1: find PM
# #   → Step2: find age


# # ------------------------------------------------------------
# # 🟡 INTERMEDIATE LEVEL AGENTS
# # ------------------------------------------------------------

# AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
# # 🔹 Same as ZERO_SHOT but optimized for chat models
# # 🔹 Works better with ChatGPT-like models
# # 💡 Use case: conversational assistants with tools


# AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# # 🔹 Adds MEMORY (conversation history)
# # 🔹 Maintains context across multiple turns
# # 🧠 Stateful agent
# # 💡 Use case: customer support chatbot


# AgentType.CONVERSATIONAL_REACT_DESCRIPTION
# # 🔹 Similar to above but for non-chat LLMs
# # 🔹 Keeps track of previous conversation
# # 💡 Use case: dialogue systems


# # ------------------------------------------------------------
# # 🔴 ADVANCED LEVEL AGENTS
# # ------------------------------------------------------------

# AgentType.OPENAI_FUNCTIONS
# # 🔹 Uses OpenAI function calling
# # 🔹 Structured JSON-based tool calling
# # 🔹 More reliable than text-based REACT
# # 🧠 LLM outputs structured function call
# # 💡 Use case: production-grade APIs, automation


# AgentType.OPENAI_MULTI_FUNCTIONS
# # 🔹 Can call multiple functions in one reasoning chain
# # 🔹 Supports complex workflows
# # 💡 Use case: multi-step automation (finance, DevOps)


# AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# # 🔹 Advanced version of REACT with structured inputs
# # 🔹 Handles complex tool arguments (JSON inputs)
# # 💡 Use case: tools requiring multiple parameters


# # ------------------------------------------------------------
# # 🚀 ENTREPRENEURIAL / PRODUCTION LEVEL INSIGHT
# # ------------------------------------------------------------
# # 🔥 REAL INDUSTRY STACK:

# # 1. STARTUP / MVP:
# # → ZERO_SHOT_REACT_DESCRIPTION
# # → Fast to build, flexible

# # 2. SCALING PRODUCT:
# # → CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# # → Adds memory, better UX

# # 3. PRODUCTION / ENTERPRISE:
# # → OPENAI_FUNCTIONS / MULTI_FUNCTIONS
# # → Structured, reliable, less hallucination

# # ------------------------------------------------------------
# # 🧠 KEY COMPARISON
# # ------------------------------------------------------------
# """
# REACT AGENTS:
# - Flexible
# - Human-like reasoning
# - Slightly less reliable (text parsing)

# FUNCTION AGENTS:
# - Structured (JSON)
# - More reliable
# - Used in production systems

# CONVERSATIONAL AGENTS:
# - Maintain memory
# - Better user experience
# """

# ============================================================
# 🧠 1. DOCSTORE (WHAT IT REALLY DOES)
# ============================================================

"""
❓ Does docstore search random documents?
👉 NO

Docstore = structured document retrieval system
NOT random internet search ❌
"""

# -----------------------------
# 🔍 TYPES OF DOCSTORE
# -----------------------------

"""
1. LOCAL DOCSTORE
   - Your own data (PDFs, text files, DB)
   - Stored in:
     → FAISS
     → ChromaDB
     → Pinecone

2. WEB DOCSTORE
   - Predefined sources (NOT full internet)
   - Example:
     → Wikipedia API
     → Custom website scraping

3. HYBRID (RAG)
   - Local + external API
"""


# -----------------------------
# 📏 DATA SIZE LIMITS
# -----------------------------

"""
Docstore doesn't have fixed limit — depends on:

1. Storage system (FAISS, Pinecone, etc.)
2. RAM / disk
3. Embedding size

Example:
- Small system: 100 MB – 1 GB docs
- Medium: 1–10 GB
- Enterprise: TBs (with vector DB)

IMPORTANT:
LLM does NOT read full data
👉 It retrieves only relevant chunks (top-k)
"""


# -----------------------------
# 🔥 KEY CONCEPT
# -----------------------------

"""
Docstore = Retrieval system
LLM = Reasoning system

👉 Together = RAG (Retrieval Augmented Generation)
"""



# ============================================================
# 💻 2. SYSTEM REQUIREMENTS (VERY IMPORTANT 🔥)
# ============================================================

"""
We divide into 3 levels:
1. Basic (learning)
2. Intermediate (projects)
3. Entrepreneurial (production)
"""


# ------------------------------------------------------------
# 🟢 BASIC LEVEL (Student / Local Machine)
# ------------------------------------------------------------

"""
Agent Types:
- ZERO_SHOT_REACT_DESCRIPTION
- SIMPLE TOOLS
"""

"""
Hardware:
- CPU: Ryzen 5 / i5 ✅
- RAM: 8 GB (minimum)
- GPU: Not required ❌
- VRAM: Not required ❌
- Storage: 20–50 GB

Why?
- Using API-based LLM (OpenAI / OpenRouter)
- No local model training
"""

"""
Use Cases:
- Calculator agent
- Weather bot
- Simple tool orchestration
"""



# ------------------------------------------------------------
# 🟡 INTERMEDIATE LEVEL (Projects / Portfolio)
# ------------------------------------------------------------

"""
Agent Types:
- CHAT_CONVERSATIONAL_REACT_DESCRIPTION
- SELF_ASK_WITH_SEARCH
- RAG-based agents
"""

"""
Hardware:
- CPU: i7 / Ryzen 7
- RAM: 16–32 GB ✅
- GPU: Optional (for embeddings)
- VRAM: 4–8 GB (optional)
- Storage: 100+ GB

Why?
- Handling vector DB
- Running embeddings locally
"""

"""
Use Cases:
- PDF chatbot
- Multi-tool assistant
- Resume analyzer
"""



# ------------------------------------------------------------
# 🔴 ADVANCED / ENTREPRENEURIAL LEVEL
# ------------------------------------------------------------

"""
Agent Types:
- OPENAI_FUNCTIONS
- MULTI_FUNCTIONS
- STRUCTURED_CHAT
- Multi-agent systems
"""

"""
Hardware (LOCAL or SERVER):

Option 1: API-based (recommended startup)
- CPU: any decent server
- RAM: 16–64 GB
- GPU: Not needed ❌

Option 2: Self-hosted LLM
- GPU: RTX 3090 / A100 / H100
- VRAM: 24–80 GB 🔥
- RAM: 64–128 GB
- Storage: 500GB–TB SSD

Why?
- Large models
- High throughput
- Low latency
"""

"""
Use Cases:
- SaaS AI assistants
- Autonomous agents
- Enterprise copilots
"""



# ============================================================
# ☁️ 3. BEST PLATFORMS (VERY IMPORTANT)
# ============================================================

"""
Choose platform based on level
"""


# ------------------------------------------------------------
# 🟢 BASIC
# ------------------------------------------------------------

"""
Platforms:
- Google Colab ✅ BEST
- Kaggle Notebooks

Why:
- Free GPU
- Easy setup
- No infra headache
"""


# ------------------------------------------------------------
# 🟡 INTERMEDIATE
# ------------------------------------------------------------

"""
Platforms:
- Colab Pro
- Local system + APIs
- HuggingFace Spaces

Why:
- More control
- Better performance
"""


# ------------------------------------------------------------
# 🔴 ENTREPRENEURIAL
# ------------------------------------------------------------

"""
Platforms:
- AWS / GCP / Azure
- Docker + Kubernetes
- Serverless (Vercel, etc.)

Why:
- Scalability
- Reliability
- Production deployment
"""



# ============================================================
# 🧠 FINAL UNDERSTANDING (CRITICAL 🔥)
# ============================================================

"""
Docstore:
👉 Not random internet
👉 Controlled retrieval system

Agents:
👉 Decide what to do

Hardware:
👉 Depends on whether you:
   - Use API (lightweight)
   - Host model (heavy)

Best strategy (for YOU):
👉 Use API + Colab → build projects → scale later
"""


# ============================================================
# 🎯 INTERVIEW KILLER ANSWER
# ============================================================
"""
"Docstore is a structured retrieval system, not random search. 
System requirements depend on whether we use API-based LLMs or self-hosted models. 
For most applications, lightweight setups with APIs are sufficient, 
while production systems may require high-memory GPUs and scalable cloud infrastructure."
"""


# # gpu details:
# # ============================================================
# # 💻 GPU GUIDE FOR AI / LLM (Basic → Advanced → Entrepreneurial)
# # ============================================================
# # ------------------------------------------------------------
# # 🟢 BASIC LEVEL GPUs (Students / Beginners)
# # ------------------------------------------------------------
# # 🔹 NVIDIA GTX 1650 (4GB VRAM)
# # - Entry-level GPU
# # - Can run:
# #     → Small ML models
# #     → TinyLlama inference (CPU+GPU hybrid)
# # - Cannot:
# #     → Fine-tune LLMs properly
# # - Best for:
# #     → Learning ML basics
# #     → Small projects


# # 🔹 NVIDIA GTX 1660 / 1660 Ti (6GB VRAM)
# # - Slightly better than 1650
# # - Can run:
# #     → Small LLMs (1B–2B)
# # - Limited for:
# #     → QLoRA (very tight memory)
# # - Good for:
# #     → Beginner experimentation


# # 🔹 NVIDIA RTX 3050 (4–8GB VRAM)
# # - First "entry AI GPU"
# # - CUDA support ✅
# # - Can run:
# #     → TinyLlama
# #     → Phi-2 (with optimization)
# # - Good for:
# #     → Basic fine-tuning experiments


# # ------------------------------------------------------------
# # 🟡 INTERMEDIATE LEVEL GPUs (Serious Projects)
# # ------------------------------------------------------------
# # 🔹 NVIDIA RTX 3060 (12GB VRAM) ⭐ BEST VALUE
# # - Most recommended GPU for students
# # - Can run:
# #     → Mistral 7B (QLoRA)
# #     → LLaMA 7B
# # - Supports:
# #     → Fine-tuning
# #     → RAG systems
# # - Perfect for:
# #     → Resume-level projects


# # 🔹 NVIDIA RTX 4060 / 4060 Ti (8–16GB VRAM)
# # - Newer architecture (Ada Lovelace)
# # - Faster + efficient
# # - Good for:
# #     → Medium LLM workflows
# #     → Multi-tool agents


# # 🔹 NVIDIA RTX 3070 / 3080 (8–12GB VRAM)
# # - High performance
# # - Can handle:
# #     → Larger batch sizes
# #     → Faster training
# # - Used in:
# #     → Research + dev setups


# # ------------------------------------------------------------
# # 🔴 ADVANCED GPUs (Professional / Research)
# # ------------------------------------------------------------
# # 🔹 NVIDIA RTX 3090 (24GB VRAM)
# # - Very powerful consumer GPU
# # - Can run:
# #     → 13B models (with tricks)
# #     → Fast QLoRA
# # - Popular among:
# #     → AI engineers / freelancers


# # 🔹 NVIDIA RTX 4090 (24GB VRAM) 🔥
# # - Current best consumer GPU
# # - Very fast training + inference
# # - Can handle:
# #     → Large LLM experiments
# # - Ideal for:
# #     → Serious AI development


# # ------------------------------------------------------------
# # 🚀 ENTREPRENEURIAL / ENTERPRISE GPUs
# # ------------------------------------------------------------
# # 🔹 NVIDIA A100 (40GB / 80GB VRAM)
# # - Data center GPU
# # - Used in:
# #     → OpenAI, Google, Meta
# # - Can run:
# #     → 70B models (distributed)
# # - Very expensive 💰


# # 🔹 NVIDIA H100 (80GB VRAM) 🔥🔥
# # - Most powerful AI GPU (2024+)
# # - Designed for:
# #     → LLM training at scale
# #     → GenAI startups
# # - Used for:
# #     → GPT-level systems


# # 🔹 NVIDIA L4 / T4 (Cloud GPUs)     # Note:  collab free: (runtime -> change runtime type) GPU T4 and TPU v5e-1 are: free in collab.        kaggle free (better choice): (settings -> accelrelator) GPU T4 X2 (30hr/week), GPU P100 (30 hr/week) , TPU v5e-8  (20 hr/week)
# # - Found in:
# #     → Google Colab
# #     → Kaggle
# # - VRAM:
# #     → T4 = 16GB
# # - Best for:
# #     → Students
# #     → QLoRA training


# # ------------------------------------------------------------
# # 🧠 IMPORTANT CONCEPTS (INTERVIEW LEVEL)
# # ------------------------------------------------------------
# # VRAM (Video RAM):
# # - Most important factor for LLMs
# # - Determines:
# #     → model size you can load
# #     → batch size

# # CUDA:
# # - NVIDIA framework for GPU computing
# # - Required for:
# #     → PyTorch
# #     → TensorFlow
# #     → bitsandbytes (QLoRA)

# # Tensor Cores:
# # - Special hardware for AI acceleration
# # - Present in RTX/A100/H100 GPUs

# # Memory Rule:
# # - 1B parameters ≈ ~2GB VRAM (approx, FP16)
# # - With QLoRA → much less needed


# # ------------------------------------------------------------
# # 🎯 WHICH GPU SHOULD YOU USE?
# # ------------------------------------------------------------
# # For YOU (8GB RAM + AMD):
# # ❌ Don't upgrade immediately
# # ✅ Use:
# #     → Google Colab (T4 GPU)
# #     → Kaggle GPU

# # Best learning path:
# #     TinyLlama → Phi-2 → Mistral (Colab)


# # ------------------------------------------------------------
# # 🧠 INTERVIEW KILLER ANSWER
# # ------------------------------------------------------------
# """
# "For LLM workloads, GPU selection depends mainly on VRAM.
# Entry-level GPUs like RTX 3050 are suitable for small models,
# while RTX 3060 (12GB) is ideal for fine-tuning with QLoRA.
# For production-scale systems, data center GPUs like A100 or H100 are used."
# """

# gpu vs tpu:-
# # ============================================================
# # ⚔️ TPU vs GPU (Basic → Advanced → Entrepreneurial)
# # ============================================================
# # 🧠 CORE DIFFERENCE:
# # GPU = General-purpose parallel processor (flexible)
# # TPU = Specialized chip for tensor operations (Google only)

# # ------------------------------------------------------------
# # 🟢 BASIC LEVEL (Students / Learning)
# # ------------------------------------------------------------
# """
# GPU:
# ✅ Easy to use (PyTorch, TensorFlow)
# ✅ Works everywhere (Colab, Kaggle, local)
# ✅ Supports most libraries (transformers, LangChain)

# TPU:
# ❌ Harder setup
# ❌ Limited support (mainly TensorFlow / JAX)
# ❌ Not beginner-friendly

# 👉 WINNER: GPU 🏆
# Reason: simplicity + ecosystem support
# """


# # ------------------------------------------------------------
# # 🟡 INTERMEDIATE LEVEL (Projects / Portfolio)
# # ------------------------------------------------------------

# """
# GPU:
# ✅ Best for:
#    - LLM fine-tuning (QLoRA)
#    - HuggingFace workflows
#    - RAG systems
# ✅ Works with:
#    - PyTorch
#    - bitsandbytes
#    - PEFT

# TPU:
# ⚠️ Good for:
#    - Large batch training
#    - TensorFlow models
# ❌ Not compatible with:
#    - bitsandbytes
#    - many LLM tools

# 👉 WINNER: GPU 🏆
# Reason: LLM ecosystem built for GPUs
# """


# # ------------------------------------------------------------
# # 🔴 ADVANCED / ENTREPRENEURIAL LEVEL
# # ------------------------------------------------------------

# """
# GPU (A100, H100):
# ✅ Industry standard
# ✅ Used by OpenAI, Meta, etc.
# ✅ Best for:
#    - LLM training
#    - Inference at scale

# TPU (v4, v5):
# ✅ Used by Google
# ✅ Very fast for:
#    - Large-scale training
#    - JAX / TensorFlow pipelines
# ❌ Limited ecosystem outside Google

# 👉 WINNER:
# - GPU → startups / general AI 🏆
# - TPU → Google-scale infra only
# """


# # ------------------------------------------------------------
# # 🧠 FINAL VERDICT
# # ------------------------------------------------------------

# """
# Beginner → GPU ✅
# Intermediate → GPU ✅
# Entrepreneurial → GPU (most cases) ✅
# Google-scale → TPU (special case)
# """


# # ============================================================
# # 🏆 TPU RANKINGS (Google TPUs)
# # ============================================================

# # 1️⃣ TPU v5 (latest)
# # - Fastest TPU
# # - Used in advanced AI research
# # - High efficiency + performance

# # 2️⃣ TPU v4
# # - Widely used in Google data centers
# # - Strong training performance

# # 3️⃣ TPU v3
# # - Available in Colab (older)
# # - Good for TensorFlow workloads

# # 4️⃣ TPU v2
# # - Legacy TPU
# # - Limited use now


# # ============================================================
# # ⚙️ TPU USE CASES
# # ============================================================

# """
# Best for:
# ✅ Large-scale training (Google-level)
# ✅ Matrix-heavy operations
# ✅ TensorFlow / JAX models

# Examples:
# - Google BERT training
# - PaLM model
# - Large recommendation systems

# Not good for:
# ❌ HuggingFace + PyTorch workflows (limited)
# ❌ QLoRA / bitsandbytes
# ❌ LangChain agents


# # ============================================================
# # 🔥 GPU vs TPU SUMMARY
# # ============================================================

# """
# GPU:
# - Flexible
# - Easy
# - Industry standard
# - Best for LLMs

# TPU:
# - Specialized
# - Harder to use
# - Best for Google-scale training
# """


# # ============================================================
# # 🎯 INTERVIEW KILLER ANSWER
# # ============================================================

# """
# "GPUs are preferred for most LLM workloads due to their flexibility 
# and ecosystem support, while TPUs are specialized accelerators used 
# primarily in large-scale TensorFlow or JAX-based systems, especially at Google."
# """

#     verbose=True  # prints reasoning steps (debugging)
# )


# # -------------------------------------------------------------
# # 🧠 WHAT HAPPENS INTERNALLY (ADVANCED FLOW)
# # -------------------------------------------------------------
# """
# User Query → LLM reads:
#     - tool names
#     - tool descriptions
    
# LLM generates:
#     Thought: "User is asking math → use Calculator"
#     Action: Calculator
#     Action Input: "2+3*4"
    
# System executes:
#     calculator_tool("2+3*4") → 14
    
# LLM continues:
#     Observation: 14
#     Final Answer: "The result is 14"
# """


# # -------------------------------------------------------------
# # 🔄 5. INTERACTIVE LOOP (USER INTERFACE)
# # -------------------------------------------------------------

# print("\n💡 Ask anything (math, weather, joke). Type 'exit' to quit.")

# while True:
#     # Take user input
#     query = input("\n🧠 Your Question: ")
    
#     if query.lower() == "exit":
#         break
    
#     # 🔥 CORE EXECUTION
#     answer = agent_executor.run(query)
    
#     # What happens here:
#     # 1. LLM analyzes query
#     # 2. Chooses tool
#     # 3. Calls tool
#     # 4. Combines result
#     # 5. Returns final answer
    
#     print("\n🤖 Answer:", answer)


# # -------------------------------------------------------------
# # 🚀 ENTREPRENEURIAL UNDERSTANDING
# # -------------------------------------------------------------
# """
# This system is NOT just a chatbot.

# It is:
# 👉 A decision-making AI system
# 👉 That can call external tools/APIs
# 👉 And orchestrate workflows

# Real-world applications:
# - AI assistants (like ChatGPT plugins)
# - Customer support bots
# - Financial advisors
# - Dev copilots
# - Autonomous agents

# You basically built:
# 👉 A mini version of ChatGPT + Plugins
# """


# # -------------------------------------------------------------
# # 🔥 ADVANCED CONCEPTS YOU USED
# # -------------------------------------------------------------
# """
# 1. Tool Abstraction
#    → Converting functions into LLM-callable APIs

# 2. Agent Architecture
#    → LLM decides "what to do next"

# 3. ReAct Framework
#    → Reason + Act loop

# 4. Prompt Engineering (hidden)
#    → LangChain internally creates prompts like:
#      "You have access to these tools..."

# 5. Zero-shot reasoning
#    → No training, pure reasoning

# 6. Function calling
#    → LLM → structured API call

# 7. Orchestration
#    → Coordinating multiple tools dynamically
# """


# # -------------------------------------------------------------
# # 🧠 INTERVIEW KILLER ANSWER 🔥
# # -------------------------------------------------------------
# """
# "I built an agentic AI system using LangChain where an LLM dynamically selects and executes tools using the ReAct framework, enabling multi-step reasoning and task orchestration."

# """

# # - The agent (LLM) reads tool descriptions and decides: use `Weather`
# # - It passes "Jaipur" to weather_tool → gets output
# # - LLM formats the answer and returns

# # 🔥 You just built your first multi-tool AI agent!
