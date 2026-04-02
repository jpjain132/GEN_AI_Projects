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
