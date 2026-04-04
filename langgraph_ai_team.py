# ✅ Build a stateful multi-agent workflow using LangGraph
# Roles: Assistant (writes content), Critic (reviews & critiques)
# LangGraph helps manage conversation state and flow

# 🔽 Install first (if not already):
# pip install langgraph langchain openai

import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI


# Define state type as dictionary (shared memory structure)
from typing import TypedDict, List

class AgentState(TypedDict):
    history: List[str]     # Conversation log
    content: str           # Current message content

# 🔐 Set your OpenAI API Key (or use OpenRouter)
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


# 🎯 Initialize the LLM
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",  # or "deepseek-chat" for OpenRouter
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],

    temperature=0.5              # Controls creativity
)

# 🧠 Role 1: Assistant generates a paragraph on the topic
def assistant_node(state: AgentState) -> AgentState:
    prompt = f"Write a short paragraph about: '{state['content']}'"
    print("\n🧠 Assistant Prompt:", prompt)
    response = llm.invoke(prompt)
    print("📝 Assistant Response:", response.content)
    return {
        "content": response.content,
        "history": state["history"] + [f"Assistant: {response.content}"]
    }

# 🧠 Role 2: Critic reviews and critiques the paragraph
def critic_node(state: AgentState) -> AgentState:
    prompt = f"Review the following and provide critique:\n\n{state['content']}"
    print("\n🔍 Critic Prompt:", prompt)
    response = llm.invoke(prompt)
    print("📋 Critic Response:", response.content)
    return {
        "content": response.content,
        "history": state["history"] + [f"Critic: {response.content}"]
    }

# 🛠️ Build LangGraph Workflow
workflow = StateGraph(AgentState)

workflow.add_node("assistant", assistant_node)
workflow.add_node("critic", critic_node)

# Define flow: assistant ➝ critic ➝ end
workflow.set_entry_point("assistant")
workflow.add_edge("assistant", "critic")
workflow.add_edge("critic", END)

# 🎬 Compile the workflow
app = workflow.compile()

# 🚀 Run the Graph
user_topic = input("🎯 Enter a topic: ")
initial_state = {
    "history": [],
    "content": user_topic
}

# 🧾 Final Output
final_state = app.invoke(initial_state)

print("\n🧾 Final Output History:")
for item in final_state["history"]:
    print("•", item)








# # # ============================================================
# # ⚔️ LANGCHAIN vs LANGGRAPH (ENTREPRENEURIAL DECISION)
# # ============================================================


# # ------------------------------------------------------------
# # 🟢 LANGCHAIN (CHAIN OF PROMPTS + TOOLS)
# # ------------------------------------------------------------

# """
# Architecture:
# User → Prompt → Tool → Prompt → LLM → Output

# ✅ PROS:
# + Simple to build
# + Fast MVP
# + Less cost
# + Easier debugging

# ❌ CONS:
# - Linear flow only
# - No complex reasoning loops
# - Hard to scale for multi-step decisions

# 👉 BEST FOR:
# ✔ RAG systems
# ✔ Chatbots
# ✔ API-based workflows
# ✔ SaaS MVPs
# """


# # ------------------------------------------------------------
# # 🔴 LANGGRAPH (MULTI-AGENT SYSTEM)
# # ------------------------------------------------------------

# """
# Architecture:
# User → Agent1 → Agent2 → Decision → Loop → Final Output

# ✅ PROS:
# + Multi-agent collaboration
# + Iterative refinement (write → critique → improve)
# + Complex decision making
# + Stateful workflows

# ❌ CONS:
# - More complex
# - Higher cost (multiple LLM calls)
# - Harder to debug

# 👉 BEST FOR:
# ✔ Autonomous systems
# ✔ AI copilots
# ✔ Multi-step reasoning apps
# ✔ Enterprise workflows
# """


# # ------------------------------------------------------------
# # 🧠 WHICH GIVES BETTER RESULTS?
# # ------------------------------------------------------------

# """
# 🔥 SIMPLE TASKS:
# → LangChain wins
# Reason: less noise, faster, cheaper

# 🔥 COMPLEX TASKS:
# → LangGraph wins ⭐
# Reason:
# - Multiple agents improve output
# - Iterative refinement increases quality

# 👉 Example:
# Write article once → LangChain
# Write + critique + improve → LangGraph ⭐
# """


# # ------------------------------------------------------------
# # 🚀 ENTREPRENEURIAL LEVEL DECISION
# # ------------------------------------------------------------

# """
# Stage 1 (MVP):
# → Use LangChain ⭐
# ✔ Fast build
# ✔ Validate idea

# Stage 2 (Scaling):
# → Add LangGraph ⭐
# ✔ Better quality
# ✔ Complex workflows

# 👉 REAL STARTUP STACK:

# LangChain:
# → RAG + tools

# LangGraph:
# → orchestration + multi-agent logic

# 👉 BEST COMBO:
# LangChain + LangGraph ⭐⭐⭐
# """







# # ============================================================
# # 🧠 PART 1: EXPLANATION OF YOUR LANGGRAPH CODE
# # ============================================================

# # ------------------------------------------------------------
# # 🛠️ Build LangGraph Workflow
# # ------------------------------------------------------------

# workflow = StateGraph(AgentState)

# """
# 👉 StateGraph = graph-based workflow manager

# - AgentState = shared memory structure
# - It defines what data flows between nodes

# Here:
# state = {
#     "history": [...],
#     "content": "current text"
# }

# 👉 Think:
# StateGraph = "pipeline + memory + control flow"
# """


# # ------------------------------------------------------------
# # ➕ Add Nodes (Agents)
# # ------------------------------------------------------------

# workflow.add_node("assistant", assistant_node)
# workflow.add_node("critic", critic_node)

# """
# 👉 add_node(name, function)

# - "assistant" → node name
# - assistant_node → function to execute

# 👉 Each node:
# - takes state as input
# - returns updated state

# 👉 So:
# assistant → generates content
# critic → reviews content
# """


# # ------------------------------------------------------------
# # 🔀 Define Flow (Edges)
# # ------------------------------------------------------------

# workflow.set_entry_point("assistant")

# """
# 👉 Entry point = where execution starts
# """

# workflow.add_edge("assistant", "critic")

# """
# 👉 After assistant finishes → go to critic
# """

# workflow.add_edge("critic", END)

# """
# 👉 After critic → stop execution
# END = special terminal node
# """


# # ------------------------------------------------------------
# # 🎬 Compile Workflow
# # ------------------------------------------------------------

# app = workflow.compile()

# """
# 👉 Converts graph definition → executable app

# Think:
# Design → Compile → Run
# """


# # ------------------------------------------------------------
# # 🚀 Run the Graph
# # ------------------------------------------------------------

# user_topic = input("🎯 Enter a topic: ")

# initial_state = {
#     "history": [],
#     "content": user_topic
# }

# """
# 👉 Initial state:
# - history empty
# - content = user input
# """


# final_state = app.invoke(initial_state)

# """
# 👉 invoke() = run the workflow

# Flow:
# initial_state
# ↓
# assistant_node
# ↓
# critic_node
# ↓
# END

# 👉 Each step updates state
# """


# # ------------------------------------------------------------
# # 🧾 Output
# # ------------------------------------------------------------

# print("\n🧾 Final Output History:")
# for item in final_state["history"]:
#     print("•", item)

# """
# 👉 final_state["history"] contains:

# [
#   "Assistant: ...",
#   "Critic: ..."
# ]

# 👉 Printed as conversation log
# """


# # ============================================================
# # 🔄 FULL FLOW VISUALIZATION
# # ============================================================

# """
# User Input
# ↓
# [Assistant Node]
# → generates paragraph
# ↓
# [Critic Node]
# → critiques paragraph
# ↓
# END
# ↓
# Final history printed
# """



# # ============================================================
# # 🧠 PART 2: LANGCHAIN vs LANGGRAPH (VERY IMPORTANT 🔥)
# # ============================================================


# # ------------------------------------------------------------
# # ⚔️ BASIC DIFFERENCE
# # ------------------------------------------------------------

# """
# LangChain:
# 👉 Linear workflows (chains)
# 👉 Tool calling, RAG, agents

# LangGraph:
# 👉 Graph-based workflows
# 👉 Multi-agent + loops + state management
# """


# # ------------------------------------------------------------
# # ⚙️ FEATURE COMPARISON
# # ------------------------------------------------------------

# """
# Feature              LangChain        LangGraph
# ---------------------------------------------------------
# Flow type            Linear           Graph (nodes + edges)
# Memory               Basic            Advanced (stateful)
# Multi-agent          Limited          Strong ⭐
# Loops                Hard             Easy ⭐
# Control flow         Simple           Complex ⭐
# Debugging            Moderate         Better tracing
# """


# # ------------------------------------------------------------
# # 🧠 UNIQUE USE CASES
# # ------------------------------------------------------------

# """
# 🔹 LangChain:

# Use for:
# ✔ RAG pipelines
# ✔ Tool-based agents
# ✔ Simple chatbots

# Example:
# User → Retriever → LLM → Answer


# 🔹 LangGraph:

# Use for:
# ✔ Multi-agent workflows
# ✔ Iterative reasoning
# ✔ Complex decision systems

# Example:
# Assistant → Critic → Refine → Loop → Final
# """


# # ------------------------------------------------------------
# # 🚀 ENTREPRENEURIAL USE CASES
# # ------------------------------------------------------------

# """
# 🔴 LangChain (Startup Level):

# 1. Document QA system (RAG)
# 2. Customer support chatbot
# 3. AI-powered search

# 👉 Focus:
# - Fast MVP
# - Simpler pipelines


# 🔴 LangGraph (Advanced Startup / Scale):

# 1. Multi-agent systems (AutoGPT style)   
# 👉 Multi-agent systems = multiple AI agents working together, each with a specific role, to solve a complex task.
# 👉 Each agent handles a part (like planning, execution, validation) and they collaborate to reach the final result.

# eg.
# Agent 1 (Writer) → writes article  
# Agent 2 (Critic) → reviews mistakes  
# Agent 3 (Editor) → improves quality  

# # Final Output:
# High-quality refined article
    
# 2. AI code reviewer (write → review → fix loop)
# 3. Financial advisor agents (analyze → validate → decide)
# 4. Autonomous workflows
# Task: Write and improve an article
    
# 👉 Focus:
# - Complex logic
# - Decision-making systems
# - Iterative refinement


# 🔥 REAL INDUSTRY STACK:

# LangChain:
# → Data retrieval + tools

# LangGraph:
# → Orchestration of multiple agents

# 👉 Combined:
# LangChain + LangGraph = FULL AI SYSTEM ⭐
# """


# # ------------------------------------------------------------
# # 🧠 WHEN TO USE WHAT
# # ------------------------------------------------------------

# """
# Use LangChain:
# ✔ Simple pipelines
# ✔ RAG
# ✔ APIs integration

# Use LangGraph:
# ✔ Multi-step reasoning
# ✔ Agent collaboration
# ✔ Loops / feedback systems

# 👉 Rule:
# Simple → LangChain
# Complex → LangGraph
# """


# # ------------------------------------------------------------
# # 🎯 INTERVIEW KILLER ANSWER
# # ------------------------------------------------------------

# """
# "LangChain is used for building linear LLM pipelines like RAG and tool-based agents, 
# while LangGraph extends it to graph-based workflows enabling multi-agent collaboration, 
# state management, and iterative reasoning, making it suitable for complex AI systems."
# """
