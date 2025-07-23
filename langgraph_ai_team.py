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
