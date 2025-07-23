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

# os.environ["OPENAI_API_KEY"] = "your_api_key"      #uncomment this line
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
# - The agent (LLM) reads tool descriptions and decides: use `Weather`
# - It passes "Jaipur" to weather_tool → gets output
# - LLM formats the answer and returns

# 🔥 You just built your first multi-tool AI agent!
