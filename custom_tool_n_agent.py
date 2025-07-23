# ✅ End-to-End Project: Custom Tools + Custom Agent with LangChain
# --------------------------------------------------------------------------------
# GOAL: Build a LangChain Agent that can use multiple **custom tools** to answer questions.
# You will create:
# 1. A math tool
# 2. A weather checker tool (mocked)
# 3. A fun fact tool
# Then combine these in an agent that chooses which tool to use.

# 🔽 Install dependencies
# pip install langchain langchain-core langchain-community openai

import os
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from typing import Optional

# --------------------------------------------------------------------------------
# 1. ✅ Custom Tools (as Python functions wrapped in LangChain Tool)
# --------------------------------------------------------------------------------

# 🧮 Tool 1: Math evaluator
def basic_calculator(query: str) -> str:
    try:
        result = eval(query)
        return f"The result of {query} is {result}."
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# 🌤 Tool 2: Mock Weather checker
def check_weather(city: str) -> str:
    # You can replace this with an API call to real weather APIs later
    weather_data = {
        "jaipur": "Sunny, 38°C",
        "mumbai": "Cloudy, 30°C",
        "delhi": "Rainy, 29°C"
    }
    city = city.lower()
    return weather_data.get(city, "Sorry, I don't have weather data for that city.")

# 🎲 Tool 3: Random Fun Fact
def get_fun_fact(_: Optional[str] = None) -> str:
    return "Did you know? Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs!"

# --------------------------------------------------------------------------------
# 2. 🔧 Wrap functions in LangChain Tool objects
# --------------------------------------------------------------------------------

custom_tools = [
    Tool(
        name="Calculator",
        func=basic_calculator,
        description="Useful for evaluating math expressions like '4 * 5' or '100 / 25'"
    ),
    Tool(
        name="WeatherChecker",
        func=check_weather,
        description="Use to find weather of a city like 'weather in Jaipur'"
    ),
    Tool(
        name="FunFact",
        func=get_fun_fact,
        description="Provides a random fun fact"
    )
]

# --------------------------------------------------------------------------------
# 3. 🔑 Load the LLM via OpenRouter
# --------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = "put_your_api_key_here"  # Replace with your real key
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3
)

# --------------------------------------------------------------------------------
# 4. 🧠 Create the Agent with tools + LLM
# --------------------------------------------------------------------------------

agent = initialize_agent(
    tools=custom_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --------------------------------------------------------------------------------
# 5. 💬 Ask questions in a loop
# --------------------------------------------------------------------------------

print("\n🤖 Ask me anything! I can calculate, give weather, or share fun facts (type 'exit' to quit).")
while True:
    query = input("\n🧠 You: ")
    if query.lower() == "exit":
        break
    response = agent.run(query)
    print("\n🤖 Response:", response)

# --------------------------------------------------------------------------------
# 🔚 Summary:
# - Tools are Python functions with descriptions.
# - LangChain's `Tool` class wraps them.
# - The Agent decides which tool to use based on question.
# - ZERO_SHOT_REACT_DESCRIPTION makes LLM decide based on function descriptions.
# --------------------------------------------------------------------------------
