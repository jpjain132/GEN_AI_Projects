# ✅ End-to-End LangChain Agent Project with Tools (ReAct Agent, Calculator, Web Search, API, SQL)
# ----------------------------------------------------------------------------------------------
# This script builds an LLM Agent that can:
# - Do math (Calculator tool)
# - Search the internet (DuckDuckGo)
# - Answer from SQL DB
# - Call public APIs (REST calls)
# - Remember past chats

import os
import requests
import sqlite3
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

from langchain.agents import load_tools
from langchain.tools import tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI

# 🔐 Set your OpenRouter key (for DeepSeek or GPT-4 via OpenRouter)
os.environ["OPENAI_API_KEY"] = "put_your_api_key_here" 
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.3
)

# 🧮 1. Add Calculator Tool
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression like 2+2*3."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression."

# 🌐 2. Web Search Tool (DuckDuckGo)
search = DuckDuckGoSearchAPIWrapper()

@tool
def web_search(query: str) -> str:
    """Search for information on the web using DuckDuckGo."""
    return search.run(query)

# 🌐 3. API Caller Tool (simple REST call)
@tool
def get_weather(city: str) -> str:
    """Calls MetaWeather public API to get weather for a city."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        return response.text
    except Exception as e:
        return str(e)

# 🗃️ 4. SQL Database Tool
# Create a dummy database if it doesn’t exist
db_path = "data.db"
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE employees (id INTEGER, name TEXT, role TEXT);")
    cursor.execute("INSERT INTO employees VALUES (1, 'Alice', 'Data Scientist');")
    cursor.execute("INSERT INTO employees VALUES (2, 'Bob', 'Backend Developer');")
    conn.commit()
    conn.close()

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 🛠️ 5. Combine all tools
tools = [
    Tool(name="Calculator", func=calculate, description="Useful for math calculations"),
    Tool(name="Search", func=web_search, description="Useful for answering general world knowledge"),
    Tool(name="Weather", func=get_weather, description="Returns weather for a given city"),
    QuerySQLDataBaseTool(db=db, llm=llm),
    InfoSQLDatabaseTool(db=db)
]

# 🧠 6. Setup Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🤖 7. Create the agent with tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 💬 8. Chat Loop
print("\n🤖 Ask me anything! (type 'exit' to quit)\n")
while True:
    user_input = input("🧠 You: ")
    if user_input.lower() == "exit":
        break
    response = agent.run(user_input)
    print("🤖 Bot:", response)

