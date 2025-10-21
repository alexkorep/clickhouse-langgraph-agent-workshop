import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Create an OpenAI LLM instance
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
)

# System prompt for the agent
SYSTEM_PROMPT = "You are a helpful assistant."


@tool
def my_encoder(text: str) -> str:
    """A simple text encoder tool."""
    print(f"Tool called with input: {text}")
    return text + text.upper()


# Create the agent executor
tools = []
agent_executor = create_react_agent(
    llm,
    tools,
)


def main():
    graph = agent_executor.get_graph()
    png_bytes = graph.draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_bytes)


if __name__ == "__main__":
    main()
