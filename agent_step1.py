import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Initialize OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# NOTE: "gpt-5-mini" may be an invalid model name. Adjust to a known available model if needed.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
)

# System prompt for the agent
SYSTEM_PROMPT = "You are a helpful assistant."

tools = []
agent_executor = create_react_agent(
    llm,
    tools,
)


def run_agent(user_question: str):
    """Run the agent with a user question."""

    # Construct proper message objects expected by langgraph
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_question),
    ]

    result = agent_executor.invoke({"messages": messages})

    final_message = result["messages"][-1]
    print(final_message.content)


def main():
    parser = argparse.ArgumentParser(description="Run the AI agent.")
    parser.add_argument(
        "question", nargs="?", help="The natural language question to ask the agent."
    )
    args = parser.parse_args()

    if not args.question:
        print('Usage: python agent.py "<your question>"')
        return

    run_agent(args.question)


if __name__ == "__main__":
    main()
