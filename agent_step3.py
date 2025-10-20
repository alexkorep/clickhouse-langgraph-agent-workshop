import os
import argparse
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
import clickhouse_connect

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
SYSTEM_PROMPT = (
    "You are a helpful database assistant. "
    "When users ask questions about data, write and execute SQL queries against ClickHouse. "
    "Always explain your results in a clear, human-friendly way."
)


@tool
def query_clickhouse(query: str) -> str:
    """
    Execute a SQL query against ClickHouse database.
    Use this tool when you need to retrieve data from the database.

    Args:
        query: A valid ClickHouse SQL query string

    Returns:
        Query results as a formatted string
    """
    try:
        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
            username=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        )

        result = client.query(query)

        if result.result_rows:
            data = [dict(zip(result.column_names, row)) for row in result.result_rows]
            return json.dumps(data, indent=2, default=str)
        else:
            return "Query executed successfully but returned no results."

    except Exception as e:
        return f"Error executing query: {str(e)}"


# Create the agent executor
tools = [query_clickhouse]
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
        print('Usage: python agent.py "<your question about the database>"')
        print('Example: python agent.py "Show me the first 5 rows from system.tables"')
        return

    run_agent(args.question)


if __name__ == "__main__":
    main()
