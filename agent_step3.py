import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
import clickhouse_connect
import json

# Load environment variables
load_dotenv()


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


# Initialize OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# NOTE: "gpt-5-mini" may be an invalid model name. Adjust to a known available model if needed.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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

# Create the agent (current version of create_react_agent does not support state_modifier; we inject a system message manually)
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
    """Main function to run the agent console app.

    Behavior:
    - If a question is passed as a command-line argument, run once and exit.
    - Otherwise, fall back to interactive REPL mode.
    """
    parser = argparse.ArgumentParser(
        description="Run the ClickHouse AI agent with a single question or in interactive mode."
    )
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
