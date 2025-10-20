import os
import argparse
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
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

    final_answer = None

    event_iterator = agent_executor.stream({"messages": messages})
    # Iterate over streamed events from the agent graph
    for event in event_iterator:
        # Each event is a mapping of node name -> state slice
        for _node, value in event.items():
            msg_list = value.get("messages", [])
            if not msg_list:
                continue

            last = msg_list[-1]

            # Handle AI messages (reasoning + potential tool calls)
            if isinstance(last, AIMessage):
                reasoning_text = last.content
                tool_calls = getattr(last, "tool_calls", None)

                if reasoning_text:
                    print(f"\n[Reasoning] {reasoning_text}")

                if tool_calls:
                    # Print each tool call input
                    for i, tc in enumerate(tool_calls, start=1):
                        name = tc.get("name")
                        args = tc.get("args")
                        try:
                            args_str = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args_str = str(args)
                        print(f"[Tool Call {i}] {name} args={args_str}")
                else:
                    # AIMessage without tool calls could be the final answer candidate
                    final_answer = reasoning_text

            # Handle tool result messages
            elif isinstance(last, ToolMessage):
                # ToolMessage.name may contain the tool name
                tool_name = getattr(last, "name", "tool")
                print(f"[Tool Result - {tool_name}] {last.content}")

    if final_answer:
        print("\n=== Final Answer ===")
        print(final_answer)


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
