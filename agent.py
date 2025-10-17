import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import clickhouse_connect

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
            columns = result.column_names
            rows = result.result_rows

            output = f"Found {len(rows)} rows:\n\n"
            output += " | ".join(columns) + "\n"
            output += "-" * (len(columns) * 15) + "\n"

            for row in rows[:10]:
                output += " | ".join(str(cell) for cell in row) + "\n"

            if len(rows) > 10:
                output += f"\n... and {len(rows) - 10} more rows"

            return output
        else:
            return "Query executed successfully but returned no results."

    except Exception as e:
        return f"Error executing query: {str(e)}"


# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

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
    print(f"\n{'='*60}")
    print(f"Question: {user_question}")
    print(f"{'='*60}\n")

    # Prepend system prompt manually
    result = agent_executor.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", user_question)]})

    final_message = result["messages"][-1]
    print(f"\nAgent Response:\n{final_message.content}\n")

    return result


def main():
    """Main function to run the agent console app."""
    print("🤖 AI Agent with ClickHouse Tool")
    print("Ask questions about your database in natural language!")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            run_agent(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
