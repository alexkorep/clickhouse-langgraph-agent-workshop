import argparse


def run_agent(user_question: str):
    """Run the agent with a user question."""
    print(user_question)


def main():
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
