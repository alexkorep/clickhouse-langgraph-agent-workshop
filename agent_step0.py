import argparse


def run_agent(user_question: str):
    """Run the agent with a user question."""
    print(user_question)


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
