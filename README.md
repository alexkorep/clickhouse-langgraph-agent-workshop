# ClickHouse LangGraph Agent Workshop

An example AI agent that uses LangGraph + LangChain with an OpenAI chat model and a custom tool to query a ClickHouse database.

## Features
- Natural language to SQL against ClickHouse
- Tool abstraction via `@tool` decorator
- Supports single-run CLI question

## Prerequisites
1. Python 3.10+
2. A running ClickHouse instance (local or remote)
3. OpenAI API key

## Installation

Create and activate a virtual environment (optional but recommended) then install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file or export env vars in your shell:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini            # optional override
CLICKHOUSE_HOST=localhost           # default is localhost
CLICKHOUSE_PORT=8123                # default is 8123
CLICKHOUSE_USER=default             # default is default
CLICKHOUSE_PASSWORD=                # optional
```
