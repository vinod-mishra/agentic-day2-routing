# Agentic AI Cohort - Day 2: Tier-Based Routing

This project implements a LangGraph workflow that routes support tickets based on user tiers (VIP vs. Standard).

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file and add your `GOOGLE_API_KEY`.

The current entrypoint is `app.py`.

## Requirements

Dependencies are listed in `requirements.txt`.

At minimum, this project needs:
- Python 3.12
- `langchain_google_genai`
- `python-dotenv`
- `langchain-core`

## Activate the environment

From the project root:

```bash
cd /Users/vinod/Documents/Agentic-AI-Cohort/agentic-day2-routing
```

### Using the included environment

If you already have the included `env/` directory, activate it first:

```bash
source env/bin/activate
```

### Create a new virtual environment (recommended)

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running the demo

```bash
python app.py
```

The script will run the workflow and print the generated support response for the configured user tier.

## What this project demonstrates

- Tier-based routing using a LangGraph state graph.
- Using a typed state object for messages and routing decisions.
- Sending chat history to a Google Gemini LLM via `langchain_google_genai`.

## Notes

- Ensure your `.env` file contains `GOOGLE_API_KEY` before running the script.
- The code currently uses `gemini-2.5-flash-lite` as the model name.

## File structure

- `app.py` — demo script with routing and LLM response logic
- `requirements.txt` — dependency list
- `README.md` — project documentation
