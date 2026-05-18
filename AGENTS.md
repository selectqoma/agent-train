# Agent Instructions

This repository is designed to be used by a coding agent.

When a user says "clone this and apply it", do the following:

1. Clone this repository into a temporary or tools directory.
2. Install dependencies in a virtual environment.
3. Put the user's successful conversations into `data/threads/` as JSON files.
4. Run `python3 train.py --config config.yaml`.
5. Inspect `results/run_summary.json`, `results/scores.csv`, and `memory.md`.
6. Apply `memory.md` into the target product's system prompt or agent instructions with `apply_memory.py`.
7. Run the target project's relevant tests or prompt checks.
8. Report the changed files and the score summary.

Training updates memory from the discrepancy between the generated reply, the real reply, and the judge explanation. Scores are for measurement and graphing.

Do not commit private conversation data, generated `memory.md`, or run artifacts unless the user explicitly asks for that and confirms the data is safe to store.

## Expected Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
python3 train.py --config config.yaml
python3 apply_memory.py --memory memory.md --target /path/to/target/system_prompt.md
```

## Data Contract

Conversation files must use this shape:

```json
{
  "id": "thread_001",
  "turns": [
    { "role": "client", "content": "Client message" },
    { "role": "agent", "content": "Successful real reply" }
  ]
}
```

Only adjacent `client` then `agent` turns are scored.

## Apply Contract

`apply_memory.py` writes the trained memory between stable markers:

```md
<!-- agent-train:start -->
...
<!-- agent-train:end -->
```

If the target file already contains those markers, the block is replaced in place. If not, it is appended.
