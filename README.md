# Agent Train

Train an LLM agent to behave more like your best real client conversations, then apply that behavior to a target agent prompt.

This is not fine-tuning. It does not update model weights. It runs an agent through successful conversation threads, compares each generated reply to the real reply, and updates a plain-text memory when the reply misses the mark.

The result is a behavioral memory you can read, edit, version, and apply into an agent system prompt. It is designed to work well when handed to a coding agent such as Claude Code:

```text
Clone https://github.com/selectqoma/agent-train and apply it to our agent prompt using these successful conversation threads.
```

## Why

RAG helps an agent retrieve facts. It does not automatically teach judgment: when to be brief, when to push, how to handle objections, what level of confidence to show, or what a good reply sounds like in your business.

This project treats past successful conversations as training examples for that behavior.

## How It Works

1. Load successful conversation threads from `data/threads/*.json`.
2. For every client turn, ask the agent to generate the next reply using its current memory.
3. Ask a judge model to grade the reply against the real successful reply on tone, precision, and coherence.
4. Ask the agent to reflect on the discrepancy between its reply, the real reply, and the judge's explanation.
5. Rewrite memory from that discrepancy.
6. Write auditable run artifacts: `scores.csv`, `run_summary.json`, and `scores.png`.
7. Apply the trained memory into a target prompt file with stable `agent-train` markers.

You can also reserve the last N threads as a held-out evaluation set with `eval_holdout`.

## What Is Included

The repo includes three small synthetic example threads so the file format is clear.

It does not include private client conversations or claim that the bundled examples prove convergence. Real results require your own successful conversations.

## Setup

```bash
git clone https://github.com/selectqoma/agent-train
cd agent-train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
```

## Configure

Copy the example config if you want to edit it:

```bash
cp config.example.yaml config.yaml
```

Default config:

```yaml
model: claude-sonnet-4-20250514
judge_model: claude-sonnet-4-20250514
threads_dir: ./data/threads
memory_file: ./memory.md
output_dir: ./results
max_threads: 50
eval_holdout: 0
```

To hold out the final 5 threads for evaluation:

```yaml
eval_holdout: 5
```

## Data Format

Add JSON files to `data/threads/`:

```json
{
  "id": "thread_01",
  "turns": [
    { "role": "client", "content": "Hi, I'm interested in..." },
    { "role": "agent", "content": "Hi Marcus, thanks for reaching out..." },
    { "role": "client", "content": "What does an engagement look like?" },
    { "role": "agent", "content": "Usually three steps..." }
  ]
}
```

Each `client` turn followed by an `agent` turn becomes one scored iteration.

## Run

```bash
python3 train.py --config config.example.yaml
```

Outputs:

- `memory.md` - the trained behavioral memory.
- `results/scores.csv` - every scored reply, tone score, precision score, coherence score, averaged score, judge reason, and whether memory changed.
- `results/run_summary.json` - model, config, thread counts, score averages.
- `results/scores.png` - graph generated from the actual score CSV.

## Apply To An Agent Prompt

After training, apply the memory into the target project's prompt or agent instruction file:

```bash
python3 apply_memory.py \
  --memory memory.md \
  --target /path/to/your/project/prompts/system.md
```

The tool writes between stable markers:

```md
<!-- agent-train:start -->
...
<!-- agent-train:end -->
```

If the markers already exist, the block is replaced in place. If not, it is appended.

## Code Agent Usage

This repo includes [AGENTS.md](AGENTS.md) and [docs/CODE_AGENT_USAGE.md](docs/CODE_AGENT_USAGE.md) so a coding agent can clone it, run the training loop, and apply the result to another project.

Typical instruction:

```text
Clone https://github.com/selectqoma/agent-train.
Use the conversation threads I provide.
Train memory, apply it to prompts/sales_agent.md, and show me the changed files plus the score summary.
Do not commit the raw conversation data.
```

## Notes

- Use successful conversations, not average ones.
- Remove private names and sensitive details before committing data.
- More threads usually help, but quality matters more than volume.
- Scores are model-judged and used for measurement, not as the memory update itself.
- A held-out eval set is recommended before using the memory in production.

## Limitations

This is a lightweight experiment, not a replacement for evals, monitoring, or human review. The judge model can be inconsistent, the memory can overfit, and training on poor examples will teach poor behavior. Use the exported artifacts to inspect what changed and why.
