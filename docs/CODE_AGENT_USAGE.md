# Code Agent Usage

Use this repository as a small training tool that a coding agent can run and then apply to a target project.

Example instruction to a coding agent:

```text
Clone https://github.com/selectqoma/agent-train.
Use the successful conversation threads I provide as training data.
Run the memory training loop.
Apply the resulting memory to our agent system prompt.
Show me the changed prompt file and the run summary.
Do not commit the raw conversation data.
```

## Workflow

1. Prepare data.

   Put anonymized successful conversations in `data/threads/*.json`.

2. Configure.

   ```bash
   cp config.example.yaml config.yaml
   ```

   Set `threads_dir`, `memory_file`, `output_dir`, and optionally `eval_holdout`.

3. Train.

   ```bash
   python3 train.py --config config.yaml
   ```

4. Review.

   Inspect:

   - `memory.md`
   - `results/scores.csv`
   - `results/run_summary.json`
   - `results/scores.png`

   `scores.csv` includes tone, precision, and coherence grades plus the averaged final score.

5. Apply to the target project.

   ```bash
   python3 apply_memory.py \
     --memory memory.md \
     --target /path/to/target/project/prompts/system.md
   ```

6. Verify.

   Run the target project's tests, evals, or prompt snapshots.

## Subagent Pattern

For agent systems with subagents, give each subagent a separate memory target:

```bash
python3 apply_memory.py --memory memory.md --target prompts/sales_agent.md
python3 apply_memory.py --memory memory.md --target prompts/support_agent.md
```

Use separate training datasets when the subagents have different jobs. Do not train one shared memory from mixed sales, support, onboarding, and escalation examples unless the resulting agent really needs to handle all of them.
