Training runs write artifacts here:

- `scores.csv` - every scored reply with split, thread id, turn index, score, and reason.
- `run_summary.json` - model, config, thread count, iteration count, and score aggregates.
- `scores.png` - score curve generated from `scores.csv`.

These files are ignored by git so private client data and run logs are not
accidentally committed.
