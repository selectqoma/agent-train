import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from agent import Agent
from judge import Judge
from plot import save_plot

console = Console()


REQUIRED_CONFIG = {
    "model",
    "threads_dir",
    "memory_file",
    "output_dir",
    "update_threshold",
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    missing = sorted(REQUIRED_CONFIG - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


def load_threads(threads_dir: str) -> list[dict]:
    paths = sorted(Path(threads_dir).glob("*.json"))
    threads = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            thread = json.load(f)
        validate_thread(thread, path)
        threads.append(thread)
    if not threads:
        raise ValueError(f"No JSON threads found in {threads_dir}")
    return threads


def validate_thread(thread: dict, path: Path) -> None:
    if "turns" not in thread or not isinstance(thread["turns"], list):
        raise ValueError(f"{path} must contain a list field named 'turns'")
    for idx, turn in enumerate(thread["turns"]):
        if turn.get("role") not in {"client", "agent"}:
            raise ValueError(f"{path} turn {idx} must use role 'client' or 'agent'")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError(f"{path} turn {idx} must contain non-empty content")


def iter_training_pairs(thread: dict):
    turns = thread["turns"]
    history: list[dict] = []
    for idx, turn in enumerate(turns):
        if turn["role"] != "client":
            continue
        if idx + 1 >= len(turns) or turns[idx + 1]["role"] != "agent":
            continue
        yield idx, history.copy(), turn["content"], turns[idx + 1]["content"]
        history.append({"role": "client", "content": turn["content"]})
        history.append({"role": "agent", "content": turns[idx + 1]["content"]})


def split_threads(threads: list[dict], holdout: int) -> tuple[list[dict], list[dict]]:
    if holdout <= 0:
        return threads, []
    if holdout >= len(threads):
        raise ValueError("eval_holdout must be smaller than the number of threads")
    return threads[:-holdout], threads[-holdout:]


def score_thread(
    *,
    split: str,
    thread: dict,
    agent: Agent,
    judge: Judge,
    threshold: float,
    update_memory: bool,
    rows: list[dict],
    iteration: int,
) -> int:
    for turn_index, history, client_msg, ground_truth in iter_training_pairs(thread):
        agent_reply = agent.respond(client_msg, history)
        score, reason = judge.score(client_msg, agent_reply, ground_truth)
        iteration += 1

        rows.append(
            {
                "iteration": iteration,
                "split": split,
                "thread_id": thread.get("id", ""),
                "turn_index": turn_index,
                "score": f"{score:.2f}",
                "updated_memory": str(update_memory and score < threshold).lower(),
                "reason": reason,
            }
        )

        color = "green" if score >= threshold else "red"
        console.print(
            f"  [{color}]{iteration:3d}[/] {split:<5} score=[{color}]{score:.1f}[/] {reason}"
        )

        if update_memory and score < threshold:
            agent.update_memory(client_msg, agent_reply, ground_truth, score, reason)
            console.print("       [dim]memory updated[/dim]")

    return iteration


def write_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "split",
                "thread_id",
                "turn_index",
                "score",
                "updated_memory",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], config: dict, train_count: int, eval_count: int) -> dict:
    scores = [float(row["score"]) for row in rows]
    train_scores = [float(row["score"]) for row in rows if row["split"] == "train"]
    eval_scores = [float(row["score"]) for row in rows if row["split"] == "eval"]

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "judge_model": config.get("judge_model", config["model"]),
        "threads": {"train": train_count, "eval": eval_count},
        "iterations": len(rows),
        "threshold": config["update_threshold"],
        "avg_score": avg(scores),
        "avg_train_score": avg(train_scores),
        "avg_eval_score": avg(eval_scores),
        "avg_first_20": avg(scores[:20]),
        "avg_last_20": avg(scores[-20:]),
    }


def run(config: dict) -> None:
    output_dir = Path(config["output_dir"])
    threshold = float(config["update_threshold"])

    threads = load_threads(config["threads_dir"])
    if config.get("max_threads"):
        threads = threads[: int(config["max_threads"])]
    train_threads, eval_threads = split_threads(threads, int(config.get("eval_holdout", 0)))

    agent = Agent(memory_file=config["memory_file"], model=config["model"])
    judge = Judge(model=config.get("judge_model", config["model"]))

    rows: list[dict] = []
    iteration = 0

    for idx, thread in enumerate(train_threads):
        console.print(f"\n[bold cyan]Train thread {idx + 1}/{len(train_threads)}[/] - {thread.get('id', '')}")
        iteration = score_thread(
            split="train",
            thread=thread,
            agent=agent,
            judge=judge,
            threshold=threshold,
            update_memory=True,
            rows=rows,
            iteration=iteration,
        )

    for idx, thread in enumerate(eval_threads):
        console.print(f"\n[bold magenta]Eval thread {idx + 1}/{len(eval_threads)}[/] - {thread.get('id', '')}")
        iteration = score_thread(
            split="eval",
            thread=thread,
            agent=agent,
            judge=judge,
            threshold=threshold,
            update_memory=False,
            rows=rows,
            iteration=iteration,
        )

    scores_path = output_dir / "scores.csv"
    summary_path = output_dir / "run_summary.json"
    plot_path = output_dir / "scores.png"

    write_scores(scores_path, rows)
    summary = summarize(rows, config, len(train_threads), len(eval_threads))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_plot([float(row["score"]) for row in rows], str(plot_path), threshold=threshold)

    table = Table(title="Training summary", show_header=False, box=None)
    table.add_row("Train threads", str(len(train_threads)))
    table.add_row("Eval threads", str(len(eval_threads)))
    table.add_row("Iterations", str(len(rows)))
    table.add_row("Avg score", str(summary["avg_score"]))
    table.add_row("Avg train score", str(summary["avg_train_score"]))
    table.add_row("Avg eval score", str(summary["avg_eval_score"]))
    table.add_row("Scores", str(scores_path))
    table.add_row("Summary", str(summary_path))
    table.add_row("Graph", str(plot_path))
    console.print("\n", table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an agent voice memory on successful email threads")
    parser.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
