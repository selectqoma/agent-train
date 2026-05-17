import csv
import json
import tempfile
import unittest
from pathlib import Path

from train import iter_training_pairs, split_threads, summarize, validate_thread, write_scores


class TrainHelpersTest(unittest.TestCase):
    def test_iter_training_pairs_uses_client_agent_pairs_with_history(self):
        thread = {
            "id": "t1",
            "turns": [
                {"role": "client", "content": "first"},
                {"role": "agent", "content": "reply one"},
                {"role": "client", "content": "second"},
                {"role": "agent", "content": "reply two"},
            ],
        }

        pairs = list(iter_training_pairs(thread))

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][2:], ("first", "reply one"))
        self.assertEqual(pairs[0][1], [])
        self.assertEqual(pairs[1][2:], ("second", "reply two"))
        self.assertEqual(
            pairs[1][1],
            [
                {"role": "client", "content": "first"},
                {"role": "agent", "content": "reply one"},
            ],
        )

    def test_split_threads_reserves_holdout_from_end(self):
        threads = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        train, eval_threads = split_threads(threads, 1)

        self.assertEqual([t["id"] for t in train], ["a", "b"])
        self.assertEqual([t["id"] for t in eval_threads], ["c"])

    def test_validate_thread_rejects_bad_role(self):
        with self.assertRaises(ValueError):
            validate_thread(
                {"turns": [{"role": "customer", "content": "hello"}]},
                Path("bad.json"),
            )

    def test_write_scores_and_summary(self):
        rows = [
            {
                "iteration": 1,
                "split": "train",
                "thread_id": "a",
                "turn_index": 0,
                "score": "6.00",
                "updated_memory": "true",
                "reason": "Too long.",
            },
            {
                "iteration": 2,
                "split": "eval",
                "thread_id": "b",
                "turn_index": 0,
                "score": "8.00",
                "updated_memory": "false",
                "reason": "Good match.",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            write_scores(path, rows)

            with open(path, newline="", encoding="utf-8") as f:
                written = list(csv.DictReader(f))

        self.assertEqual(written[0]["score"], "6.00")

        summary = summarize(
            rows,
            {"model": "model-a", "judge_model": "model-b", "update_threshold": 7.0},
            train_count=1,
            eval_count=1,
        )

        self.assertEqual(summary["avg_score"], 7.0)
        self.assertEqual(summary["avg_train_score"], 6.0)
        self.assertEqual(summary["avg_eval_score"], 8.0)


if __name__ == "__main__":
    unittest.main()
