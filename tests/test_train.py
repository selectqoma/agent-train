import csv
import json
import tempfile
import unittest
from pathlib import Path

from apply_memory import END_MARKER, START_MARKER, apply_memory, build_block
from judge import Judge
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
                "tone_score": "5.00",
                "precision_score": "6.00",
                "coherence_score": "7.00",
                "score": "6.00",
                "updated_memory": "true",
                "reason": "Too long.",
            },
            {
                "iteration": 2,
                "split": "eval",
                "thread_id": "b",
                "turn_index": 0,
                "tone_score": "8.00",
                "precision_score": "8.00",
                "coherence_score": "8.00",
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
            {"model": "model-a", "judge_model": "model-b"},
            train_count=1,
            eval_count=1,
        )

        self.assertEqual(summary["avg_score"], 7.0)
        self.assertEqual(summary["avg_train_score"], 6.0)
        self.assertEqual(summary["avg_eval_score"], 8.0)


class JudgeParsingTest(unittest.TestCase):
    def test_extract_score_clamps_values(self):
        self.assertEqual(Judge._extract_score("TONE: 12", "TONE"), 10.0)
        self.assertEqual(Judge._extract_score("TONE: -1", "TONE"), 0.0)

    def test_extract_score_requires_label(self):
        with self.assertRaises(ValueError):
            Judge._extract_score("SCORE: 8", "TONE")


class ApplyMemoryTest(unittest.TestCase):
    def test_build_block_requires_memory(self):
        with self.assertRaises(ValueError):
            build_block("")

    def test_apply_memory_appends_or_replaces_marked_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memory = tmp_path / "memory.md"
            target = tmp_path / "system.md"
            memory.write_text("- Use short replies.\n", encoding="utf-8")
            target.write_text("Base prompt.\n", encoding="utf-8")

            apply_memory(str(memory), str(target))
            first = target.read_text(encoding="utf-8")

            self.assertIn("Base prompt.", first)
            self.assertIn(START_MARKER, first)
            self.assertIn("- Use short replies.", first)

            memory.write_text("- Ask one question at a time.\n", encoding="utf-8")
            apply_memory(str(memory), str(target))
            second = target.read_text(encoding="utf-8")

            self.assertIn("- Ask one question at a time.", second)
            self.assertNotIn("- Use short replies.", second)
            self.assertEqual(second.count(START_MARKER), 1)
            self.assertEqual(second.count(END_MARKER), 1)


if __name__ == "__main__":
    unittest.main()
