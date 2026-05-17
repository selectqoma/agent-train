import re

import anthropic


class JudgeResult(dict):
    @property
    def score(self) -> float:
        return self["score"]

    @property
    def reason(self) -> str:
        return self["reason"]


class Judge:
    def __init__(self, model: str):
        self.client = anthropic.Anthropic()
        self.model = model

    def score(self, client_msg: str, agent_reply: str, ground_truth: str) -> JudgeResult:
        prompt = f"""Evaluate an email agent reply against the real successful reply.

Client message:
{client_msg}

Agent reply:
{agent_reply}

Real successful reply:
{ground_truth}

Grade each criterion from 0 to 10:
- Tone: voice, warmth, directness, and whether it sounds like the same kind of person.
- Precision: answers the right thing, omits what should be omitted, and avoids unsupported claims.
- Coherence: structure, clarity, length, and flow.

Do not reward copying names or private details. Reward matching judgment and style.

Respond exactly like this:
TONE: <number>
PRECISION: <number>
COHERENCE: <number>
REASON: <one sentence>"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        tone = self._extract_score(text, "TONE")
        precision = self._extract_score(text, "PRECISION")
        coherence = self._extract_score(text, "COHERENCE")
        reason_match = re.search(r"REASON:\s*(.+)", text, flags=re.DOTALL)

        score = round((tone + precision + coherence) / 3, 2)
        reason = reason_match.group(1).strip() if reason_match else ""
        return JudgeResult(
            tone_score=tone,
            precision_score=precision,
            coherence_score=coherence,
            score=score,
            reason=reason,
        )

    @staticmethod
    def _extract_score(text: str, label: str) -> float:
        match = re.search(rf"{label}:\s*(-?[0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            raise ValueError(f"Judge response did not contain a {label} line: {text!r}")
        return max(0.0, min(10.0, float(match.group(1))))
