import re

import anthropic


class Judge:
    def __init__(self, model: str):
        self.client = anthropic.Anthropic()
        self.model = model

    def score(self, client_msg: str, agent_reply: str, ground_truth: str) -> tuple[float, str]:
        prompt = f"""Evaluate an email agent reply against the real successful reply.

Client message:
{client_msg}

Agent reply:
{agent_reply}

Real successful reply:
{ground_truth}

Score from 0 to 10 using these criteria:
- Voice and tone match
- Correct content and omissions
- Length, structure, and directness
- Naturalness

Do not reward copying names or private details. Reward matching judgment and style.

Respond exactly like this:
SCORE: <number>
REASON: <one sentence>"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text, flags=re.DOTALL)

        if not score_match:
            raise ValueError(f"Judge response did not contain a SCORE line: {text!r}")

        score = max(0.0, min(10.0, float(score_match.group(1))))
        reason = reason_match.group(1).strip() if reason_match else ""
        return score, reason
