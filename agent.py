from pathlib import Path

import anthropic


class Agent:
    def __init__(self, memory_file: str, model: str):
        self.client = anthropic.Anthropic()
        self.model = model
        self.memory_path = Path(memory_file)
        self.memory = self._load_memory()

    def _load_memory(self) -> str:
        if not self.memory_path.exists():
            return ""
        return self.memory_path.read_text(encoding="utf-8").strip()

    def _save_memory(self, content: str) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(content.strip() + "\n", encoding="utf-8")
        self.memory = content.strip()

    def respond(self, client_msg: str, history: list[dict]) -> str:
        messages = []
        for turn in history:
            role = "user" if turn["role"] == "client" else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": client_msg})

        system = f"""You are replying to a client.

Your memory captures the voice, judgment, and behavior learned from successful past conversations.

<memory>
{self.memory or "No memory yet. Be clear, useful, concise, and natural."}
</memory>

Reply only with the email response. Do not explain your reasoning."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    def update_memory(
        self,
        client_msg: str,
        agent_reply: str,
        ground_truth: str,
        score: float,
        reason: str,
    ) -> str:
        prompt = f"""You are improving the memory used by an email agent.

The agent scored {score:.1f}/10 against a real successful reply.

Client message:
{client_msg}

Agent reply:
{agent_reply}

Real successful reply:
{ground_truth}

Judge reason:
{reason}

Current memory:
{self.memory or "(empty)"}

Rewrite the memory so the agent is more likely to match the real reply next time.
Keep durable behavioral rules. Avoid memorizing private facts, names, or one-off details.
Use concise bullet points. Return only the updated memory."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        new_memory = response.content[0].text.strip()
        self._save_memory(new_memory)
        return new_memory
