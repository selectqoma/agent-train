import argparse
from pathlib import Path


START_MARKER = "<!-- agent-train:start -->"
END_MARKER = "<!-- agent-train:end -->"


def build_block(memory: str, heading: str = "Agent Voice Memory") -> str:
    memory = memory.strip()
    if not memory:
        raise ValueError("Memory file is empty")

    return f"""{START_MARKER}
## {heading}

Use these behavioral rules when writing replies:

{memory}
{END_MARKER}"""


def apply_memory(memory_file: str, target_file: str, heading: str = "Agent Voice Memory") -> str:
    memory = Path(memory_file).read_text(encoding="utf-8")
    block = build_block(memory, heading)
    target = Path(target_file)

    if target.exists():
        content = target.read_text(encoding="utf-8")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        content = ""

    if START_MARKER in content and END_MARKER in content:
        before = content.split(START_MARKER, 1)[0].rstrip()
        after = content.split(END_MARKER, 1)[1].lstrip()
        updated = "\n\n".join(part for part in [before, block, after] if part)
    else:
        updated = "\n\n".join(part for part in [content.rstrip(), block] if part)

    target.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply trained agent memory to a target prompt file")
    parser.add_argument("--memory", default="memory.md", help="Trained memory file")
    parser.add_argument("--target", required=True, help="Prompt/system-instruction file to update")
    parser.add_argument("--heading", default="Agent Voice Memory")
    args = parser.parse_args()

    path = apply_memory(args.memory, args.target, args.heading)
    print(f"Applied memory to {path}")


if __name__ == "__main__":
    main()
