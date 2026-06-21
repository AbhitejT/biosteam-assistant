"""Minimal terminal chat for the BioSTEAM Assistant (no web UI required).

Usage:  python -m biosteam_ai.cli
"""
from __future__ import annotations

from .orchestrator import Orchestrator


def main() -> None:
    try:
        orch = Orchestrator()
    except Exception as exc:
        print(f"Cannot start assistant: {exc}")
        return

    print("BioSTEAM Assistant (type 'exit' to quit)\n")
    while True:
        try:
            prompt = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        answer = orch.ask(
            prompt,
            on_tool_call=lambda name, args: print(f"  [tool] {name} {args or ''}"),
        )
        print(f"\nassistant > {answer}\n")


if __name__ == "__main__":
    main()
