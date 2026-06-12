from __future__ import annotations
import datetime


class Exporter:
    """Converts session data to exportable text formats."""

    @staticmethod
    def chat_to_markdown(messages: list[dict], filename: str) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        out = [f"# ScholarMind Session\n**Paper:** {filename}  \n**Date:** {ts}\n\n---"]
        for m in messages:
            if m["role"] == "user":
                out.append(f"\n### ❓ {m['content']}\n")
            else:
                out.append(f"\n**Answer** *(conf: {m.get('confidence','')})*\n\n{m['content']}\n")
                for i, s in enumerate(m.get("sources", []), 1):
                    out.append(f"> [{i}] p.{s['page']} — {s['content']}\n")
        return "\n".join(out)

    @staticmethod
    def summary_to_markdown(summary: dict, filename: str) -> str:
        kw = ", ".join(summary.get("keywords", []))
        return (
            f"# Summary — {filename}\n\n"
            f"## In One Line\n{summary.get('one_liner', '')}\n\n"
            f"## Problem\n{summary.get('problem', '')}\n\n"
            f"## Method\n{summary.get('method', '')}\n\n"
            f"## Results\n{summary.get('results', '')}\n\n"
            f"## Contribution\n{summary.get('contribution', '')}\n\n"
            f"## Limitations\n{summary.get('limitations', '')}\n\n"
            f"## Keywords\n{kw}\n"
        )
