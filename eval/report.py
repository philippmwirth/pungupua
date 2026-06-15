"""Shared markdown report utilities for eval scripts."""


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Return lines forming a GFM markdown table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for k, cell in enumerate(row):
            widths[k] = max(widths[k], len(cell))

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[k]) for k, c in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return [fmt(headers), sep] + [fmt(row) for row in rows]
