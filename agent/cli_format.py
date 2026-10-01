"""
cli_format.py — pretty-prints markdown tables inside agent responses for
the plain-text CLI.

The agent (correctly) produces valid markdown, including pipe tables --
but a plain terminal doesn't render markdown, so a table shows up as raw
"| a | b |" text instead of an aligned grid. This module detects markdown
table blocks inside a larger response (which may also contain ordinary
prose before/after the table) and replaces just those blocks with an
aligned ASCII table via `tabulate`, leaving everything else untouched.
"""

import re
import shutil
import textwrap

from tabulate import tabulate

_SEPARATOR_RE = re.compile(r"^[\s\-:|]+$")


def _column_width_for_terminal(num_columns: int) -> int:
    """Picks a per-column max width so the rendered table fits the
    current terminal width, instead of a fixed guess that might still
    wrap awkwardly on a narrower window or waste space on a wider one."""
    try:
        terminal_width = shutil.get_terminal_size().columns
    except OSError:
        terminal_width = 100  # fallback if terminal size can't be detected

    # Each column also carries ~3 chars of border/padding overhead
    # ("| " + " " before the next "|"), plus one for the table's own
    # outer border.
    overhead_per_column = 3
    usable_width = max(terminal_width - 1, 40)
    width = usable_width // num_columns - overhead_per_column

    return max(10, min(width, 40))  # keep it within a sane readable range


def _wrap(text: str, width: int) -> str:
    """Wraps text at word boundaries to the given width, embedding real
    newlines -- tabulate renders multi-line cell content natively.
    break_long_words=False avoids ugly mid-word splits like
    'electronic/s'; a single long word may modestly exceed `width`
    rather than fracture, which reads better."""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def pretty_print_tables(text: str) -> str:
    """Finds markdown pipe-table blocks in `text` and replaces each with
    an aligned ASCII table. Non-table text is returned unchanged. Safe to
    call on text with no tables at all -- returns it as-is."""
    lines = text.split("\n")
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        is_table_start = (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and _SEPARATOR_RE.match(lines[i + 1].strip())
            and "|" in lines[i + 1]
        )

        if not is_table_start:
            output_lines.append(line)
            i += 1
            continue

        # Collect the contiguous table block: header, separator, data rows.
        header = _split_row(line)
        block_end = i + 2
        data_rows = []
        while block_end < len(lines) and lines[block_end].strip().startswith("|"):
            data_rows.append(_split_row(lines[block_end]))
            block_end += 1

        # Only treat it as a real table if rows have a consistent column
        # count -- otherwise leave the raw text alone rather than risk
        # mangling something that merely looked table-ish.
        if data_rows and all(len(r) == len(header) for r in data_rows):
            col_width = _column_width_for_terminal(len(header))
            wrapped_header = [_wrap(h, col_width) for h in header]
            wrapped_rows = [[_wrap(cell, col_width) for cell in row] for row in data_rows]
            rendered = tabulate(wrapped_rows, headers=wrapped_header, tablefmt="grid")
            output_lines.append(rendered)
        else:
            # Not a well-formed table -- emit the original lines untouched.
            output_lines.extend(lines[i:block_end])

        i = block_end

    return "\n".join(output_lines)