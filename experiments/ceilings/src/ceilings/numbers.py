"""Entries for the paper's number registry (`paper/numbers.json`).

The paper's camera-ready checker reruns each entry's command and fails if the output no longer
matches. `ceilings show` prints one result per line in a fixed format; this module turns chosen
results into registry entries whose command is `ceilings show ...`, run from
`experiments/ceilings` (relative to `paper/`).
"""

from __future__ import annotations

import re

import pandas as pd

CWD_FROM_PAPER = "../experiments/ceilings"


def show_line(row: dict) -> str:
    """One stable, greppable line per gain-table row."""
    return (
        f"gain model={row['model']} family={row['family']} arm={row['arm']} "
        f"delta={row['delta']} n={row['n']} gain={row['gain']:+.3f} "
        f"ci=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
        f"baseline_acc={row['baseline_acc']:.3f} arm_acc={row['arm_acc']:.3f}"
    )


def entries(table: pd.DataFrame, root: str, rows: list[dict] | None = None,
            note_prefix: str = "") -> list[dict]:
    """Registry entries for the given gain-table rows (all rows when None)."""
    out = []
    for row in (rows if rows is not None else table.to_dict("records")):
        line = show_line(row)
        cmd = (f"ceilings --root {root} show --model {row['model']} --family {row['family']} "
               f"--arm {row['arm']} --delta {row['delta']}")
        out.append({
            "value": f"{row['gain']:+.3f}",
            "command": cmd,
            "cwd": CWD_FROM_PAPER,
            "expect": re.escape(line),
            "note": f"{note_prefix}gain of {row['arm']} over baseline, {row['family']}, "
                    f"{row['model']}, delta {row['delta']}",
        })
    return out
