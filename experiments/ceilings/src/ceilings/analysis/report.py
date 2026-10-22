"""Verdicts for the pre-registered predictions (FR-020) and report files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ceilings.analysis import changepoint
from ceilings.analysis.equivalence import holm, superiority, tost
from ceilings.analysis.gains import gain_table, paired_diffs
from ceilings.records import Prediction

ALPHA = 0.05
DIAGONAL = {"H1": (["info"], "E"), "H2": (["reason"], "K"), "H3": (["comm_parity", "comm_ip"], "M")}
OFF_DIAGONAL = ["E", "K", "M", "A1", "P1", "P2"]
STATEMENTS = {
    "H1": ("On information-bound tasks, E helps; K, M, A1, P1 and P2 do not", "Thm 1, Cor 1"),
    "H2": (
        "On reasoning-bound tasks, K helps; E (restated evidence), M, A1, P1, P2 do not",
        "Thm 2, Cor 3",
    ),
    "H3": ("On communication-bound tasks, M helps; K, same-kind E, A1, P1, P2 do not", "Thm 5"),
    "H4": ("A1 (more agents, same totals) gives no gain on any family", "Cor 1, Thm 3"),
    "H5": ("A2 gain equals the gain from the same compute given to one agent", "Thm 3"),
    "H6": ("Information-bound error never falls significantly below the Fano floor", "Cor 2"),
    "H7": ("Inner-product splits stay flat then rise; parity splits jump at 1 bit", "Thm 5"),
}


def _tests_for_model(df: pd.DataFrame, model: str, delta: str, margin: float) -> list[dict]:
    """Every pre-registered test for one model, with raw p-values."""
    tests: list[dict] = []
    families = set(df[df["model"] == model]["family"])
    for hid, (fams, match_arm) in DIAGONAL.items():
        for fam in fams:
            if fam not in families:
                continue
            d = paired_diffs(df, model, fam, match_arm, delta)["d_correct"].to_numpy()
            if len(d) >= 2:
                tests.append(
                    {
                        "h": hid,
                        "family": fam,
                        "arm": match_arm,
                        "kind": "superiority",
                        "gain": float(d.mean()),
                        "p": superiority(d),
                        "p_equiv": tost(d, margin),
                    }
                )
            for arm in [a for a in OFF_DIAGONAL if a != match_arm]:
                d = paired_diffs(df, model, fam, arm, delta)["d_correct"].to_numpy()
                if len(d) >= 2:
                    tests.append(
                        {
                            "h": hid,
                            "family": fam,
                            "arm": arm,
                            "kind": "equivalence",
                            "gain": float(d.mean()),
                            "p": tost(d, margin),
                            "p_sup": superiority(d),
                        }
                    )
    for fam in sorted(families):
        d = paired_diffs(df, model, fam, "A1", delta)["d_correct"].to_numpy()
        if len(d) >= 2:
            tests.append(
                {
                    "h": "H4",
                    "family": fam,
                    "arm": "A1",
                    "kind": "equivalence",
                    "gain": float(d.mean()),
                    "p": tost(d, margin),
                    "p_sup": superiority(d),
                }
            )
        d = paired_diffs(df, model, fam, "A2", delta, reference="A2_single_ref", ref_delta=delta)[
            "d_correct"
        ].to_numpy()
        if len(d) >= 2:
            tests.append(
                {
                    "h": "H5",
                    "family": fam,
                    "arm": "A2-A2_single_ref",
                    "kind": "equivalence",
                    "gain": float(d.mean()),
                    "p": tost(d, margin),
                    "p_sup": superiority(d),
                }
            )
    adj = holm([t["p"] for t in tests])
    for t, p in zip(tests, adj, strict=True):
        t["p_adj"] = p
    return tests


def _verdict(tests: list[dict], margin: float) -> str:
    if not tests:
        return "inconclusive"
    contradicted = any(
        (t["kind"] == "equivalence" and t["p_sup"] < ALPHA and t["gain"] > margin)
        or (t["kind"] == "superiority" and t["p_equiv"] < ALPHA)
        for t in tests
    )
    if contradicted:
        return "contradicted"
    if all(t["p_adj"] < ALPHA for t in tests):
        return "supported"
    return "inconclusive"


def _combine(per_model: dict[str, str]) -> str:
    """Majority rule across model families (SC-005); with fewer than 3, all must agree."""
    values = list(per_model.values())
    if not values:
        return "inconclusive"
    need = 2 if len(values) >= 3 else len(values)
    for v in ("contradicted", "supported"):
        if values.count(v) >= need:
            return v
    return "inconclusive"


def fano_verdict(df: pd.DataFrame, floors: dict[tuple[int, float], float]) -> tuple[str, list]:
    """H6: in each (T, p) cell, is the error rate significantly below the floor? (binomial test)"""
    cells = []
    sweep = df[df["is_sweep"] & (df["family"] == "info")]
    for (model, T, p), g in sweep.groupby(["model", "T", "flip_rate"]):
        floor = floors.get((int(T), float(p)), 0.0)
        errors = int((~g["correct"]).sum())
        n = len(g)
        pval = (
            float(stats.binomtest(errors, n, floor, alternative="less").pvalue)
            if floor > 0
            else 1.0
        )
        cells.append(
            {
                "model": model,
                "T": int(T),
                "p": float(p),
                "n": n,
                "error": errors / n,
                "floor": floor,
                "p_below": pval,
            }
        )
    if not cells:
        return "inconclusive", cells
    adj = holm([c["p_below"] for c in cells])
    for c, a in zip(cells, adj, strict=True):
        c["p_adj"] = a
    tested = [c for c in cells if c["floor"] > 0]
    if any(c["p_adj"] < ALPHA for c in tested):
        return "contradicted", cells
    return ("supported" if tested else "inconclusive"), cells


def threshold_verdict(df: pd.DataFrame) -> tuple[str, dict]:
    """H7 from the bit-mode message sweeps."""
    detail: dict = {}
    sweep = df[df["is_sweep"] & df["family"].isin(["comm_ip", "comm_parity"])]
    ok_ip = ok_par = None
    for (model, fam), g in sweep.groupby(["model", "family"]):
        L = g["L"].to_numpy(dtype=float)
        y = g["correct"].to_numpy(dtype=float)
        acc = g.groupby("L")["correct"].mean().to_dict()
        if fam == "comm_ip":
            res = changepoint.compare(L, y)
            res["accuracy_by_L"] = acc
            detail[f"{model}/comm_ip"] = res
            ok_ip = res["threshold_wins"]
        else:
            ceiling = acc[max(acc)]
            jump = acc.get(1.0, acc.get(1, 0.0))
            detail[f"{model}/comm_parity"] = {
                "accuracy_by_L": acc,
                "acc_L1": jump,
                "ceiling": ceiling,
            }
            ok_par = jump >= 0.9 * ceiling
    if ok_ip is None or ok_par is None:
        return "inconclusive", detail
    if ok_ip and ok_par:
        return "supported", detail
    return "contradicted", detail


def assign_verdicts(
    df: pd.DataFrame, delta: str = "medium", margin: float = 0.05, floors: dict | None = None
) -> tuple[list[Prediction], list[dict]]:
    all_tests: list[dict] = []
    per_h: dict[str, dict[str, str]] = defaultdict(dict)
    for model in sorted(set(df["model"])):
        tests = _tests_for_model(df, model, delta, margin)
        for t in tests:
            t["model"] = model
        all_tests += tests
        for hid in ["H1", "H2", "H3", "H4", "H5"]:
            per_h[hid][model] = _verdict([t for t in tests if t["h"] == hid], margin)
    preds = []
    for hid in ["H1", "H2", "H3", "H4", "H5"]:
        statement, source = STATEMENTS[hid]
        preds.append(
            Prediction(
                prediction_id=hid,
                statement=statement,
                paper_source=source,
                test="superiority + TOST, Holm-adjusted",
                verdict=_combine(per_h[hid]),
                detail={"per_model": per_h[hid]},
            )
        )
    v6, cells = fano_verdict(df, floors or {})
    preds.append(
        Prediction(
            prediction_id="H6",
            statement=STATEMENTS["H6"][0],
            paper_source=STATEMENTS["H6"][1],
            test="binomial floor check, Holm",
            verdict=v6,
            detail={"cells": cells},
        )
    )
    v7, detail = threshold_verdict(df)
    preds.append(
        Prediction(
            prediction_id="H7",
            statement=STATEMENTS["H7"][0],
            paper_source=STATEMENTS["H7"][1],
            test="change-point vs linear",
            verdict=v7,
            detail=detail,
        )
    )
    return preds, all_tests


def write_reports(
    df: pd.DataFrame,
    out: Path | str,
    delta: str = "medium",
    floors: dict | None = None,
    n_boot: int = 10_000,
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = gain_table(df, n_boot=n_boot)
    table.to_csv(out / "gain_table.csv", index=False)
    preds, tests = assign_verdicts(df, delta=delta, floors=floors)
    (out / "predictions.json").write_text(
        json.dumps([p.model_dump() for p in preds], indent=2, default=_json_default)
    )
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    return {"gain_table": table, "predictions": preds, "tests": tests}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
