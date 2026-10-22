"""`ceilings` command-line interface (contracts/cli.md). Exit codes: 0 ok, 2 gate/validation
failure, 3 pre-registration hash mismatch, 4 budget exceeded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ceilings import experiment, prereg, tasks
from ceilings.agents import make_policy
from ceilings.audit import audit
from ceilings.runner import BudgetExceeded, load_runs, run_grid


def _difficulty(text: str | None, family: str) -> dict:
    if not text:
        return experiment.DEFAULT_DIFFICULTY[family]
    out = {}
    for part in text.split(","):
        key, value = part.split("=")
        out[key] = float(value) if "." in value else int(value)
    return out


def cmd_generate(a) -> int:
    for fam in a.family.split(","):
        insts = tasks.generate_many(fam, a.n, a.split, a.seed, **_difficulty(a.difficulty, fam))
        print(tasks.save_instances(insts, Path(a.root)))
    return 0


def cmd_prepare(a) -> int:
    policy = make_policy(a.model)
    failed = False
    for fam in a.family.split(","):
        res = experiment.prepare(
            fam, policy, a.root, a.n_main, a.n_pilot, _difficulty(a.difficulty, fam)
        )
        cal = res["calibration"]
        unmatched = [
            f"{arm}/{lvl}"
            for arm, v in res["table"]["arms"].items()
            for lvl, e in v.items()
            if not e["matched"]
        ]
        print(
            f"{fam}: gate {'PASS' if cal.passed else 'FAIL'} (baseline {cal.baseline_acc:.2f}, "
            f"chance {cal.chance_acc:.2f}, full {cal.matched_full_acc:.2f}); "
            f"unmatched cells: {unmatched or 'none'}"
        )
        failed |= not cal.passed
    return 2 if failed else 0


def cmd_run(a) -> int:
    policy = make_policy(a.model)
    total = 0
    for fam in a.family.split(","):
        table = experiment.load_table(a.root, fam, policy.model_id)
        version = experiment.tasks.GENERATORS[fam](0).family_version
        insts = tasks.load_instances(Path(a.root), fam, version, a.split)
        cells = experiment.main_cells(
            insts, table, deltas=tuple(a.delta.split(",")), arm_ids=tuple(a.arms.split(","))
        )
        total += len(
            run_grid(
                cells,
                policy,
                a.root,
                a.max_cost,
                require_frozen=a.split == "main" and a.require_frozen,
            )
        )
    print(f"new runs: {total}")
    return 0


def cmd_sweep(a) -> int:
    policy = make_policy(a.model)
    n_new = 0
    if a.family in ("info", "all"):
        Ts = [int(t) for t in a.T.split(",")]
        ps = [float(p) for p in a.flip_rate.split(",")]
        cells, floors = experiment.info_sweep_cells(a.k, Ts, ps, a.n)
        n_new += len(run_grid(cells, policy, a.root, a.max_cost))
        (Path(a.root) / "floors.json").write_text(
            json.dumps([{"T": T, "p": p, "floor": f} for (T, p), f in floors.items()])
        )
    if a.family in ("comm", "all"):
        Ls = [int(x) for x in a.L.split(",")]
        for variant in ("parity", "ip"):
            n_new += len(
                run_grid(
                    experiment.comm_sweep_cells(variant, a.n_bits, Ls, a.n),
                    policy,
                    a.root,
                    a.max_cost,
                )
            )
    print(f"new sweep runs: {n_new}")
    return 0


def _floors(root: Path) -> dict:
    path = root / "floors.json"
    if not path.exists():
        return {}
    return {(r["T"], r["p"]): r["floor"] for r in json.loads(path.read_text())}


def cmd_analyze(a) -> int:
    from ceilings.analysis.gains import runs_frame
    from ceilings.analysis.report import write_reports

    df = runs_frame(load_runs(a.root))
    res = write_reports(df, a.out, delta=a.delta, floors=_floors(Path(a.root)), n_boot=a.n_boot)
    for p in res["predictions"]:
        print(f"{p.prediction_id}: {p.verdict}")
    return 0


def cmd_audit(a) -> int:
    records = load_runs(a.root)
    tables = {}
    for path in Path(a.root).glob("arms/*/*.json"):
        t = json.loads(path.read_text())
        tables[f"{t['model_id']}/{t['family']}"] = t
    res = audit(records, prereg.current_hash(), tables)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "audit.json").write_text(json.dumps(res, indent=2))
    print(
        "audit",
        "PASSED" if res["passed"] else "FAILED",
        {k: len(v) for k, v in res["checks"].items()},
    )
    return 0 if res["passed"] else 2


def cmd_prereg(a) -> int:
    if a.action == "freeze":
        print(prereg.freeze())
    else:
        print(prereg.check_hash())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ceilings")
    p.add_argument("--root", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--family", required=True)
    g.add_argument("--difficulty")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--split", default="main")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    pr = sub.add_parser("prepare", help="generate + calibration gate + cost matching")
    pr.add_argument("--family", default="info,reason,comm_parity,comm_ip")
    pr.add_argument("--model", default="sim-bounded")
    pr.add_argument("--difficulty")
    pr.add_argument("--n-main", type=int, default=100)
    pr.add_argument("--n-pilot", type=int, default=40)
    pr.set_defaults(fn=cmd_prepare)

    r = sub.add_parser("run")
    r.add_argument("--family", default="info,reason,comm_parity,comm_ip")
    r.add_argument("--model", default="sim-bounded")
    r.add_argument("--arms", default="baseline,E,K,M,A1,A2,A2_single_ref,P1,P2")
    r.add_argument("--delta", default="small,medium,large")
    r.add_argument("--split", default="main")
    r.add_argument("--max-cost", type=float, default=None)
    r.add_argument("--require-frozen", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep")
    s.add_argument("--family", default="all", choices=["info", "comm", "all"])
    s.add_argument("--model", default="sim-bounded")
    s.add_argument("--T", default="0,1,2,3,4,6,8,12,16")
    s.add_argument("--flip-rate", default="0.0,0.1,0.2")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--L", default="0,1,2,4,6,8,10,12,14,16")
    s.add_argument("--n-bits", type=int, default=16)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--max-cost", type=float, default=None)
    s.set_defaults(fn=cmd_sweep)

    an = sub.add_parser("analyze")
    an.add_argument("--out", default="reports")
    an.add_argument("--delta", default="medium")
    an.add_argument("--n-boot", type=int, default=10_000)
    an.set_defaults(fn=cmd_analyze)

    au = sub.add_parser("audit")
    au.add_argument("--out", default="reports")
    au.set_defaults(fn=cmd_audit)

    pg = sub.add_parser("prereg")
    pg.add_argument("action", choices=["freeze", "check"])
    pg.set_defaults(fn=cmd_prereg)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except prereg.PreregHashMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except BudgetExceeded as e:
        print(f"error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
