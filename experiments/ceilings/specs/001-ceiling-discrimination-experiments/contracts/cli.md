# Contract: `ceilings` command-line interface

Every command is idempotent and resumable. Every command except `generate` and `prereg freeze` refuses to run if `prereg/config_frozen.json` doesn't match `prereg/HASH`. Exit codes: `0` success, `2` validation or gate failure, `3` pre-registration hash mismatch, `4` budget exceeded.

| Command | Purpose | Key options | Output |
|---|---|---|---|
| `ceilings generate` | Generate task instances for a family | `--family`, `--difficulty k=…,p=…,d=…,n=…`, `--n`, `--split`, `--seed` | `data/instances/<family>/<version>/*.jsonl` |
| `ceilings theory` | Compute exact theory quantities and attach them to instances | `--family` | Instances updated in place with `theory` (new version) |
| `ceilings calibrate` | Run the calibration gate (FR-007) on calibration-split instances | `--family`, `--model`, `--max-cost` | `data/calibration/*.json`; exit 2 if the gate fails |
| `ceilings match` | Pilot cost-matching: tune each arm's config to hit Δ within ±5% | `--family`, `--model`, `--delta small,medium,large` | `data/arms/<family>/<model>.json` with `matched` flags |
| `ceilings prereg freeze` | Freeze predictions, margins, n and analysis plan, then write the hash | `--config prereg/config.json` | `prereg/config_frozen.json`, `prereg/HASH` |
| `ceilings run` | Execute runs for a grid of family × arm × Δ × model | `--family`, `--arms`, `--delta`, `--model`, `--split main`, `--concurrency`, `--batch/--no-batch`, `--max-cost` | `data/runs/<model>/<family>/*.jsonl` + transcripts |
| `ceilings sweep` | Story 2 sweeps: tool calls T and flip rate p (info); message bits L (comm, bit mode) | `--family`, `--T 0..Tmax`, `--flip-rate 0,0.1,0.2`, `--L 0,1,2,4,8,n/2,n` | Run records tagged `sweep` |
| `ceilings analyze` | Compute gains, intervals, TOST, Holm, interaction and change-point fits; assign verdicts | `--out reports/` | `reports/predictions.json`, `reports/gain_table.csv`, figures |
| `ceilings mask` | Build masked gain tables for the blind diagnostic (Story 4) | `--split heldout_blind` | `blind/masked/*.csv`, `blind/key.enc` (encrypted key) |
| `ceilings unmask` | Reveal truth labels after every DiagnosticCall is locked | `--calls blind/calls.json` | `reports/diagnostic.json` |
| `ceilings audit` | Integrity checks: hashes, ledger totals, bit-mode compliance, Δ matching, refusals | | `reports/audit.json`; exit 2 on any failure |
| `ceilings report` | Assemble the final report against the pre-registration, listing deviations | | `reports/final.md` |

## Added during implementation

| Command | Purpose | Key options | Output |
|---|---|---|---|
| `ceilings calibrate` | Per-model difficulty search: try each family's difficulty grid until the gate passes | `--model`, `--n`, `--grid full\|quick`, `--concurrency` | `<root>/difficulty/<family>__<model>.json` (`chosen` is null if nothing passes) |
| `ceilings prepare` | Generate instances at the searched (or given) difficulty, run the gate, match costs | `--no-match` (smoke tests only; cells marked unmatched), `--delta`, `--n-match` | instances, `calibration/`, `arms/` |
| `ceilings check` | Did the runs go as designed? Integrity (FAIL), quality rates (WARN), expectations | `--model` | `reports/checks.json`; exit 2 on any integrity failure |
| `ceilings show` | Print gain rows in a fixed format, for the paper's number registry | `--model --family --arm --delta` | stdout |
| `ceilings numbers` | Write `paper/numbers.json` entries whose command is `ceilings show ...` | as `show`, `--out` | JSON entries |

`ceilings analyze` also takes `--min-n` (default: `prereg/config.json` `sample_sizes.per_cell`) and
`--min-models` (default 2). It reports **not evaluable** when a gate failed, n is too small, or an
arm is missing. Every model-calling command takes `--concurrency`.
