# ceilings: testing the Coffee Law's three ceilings

An experiment harness for the reviewer's request on *The Coffee Law* (Bhatt, 2026): separate the
**information**, **reasoning** and **communication** ceilings of agent teams under matched
resource budgets.

- Design: [specs/001-ceiling-discrimination-experiments/](specs/001-ceiling-discrimination-experiments/) (spec, plan, research, data model, contracts, tasks)
- Background: [litreview-coffee-law.md](litreview-coffee-law.md) (an explainer of the paper)
- Paper and formal proofs: [../../paper/](../../paper/), [../../formal_proofs/](../../formal_proofs/)
- Notebook: [notebooks/ceilings_small_run.ipynb](notebooks/ceilings_small_run.ipynb)

## Install (Python 3.10+)

```bash
pip install -e ".[dev,notebook]"
```

## Run the small suite

The default model is `sim-bounded`, a simulated agent whose ceilings are known by construction.
It needs no API key and costs nothing, and it validates the pipeline, not the theory.

```bash
pytest -q                                  # 68 tests: theory, contracts, generators, analysis, end to end
ceilings prepare --n-main 200 --n-pilot 200  # instances + calibration gate + cost matching
ceilings run                               # 9 arms x 3 deltas x 4 families
ceilings sweep --n 150                     # Fano-floor (H6) and message-bit (H7) sweeps
ceilings analyze                           # reports/gain_table.csv, predictions.json, tests.csv
ceilings audit                             # reports/audit.json
```

Or open the notebook, which does all of the above and plots the results.

## On GCP

This project lives in `experiments/ceilings/` of the private repository `Ishan-K-S/coffeelaw`.
The rest of the repository is the upstream paper code from `mbhatt1/coffeelaw`.

```bash
git clone https://github.com/Ishan-K-S/coffeelaw.git   # private: authenticate first
cd coffeelaw/experiments/ceilings
```

- **Vertex AI Workbench or a VM with Jupyter:** open `notebooks/ceilings_small_run.ipynb`. Its
  first cell installs the package.
- **Docker:**
  ```bash
  docker build -t ceilings .
  docker run --rm -v "$PWD":/work ceilings                    # test suite
  docker run --rm -p 8888:8888 -v "$PWD":/work ceilings \
    jupyter notebook --ip 0.0.0.0 --allow-root --no-browser    # notebook server
  ```

## Real models

`claude-opus-5-5` is wired in (`src/ceilings/adapters/anthropic_claude.py`). It sets effort
explicitly, uses structured output for the answer, and keeps refusal fallbacks **off** so that no
model substitution can occur. Families B and C must be chosen before a real main run (tasks T038,
T039), and every model-calling command takes `--max-cost`. Freeze the pre-registration first:

```bash
ceilings prereg freeze
```
