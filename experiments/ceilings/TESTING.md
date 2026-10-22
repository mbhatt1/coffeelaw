# Test runs before handing over

What was run to check the harness end to end with a real model, what it found, and what changed
as a result. None of these runs is evidence about the Coffee Law: the model is far too small, and
n is tiny.

## Setup

- Model: `ollama/gemma3:1b` on an Apple M2 Pro laptop, through Ollama.
- Throttled: one model call at a time, Ollama busy at most 50% of the time
  (`CEILINGS_OLLAMA_DUTY=0.5`), 4 threads, short reasoning budgets, and 60 s cool-downs.
- Watchdog (`scripts/watchdog.py`): harness container capped at 1.5 CPUs and 2 GB; kill on any macOS
  thermal warning, Ollama above 300% CPU for 60 s, or 2.5 h. Its kill path was tested first with a
  CPU-burning container (killed after 12 s).
- After each run, three independent reviewers (subagents with fresh context, reading files only)
  audited the transcripts, the model's behaviour, and the statistics.

## Run 1: `scripts/local_smoke.sh` (236 team runs, about 75 min)

The run itself was clean. Ollama's CPU peaked at 64% with no thermal warnings, every answer
parsed, and no runs were invalid. An independent integrity audit rebuilt every prompt character
for character and found nothing wrong in leakage, private-evidence isolation, message budgets, arm
configurations or token ledgers.

gemma3:1b failed every calibration gate. Even given everything it needed (every attribute probed
exactly), it picked the right host among 4 only half the time. The gate rejected it as designed.

The reviews also found these harness and analysis problems, now fixed:

| Problem | Fix |
|---|---|
| The answer schema asked for the answer *before* the probabilities, so constrained decoding made the model commit first. In about half of the reasoning-family calls the answer contradicted the model's own probabilities. | Probabilities come first; the prompt says they must sum to 1 and the answer must be the most probable option; `ceilings check` reports both rates. |
| Agent A was never told what agent B's bits meant, and B was told it *could* send its list. B sent `0` in all 40 messages. | The message format is fixed per family and stated to both agents (parity: one parity bit; inner product: one bit per skill, in order). |
| The reasoning arm was too short to work (every reasoning stage hit its cap), and the prompt mixed "answer" with "reason". | Larger default budgets, with guidance of about 20+ tokens per reasoning step; the reasoning turn says not to answer yet; a truncation note goes to the final turn. |
| The information gate was nearly unpassable even for a perfect reader (3 noisy probes per attribute top out around 92%). | Full strength uses 5 probes per attribute when the probe is noisy (Bayes-optimal accuracy about 97%). |
| The information prompt described using a probe in arms that had none. | It now describes probe *reports* the team may receive, and says plainly when probes are exact. |
| The answer options were always in the same order and temperature was left to each model's default; the model favoured the last-named option. | Options are shuffled per call; the even/odd question alternates its wording; temperature is always sent (default 0.7). |
| The parity and inner-product families shared the same private lists. | The variant is part of the seed (instance version 2). |
| Sweep instances were not saved, so their runs were silently skipped by the checks. | Saved; a run without its instance is now an integrity failure. |
| **Statistics:** identical outcomes gave an equivalence p-value of 0 (zero standard error), and a rule outside the pre-registration turned "the matching arm did not help" into "contradicted" (H3). | Exact McNemar superiority and Newcombe paired-interval TOST; each hypothesis is contradicted only by its pre-registered falsifier, with Holm-adjusted p-values. |
| **Statistics:** H7 was "contradicted" by an overfitted model comparison on a flat line at n=4. | Instance-grouped cross-validation, clipped fits, and contradiction only on positive evidence of a steady rise from the first bits. |
| **Statistics:** H6 was "supported" from cells that could never reject. | Only cells that could reject at all count; otherwise inconclusive. |
| **Statistics:** verdicts ignored failed gates and sample size, and one model could decide. | A **not evaluable** verdict (with reasons) for a failed gate, n below the pre-registered size, or a missing arm; at least 2 evaluable models must agree. |

## Run 2: `scripts/local_confirm.sh` (248 team runs, about 1 h 50 min)

This run used every arm (including P1, A2 and A2_single_ref) and both sweeps, at fixed easy
difficulties, with the fixes above applied. Ollama's CPU peaked at 62%, with no thermal warnings
in 120 samples.

- **Integrity:** all 7 checks pass on all 248 runs, now including the sweep runs.
- **Analysis:** every prediction is **not evaluable**, with the reason "calibration gate failed"
  for each family. That is the correct outcome for a model that failed every gate. No arm was
  missing, and nothing was falsely "contradicted".
- **Information family, read directly from transcripts:** the probe reports were correct and
  unambiguous. For example, "rack A, row back, zone west" matches exactly one row of the table,
  host-06. gemma answered host-07, the table's last row, in almost every run, with or without
  evidence. That is model weakness; the option order in the answer schema was shuffled as
  intended.
- **Quality warnings now visible:** the answer is not the model's own most probable option in up
  to 67% of calls in a cell, the probabilities rarely sum to 1, and the reasoning stage ran past
  even 320 tokens. These are gemma's, and `ceilings check` reports them.
- **One more harness problem:** agent B followed the message format in 0 of 12 runs. Its replies
  were literally "0s and 1s", "0s" and "0s and 0s", copied from the instruction's wording.
  - Fix: the instruction no longer contains a phrase that can be copied as an answer.
  - Fix: B's private notes now include its message already written in the agreed format. This is
    a function of B's own list, so it adds no information, but B's encoding work no longer
    confounds the communication test.

## Run 3: communication families only, after the message fix (72 team runs, about 12 min)

This run covered the baseline and M arms plus the message-bit sweep (L = 0, 1, 4) for
`comm_parity` and `comm_ip` at n = 4. Ollama's CPU peaked at 58%, with no thermal warnings.

- **Agent B followed the message format in 28 of 28 messages** (0 of 12 in run 2), with no
  rejections. Every integrity check passes.
- Agent A still mostly failed to combine B's bit with its own list. For example, comm_parity with
  the correct parity bit was right in 2 of 6 runs. The channel now delivers the information
  correctly, and using it is the model's job. A 1B model can't, which is why every gate fails.

## Final independent verification (runs 2 and 3)

A fresh reviewer checked every fix against the data from runs 2 and 3:

- **Probabilities first:** all 1040 answer calls returned `probabilities` before `answer`, and
  1040/1040 prompts state the answer rules. Option order varied per call (info: 210 distinct orders
  in 212 calls).
- **Information prompt:** 236/236 info prompts describe probe reports, with correct rotation and
  probe results.
- **Communication:** both agents saw the message format, A saw B's bits in 112/112 calls, and B
  sent its ready-made message in 28/28 cases. The parity and inner-product families never share
  private lists, and the even/odd wording alternates.
- **Checks and analysis:** every sweep run has its saved instance, and every verdict is "not
  evaluable" with gate-failure reasons.

It also found four loose ends, all fixed afterwards and covered by tests:

- The A2 arm called the model at each relay hop even when the bit budget was 0. No message calls
  are made now when nothing can cross.
- The phrase "0s and 1s" was still in the inner-product format text and in the channel's retry
  text. It is now removed from every prompt.
- Transcripts did not record the reasoning turn or the final-answer turn. Every turn sent is now
  logged under `call.sent`.
- The H6 and H7 "not evaluable" verdicts gave no reasons; they now do.

Not exercised with a real model: the noisy-probe information path (5 probes per attribute), the
flip-rate sweep, and the default (unthrottled) reasoning budgets. The simulated-agent tests cover
the noisy path. Runs 2 and 3 used `--min-n 0 --min-models 1` because they are smoke tests; the
sample-size and 2-model rules are covered by unit tests.

## What this means for real runs

- The pipeline, the transcripts and the checks work end to end with a real local model, under hard
  resource caps.
- gemma3:1b is too weak to show any ceiling cleanly. Expect to need models that pass
  `ceilings calibrate`; families no setting passes are reported, not hidden.
- Before trusting a new model, read a few transcripts per family. Use
  `ceilings check --model <m>` for the rates (parse failures, answers that contradict their own
  probabilities, rejected messages, truncated reasoning) and look at the raw replies it points to.
