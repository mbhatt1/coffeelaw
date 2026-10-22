# Contract: internal interfaces

These are the seams that keep each ceiling's mechanics separate and testable. They are described as behaviour, not code.

## TaskGenerator

- **Input**: family version, difficulty parameters, seed, split.
- **Output**: a `TaskInstance` (see data-model.md).
- **Guarantees**:
  - Deterministic: same inputs give an identical instance and `instance_id`.
  - For `info` families, the hidden answer comes only from the seeded generator and never appears in `shared_evidence` or `private_evidence`.
  - For `comm_*` families, each agent's evidence alone has `I_held_by_agent ≤ 0.01` bits, and the team's combined evidence has `I_held_team = H_theta`.
  - `theory` quantities are computed exactly by the shared theory module (the same formulas as `litreview-coffee-law-check.py`).

## ToolSimulator

- **Input**: `ToolSpec`, the instance's hidden answer, a query (the bit index for `bit_probe`), a seeded RNG stream.
- **Output**: one symbol from `answer_alphabet` (`bit_probe`), or a fixed evidence passage (`oracle`).
- **Guarantees**:
  - Flips are independent across calls, with probability exactly `flip_rate`.
  - Malformed queries return a fixed error symbol from the alphabet. They never return free text that could leak information.
  - Every call is appended to the ledger with `charge_tokens`.

## ModelAdapter (one per model family)

- **Input**: a list of messages, a reasoning-depth level (`low`…`max` or the family's equivalent), an answer schema (`answer_space`), a seed or replicate index.
- **Output**: `{final_answer, probabilities, refusal, usage: {input, output, reasoning}, request_id}`.
- **Guarantees**:
  - The final answer comes back through structured output matching the answer schema. A parse failure is retried once, then recorded as `final_answer = null`.
  - **No silent model substitution**: refusal fallbacks are disabled (research R6), and `usage`/`model` from the response must match the requested model or the run is marked invalid.
  - Usage is taken from the provider's response (or counted locally for the open-weights family), never estimated.
  - **Claude adapter specifics**:
    - Model `claude-opus-5-5`.
    - Reasoning depth via `output_config.effort`, always set explicitly (the model's default is `medium`).
    - No thinking-budget or sampling parameters (the model rejects them).
    - Streaming for long outputs.
    - Message Batches for batchable rounds, with results keyed by `custom_id` = `run_id`.

## Channel (inter-agent messaging)

- **Input**: sender, receiver, message, budget mode (`tokens:B` or `bits:L`).
- **Guarantees**:
  - The only path between agents. There is no shared memory or scratchpad.
  - In `bits:L` mode, a message not matching `[01]{0,L}` is rejected. The sender gets one retry; after that the empty message is delivered.
  - In `tokens:B` mode, the cumulative message tokens per direction never exceed B. Excess is truncated at the boundary and the truncation is logged.
  - Every delivered message is recorded in the transcript and the ledger.

## TeamOrchestrator

- **Input**: instance, arm config, ModelAdapter, Channel, ToolSimulator.
- **Behaviour**: runs a fixed protocol per arm:
  1. agents read their evidence;
  2. tool rounds (if allowed);
  3. message rounds (if allowed);
  4. answer aggregation (single answerer, or majority / probability averaging for K and A2).
- **Guarantees**:
  - The protocol is identical across arms except for the one resource the arm changes, so A1 differs from baseline *only* in agent count with the same totals.
  - The ledger total equals the sum of adapter usage, tool charges and message tokens.

## Analysis

- **Input**: run records, frozen pre-registration.
- **Output**: one `Prediction` record per H1–H9, the gain table, figures.
- **Guarantees**:
  - Uses only the tests named in the pre-registration.
  - Excluded instances are dropped by the pre-registered rules only.
  - Every number in the final report traces to a stored run record.
