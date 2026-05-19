# Rank-GRPO Algorithm Parity Test

Golden-value end-to-end guardrail for the Rank-GRPO recipe. Deterministic seeded
inputs run through the full algorithm pipeline; every intermediate and final
output is compared against committed golden values.

## Purpose

Rank-GRPO's correctness depends on five pure functions (reward parsing, token
segmentation, advantage normalization, loss computation, length shaping) plus
one engine patch (`div_` hook). When the training backend changes — FSDP2 to
DDP, CPU prefix trie to GPU vector trie — none of these functions should produce
different outputs. This test catches regressions before they reach a training run.

## Components exercised

| # | Component | File | What is tested |
|---|-----------|------|----------------|
| 1 | Reward parsing | `rankgrpo_reward.py` | `rank_rewards_from_text` — format stripping, regex parsing, year extraction, catalog matching, seen-title exclusion, year tolerance |
| 2 | Token segmentation | `rankgrpo_algorithm.py` | `_segment_rank_tokens` — separator detection, segment id assignment, overflow detection, EOS mask |
| 3 | Advantage normalization | `rankgrpo_algorithm.py` | `compute_rank_grpo_advantage` — per-prompt group mean centering, std normalization, length shaping (EOS reward, early-stop penalty, overflow penalty) |
| 4 | TRL-matched loss | `rankgrpo_loss.py` | `_trl_clipped_pg_loss` + `_compute_item_mean_log_ratio` — per-item geometric-mean log-ratios, exp-coef clip, per-token KL, seq-mean-token-mean aggregation |
| 5 | verl-default loss | `rankgrpo_loss.py` | `_item_level_log_prob` + `compute_policy_loss_vanilla` — dual-clip PPO with item-level importance sampling |
| 6 | Gradient hooks | `grad_hooks.py` | `install_grad_hooks` — verifies `FSDPEngineWithLMHead.prepare_model_outputs` is patched; verifies `torch.Tensor.div_` is restored after the call |
| 7 | Agent loop grouping | `rankgrpo_agent_loop.py` | `RankGRPOAgentLoopWorker._group_repeated_prompts` — contiguous prompt grouping, interleaved handling |

## Test data design

Six hand-crafted prompt/response pairs cover the reward and advantage edge cases:

| Case | Description | Expected behavior |
|------|-------------|-------------------|
| A — all-matched | 5 items, all in catalog, all match ground truth, no seen overlap | All 5 positions = 1.0 reward |
| B — partial-match | 5 items, positions 0,2,4 match GT; 1,3 are wrong titles | Rewards = `[1, 0, 1, 0, 1]` |
| C — seen-excluded | 3 items, position 1 was already seen by the user | Rewards = `[1, 0, 1]` (position 1 skipped) |
| D — early-stop | EOS after 3 items with `rec_num=5`, all 3 items match | 3× reward=1 + early-stop EOS penalty applied |
| E — overflow | 7 items generated for `rec_num=5`, items 5-6 are overflow tokens | First 5 positions matched, overflow tokens get extra penalty |
| F — empty | Response is garbled, no valid recs parsed | All zeros, no EOS, no length shaping triggered |

Three prompts are duplicated (A and A', B and B') to exercise per-prompt group
advantage normalization — the duplicated prompts form a group whose mean reward
is used for centering.

## Golden values

The test commits golden `.pt` tensors and scalar values in
`tests/rankgrpo_golden/`. Each file corresponds to an intermediate or final
output:

```
tests/rankgrpo_golden/
  rank_rewards.pt             # per-position rewards (N, rec_num)
  advantages.pt               # token-level advantages after group norm + length shaping (N, T)
  rank_seg_ids.pt             # token-to-item assignment (N, T)
  rank_token_mask.pt          # valid rank tokens (N, T)
  item_token_mask.pt          # rank + overflow token mask (N, T)
  overflow_token_mask.pt      # overflow token mask (N, T)
  eos_mask.pt                 # EOS token positions (N, T)
  loss_trl_match.pt           # scalar: aggregated policy loss (trl_match path)
  loss_verl_default.pt        # scalar: aggregated policy loss (verl_default path)
  pg_clipfrac_trl.pt          # scalar: clip fraction (trl_match path)
  item_mean_log_ratio.pt      # per-token item-level log importance weights (N, T)
```

### Golden-value contract

- All tensors are `torch.float32` (scalars) or `torch.float32` / `torch.long` (batched tensors).
- Golden values are generated with a fixed seed (`torch.manual_seed(42)`) and
  deterministic tokenizer behavior.
- The `rec_num`, `rank_separator`, `year_tolerance`, `exclude_seen`, and
  `apply_extra_length_shaping` config values are embedded in the golden
  generation script and committed alongside the tensor files.
- Any intentional algorithm change that modifies these values must regenerate
  the golden files and document the reason.

## Test execution

```bash
# Run from repo root
python tests/test_rankgrpo_parity.py

# With verbose diff output on failure
python tests/test_rankgrpo_parity.py --verbose

# Regenerate golden values (after intentional algorithm change)
python tests/test_rankgrpo_parity.py --regenerate
```

The test script:
1. Builds the hand-crafted test dataset (prompts, ground truths, catalogs).
2. Runs `rank_rewards_from_text` per response and checks against `rank_rewards.pt`.
3. Runs `compute_rank_grpo_advantage` (advantage normalization + length shaping)
   and checks `advantages.pt`, `rank_seg_ids.pt`, `rank_token_mask.pt`,
   `item_token_mask.pt`, `overflow_token_mask.pt`, `eos_mask.pt`.
4. Loads a toy model (Qwen2.5-0.5B-Instruct), runs one forward pass per loss
   path, and checks `loss_trl_match.pt`, `loss_verl_default.pt`,
   `pg_clipfrac_trl.pt`, `item_mean_log_ratio.pt`.
5. Verifies `install_grad_hooks()` patches and restores correctly.
6. Verifies `_group_repeated_prompts` groups contiguous repeated prompts and
   correctly handles interleaved prompts.

### Exit codes

- `0` — all golden values match within tolerance (`rtol=1e-5`, `atol=1e-8` for
  loss scalars; exact match for integer tensors).
- `1` — one or more golden values differ. Diff is printed to stderr.

## Backend-agnostic design

The test exercises the pure algorithm functions directly — it does **not**
require a running Ray cluster, GPUs, or a specific training backend. Loss-mode
tests load a real HF model via `AutoModelForCausalLM` but run on CPU and do not
wrap with FSDP or DDP. This is intentional: the same test must pass identically
before and after the DDP backend is introduced.

### What the test does NOT cover

- FSDP/DDP wrapping correctness (separate integration test)
- Ray worker group communication
- vLLM rollout server interaction
- Full training loop convergence
- Checkpoint save/load format

These belong in integration or smoke tests, not in the algorithm parity
guardrail.

## Guardrail checklist

Before claiming any major architecture change is correct, run:

- [ ] `python tests/test_rankgrpo_parity.py` passes (all golden values match)
- [ ] `python tests/test_rankgrpo_loss_modes.py` passes (loss function invariants)
- [ ] `scripts/.match_rankgrpo.sh` completes at least one training step without
      OOM, NaN loss, or gradient collapse
- [ ] Training metrics (`actor/pg_loss`, `actor/pg_clipfrac`, `eval/reward_total`)
      are within 5% of the FSDP2 baseline for the first 50 steps
