# Aligning verl-gr RankGRPO with TRL — Progress by Target Item

Goal: make verl-gr RankGRPO match or exceed TRL's convergence rate and compute
efficiency.

Reference target analysis:
`docs/verl_gr/rankgrpo/aligning_target_rankgrpo.md`.

This document is organized in the same order as the target analysis. It tracks
what has been aligned, what remains different, and what still needs evidence
from fresh runs.

Status legend:

- **Done**: code/config default has been changed and the target behavior is now
  represented by the default launch path.
- **Partial**: a likely cause has been addressed, but the run evidence is not
  complete yet.
- **Pending**: known target item is not implemented or not yet tested.

## Current Status Snapshot

| Target item | Status | Current state | Remaining work |
|---|---|---|---|
| 1.1 Effective batch size | **Done** | good `fp32opt` run uses 6 unique prompts, 8 rollouts, 48 generated sequences, 6 accumulation micro-batches, 1 optimizer step | Continue using this as baseline |
| 1.2 PPO clip ratio | **Done** | verl-gr defaults to `[0.94, 1.08]`, matching TRL | Monitor `actor/pg_clipfrac` |
| 1.3 PPO epochs / sequence reuse | **Done** | good `fp32opt` run shows `ppo_epochs=1`, matching TRL `mu=1` | Keep no override in future runs |
| 1.4 Other aligned hparams | **Done** | LR, KL coefficient, Adam betas, shuffle behavior, rollout count, seed, and actor dtype defaults are aligned in `fp32opt` | Keep these fixed for follow-ups |
| 1.5 Distributed backend | **Pending / known difference** | TRL uses DeepSpeed ZeRO-3 + colocated vLLM TP=2; the good verl-gr run still uses FSDP2 + Ray hybrid engine with rollout TP=1/DP=2 | Test TP=2 separately if needed |
| 2. Compute performance | **Pending** | Batch work per optimizer step is aligned; structural overhead remains | Measure wall-clock and phase timing |
| 3. Convergence analysis | **Done for KL** | good run `g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt` shows `actor/kl_loss` growing in the same range as TRL | Continue reward/throughput comparisons |
| 5. Recommended fixes | **Partial** | confirmed-good defaults are in place through `fp32opt`; old-logprob and TP=2 experiments are not part of the good run | Keep follow-ups separate |
| 6. Verification plan | **Pending** | checklist exists below | Run and record evidence |

## 1. Hyperparameter Analysis

### 1.1 Effective Batch Size: Done

Accomplished:

- Corrected the interpretation of TRL's `generation_batch_size`.
- Moved TRL-alignment defaults into `scripts/run_rankgrpo.sh`, so
  `.match_rankgrpo.sh` only keeps endpoint-specific GPU/Ray/output setup.
- Added `data.gen_batch_size=6` as unique prompts per optimizer step.
- Set actor `global_batch_size = mini_batch_size = gen_batch_size × rollout.n`
  in `verl_080_dev/verl/trainer/ppo/ray_trainer.py`, so one actor update uses
  one mini-batch and lets the engine accumulate gradients across micro-batches.
- Disabled dynamic actor micro-batching for the aligned run and set
  `ppo_micro_batch_size_per_gpu=4`, giving 6 micro-batches per optimizer step on
  2 GPUs.

Current verl-gr defaults:

```text
TRAIN_BATCH_SIZE              = 1   unique prompt per micro-batch
GRADIENT_ACCUMULATION_STEPS   = 6
GEN_BATCH_SIZE                = 1 × 6 = 6 unique prompts per optimizer step
ROLLOUT_N                     = 8
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU = 4 sequences/GPU

Total generated sequences per optimizer step = 6 × 8 = 48
```

#### TRL Batch Logic

TRL has three related but different quantities:

- `num_generations`: GRPO group size `G`. Each unique prompt is sampled `G`
  times so rewards can be normalized within the same-prompt group.
- `per_device_train_batch_size × num_processes`: generated sequence slots
  processed by one global forward/backward micro-step.
- `gradient_accumulation_steps`: number of micro-steps accumulated before
  `optimizer.step()`.

Therefore TRL's `generation_batch_size` is **not** the number of unique prompts.
It is the number of generated sequence slots consumed before one optimizer step:

```text
generation_batch_size
  = per_device_train_batch_size × num_processes × gradient_accumulation_steps
  = 4 × 2 × 6
  = 48 generated sequence slots
```

Because `num_generations=8`, those 48 slots correspond to:

```text
unique prompts per optimizer step
  = generation_batch_size / num_generations
  = 48 / 8
  = 6 unique prompts
```

In the current 2-GPU reference run, `gradient_accumulation_steps=6` happens to
equal the number of unique prompts per optimizer step. That equality is
incidental:

```text
per_device_train_batch_size × num_processes
  = 4 × 2
  = 8 generated sequence slots per micro-step
  = num_generations
```

So each micro-step contains exactly one prompt group:

```text
micro-step 1: prompt A × 8 generations
micro-step 2: prompt B × 8 generations
...
micro-step 6: prompt F × 8 generations
optimizer.step()
```

If `per_device_train_batch_size` changed to 8 while keeping
`num_processes=2`, `gradient_accumulation_steps=6`, and `num_generations=8`,
then:

```text
generation_batch_size = 8 × 2 × 6 = 96 slots
unique prompts        = 96 / 8 = 12 prompts
```

`gradient_accumulation_steps` would still be 6, but each micro-step would
contain 16 generated sequence slots = 2 unique prompts × 8 generations. This is
why `gradient_accumulation_steps` cannot be treated as the unique-prompt batch
size in general.

#### verl-gr Mapping

```text
verl-gr:
  DataLoader yields gen_batch_size=6 unique prompts
  Generate 6 × 8 = 48 completions
  _update_actor:
    global_batch_size = mini_batch_size = 48
    → train_mini_batch creates 1 mini-batch of 48 seq (24 seq/GPU)
  Engine.train_batch:
    Splits 24 seq/GPU into micro-batches of 4 seq/GPU
    → 6 micro-batches per optimizer step when ppo_epochs=1
    → 1 optimizer.step()
```

Progress-bar denominator implication:

```text
383013 dataset prompts // 6 unique prompts per optimizer step = 63835 steps
```

The previous `gen_batch_size=48` verl-gr setting meant 48 unique prompts × 8
rollouts = 384 generated sequences per update, which was not equivalent to TRL.

Remaining verification:

- Confirm `data.gen_batch_size=6` in the Hydra dump.
- Confirm actor `global_batch_size=48` and `mini_batch_size=48` in debug logs.
- Confirm one optimizer update per 6 unique prompts.

### 1.2 PPO Clip Ratio: Done

Accomplished:

- `scripts/run_rankgrpo.sh` now defaults:

```bash
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.06}"
PPO_CLIP_RATIO_HIGH="${PPO_CLIP_RATIO_HIGH:-0.08}"
```

- These values are passed to:

```text
actor_rollout_ref.actor.clip_ratio=0.06
actor_rollout_ref.actor.clip_ratio_low=0.06
actor_rollout_ref.actor.clip_ratio_high=0.08
```

This matches TRL's effective clip range `[0.94, 1.08]`.

Remaining verification:

- Monitor `actor/pg_clipfrac`; it should not immediately saturate under the
  aligned default.

### 1.3 PPO Epochs / Sequence Reuse: Done

Accomplished:

- `scripts/run_rankgrpo.sh` now defaults `PPO_EPOCHS=1`.
- This matches TRL's `--mu 1` / `num_iterations=1`.
- Each generated sequence is used for one forward/backward pass before the next
  optimizer step.

Remaining verification:

- Confirm launched jobs do not override `PPO_EPOCHS`.
- Compare `actor/pg_loss`, KL, and reward with one pass per generated batch.

### 1.4 Other Aligned Hyperparameters: Done

Current defaults aligned with TRL:

| Parameter | TRL | verl-gr |
|---|---|---|
| Learning rate | `1e-6` | `LEARNING_RATE=1e-6` |
| KL coefficient | `1e-3` | `KL_COEF=1e-3` |
| Adam beta1/beta2 | `0.9 / 0.99` | `0.9 / 0.99` |
| Rollouts per prompt | `num_generations=8` | `ROLLOUT_N=8` |
| Prompt/completion length | `2048 / 1024` | `2048 / 1024` |
| Seed | `3407` | `SEED=3407` |
| Train shuffle | enabled | `DATA_SHUFFLE=True` |
| Validation shuffle | disabled | `VALIDATION_SHUFFLE=False` |
| Actor train dtype | fp32 master/mixed bf16 | `ACTOR_MODEL_DTYPE=fp32` + FSDP mixed precision |

Accomplished:

- Validation now follows TRL's `--no-val_shuffle` behavior.
- The trainable actor now defaults to fp32 loading to avoid quantizing
  `lr=1e-6` AdamW updates into bf16 parameters.

Confirmed in the good `fp32opt` run:

- Log shows `Train/validation shuffle: True/False`.
- Hydra dump shows actor `model_dtype: 'fp32'`.
- Hydra dump shows `ppo_epochs: 1`.
- Hydra dump shows `validation_shuffle: False`.

### 1.5 Distributed Backend Differences: Pending / Known Difference

Not aligned yet:

| Aspect | TRL | verl-gr current default |
|---|---|---|
| Training backend | DeepSpeed ZeRO-3 | FSDP2 |
| Runtime topology | Accelerate trainer process | Ray hybrid engine |
| vLLM integration | colocated | Ray rollout workers |
| vLLM tensor parallelism | TP=2 | TP=1, DP=2 in the confirmed good `fp32opt` run |

Current decision:

- Keep the confirmed good `fp32opt` run as the baseline. Do not fold TP=2 into
  that baseline until a separate TP=2 run proves it preserves KL/reward behavior.

Remaining verification:

- Compare a future TP=2 run against the confirmed TP=1/DP=2 `fp32opt` run under
  the same batch/clip/epoch/dtype settings.
- Compare checkpoint parameter drift and optimizer-state dtypes between TRL and
  verl-gr.

## 2. Compute Performance Analysis

### 2.1 Per-Optimizer-Step Work Breakdown: Partial

Accomplished:

- The amount of training data per optimizer step is now aligned:

```text
6 unique prompts × 8 rollouts = 48 generated sequences
```

Still different:

- TRL computes generation, old policy log-probs, and reference log-probs in a
  more colocated/inlined path.
- verl-gr still performs separate Ray phases for generation, old log-prob,
  reference log-prob, and actor update.

Current verl-gr step shape:

```text
1. vLLM rollout: 6 prompts × 8 = 48 completions
2. old_log_prob: separate actor forward over 48 sequences
3. ref_log_prob: separate ref forward over 48 sequences
4. actor update: 6 fixed micro-batches, 1 optimizer step
```

### 2.2 Why verl-gr Is Slower: Pending

Known remaining speed differences:

- Separate `old_log_prob` forward pass.
- Separate `ref_log_prob` forward pass.
- Ray RPC/DataProto boundaries between phases.
- vLLM remains TP=1/DP=2 in the confirmed good run, unlike TRL's colocated TP=2
  group.
- FSDP2/Ray memory layout differs from DeepSpeed ZeRO-3.

No fix is marked done for these structural speed items yet. The batch fix makes
the comparison fair, but it does not remove the extra forward/RPC overhead.

### 2.3 Timing Budget: Pending

Remaining evidence to collect:

- Wall-clock time per 100 optimizer steps.
- Rollout generation tokens/sec.
- old/ref log-prob phase time.
- Actor update phase time.
- End-to-end comparison against the TRL run after batching and dtype alignment.

## 3. Training Convergence Analysis

### 3.1 Advantage Noise / Batch-Size Mismatch: Done

Accomplished:

- Both systems now use 6 prompt groups per optimizer step.
- Both systems use 8 rollouts per prompt.
- The previous "TRL has 48 unique prompts while verl-gr has 6" concern is
  resolved; it came from treating TRL generated slots as unique prompts.

Remaining verification:

- Compare reward variance and eval reward slope after the fresh aligned run.

### 3.2 Clip Range: Done

Accomplished:

- Both systems now use the same item-level trust region, `[0.94, 1.08]`.

Remaining verification:

- Compare `actor/pg_clipfrac` and KL against TRL.

### 3.3 PPO Epochs: Done

Accomplished:

- verl-gr uses `PPO_EPOCHS=1`, matching TRL `mu=1`.
- Historical concerns about 12 repeated PPO epochs no longer apply to the
  default aligned launch.

Remaining verification:

- Confirm no launch-time override.

### 3.4 Sample Diversity and Generalization: Done

Accomplished:

- Both systems now consume 6 new unique prompts per optimizer step.
- Both systems therefore cover a dataset of `N` prompts in about `N / 6`
  optimizer steps.

Remaining verification:

- Confirm progress-bar denominator near `383013 // 6 = 63835`.

### 3.5 KL Divergence Dynamics: Done for the confirmed fp32 actor run

Accomplished:

- Batch size, clip range, PPO epochs, shuffle behavior, and actor train dtype
  have been aligned.
- `ACTOR_MODEL_DTYPE=fp32` is now the default, addressing the observed flat
  `actor/kl_loss` failure mode caused by bf16 actor parameters and bf16 AdamW
  moments at `lr=1e-6`.
- The good run `g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt` confirms
  KL is no longer flat: `actor/kl_loss` went from `0.000167` at step 10 to
  `0.020529` at step 360. The TRL reference was `0.000064` at step 10 and
  `0.021844` at step 360.

Still open:

- Reward and throughput still need comparison after accepting the KL behavior.
- Remaining backend differences still include old/ref log-prob plumbing, vLLM
  topology, FSDP2 vs ZeRO-3, and Ray boundaries.

Next checks:

- Keep `fp32opt` as the baseline for follow-up experiments.
- Compare reward, clipfrac, throughput, and checkpoint drift at the same steps.
- Test old-log-prob semantics and rollout TP=2 only as separate experiments.

## 4. Root Causes Summary

### 4.1 Primary Convergence Causes

| Cause | Status | What changed | Remaining work |
|---|---|---|---|
| Unique prompts per step | **Done** | `GEN_BATCH_SIZE=6`; actor global/mini batch = 48 seq | Verify fresh logs |
| Clip epsilon | **Done** | `0.06 / 0.08` defaults | Monitor clipfrac |
| PPO epochs | **Done** | `PPO_EPOCHS=1` default | Confirm no override |
| Actor update precision | **Done for KL** | `ACTOR_MODEL_DTYPE=fp32` default; `fp32opt` run shows KL growth comparable to TRL | Continue reward/drift checks |

### 4.2 Secondary Speed Causes

| Cause | Status | Current state |
|---|---|---|
| Separate forward passes | **Pending** | old/ref log-probs still separate |
| vLLM TP topology | **Pending** | confirmed good run uses TP=1/DP=2; TP=2 should be tested separately |
| Distributed backend | **Pending** | FSDP2/Ray remains different from ZeRO-3 |
| Ray RPC overhead | **Pending** | not optimized yet |

## 5. Recommended Fixes from Target Doc

### Already Applied

- Correct unique-prompt batch size: `GEN_BATCH_SIZE=6`.
- Fixed micro-batching for gradient accumulation:
  `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=4`.
- Single actor mini-batch per update:
  `global_batch_size = mini_batch_size = gen_batch_size × rollout.n`.
- Tight TRL clip range: `0.06 / 0.08`.
- Single pass per generated sequence: `PPO_EPOCHS=1`.
- Train shuffle enabled and validation shuffle disabled.
- fp32 actor load for trainable parameters: `ACTOR_MODEL_DTYPE=fp32`.

### Not Applied Yet

- `old_log_prob_mode=current`
  - Not part of the good `fp32opt` run.
  - Expected impact: closer TRL PPO-anchor semantics; needs a separate test
    before being folded into defaults.
- `RANKGRPO_BYPASS_OLD_LOG_PROB=True`
  - Current default: `False`.
  - Expected impact: speed improvement by avoiding one separate actor forward.
  - Convergence impact: should be tested carefully because it changes which
    old log-prob source is trusted.
- `ROLLOUT_TENSOR_PARALLEL_SIZE=2`
  - Current default for the good `fp32opt` run: `1`.
  - Expected impact: closer to TRL's vLLM TP=2 generation topology and possibly
    better rollout throughput.

## 6. Verification Plan

Fresh aligned run checklist:

- [x] Hydra/logs show `data.gen_batch_size=6`.
- [x] Hydra dump shows `actor_rollout_ref.actor.ppo_epochs=1`.
- [x] Hydra dump shows clip low/high `0.06 / 0.08`.
- [x] Hydra dump shows `actor_rollout_ref.actor.fsdp_config.model_dtype=fp32`.
- [x] Logs show total actor batch of 48 generated sequences per optimizer step.
- [x] Logs show 6 fixed micro-batches of 4 seq/GPU per optimizer step.
- [ ] Progress denominator is about `383013 // 6 = 63835`.
- [x] `actor/kl_loss` grows comparably to TRL `train/kl` in the good `fp32opt`
  run.
- [ ] Checkpoint parameter drift is no longer bf16-quantized away.
- [ ] `eval/reward_total` slope is compared against the TRL baseline.
- [ ] Wall-clock time per 100 optimizer steps is recorded.
- [ ] If convergence still differs beyond KL, test old-logprob semantics
  separately.
- [ ] If speed or generation behavior still differs, test rollout TP=2
  separately.

## Implementation Inventory

| File | Current role |
|---|---|
| `scripts/run_rankgrpo.sh` | Owns alignment defaults and passes Hydra overrides |
| `scripts/.match_rankgrpo.sh` | Endpoint-specific GPU/Ray/output/resume wrapper |
| `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml` | RankGRPO base config; actor dtype default is fp32 |
| `verl_080_dev/verl/trainer/ppo/ray_trainer.py` | Actor update uses `gen_batch_size × rollout.n` and one full mini-batch |
| `verl_gr/recipes/rankgrpo/rankgrpo_loss.py` | TRL-matched RankGRPO PPO loss path |
| `verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py` | Per-prompt-group RankGRPO advantage computation |
