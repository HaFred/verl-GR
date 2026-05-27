# Aligning verl-gr RankGRPO with TRL — Progress Log

Goal: Make verl-gr RankGRPO match or exceed TRL's convergence rate and compute efficiency.

Reference: `aligning_rankgrpo.md` (root cause analysis).

---

## Current correction — 2026-05-27: TRL `generation_batch_size` counts generation slots

**Status: finished.** `run_rankgrpo.sh` now owns the TRL-alignment defaults,
and `.match_rankgrpo.sh` only keeps endpoint-specific GPU/Ray/output setup.
The default verl-gr RankGRPO launch now matches TRL's current batching and
optimizer-update behavior: `6` unique prompts, `8` generations per prompt,
`48` generated sequences, `6` accumulation micro-batches, and `1` optimizer
step.

The earlier version of this note treated TRL's `generation_batch_size=48` as
48 unique prompts. That was wrong for the current TRL RankGRPO code path.

In TRL:

- `generation_batch_size = per_device_train_batch_size × num_processes × gradient_accumulation_steps`
- For the current 2-GPU reference: `4 × 2 × 6 = 48`
- `RepeatSampler` then uses `batch_size = generation_batch_size // num_generations`
- With `num_generations=8`, that is `48 // 8 = 6` unique prompts
- Each optimizer update therefore sees `6 unique prompts × 8 generations = 48 generated sequences`

In verl-gr:

- `data.gen_batch_size` is measured in unique prompts, not repeated generation slots
- To match TRL's current optimizer-update behavior, `data.gen_batch_size` must be `6`, not `48`
- With `rollout.n=8`, verl-gr also produces `6 × 8 = 48` generated sequences per optimizer update

This correction explains the observed progress-bar denominators:

- TRL: `383013 // 6 = 63835` optimizer updates
- verl-gr with the previous `gen_batch_size=48`: about `383013 // 48 = 7975` optimizer updates
- verl-gr with corrected `gen_batch_size=6`: about `383013 // 6 = 63835` optimizer updates

## Change 1 — 2026-05-26: `use_dynamic_bsz` knob + gradient accumulation support

### Motivation

The original goal was to stop verl-gr from taking multiple small optimizer steps
inside one rollout batch. That remains valid, but the target batch size has been
corrected: current TRL sees **6 unique prompts per optimizer update**, not 48.
The desired behavior is:

- 6 unique prompts per optimizer update
- 8 rollouts per prompt
- 48 generated sequences per optimizer update
- 6 gradient-accumulation micro-batches
- 1 optimizer step after all micro-batches
- A progress-bar denominator matching TRL's optimizer-step count

### Changes

#### 1a. New env vars (`scripts/run_rankgrpo.sh`)

| Var | Default | Purpose |
|-----|---------|---------|
| `USE_DYNAMIC_BSZ` | `False` in `run_rankgrpo.sh` when gradient accumulation is enabled | Knob to disable dynamic token-based micro-batching so fixed micro-batches reproduce TRL-style accumulation |
| `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU` | `4` | Fixed micro-batch size (seq/GPU) when `USE_DYNAMIC_BSZ=False`. 4 = 8 rollout sequences split over 2 GPUs |
| `GRADIENT_ACCUMULATION_STEPS` | `6` | Number of micro-batches per optimizer step. `run_rankgrpo.sh` sets `gen_batch_size = 1 × 6 = 6` unique prompts |
| `GEN_BATCH_SIZE` | (computed) | `1 × 6 = 6` unique prompts per global step, matching TRL's `generation_batch_size=48` slots with 8 generations |

#### 1b. Wired through `run_rankgrpo.sh`

- Owns the TRL-alignment defaults so all launch wrappers get the same behavior
- Passes `data.gen_batch_size`, `actor_rollout_ref.actor.use_dynamic_bsz`, and `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` to Hydra CLI

#### 1c. Config default (`configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml`)

- Added `ppo_micro_batch_size_per_gpu` under `actor_rollout_ref.actor` and override it from the match script (only effective when `use_dynamic_bsz=False`)

#### 1d. Core change (`verl_080_dev/verl/trainer/ppo/ray_trainer.py:_update_actor`)

- `global_batch_size` now uses `gen_batch_size × rollout.n` (total sequences per optimizer step)
- `mini_batch_size` is set to `global_batch_size` — creating a single mini-batch containing all data
- Gradient accumulation happens at the **micro-batch** level inside the FSDP engine, not at the mini-batch level in `train_mini_batch`

### How verl's batch hierarchy maps to TRL's gradient accumulation

verl has a three-level batch hierarchy. Understanding this is critical to see why the implementation correctly aligns with TRL.

#### verl's three-level batch structure

```
┌─────────────────────────────────────────────────────────────┐
│  Global Batch  (global_batch_size)                          │
│  All sequences processed before one optimizer.step()        │
│  e.g. 6 prompts × 8 rollouts = 48 sequences                 │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Mini-Batch  (mini_batch_size)                        │  │
│  │  Subset of global batch processed by train_mini_batch │  │
│  │  Each mini-batch → one call to engine.train_batch()   │  │
│  │  Multiple mini-batches → multiple optimizer.step()s   │  │
│  │  e.g. 48 sequences (1 mini-batch = full global)       │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Micro-Batch  (micro_batch_size_per_gpu)        │  │  │
│  │  │  Subset of mini-batch processed by GPU in one   │  │  │
│  │  │  forward+backward pass.                         │  │  │
│  │  │  Gradients ACCUMULATE across micro-batches.     │  │  │
│  │  │  Optimizer steps only after ALL micro-batches.  │  │  │
│  │  │  e.g. 4 seq/GPU (6 micro-batches per epoch)     │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key rule**: Only micro-batch boundaries accumulate gradients. Mini-batch boundaries always trigger `optimizer.step()` + `zero_grad()`. Therefore, to get true gradient accumulation, we must have exactly **one mini-batch** and **multiple micro-batches**.

#### How this maps to TRL

TRL's gradient accumulation (`gradient_accumulation_steps=6`) works as follows:

```
TRL:
  generation_batch_size=48 repeated generation slots
  num_generations=8 → 6 unique prompts
  Generate 6 prompts × 8 = 48 completions
  Split into 6 chunks of 8 seq (1 prompt × 8 rollouts)
  For each chunk:
    forward + backward (accumulate gradients, no optimizer step)
  After all 6 chunks:
    optimizer.step() + zero_grad()
  → 1 optimizer step per 6 unique prompts / 48 generation slots
```

verl-gr achieves the same with:

```
verl-gr:
  DataLoader yields gen_batch_size=6 unique prompts
  Generate 6 × 8 = 48 completions
  _update_actor:
    global_batch_size = mini_batch_size = 48
    → train_mini_batch creates 1 mini-batch of 48 seq (24 seq/GPU)
  Engine.train_batch (FSDP2):
    Splits 24 seq/GPU into micro-batches of 4 seq/GPU
    → 6 micro-batches per epoch
    For each micro-batch:
      forward + backward (accumulate gradients, no optimizer step)
    After all 6 micro-batches:
      optimizer.step() + zero_grad()
  → 1 optimizer step per 6 unique prompts / 48 generation slots
```

The mapping is:

| TRL concept | verl-gr equivalent | Value |
|---|---|---|
| Generation batch | `gen_batch_size × rollout.n` | 48 seq |
| Gradient accumulation steps | Number of micro-batches | 6 |
| Per-micro-batch (seq/GPU) | `ppo_micro_batch_size_per_gpu` | 4 |
| Unique prompts per opt step | `gen_batch_size` | 6 |
| Optimizer steps per generation | 1 mini-batch × 1 epoch | 1 |

**Important**: `ppo_epochs` multiplies everything — each epoch does a full forward+backward pass through all micro-batches with its own optimizer step. For strict TRL optimizer-step alignment, keep `ppo_epochs=1` with `num_iterations=1`.

#### Why the previous approach (multiple mini-batches) was wrong

The initial implementation set `mini_batch_size < global_batch_size`, creating multiple mini-batches. But `train_mini_batch` calls `engine.train_batch()` for **each** mini-batch, and each call does its own `optimizer.step()`. This meant:

```
WRONG (previous):
  global_batch_size=48, mini_batch_size=8
  → 6 mini-batches, each calls train_batch()
  → 6 separate optimizer.step() calls
  → Each step sees only 1 unique prompt (8 seq)
  → NOT gradient accumulation — just 6 small optimizer steps
```

The corrected approach ensures:

```
CORRECT (current):
  global_batch_size=48, mini_batch_size=48
  → 1 mini-batch, 1 call to train_batch()
  → Engine splits into 6 micro-batches of 4 seq/GPU
  → Gradients accumulate, 1 optimizer.step()
  → Each step sees 6 unique prompts (48 seq)
  → TRUE gradient accumulation matching TRL
```

### How to use

**Defaults now match TRL structure (48 generation slots/update):**

```bash
bash scripts/.match_rankgrpo.sh
```

With defaults: `TRAIN_BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS=6` → `GEN_BATCH_SIZE=6` unique prompts/step, matching TRL's `generation_batch_size=48` repeated slots with `num_generations=8`.

**Full TRL alignment (tight clip, fixed micro-batching):**

```bash
USE_DYNAMIC_BSZ=False \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=4 \
  bash scripts/.match_rankgrpo.sh \
    actor_rollout_ref.actor.clip_ratio=0.06 \
    actor_rollout_ref.actor.clip_ratio_low=0.06 \
    actor_rollout_ref.actor.clip_ratio_high=0.08 \
    actor_rollout_ref.actor.ppo_epochs=1
```

This gives:
- 6 unique prompts per optimizer step (matches TRL's 48 slots / 8 generations)
- 6 × 8 = 48 completions generated per step (matches TRL)
- 1 mini-batch of 48 seq → engine splits into **6 micro-batches** of 4 seq/GPU
- Gradients accumulate across 6 micro-batches → 1 optimizer step (matches TRL's grad_accum=6)
- Per GPU per micro-batch: 4 seq
- clip ratio [0.94, 1.08] (matches TRL)
- 1 PPO epoch / `num_iterations=1` for strict optimizer-update alignment

### Verification

TODO after running:
- [ ] Confirm `data.gen_batch_size` shows correct value in Hydra config dump
- [ ] Confirm `global_batch_size` vs `mini_batch_size` in `_update_actor` logs
- [ ] Compare `eval/reward_total` convergence rate vs TRL baseline
- [ ] Compare wall-clock time per 100 optimizer steps
- [ ] Monitor `actor/pg_clipfrac` — should stay < 0.3 with tighter clip

### Files modified

| File | Change |
|------|--------|
| `scripts/run_rankgrpo.sh` | Defaults to `TRAIN_BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS=6`, computed `GEN_BATCH_SIZE=6`; documents that verl-gr batch sizes are unique prompts while TRL `generation_batch_size` is repeated generation slots |
| `scripts/.match_rankgrpo.sh` | Keeps only endpoint-specific GPU/Ray/output/resume setup; delegates TRL-alignment defaults to `run_rankgrpo.sh` |
| `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml` | `ppo_mini_batch_size` 6→8; +`ppo_micro_batch_size_per_gpu: null` |
| `verl_080_dev/verl/trainer/ppo/ray_trainer.py` | `_update_actor`: `global_batch_size` computed from `gen_batch_size × n`; `mini_batch_size = global_batch_size` (one mini-batch, gradient accumulation via micro-batches in engine) |

### TRL alignment map

| TRL parameter | TRL value | verl-gr equivalent | verl-gr value |
|---|---|---|---|
| `per_device_train_batch_size` | 4 | sequences/GPU/micro-batch after rollouts | 4 seq/GPU |
| `num_processes` | 2 | `N_GPUS` | 2 |
| `gradient_accumulation_steps` | 6 | `GRADIENT_ACCUMULATION_STEPS` | 6 |
| Generation slots per opt step | 4×2×6 = 48 | `GEN_BATCH_SIZE × ROLLOUT_N` | 6×8 = 48 |
| Unique prompts per opt step | 48 / 8 = 6 | `GEN_BATCH_SIZE` = 1×6 | 6 |
| Rollouts per prompt | 8 | `ROLLOUT_N` | 8 |
| Seq per micro-batch | 8 global (4/GPU) | `TRAIN_BATCH_SIZE × ROLLOUT_N` | 8 global (4/GPU) |
| Total seq per opt step | 48 | `GEN_BATCH_SIZE × ROLLOUT_N` | 48 |
| `num_iterations` (mu) | 1 | `ppo_epochs` | 1 |
