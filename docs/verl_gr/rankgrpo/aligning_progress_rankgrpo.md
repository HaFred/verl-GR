# Aligning verl-gr RankGRPO with TRL — Progress Log

Goal: Make verl-gr RankGRPO match or exceed TRL's convergence rate and compute efficiency.

Reference: `aligning_rankgrpo.md` (root cause analysis).

---

## Change 1 — 2026-05-26: `use_dynamic_bsz` knob + gradient accumulation support

### Motivation

The verl-gr actor sees only 6 unique prompts per optimizer update vs TRL's 48 (8× fewer).
This causes:

- Noisy GRPO advantage estimates (6 groups vs 48 for normalization)
- 8× fewer unique training examples per optimizer step
- The `ppo_epochs=12` workaround wastes compute without fixing the root cause

### Changes

#### 1a. New env vars (`scripts/run_rankgrpo.sh`)

| Var | Default | Purpose |
|-----|---------|---------|
| `USE_DYNAMIC_BSZ` | `True` | Knob to disable dynamic token-based micro-batching |
| `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU` | `32` | Fixed micro-batch size (seq/GPU) when `USE_DYNAMIC_BSZ=False`. 32 = 4 prompts × 8 rollouts, matching TRL's `per_device_train_batch_size=4` |
| `GRADIENT_ACCUMULATION_STEPS` | `6` | Number of mini-batches per optimizer step. `gen_batch_size = TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`. Default 6 matches TRL |
| `GEN_BATCH_SIZE` | (computed) | `8 × 6 = 48` unique prompts per global step — matches TRL exactly (4 prompts/GPU × 2 GPUs × 6 accumulation steps) |

#### 1b. Wired through `.match_rankgrpo.sh`

- Exports the same env vars with defaults
- Passes `data.gen_batch_size`, `actor_rollout_ref.actor.use_dynamic_bsz`, and `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` to Hydra CLI

#### 1c. Config default (`configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml`)

- Added `ppo_micro_batch_size_per_gpu: null` under `actor_rollout_ref.actor` (only effective when `use_dynamic_bsz=False`)

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
│  e.g. 48 prompts × 8 rollouts = 384 sequences               │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Mini-Batch  (mini_batch_size)                        │  │
│  │  Subset of global batch processed by train_mini_batch │  │
│  │  Each mini-batch → one call to engine.train_batch()   │  │
│  │  Multiple mini-batches → multiple optimizer.step()s   │  │
│  │  e.g. 384 sequences (1 mini-batch = full global)      │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Micro-Batch  (micro_batch_size_per_gpu)        │  │  │
│  │  │  Subset of mini-batch processed by GPU in one   │  │  │
│  │  │  forward+backward pass.                         │  │  │
│  │  │  Gradients ACCUMULATE across micro-batches.     │  │  │
│  │  │  Optimizer steps only after ALL micro-batches.  │  │  │
│  │  │  e.g. 32 seq/GPU (6 micro-batches per epoch)    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key rule**: Only micro-batch boundaries accumulate gradients. Mini-batch boundaries always trigger `optimizer.step()` + `zero_grad()`. Therefore, to get true gradient accumulation, we must have exactly **one mini-batch** and **multiple micro-batches**.

#### How this maps to TRL

TRL's gradient accumulation (`gradient_accumulation_steps=6`) works as follows:

```
TRL:
  Generate 48 prompts × 8 = 384 completions
  Split into 6 chunks of 64 seq (8 prompts × 8 rollouts)
  For each chunk:
    forward + backward (accumulate gradients, no optimizer step)
  After all 6 chunks:
    optimizer.step() + zero_grad()
  → 1 optimizer step per 48 unique prompts
```

verl-gr achieves the same with:

```
verl-gr:
  DataLoader yields gen_batch_size=48 prompts
  Generate 48 × 8 = 384 completions
  _update_actor:
    global_batch_size = mini_batch_size = 384
    → train_mini_batch creates 1 mini-batch of 384 seq (192 seq/GPU)
  Engine.train_batch (FSDP2):
    Splits 192 seq/GPU into micro-batches of 32 seq/GPU
    → 6 micro-batches per epoch
    For each micro-batch:
      forward + backward (accumulate gradients, no optimizer step)
    After all 6 micro-batches:
      optimizer.step() + zero_grad()
  → 1 optimizer step per 48 unique prompts
```

The mapping is:

| TRL concept | verl-gr equivalent | Value |
|---|---|---|
| Generation batch | `gen_batch_size × rollout.n` | 384 seq |
| Gradient accumulation steps | Number of micro-batches | 6 |
| Per-micro-batch (seq/GPU) | `ppo_micro_batch_size_per_gpu` | 32 |
| Unique prompts per opt step | `gen_batch_size` | 48 |
| Optimizer steps per generation | 1 mini-batch × 1 epoch | 1 |

**Important**: `ppo_epochs` multiplies everything — each epoch does a full forward+backward pass through all micro-batches with its own optimizer step. With `ppo_epochs=12`, we get 12 optimizer steps on the same 48 prompts. This is the multi-epoch PPO behavior, which is reduced when using tighter clip ratios.

#### Why the previous approach (multiple mini-batches) was wrong

The initial implementation set `mini_batch_size < global_batch_size`, creating multiple mini-batches. But `train_mini_batch` calls `engine.train_batch()` for **each** mini-batch, and each call does its own `optimizer.step()`. This meant:

```
WRONG (previous):
  global_batch_size=384, mini_batch_size=64
  → 6 mini-batches, each calls train_batch()
  → 6 separate optimizer.step() calls
  → Each step sees only 8 unique prompts (64 seq)
  → NOT gradient accumulation — just 6 small optimizer steps
```

The corrected approach ensures:

```
CORRECT (current):
  global_batch_size=384, mini_batch_size=384
  → 1 mini-batch, 1 call to train_batch()
  → Engine splits into 6 micro-batches of 32 seq/GPU
  → Gradients accumulate, 1 optimizer.step()
  → Each step sees 48 unique prompts (384 seq)
  → TRUE gradient accumulation matching TRL
```

### How to use

**Defaults now match TRL structure (48 unique prompts/step):**

```bash
bash scripts/.match_rankgrpo.sh
```

With defaults: `TRAIN_BATCH_SIZE=8`, `GRADIENT_ACCUMULATION_STEPS=6` → `GEN_BATCH_SIZE=48` unique prompts/step, matching TRL's `4×2×6=48`.

**Full TRL alignment (tight clip, fixed micro-batching):**

```bash
USE_DYNAMIC_BSZ=False \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=32 \
  bash scripts/.match_rankgrpo.sh \
    actor_rollout_ref.actor.clip_ratio=0.06 \
    actor_rollout_ref.actor.clip_ratio_low=0.06 \
    actor_rollout_ref.actor.clip_ratio_high=0.08 \
    actor_rollout_ref.actor.ppo_epochs=4
```

This gives:
- 48 unique prompts per optimizer step (matches TRL's 4×2×6=48)
- 48 × 8 = 384 completions generated per step (matches TRL)
- 1 mini-batch of 384 seq → engine splits into **6 micro-batches** of 32 seq/GPU
- Gradients accumulate across 6 micro-batches → 1 optimizer step (matches TRL's grad_accum=6)
- Per GPU per micro-batch: 32 seq = 4 prompts × 8 rollouts (matches TRL's per_device_train_batch_size=4)
- clip ratio [0.94, 1.08] (matches TRL)
- 4 PPO epochs (reduced from 12, since more data + tighter clip = less reuse needed)

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
| `scripts/run_rankgrpo.sh` | `TRAIN_BATCH_SIZE` default 6→8; +`GRADIENT_ACCUMULATION_STEPS` (default 6), +`GEN_BATCH_SIZE` (computed 8×6=48), +`USE_DYNAMIC_BSZ`, +`ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU` (default 32), +`data.gen_batch_size` CLI arg, +`ppo_micro_batch_size_per_gpu` CLI arg |
| `scripts/.match_rankgrpo.sh` | `TRAIN_BATCH_SIZE` default 6→8; +env var exports, +CLI passthrough, updated comments |
| `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml` | `ppo_mini_batch_size` 6→8; +`ppo_micro_batch_size_per_gpu: null` |
| `verl_080_dev/verl/trainer/ppo/ray_trainer.py` | `_update_actor`: `global_batch_size` computed from `gen_batch_size × n`; `mini_batch_size = global_batch_size` (one mini-batch, gradient accumulation via micro-batches in engine) |

### TRL alignment map

| TRL parameter | TRL value | verl-gr equivalent | verl-gr value |
|---|---|---|---|
| `per_device_train_batch_size` | 4 | `TRAIN_BATCH_SIZE / N_GPUS` | 8/2 = 4 |
| `num_processes` | 2 | `N_GPUS` | 2 |
| `gradient_accumulation_steps` | 6 | `GRADIENT_ACCUMULATION_STEPS` | 6 |
| Prompts per micro-batch | 4×2 = 8 | `TRAIN_BATCH_SIZE` | 8 |
| Unique prompts per opt step | 4×2×6 = 48 | `GEN_BATCH_SIZE` = 8×6 = 48 |
| Rollouts per prompt | 8 | `ROLLOUT_N` | 8 |
| Seq per micro-batch | 64 (32/GPU) | `TRAIN_BATCH_SIZE × n` | 64 (32/GPU) |
| Total seq per opt step | 384 | `GEN_BATCH_SIZE × n` | 384 |
| `num_iterations` (μ) | 1 | `ppo_epochs` | 12→4 (to be tuned) |
