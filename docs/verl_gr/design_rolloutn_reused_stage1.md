# Design: Rollout-N Shared Stage-1 CoT Reuse (OpenOneRec GRPO)

## Background

In OpenOneRec GRPO, each prompt is repeated `rollout.n` times (for example `ROLLOUT_N=8`) to form a GRPO group.  
Before this feature, the two-stage rollout executed full stage-1 CoT generation and stage-2 item generation independently for every repeated row in the group.

The feature introduced in commits `69e9d96724773da293be947afb973df406727eeb` and `a75be703fcb96bf8cc55e011afc3092a2eb3f605` adds:

1. A **shared stage-1 CoT path** for repeated prompts in the same GRPO group.
2. A **compare mode** that keeps default training behavior but measures feature-path quality/performance side-by-side.

---

## 1) What We Have Done So Far

### 1.1 Config + launcher wiring

- Added rollout config flag:
  - `actor_rollout_ref.rollout.compare_vanilla_vs_stage1_reuse` (default `false`)
  - in `configs/verl_gr/openonerec/grpo_trainer.yaml`
- Updated launcher `scripts/run_openonerec_grpo.sh`:
  - default `ROLLOUT_N=8`
  - env flag `COMPARE_VANILLA_ROLLOUT_WITH_STAGE1_REUSE_STAGE2_FORCE_RANDOMNESS` (default `false`)
  - normalized into `ROLLOUT_CMP_VANILLA_VS_REUSE` and passed to Hydra override:
    - `actor_rollout_ref.rollout.compare_vanilla_vs_stage1_reuse=...`
- Launcher now prints compare-mode status at startup.

### 1.2 Rollout implementation (TwoStagevLLMRollout)

Implemented in `verl_gr/workers/rollout/two_stage_vllm_rollout.py`.

Key additions:

- `resolve_rollout_n(...)`: reads GRPO repeat size from meta/kwargs/config.
- Repeat-layout and uid consistency checks:
  - `repeat_interleave_layout(...)`
  - `_uids_match_interleaved_repeat(...)`
- Compare-flag resolver:
  - `_compare_vanilla_flag(...)`
- New feature path:
  - `_feature_rollout_hybrid_grpo(...)`
  - For each GRPO group:
    1. Run stage-1 CoT once on group row 0.
    2. Reuse that CoT for stage-2.
    3. Row 0 uses deterministic stage-2 beam search.
    4. Rows `1..n-1` use stochastic stage-2 sampling with distinct deterministic seeds.
- Existing default path preserved:
  - `_two_stage_generation(...)` still performs full two-stage generation per repeated row.

Compare-mode behavior in `generate_sequences(...)`:

- If compare flag is enabled and batch layout is valid:
  1. Run **default** rollout.
  2. Run **feature** rollout.
  3. Build rollout-level metrics via `_build_rollout_cmp_metrics(...)`.
  4. Return **default rollout output** for training (no behavior change to optimization target).
  5. Store feature output in metadata (`openonerec_cmp_feature_rollout`) for trainer-side mirror metrics.
- Otherwise: fallback to default `_two_stage_generation(...)`.

### 1.3 Worker metric propagation to timing stream

Implemented in `verl_gr/recipes/openonerec/onerec_fsdp_workers.py`:

- Overrode `generate_sequences(...)` in `OneRecActorRolloutRefWorker`.
- Reads rollout compare scalars from `out.meta_info["openonerec_rollout_cmp"]`.
- Copies scalars into `out.meta_info["timing"]` as:
  - `openonerec_cmp_<metric_name>`

This makes rollout compare metrics visible in timing-oriented TensorBoard streams.

### 1.4 Trainer-side GRPO compare metrics hooks

Implemented in:

- `verl_gr/trainers/rollout_cmp_grpo_hooks.py` (new file)
- `verl_gr/trainers/rl_trainer.py` (`RLTrainer.fit()` installs hooks)

What hooks do:

1. Monkey-patch `verl.trainer.ppo.ray_trainer.compute_advantage`.
2. If compare mode produced `openonerec_cmp_feature_rollout`, run a mirror reward+advantage pass on feature generations.
3. Compute default-vs-feature GRPO metrics, including:
   - score mean/std
   - advantage mean
   - mean absolute diffs (score/advantage)
   - within-group variance (score/advantage)
   - optional advantage Pearson correlation
4. Monkey-patch `compute_data_metrics` to append pending compare metrics into final logged metrics.

Important: this remains a **measurement path**. The training batch used for optimization is still the default rollout output.

---

## 2) How To Use It

## 2.1 Default usage (safe baseline)

No extra flag required. Current launcher defaults already include:

- `ROLLOUT_N=8`
- compare flag disabled

Run:

```bash
bash scripts/run_openonerec_grpo.sh
```

Behavior:

- Uses regular two-stage rollout for all repeated rows.
- No default-vs-feature compare metrics are computed.

## 2.2 Enable compare mode (recommended for evaluation)

Set env flag before launch:

```bash
COMPARE_VANILLA_ROLLOUT_WITH_STAGE1_REUSE_STAGE2_FORCE_RANDOMNESS=true \
bash scripts/run_openonerec_grpo.sh
```

Behavior:

- For eligible GRPO batches (`rollout.n > 1` with interleaved repeat + uid-consistent groups):
  - Runs default rollout and shared-stage1 feature rollout.
  - Trains on default rollout output.
  - Logs rollout timing/throughput/diversity deltas and trainer-side GRPO quality deltas.
- For ineligible batches, it gracefully falls back to default rollout only.

## 2.3 Relevant knobs

- `ROLLOUT_N` (launcher) -> `actor_rollout_ref.rollout.n`
  - Must be `> 1` to realize intra-group sharing value.
- `COMPARE_VANILLA_ROLLOUT_WITH_STAGE1_REUSE_STAGE2_FORCE_RANDOMNESS`
  - `true`: enable dual-path compare measurement.
  - `false`: baseline only.
- Existing rollout params still apply:
  - `stage1_max_tokens`
  - `stage2_beam_size`
  - `stage2_num_tokens` / `stage2_max_tokens`
  - `temperature`, `top_p`, `top_k`

## 2.4 What to monitor

Rollout compare timing metrics (prefixed in timing):

- `openonerec_cmp_default_wall_s`
- `openonerec_cmp_feature_wall_s`
- `openonerec_cmp_default_throughput_resp_tok_per_s`
- `openonerec_cmp_feature_throughput_resp_tok_per_s`
- `openonerec_cmp_default_mean_within_group_resp_len_var`
- `openonerec_cmp_feature_mean_within_group_resp_len_var`
- `openonerec_cmp_default_mean_within_group_len_distinct_frac`
- `openonerec_cmp_feature_mean_within_group_len_distinct_frac`

Trainer-side GRPO compare metrics:

- `openonerec_cmp_grpo/score_seq_mean_default`
- `openonerec_cmp_grpo/score_seq_mean_feature`
- `openonerec_cmp_grpo/mean_abs_score_seq_diff`
- `openonerec_cmp_grpo/advantage_seq_mean_default`
- `openonerec_cmp_grpo/advantage_seq_mean_feature`
- `openonerec_cmp_grpo/mean_abs_advantage_seq_diff`
- `openonerec_cmp_grpo/within_group_adv_var_mean_default`
- `openonerec_cmp_grpo/within_group_adv_var_mean_feature`

---

## Current scope and intent

- Implemented scope is **instrumented compare mode** plus **shared-stage1 feature rollout path**.
- Current training objective remains conservative: optimize using default rollout output.
- This gives us a low-risk way to validate speed/diversity/quality impact before deciding whether to switch training to the feature path in a later phase.
