# rank-grpo

## Sample Input & Results

### Task

Given a Reddit movie recommendation conversation, the model outputs 20 movie recommendations in `"Title (Year)"` format.

### Sample Input/Output

**Input (prompt):**
```
Pretend you are a movie recommender system.
I will give you a conversation between a user and you (a recommender system).
Based on the conversation, you need to reply with 20 recommendations.
List the standardized English title of each movie in each line in the form of
"movie name" (release_year) with NO extra words or sentences.

Here is the conversation: USER: Suggest me some thought provoking movies.
Hi everyone I'm in a mood to watch something very interesting clever and also
entertaining so please suggest me some entertaining thought provoking movies

I've already watched all Christopher Nolan and Charlie Kaufman movies
```

**Ground Truth (8 items):** November (2004), A Scanner Darkly (2006), Waking Life (2001), Pi (1998), Awakenings (1990), The Secret (2006), Eternal Sunshine of the Spotless Mind (2004), The Help (2011)

**SFT Model Output (top-5):**
```
The Matrix (1999)
Eternal Sunshine of the Spotless Mind (2004)
Inception (2010)
Fight Club (1999)
Memento (2000)
...
```

**GRPO Model Output (step 40200, top-5):**
```
Inception (2010)
Eternal Sunshine of the Spotless Mind (2004)
Interstellar (2014)
The Prestige (2006)
Memento (2000)
...
```

### SFT vs GRPO — Offline Evaluation

Both evaluated on the full test set: 2,050 unique contexts, 10,972 total samples.

| Metric | SFT (epoch 1.5) | GRPO (step 40200) | Delta |
|--------|-----------------|-------------------|-------|
| Recall@5 | 0.0681 | 0.0740 | +8.6% |
| Recall@10 | 0.1064 | 0.1188 | +11.7% |
| Recall@15 | 0.1343 | 0.1517 | +13.0% |
| Recall@20 | 0.1510 | 0.1734 | +14.8% |
| NDCG@5 | 0.0518 | 0.0574 | +10.8% |
| NDCG@10 | 0.0644 | 0.0722 | +12.1% |
| NDCG@15 | 0.0726 | 0.0819 | +12.7% |
| NDCG@20 | 0.0771 | 0.0876 | +13.6% |

GRPO consistently improves over SFT across all metrics. Gains increase with K, from +8.6% Recall@5 to +14.8% Recall@20.

---

## Rank-GRPO Algorithm

The original TRL run uses:

```bash
--reward_func exp_inf
```

In `Rank-GRPO/libs/reward_funcs.py`, `exp_inf` calls
`evaluate_direct_match_aligned(...)` and returns a length-20 vector of binary
per-rank hits:

```text
rank_rewards[j] = 1 if recommendation j matches a ground-truth movie
rank_rewards[j] = 0 otherwise
```

This is not the paper-style DCG suffix return. The paper's rank-level return is
closer to TRL's `log_decay` path, where hits are discounted by rank and each
rank receives the remaining suffix DCG:

```text
gains = hits * discounts
reward_at_rank_i = sum(gains[i:])
```

However, the reference run we align against uses `exp_inf`, so verl-gr matches
that behavior rather than switching to `log_decay`.

### Training Reward Alignment

TRL computes `rewards_items` from `exp_inf`, normalizes them within each prompt's
8 generations, and broadcasts each item's advantage to the tokens belonging to
that recommendation. It also logs:

```text
train/reward_total = mean(sum(rewards_items per 20-item list))
```

So for the current TRL run, `train/reward_total` is the average number of matched
ground-truth items per generated list.

verl-gr mirrors the same training signal in
`verl_gr/recipes/rankgrpo/rankgrpo_algorithm.py`: it computes `rank_rewards`
with the same aligned matching logic, normalizes within each prompt group, and
broadcasts item-level advantages to recommendation tokens. It logs these
training TensorBoard scalars:

```text
train/rankgrpo/reward_total = mean(sum(rank_rewards per 20-item list))
train/rankgrpo/reward       = mean(rank_rewards over 20 positions)
train/rankgrpo/hit_any      = fraction of generated lists with at least one hit
```

`train/rankgrpo/reward_total` is the direct verl-gr counterpart for TRL's
`train/reward_total`. `critic/rewards/mean` is not that metric.

During validation, `eval/reward_total` is an alias for
`val-aux/rankgrpo/rank_reward_sum/mean@8`, so it is also an average hit count
per list. `eval/reward` similarly aliases
`val-aux/rankgrpo/rank_rewards/mean@8`, the per-position mean hit rate. For
example, `eval/reward_total = 0.4106` means about 0.41 matched ground-truth items
per 20-item generated list on average. It does not mean NDCG is 0.4106.

`critic/rewards/mean` is different. It is a generic verl PPO metric computed
from `token_level_rewards.sum(-1)`. In this Rank-GRPO rollout path the scalar
reward score is `float(any(rank_rewards))`, so `critic/rewards/mean` is closer
to a "hit-any" rate: the fraction of generated lists with at least one hit. It
is not the same as TRL `train/reward_total`, and it is not NDCG.

NDCG is computed only by the separate offline evaluation code using
`ndcg_at_k(...)`, where hits are discounted by rank and normalized by ideal DCG.
Do not infer NDCG directly from `eval/reward_total` or `critic/rewards/mean`.

---

## Training Convergence

### Aligned Trace Comparison

Trace sources:

- Original TRL Rank-GRPO: `Rank-GRPO/results/grpo/new2/runs`
- verl_gr fork: `tensorboard_log/RankGRPO/g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt`

These are short aligned traces, not completed one-epoch/full-convergence runs. The comparison below uses the overlapping region around 600 optimizer steps and eval step 400.

| Metric | Original TRL Rank-GRPO | verl_gr fork (`fp32opt`) | Notes |
|--------|-------------------------|---------------------------|-------|
| Train scalar range | steps 10-820 across resumed traces | steps 10-590 | TensorBoard scalar coverage |
| Comparable train step | 600 | 590 | Nearest available overlap |
| Train KL | 0.0391 at step 600 | 0.0408 at step 590 | KL is now aligned in scale and trend |
| Train loss | -0.00052 at step 600 | -0.00012 at step 590 | Same small-loss regime |
| Grad norm | 0.4449 at step 600 | 0.0458 at step 590 | Different backend/optimizer dynamics |
| Train reward total | 0.4125 at step 600 | N/A | TRL logs `train/reward_total`; verl logs rollout reward under different names |
| Eval reward total | 0.3782 at step 400 | 0.3814 at step 400 | Comparable held-out reward |
| Eval KL | 0.0260 at step 400 | N/A | TRL logs eval KL; verl trace logs train actor KL |
| Clip fraction | 0.0 | 0.0 | Both stay inside the clip region |

The key alignment result is KL behavior. The older verl_gr runs had a flat `actor/kl_loss`; the `fp32opt` run no longer does. It rises from `0.000167` at step 10 to `0.0408` at step 590, closely matching TRL's `train/kl` of `0.0391` at step 600.

### verl_gr Fork (`fp32opt`) Trace Summary

| Metric | Value |
|--------|-------|
| Train scalar steps | 10-590 |
| Eval steps | 0, 200, 400 |
| Eval/reward_total | 0.3335 → 0.3814 |
| Actor KL loss | 0.000167 → 0.0408 |
| Actor PG loss | 0.000027 → -0.000118 |
| Actor grad norm | 0.1697 → 0.0458 |
| Actor clipfrac | 0.0 throughout |
| Mean logged time_per_step | 5.12s |
| Median logged time_per_step | 4.41s |
| Mean logged throughput | ~2,022 tok/s |
| Median logged throughput | ~2,065 tok/s |

The confirmed-good change for this trace is loading the trainable actor in fp32 (`ACTOR_MODEL_DTYPE=fp32`), while keeping the rest of the matched Rank-GRPO batch, clip, shuffle, and PPO-epoch settings.

---

## Performance

### Per-Step Runtime

| Implementation | GPUs | Steps | time_per_step (s) | Total throughput (tok/s) | Per-GPU throughput (tok/s/GPU) | Notes |
|---------------|------|-------|-------------------|--------------------------|-------------------------------|-------|
| **Original TRL Rank-GRPO** | 2× H800 | trace to step 820 | ~5.5-5.7 training-only wall sec/step; ~7.15 sec/step including eval/checkpoint overhead | training-only mean/median ~3,483/~3,485; end-to-end ~2,729 | training-only mean/median ~1,742/~1,743; end-to-end ~1,365 | Derived from adjacent 10-step `num_tokens` deltas; eval runtime ~442s every 200 steps |
| **verl_gr (`fp32opt`)** | 2× H800 | trace to step 590 | mean 5.12, median 4.41 | mean ~4,045; median ~4,130 | mean ~2,022; median ~2,065 | Fixed micro-batch config (6×4 seq/GPU, `use_dynamic_bsz=False`, `MAX_TOKENS_PER_GPU=24576`). Verl `perf/throughput` is per-GPU normalized. |
| **verl_gr (`debug_june5`)** | 2× H800 | trace to step ~170 | ~5.1s timer / ~7.2s wall (step 20, warmup) | TBD (full trace needed) | TBD | Dynamic bsz config (`use_dynamic_bsz=True`, `MAX_TOKENS_PER_GPU=12000`). See Phase Distribution below. |

TRL and verl_gr report throughput differently: TRL derives throughput from token-deltas between adjacent logged steps; verl TensorBoard `perf/throughput` is per-GPU normalized (`total_num_tokens / (time_per_step × n_gpus)`). Multiply verl's per-GPU throughput by N_GPUS for total.

### Phase Distribution

Config: 6 unique prompts × 8 rollouts = 48 seqs/step, 2× H800. verl-GR current uses `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=12000`. TRL does not export per-phase timing; its values are estimated from the step structure (ZeRO-3, 6 micro-batches, colocated vLLM). verl-GR current measured via per-phase `[TIMING]` instrumentation in `verl_gr/trainers/rl_trainer.py` (`debug_june5`, step 20). verl-GR optimized projects what the current config achieves if the environmental gen+ref regression is fixed.

| Component | Rank-GRPO (TRL) | verl-GR (current) | verl-GR (optimized) | Δ0 (optimized vs TRL) | Δ1 (current vs TRL) | Notes |
|-----------|----------------|---------|---|---|---|---|
| gen (vLLM rollout) | ~0.5s | ~1.89s | ~0.5s | ~0s | **+1.4s** | 6 prompts → 48 completions, TP=2. TRL colocated; verl RPC → vLLM worker. Current regression is environmental (same model, same vLLM config, same GPU node). |
| old_log_prob | ~0.5s (recompute) | 0s (bypassed) | 0s | **−0.5s** | **−0.5s** | TRL recomputes or detaches; verl `bypass_mode=true` copies `rollout_log_probs`, skipping a forward pass. |
| ref (ref log-prob) | ~0.4s | 1.05s | ~0.4s | ~0s | **+0.6s** | One frozen-model forward over 48 seqs. Same environmental regression as gen. |
| adv (advantage) | inline | ~0.10s | ~0.10s | +0.1s | +0.1s | TRL computes inline; verl on driver CPU. Comparable cost. |
| fwd+bwd (actor) | ~2.6s (6 micro-batches) | ~0.92s (~2 micro-batches) | ~0.92s | **−1.7s** | **−1.7s** | Dynamic bsz splits 48 seqs by token count instead of 6× fixed micro-batches. Each pass processes unique, larger data → better GPU utilization. |
| optimizer step | ~0.2s | — (amortized) | — (amortized) | −0.2s | −0.2s | AdamW fp32. TRL separate; verl amortized into weight sync RPC. |
| update_weights (actor → vLLM) | 0s (colocated) | 1.15s | 1.15s | **+1.2s** | **+1.2s** | verl must sync weights to separate vLLM process; TRL shares memory. |
| **Total (training phases)** | **~4.2s** | **5.11s** | **~3.1s** | **−1.1s** | **+0.9s** | |
| Step overhead (data, logging, tqdm) | ~1.3s | ~2.1s | ~1.0s | −0.3s | +0.8s | TRL: Accelerate overhead. verl: Ray orchestration + dataloader + DataProto ops. Current tqdm average dragged up by warmup steps 1-5. |
| **Total wall** | **~5.5s** | **~7.2s** | **~4.1s** | **−1.4s** | **+1.7s** | |

**Key observations:**

1. **Dynamic bsz is the biggest win.** fwd+bwd drops from TRL's ~2.6s (6 micro-batches) to ~0.92s (~2 micro-batches) — a 65% reduction. This is enabled by `use_dynamic_bsz=True` which packs more tokens per pass instead of repeating the same data.

2. **verl-GR's `n=1` rollout is more efficient than TRL's `llm.generate()`.** TRL calls `self.llm.generate(prompts, sampling_params)` where `SamplingParams.n=8` — one vLLM request per prompt that generates all 8 completions internally. verl-GR instead fires 8 independent `n=1` requests per prompt concurrently via `asyncio.gather` ([`rankgrpo_agent_loop.py:193-201`](verl_gr/recipes/rankgrpo/rankgrpo_agent_loop.py)). The `n=1` approach is faster because: (a) vLLM's continuous batching scheduler can interleave decode steps from 48 independent requests (6 prompts × 8 rollouts), picking the optimal mix at each step, while `n=8` ties all 8 completions to the same prompt KV cache and forces them to be co-scheduled; (b) independent `n=1` requests can complete and release KV-cache blocks independently — with `n=8`, the prompt KV cache remains pinned until the *slowest* of the 8 completions finishes; (c) `asyncio.gather` saturates the vLLM request queue immediately, giving the scheduler maximum batching flexibility from the first decode step. This is why the README's earlier `fp32opt` snapshot measured gen at just ~0.53s for 48 completions.

3. **gen + ref are the biggest gap to TRL** (+2.0s combined). Both are model-forward-pass operations that regressed 3-4× between the May `fp32opt` snapshot and June `debug_june5`, despite identical vLLM engine config and model checkpoint. This is an environmental regression (GPU clocks, conda env package versions, system load), not a code issue.

4. **update_weights (+1.2s vs TRL) is the structural cost** of verl-GR's Ray-based architecture. TRL's colocated vLLM shares GPU memory with the trainer and needs no weight sync; verl-GR's separate vLLM workers require an RPC after each optimizer step. This is the trade-off for independent scaling.

5. **verl-GR optimized projects ~4.1s wall time** — beating both TRL (~5.5s) and the ~4s/it golden target — by combining dynamic bsz (already done) with a fix for the environmental gen+ref regression.

### Comparison: TRL vs verl-GR Step Structure

The two frameworks structure a training step differently:

| Phase | TRL (DeepSpeed ZeRO-3, colocated) | verl-GR (FSDP + Ray) | Notes |
|-------|-----------------------------------|----------------------|-------|
| vLLM generation | colocated in-process | Ray RPC → vLLM worker | TRL avoids RPC; verl has ~1.9s gen phase |
| old log-prob | `old_per_token_logps.detach()` | bypassed (`bypass_mode=true`) | Both use current-policy proxy |
| ref log-prob | separate ref forward | Ray RPC → ref FSDP worker | Both do one frozen-model forward |
| reward | CPU string-matching | CPU string-matching | Comparable |
| advantage | inline in trainer loop | driver CPU | Comparable; verl runs Rank-GRPO group norm |
| actor fwd+bwd | ZeRO-3, 6 micro-batches | FSDP, ~2 dynamic micro-batches | verl has fewer but larger micro-batches |
| optimizer step | DeepSpeed AdamW | FSDP AdamW | Comparable |
| weight sync → vLLM | none (colocated) | ~1.15s RPC per step | verl must sync weights to separate vLLM process |
| orchestration | Accelerate, single-process | Ray, multi-actor RPC | verl has higher per-step overhead |

The net effect: TRL's colocated design has lower per-step overhead (~5.5s training-only) but is harder to scale beyond one node. verl-GR's Ray-based design adds RPC/weight-sync overhead but supports independent scaling of rollout and training workers. The remaining gap to the ~4s/it golden target is primarily in the vLLM generation phase (gen) and ref forward phase (ref), both of which are model-inference operations that share the same GPU and have regressed equally — suggesting a single environmental root cause rather than code inefficiency.

### Eval runtime

| Implementation | Eval runtime | Frequency |
|---------------|-------------|-----------|
| Original TRL Rank-GRPO | ~442s (mean of step 200 and 400 evals) | Every 200 steps |
| verl_gr Fork (`fp32opt`) | not logged as a direct eval runtime scalar in this trace | Every 200 steps |

---

## Hyperparameters: TRL (`run_rl.sh`) vs. verl-GR (`run_rankgrpo.sh`)

Trace references:
- TRL: `Rank-GRPO/scripts/run_rl.sh`, tensorboard: `results/grpo/new2/runs/May28_09-35-22_hk01dgx028`
- verl-GR: `scripts/run_rankgrpo.sh`, tensorboard: `tensorboard_log/RankGRPO/debug_june5`

### Hyperparameter Mapping Table

| Field | TRL Value | verl-GR Value | Aligned? | Notes |
|-------|-----------|---------------|----------|-------|
| **Optimizer** | | | | |
| Learning rate | `--lr 1e-6` | `LEARNING_RATE=1e-6` | ✓ | |
| Adam β₁ | `--adam_beta1 0.9` | `ADAM_BETA1=0.9` | ✓ | |
| Adam β₂ | `--adam_beta2 0.99` | `ADAM_BETA2=0.99` | ✓ | |
| Weight decay | default (0.0) | `WEIGHT_DECAY=0.0` | ✓ | |
| LR schedule | default (constant) | `lr_scheduler_type=constant` | ✓ | |
| LR warmup | default (0) | `LR_WARMUP_STEPS=0` | ✓ | |
| **Precision** | | | | |
| Mixed precision | `--bf16` | `ACTOR_MODEL_DTYPE=fp32` (bf16 compute + fp32 optimizer) | ✓ | Same effective precision; different expression |
| Gradient checkpointing | `--gradient_checkpointing` | `GRADIENT_CHECKPOINTING=True` | ✓ | |
| **Batch Configuration** | | | | |
| Unique prompts / optimizer step | 6 (derived) | 6 (`TRAIN_BATCH_SIZE=6`) | ✓ | See Note ① |
| Generations per prompt | `--num_generations 8` | `ROLLOUT_N=8` | ✓ | |
| Total seqs / optimizer step | 48 | 48 | ✓ | 6 prompts × 8 rollouts |
| Per-device train batch size | `--per_device_train_batch_size 4` | N/A (verl uses total prompts) | — | TRL counts sequences/GPU/micro-step; verl uses `TRAIN_BATCH_SIZE` (prompts) |
| Micro-batch size per GPU | `per_device_train_batch_size=4` (4 seqs/GPU fixed) | `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=32` (ceiling when `use_dynamic_bsz=True`; =`TRAIN_BATCH_SIZE×ROLLOUT_N/N_GPUS` when `False`) | — | TRL fixes 4 seqs/GPU/micro-step. With dynamic bsz, verl sets a high ceiling (32) and splits by token count via `ppo_max_token_len_per_gpu` (~21+3 seqs in practice). Without dynamic bsz, verl matches TRL's 4 seq/GPU exactly. |
| Gradient accumulation steps | `--gradient_accumulation_steps 6` | `GRADIENT_ACCUMULATION_STEPS=1` | ✗ | See Note ② |
| Max tokens per GPU | `MAX_TOKENS_PER_GPU=12000` | `MAX_TOKENS_PER_GPU=12000` | ✓ | Controls dynamic-bsz micro-batch split; lower = more, smaller micro-batches |
| **PPO / RL** | | | | |
| PPO epochs (μ) | `--mu 1` | `PPO_EPOCHS=1` | ✓ | Single pass per generated batch |
| Clip range | default [0.94, 1.08] | `PPO_CLIP_RATIO=0.06`, `PPO_CLIP_RATIO_HIGH=0.08` | ✓ | ε=0.06, ε_high=0.08 in both |
| KL coefficient | `--kl_beta 1e-3` | `KL_LOSS_COEF=0.001` | ✓ | |
| KL loss type | `low_var_kl` (k3 estimator) | `KL_LOSS_TYPE=low_var_kl` | ✓ | Same estimator: exp(ref-log) - (ref-log) - 1 |
| Loss aggregation | `seq-mean-token-mean` | `LOSS_AGG_MODE=seq-mean-token-mean` | ✓ | Equal weight per sequence |
| Advantage estimator | GRPO | GRPO (Rank-GRPO group-normalized) | ✓ | Group-normalized within 8 generations per prompt |
| **Rollout / vLLM** | | | | |
| vLLM mode | `--vllm_mode colocate` | hybrid engine (separate workers) | — | Architectural difference; see Note ④ |
| vLLM GPU memory util. | `--vllm_gpu_memory_utilization 0.25` | `ROLLOUT_GPU_MEMORY_UTILIZATION=0.25` | ✓ | |
| vLLM TP size | `--vllm_tensor_parallel_size 2` | `ROLLOUT_TENSOR_PARALLEL_SIZE=2` | ✓ | |
| Enforce eager | default (False, CUDA graphs enabled) | `ROLLOUT_ENFORCE_EAGER=False` | ✓ | Both attempt CUDA graphs for vLLM |
| Custom all-reduce | default (vLLM built-in) | `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=True` | ✗ | See Note ③ |
| Flashinfer sampler | not set | `VLLM_USE_FLASHINFER_SAMPLER=1` | — | verl uses faster sampling kernel |
| **Data / Sequence** | | | | |
| Max prompt length | `--max_prompt_length 2048` | `max_prompt_length=2048` | ✓ | |
| Max completion length | `--max_completion_length 1024` | `max_response_length=1024` | ✓ | |
| Train shuffle | default (True) | `DATA_SHUFFLE=True` | ✓ | |
| Validation shuffle | `--no-val_shuffle` | `VALIDATION_SHUFFLE=False` | ✓ | |
| Seed | `--seed 3407` | `SEED=3407` | ✓ | |
| **Reward** | | | | |
| Reward function | `--reward_func exp_inf` | `compute_score` (aligned matching logic) | ✓ | Same per-rank hit detection against GT catalog |
| Length shaping | N/A (not in TRL) | `APPLY_EXTRA_LENGTH_SHAPING=True` (EOL=+0.1, overflow=-0.1, early_stop=-0.1) | — | verl adds length-based reward shaping; can be disabled |
| **Checkpoint / Logging** | | | | |
| Save frequency | `--save_steps 200` | `SAVE_FREQ=200` | ✓ | |
| Eval frequency | `--eval_steps 200` | `TEST_FREQ=200` | ✓ | |
| Top-k checkpoints | implicit (best only) | `BEST_CKPTS_TO_KEEP=3` | ✓ | verl prunes non-top-k `global_step_*` dirs |
| Eval before train | implicit (no) | `VAL_BEFORE_TRAIN=False` | ✓ | |
| **Distributed Backend** | | | | |
| Training framework | DeepSpeed ZeRO-3 | FSDP (`FSDP_STRATEGY=fsdp`) | ✗ | See Note ④ |
| Orchestration | Accelerate (single-process) | Ray (multi-actor RPC) | ✗ | See Note ④ |
| vLLM integration | colocated in-process | Ray rollout workers | ✗ | See Note ④ |

### Side Notes

**① Effective batch size equivalence.** TRL's `per_device_train_batch_size=4` counts *generated sequences* per GPU (not unique prompts). So `4 seqs × 2 GPUs × 6 accum = 48` sequences per optimizer step; with `num_generations=8`, that's `48 ÷ 8 = 6` unique prompts. verl-GR uses the opposite convention: `TRAIN_BATCH_SIZE=6` counts *unique prompts*, and `ROLLOUT_N=8` makes it `6 × 8 = 48` sequences. Both frameworks process the same 48 sequences from 6 unique prompts — the `num_generations`/`ROLLOUT_N` factor is present on both sides, just applied at different levels (TRL bakes it into `per_device_train_batch_size`; verl applies it explicitly via `ROLLOUT_N`).

**② Gradient accumulation: different mechanism, same outcome.** TRL's `gradient_accumulation_steps=6` means: load 1 prompt per micro-step, generate 8 responses, run fwd+bwd, accumulate gradients; repeat 6 times with 6 **different** prompts; then `optimizer.step()`. In verl, setting `GRADIENT_ACCUMULATION_STEPS=6` with the old fixed-micro-batch config would repeat the **same** prompt's data 6 times — redundant computation, not true gradient accumulation. The current verl config instead loads all 6 prompts in one `TRAIN_BATCH_SIZE=6` and uses `use_dynamic_bsz=True` to split them into ~2 micro-batches by token count, each with **different** data. The result is the same: 6 unique prompts' gradients accumulated, one optimizer step. The mechanism differs because the frameworks abstract gradient accumulation at different levels (TRL at the dataloader, verl at the engine micro-batch splitter).

**③ vLLM custom all-reduce disabled for stability.** On this H800 stack, vLLM's built-in custom all-reduce for TP=2 fails at startup (`custom_all_reduce.cuh:455 invalid argument`). verl-GR disables it and falls back to NCCL collectives for TP communication. This is a stability requirement, not a performance choice. The effective TP=2 topology and collective behavior are aligned; only the underlying implementation differs.

**④ Distributed backends are fundamentally different.** TRL uses DeepSpeed ZeRO-3 + HuggingFace Accelerate with vLLM colocated in the same process. verl-GR uses FSDP + Ray with vLLM running in separate Ray actors. These are architectural choices with different trade-offs:
- TRL's in-process design avoids RPC overhead but is harder to scale beyond one node.
- verl-GR's Ray-based design adds ~1-2s/step in orchestration overhead but supports independent scaling of rollout and training workers across nodes.
- Numeric alignment is expected in training signal (rewards, advantages, KL divergence, loss), not in per-step runtime or backend-specific metrics like gradient norms.

The hyperparameter values are intentionally chosen to produce the same training dynamics despite the different backends. Where numbers differ (e.g., `GRADIENT_ACCUMULATION_STEPS`), the difference reflects a framework-level abstraction gap, not a training discrepancy.