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
| **Original TRL Rank-GRPO** | 2× H800 | trace to step 820 | ~5.5-5.7 training-only wall sec/step; ~7.15 sec/step in the step 10-600 trace including eval/checkpoint overhead | training-only mean/median ~3,483/~3,485; end-to-end step 10-600 ~2,729 | training-only mean/median ~1,742/~1,743; end-to-end step 10-600 ~1,365 | Derived from adjacent 10-step `num_tokens` deltas; eval runtime is ~442s every 200 steps |
| **verl_gr Fork (`fp32opt`)** | 2× H800 | trace to step 590 | mean 5.12, median 4.41 | mean ~4,045; median ~4,130 | mean ~2,022; median ~2,065 | TensorBoard `perf/throughput` is per GPU; total throughput is `perf/throughput × 2` |

The old table mixed incompatible throughput definitions. TRL's derived throughput above is total tokens per wall second unless explicitly divided by 2, while verl's TensorBoard `perf/throughput` is already normalized per GPU as `total_num_tokens / (time_per_step × n_gpus)`. With consistent definitions, verl_gr has both shorter logged step time and higher token throughput on this trace. The KL/reward behavior is comparable by step 400-600; a final speed claim should still use a controlled wall-clock comparison because TRL and verl_gr report timing differently.

### Distribution Analysis

**Key finding:** the current `fp32opt` trace is still not vLLM-rollout bound. vLLM rollout is ~10% of logged step time, while actor update plus weight synchronization is ~55%. Compared with the older May 26 run, `update_actor` is much lower and `update_weights` is now a major visible component.

| Phase | Mean Time | % of Step |
|---|---|---|
| gen (vLLM rollout) | 0.53s | 10% |
| update_actor (FSDP train step) | 1.78s | 34% |
| update_weights (actor → rollout sync) | 1.06s | 20% |
| old_log_prob | 0.47s | 9% |
| ref | 0.44s | 9% |
| adv | 0.09s | 2% |
| Other/overhead | ~0.83s | 16% |
| **Total logged step** | **~5.20s** | **100%** |

These numbers are means from the current `g2_3_trlmatch_ppoegradaccu6_trainshuffleOn_fp32opt` TensorBoard timing scalars through step 1370. Validation (`timing_s/testing`, ~591s when it runs) and checkpoint saving (`timing_s/save_checkpoint`, ~4.15s when it runs) are logged separately from the regular per-step phase distribution.

### Eval runtime

| Implementation | Eval runtime | Frequency |
|---------------|-------------|-----------|
| Original TRL Rank-GRPO | ~442s (mean of step 200 and 400 evals) | Every 200 steps |
| verl_gr Fork (`fp32opt`) | not logged as a direct eval runtime scalar in this trace | Every 200 steps |
