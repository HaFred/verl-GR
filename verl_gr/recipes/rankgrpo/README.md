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

### Original Rank-GRPO (Qwen2.5-0.5B-Instruct, 2× GPU, 63,835 steps)

| Metric | Start | End |
|--------|-------|-----|
| Train reward (position-avg) | ~0.053 | 0.027 |
| Train reward total | ~1.05 | 0.53 |
| Train KL | ~0.0 | 0.52 |
| Train entropy | ~0.80 | 0.50 |
| Train loss | ~+0.05 | -0.0002 |

The model converges from high initial reward (exploration) to a stable policy with moderate KL divergence. Reward-per-position follows the expected rank-grpo pattern: higher reward at earlier positions, decaying toward position 20.

### verl_gr Fork (Qwen2.5-0.5B-Instruct, 2× H800, converged)

| Metric | Value |
|--------|-------|
| Total steps | 17,570 |
| Final eval/reward_total | 0.3515 |
| Final eval/reward (per-position) | 0.0176 |
| val-core reward mean@8 | 0.2454 |
| val-core reward best@8/mean | 0.3940 |
| Final actor loss | -0.0001 |
| Final KL loss | 0.0020 |
| Final grad norm | 0.5781 |
| time_per_step | ~2.74s |
| Throughput | ~3,180 tok/s |

Training saturated around step 14,000 at eval/reward_total ≈ 0.35. The model converged from ~0.33 at step 0 to ~0.35 at the end, with best@8 rewards peaking at 0.394.

---

## Performance

### per-time_per_step

| Implementation | GPUs | Steps | time_per_step (s) | Throughput (tok/s) | Notes |
|---------------|------|-------|-------------------|---------------------|-------|
| **Original Rank-GRPO** | 2× GPU | 63,835 | ~5.56 | — | 0.180 steps/s, 1.08 samples/s, ~99h total |
| **verl_gr Fork** | 2× H800 | 17,570 | ~2.74 | ~3,180 | Converged — eval/reward_total ≈ 0.35 |

The fork achieves ~2.74s/step on 2× H800 (2,240 tokens/step) at convergence. The original runs ~5.6s/step on 2 GPUs with higher token counts. At comparable token counts, the fork is competitive.

### Eval runtime

| Implementation | Eval runtime | Frequency |
|---------------|-------------|-----------|
| Original | ~412s (mean) | Every 200 steps |
| Fork | ~364s (mean) | Every 200 steps (87 eval points over 17,570 steps) |
