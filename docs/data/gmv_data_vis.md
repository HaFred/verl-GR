# GMV-Enriched GRPO Training Data

## Overview

The pipeline produces 3 task types from OpenOneRec-RecIF (162K users), adding price × engagement-funnel reward annotations for GMV-targeting RLVR.

## Data Sources

**Input** (OpenOneRec-RecIF):
- `OpenOneRec-RecIF/snapshots/8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet` — 162,074 users with video/ad/goods history, engagement labels, user profiles, CoT reasoning
- `video_ad_pid2sid.parquet`, `product_pid2sid.parquet`, `pid2caption.parquet` — ID mappings

**Intermediate** (stage 1 outputs used by assembly scripts):
- `outputs/item_price_lookup.parquet` — 57,730 products with prices from shopping cart data
- `outputs/category_price_lookup.parquet` — 2,053 categories with median prices
- `outputs/user_item_engagement.parquet` — 12.5M user×item rows with funnel scores (0.1–1.0)
- `outputs/price_stats.json` — global price distribution, tier thresholds


## Data Fabrication

The `rl_data_enriched/` parquets are produced by a 3-stage pipeline, invoked by `genrec_data/run_pipeline.sh`:

```
Source Dataset (162K users)
    │
    ▼
Stage 1: Signal Extraction ─────────────────────────────
  scripts/01_build_price_lookup.py
    → Parse shopping cart JSON from 88K user profiles
    → Extract 57,730 real product prices (¥0.01–¥9,999.90)
    → Compute 2,053 category-median prices as fallback
    → Output: item_price_lookup.parquet, category_price_lookup.parquet, price_stats.json

  scripts/02_compute_engagement_scores.py
    → Scan video/ad/goods history + target labels per user
    → Assign funnel_score per user×item: 0.1 (viewed) to 1.0 (purchased)
    → Output: user_item_engagement.parquet (12.5M rows)

  scripts/03_extract_cot_samples.py
    → Filter reco_cot non-null rows (3,646 samples)
    → Parse CoT reasoning chains, enrich with price tier
    → Output: cot_samples.parquet
    │
    ▼
Stage 2: Dataset Assembly ──────────────────────────────
  scripts/04_build_ranked_product_task.py
    → Combines video history + product history + price tier into prompts
    → Builds rank_labels with BOTH viewed (funnel=0.1, CTR) and purchased
      (funnel=1.0, CVR) items, sorted by price × funnel descending
    → Output: ranked_product.parquet (115,745 samples)

  scripts/05_build_purchase_prediction_task.py
    → Binary purchase prediction with candidate product price
    → Output: purchase_pred.parquet (55,000 samples)

  scripts/06_build_cot_reasoning_task.py
    → Enriches CoT samples with price context from user profiles
    → Output: cot_reason.parquet (3,646 samples)
    │
    ▼
Stage 3: Merge & Split ─────────────────────────────────
  scripts/07_merge_and_split.py
    → Merges all 3 task parquets
    → Stratified split: train (170K), test (3K), eval_gmv (1K)
    → Output: rl_data_enriched/{train,test,eval_gmv}.parquet
```

To regenerate after code changes, run from `genrec_data/`:
```bash
bash run_pipeline.sh          # full pipeline (all stages)
bash run_pipeline.sh --skip-stage1  # only stages 2+3
```


**Final** (GRPO training inputs):


| File | Purpose |
|------|---------|
| `rl_data_enriched/train.parquet` | GRPO training (170,391 samples) |
| `rl_data_enriched/test.parquet` | Validation during training (3,000 samples) |
| `rl_data_enriched/eval_gmv.parquet` | Standalone GMV eval — Ranked Product only (1,000 samples) |

### Task Breakdown (train)

```
Source                    Samples    Purpose
──────────────────────────────────────────────
RecIF_RankedProduct       113,745    Product ranking with CTR×CVR×Price NDCG
RecIF_PurchasePred         54,000    Binary purchase prediction with price weight
RecIF_CotReason             2,646    CoT reasoning enriched with price context
──────────────────────────────────────────────
Total                     170,391
```

---

## Task 1: Ranked Product (`RecIF_RankedProduct`)

**Prompt format**: video watch history (SID) + product history (SID) + price tier → ranked product SID list.

**Example (abridged)**:

```
System:  你是一个跨域推荐专家，擅长分析用户的观看习惯、购物偏好和消费能力，预测用户的商品兴趣并排序。

User:    用户长时间观看的视频：<|sid_begin|><s_a_849><s_b_6267><s_c_7692><|sid_end|> ...
        用户最近购买/加购的商品：<|sid_begin|><s_a_3899><s_b_3899><s_c_4370><|sid_end|> ...
        用户消费水平：¥低档
        请推荐用户接下来最可能购买的商品，从高到低排列。

Assistant: <|sid_begin|><s_a_3717><s_b_7001><s_c_4692><|sid_end|>
          <|sid_begin|><s_a_2564><s_b_2272><s_c_857><|sid_end|> ...
          (20 SID tokens, sorted by price × funnel descending)
```

**Metadata** (`rank_labels`): encodes the GMV hierarchy for NDCG reward computation:

```json
{
  "uid": 22314,
  "task": "ranked_product",
  "price_tier": "¥低档",
  "rank_labels": [
    {"price": 16.88, "funnel": 1.0},   // ← purchased (CVR)
    {"price": 16.88, "funnel": 1.0},
    ...                                    // ← 10 purchased items
    {"price": 6.64,  "funnel": 0.1},   // ← viewed only (CTR)
    {"price": 6.64,  "funnel": 0.1},
    ...                                    // ← 10 viewed-only items
  ]
}
```

**CTR×CVR encoding** — each sample has ~20 items split between two funnel levels:

| Funnel | Meaning | Count (avg) | GMV Role |
|--------|---------|-------------|----------|
| 1.0 | Purchased / converted | ~10 items | CVR |
| 0.1 | Viewed but not purchased | ~10 items | CTR |

**97.6% of ranked product samples** have both funnel levels (0.1 and 1.0), enabling the NDCG reward to differentiate CTR from CVR.

**Price distribution** (across all rank_labels, 5K sample):

```
min: ¥0.01   p10: ¥5.90   median: ¥16.88   p90: ¥35.90   max: ¥9,999.90
```

Prices come from 57,730 unique products extracted from real user shopping cart data, falling back to category-median or global-median (¥16.88) for items without explicit pricing.

---

## Task 2: Purchase Prediction (`RecIF_PurchasePred`)

**Prompt format**: video history + product history + candidate product with price → purchase decision.

**Example (abridged)**:

```
User:    用户浏览过的视频内容：<|sid_begin|>...<|sid_end|> ...
        用户最近购买的商品：<|sid_begin|>...<|sid_end|> ...
        用户消费水平：¥中档
        候选商品：<|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|> (¥9.90)
        请判断该用户是否会购买此商品，并说明理由。
```

**Metadata**:

```json
{"uid": 8508, "task": "purchase_pred", "label": 0, "price": 9.90, "price_tier": "¥中档"}
```

Reward: `correct_prediction × log(1 + price) / log(1 + 10000)`.

---

## Task 3: CoT Reasoning (`RecIF_CotReason`)

**Prompt format**: candidate video descriptions + user profile → reasoned recommendation.

**Metadata**:

```json
{"uid": 35395, "task": "cot_reason", "price_tier": "¥中档"}
```

The model generates `<think>` reasoning chains explaining why a user would engage with recommended content. Reward = `0.3 × think_format_quality + 0.7 × (SID_correct × price_weight)`.

---

## How the Reward Works (for GRPO Training)

The recipe at `verl_gr/recipes/openonerec/onerec_recipe.py` dispatches on `extra_info["task"]`:

```
GMV_REWARD=1 (env var)
  ├── task="ranked_product" → NDCG with relevance = funnel × log(1+price)/log(1+max)
  ├── task="purchase_pred"  → binary_correct × price_weight
  ├── task="cot_reason"     → 0.3 × think_format + 0.7 × SID_hit × price_weight
  └── (no task / legacy)    → pure SID pass@1 matching
```

**NDCG (for ranked_product)**: computed over the beam search output (32 beams). Each predicted SID is scored by the relevance of the ground-truth item it matches. The IDCG (ideal ordering) puts purchased items ahead of viewed items, both weighted by price. This captures:

```
GMV ≈ DCG where relevance = funnel × price_weight
    = Σ(CTR items × 0.1 × price_w) + Σ(CVR items × 1.0 × price_w)
    ∝ CTR × CVR × Price
```

---

## How to Use

### Training

The launch script at `verl_gr_fork_static_integra/scripts/.match_gmvscoring_openonerec_grpo.sh` sets:

```bash
export GMV_REWARD=1
export DATA_DIR=.../vanilla_onerec_customized_gmv_data/outputs/rl_data_enriched
```

This points the GRPO trainer at the enriched data and enables the GMV reward dispatch path.

### Eval

Run `08_eval_gmv.py` on the eval split (Ranked Product only, 1000 samples disjoint from train/test):

```bash
python genrec_data/08_eval_gmv.py \
  --model_path <checkpoint> \
  --eval_data outputs/rl_data_enriched/eval_gmv.parquet \
  --output_json /tmp/gmv_result.json
```

Metrics reported: Expected-GMV@K, Price-NDCG@K, GMV-Recall@K, Funnel-NDCG@K, Hit@K.
