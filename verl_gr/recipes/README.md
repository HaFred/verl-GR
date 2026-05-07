# `verl_gr/recipes`

Generative recommendation systems (GenRecSys) use LLMs to produce recommendation rankings. 
Technically, their approaches are two-fold: output tokens can be treated either as semantic IDs (SIDs) embedded for products, goods, or items, or as natural-language representations of ranked items.

 `verl-gr` currently supports three recipes for GRPO training:
* OpenOneRec
* MiniOneRec
* Rank-GRPO

We picked these three works for the initial release as they cover the two major routes of GenRecSys, and their intersection.

```
+----------------------------------------------------------------------------+
| OpenOneRec GRPO recipe                                                     |
|                                                                            |
| Rollout: custom two-stage AgentLoopWorker                                  |
|   user history -> policy LLM                                               |
|                -> stage 1: natural-language thinking context               |
|                -> stage 2: beam-search SID generation                      |
|                -> beam-width SIDs as ranking results                       |
|                                                                            |
| Optimization: vanilla GRPO reward/advantage/loss path                      |
|                                                                            |
| Route: intersection of NL thinking and SID output                          |
+----------------------------------------------------------------------------+
```

```
+----------------------------------------------------------------------------+
| MiniOneRec GRPO recipe                                                     |
|                                                                            |
| Rollout: custom constrained-beam AgentLoopWorker                           |
|   user history -> policy LLM                                               |
|                -> constrained beam-search SID generation                   |
|                -> beam-width SIDs as ranking results                       |
|                                                                            |
| Optimization: vanilla GRPO reward/advantage/loss path                      |
|                                                                            |
| Route: SID route                                                           |
+----------------------------------------------------------------------------+
```

```
+----------------------------------------------------------------------------+
| Rank-GRPO recipe                                                           |
|                                                                            |
| Rollout: vanilla verl async single-turn LLM rollout                        |
|   user request/history -> policy LLM                                       |
|                        -> natural-language ranked items                    |
|                                                                            |
| Optimization: Rank-GRPO-specific rank-wise reward, advantage, and loss     |
|                                                                            |
| Route: natural-language ranking route                                      |
+----------------------------------------------------------------------------+
```