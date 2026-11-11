# Model card — Invoice and Receipt Extraction

No model has been trained yet. Every value below must come from a tracked run.

| Field | Value | Run ID |
|---|---|---|
| Model version | | |
| Base model, revision and licence | `Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546d2b821cf421d4f5eb2576359d3770cd3`; licence `qwen-research` (research licence: check it permits your use before any commercial deployment) | T-02 |
| Adapter size (trainable parameters) | | |
| Data version | | |
| Git tag | | |
| Training time and hardware | | |

## Results on the test split
| Metric | Baseline (zero-shot) | Small model | Gate |
|---|---|---|---|
| Field F1 | | | see SPEC §5 |
| Total accuracy | | | ≥ 0.95 |
| Date accuracy | | | ≥ 0.93 |
| Vendor accuracy | | | reported |
| Document exact match | | | reported |
| JSON validity (unrepaired) | | | ≥ 0.99 |
| Straight-through rate | n/a | | ≥ 0.60 |
| Precision of auto | n/a | | ≥ 0.98 |
| Cost per 1,000 documents | | | ≤ 20% of baseline |
| Documents per minute | | | reported |

## Confidence
State whether token-probability confidence ranked errors better than random (T-11) and whether routing uses it.

## Intended use
Extracting header fields and line items from single-page printed receipts similar to the training data, with review of flagged documents.

## Not intended for
Handwritten or multi-page documents; unreviewed posting to financial systems; document types not represented in training.

## Known failure modes
_Fill from per-field results and reviewer notes._
