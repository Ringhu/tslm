# Data schema (JSONL) - SchemaV1

Each line is a JSON object.

## Required fields

- `id` (string): unique sample id.
- `values` (list[float] OR list[list[float]]):
  - univariate: `[T]`
  - multivariate: `[T][D]` (time-major)
- `text` (string): the target sequence for SFT (Chinese for now).
  - Recommended: a *single* caption string for v1.
  - Optional: you can make it JSON text (model outputs JSON), see below.

## Optional but recommended fields

- `stats` (object):
  - `mean`, `std`, `min`, `max` (float)
  - `length` (int)
  - For multivariate, you can store lists: `mean: [D]`, etc.
- `claims` (list[object]): structured facts from your claim engine, for:
  - training (aux losses / filtering)
  - evaluation (fact metrics)
- `segments` (list[object]): if you later use sliding windows / dense captions.
- `meta` (object): dataset name, split, label, units, sampling rate…

## Output format variants

### Variant A (simplest): plain caption

`text` is a plain natural-language caption.

### Variant B (recommended for factual evaluation): JSON + caption

`text` is a JSON string. Example:

```json
{
  "facts": {
    "trend": "down",
    "delta_pct": -143.9,
    "peak": {"idx": 220, "value": 1.80},
    "valley": {"idx": 23, "value": -1.03}
  },
  "caption": "整体呈明显下降趋势…"
}
```

This avoids **parsing natural language** during evaluation.
