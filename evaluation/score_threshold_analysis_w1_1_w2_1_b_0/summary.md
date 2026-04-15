# Score Threshold Analysis

- DB: `/Users/liuyuheng/Desktop/antigravityTest/spatial_db_origin`
- Score source: `object_meta.jsonl`
- Count: `1110`
- Mean / Median: `0.4578` / `0.4983`
- p10 / p25 / p75 / p90: `0.3318` / `0.4413` / `0.5000` / `0.5000`
- Sigmoid curve: `sigmoid_po_curve.png`
- Score histogram: `object_r_score_histogram.png`

## Candidate Thresholds

| threshold | num_below | num_at_or_above | drop_rate | keep_rate |
| --- | ---: | ---: | ---: | ---: |
| 0.2 | 0 | 1110 | 0.0000 | 1.0000 |
| 0.4 | 225 | 885 | 0.2027 | 0.7973 |
| 0.6 | 1110 | 0 | 1.0000 | 0.0000 |
| 0.8 | 1110 | 0 | 1.0000 | 0.0000 |
