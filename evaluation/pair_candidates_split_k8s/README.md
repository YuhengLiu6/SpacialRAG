# Object Instance Pair Annotation Candidates

Generated at: `2026-03-16T00:38:41.187737+00:00`
DB: `spatial_db_split_k8s`

Each folder in `pairs/` contains:
- `pair_manifest.json`
- source image for object A
- source image for object B

Create manual labels in `evaluation/object_instance_pairs.jsonl` using:

```json
{"pair_id":"cand_000001","db_dir":"spatial_db_split_k8s","obj_a_id":1,"obj_b_id":2,"is_same_instance":true,"split":"dev","notes":"same chair"}
```

Suggested labels in `candidates.jsonl` are heuristics only and must be verified by a human annotator.
