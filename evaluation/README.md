# Object Instance Evaluation

This directory stores manual same-instance pair annotations and evaluation outputs for the offline VLM description + GraphRAG similarity study.

Files:

- `object_instance_pairs.jsonl`: human-labeled ground-truth pairs used by `spatial_rag.object_instance_eval`
- `object_instance_pairs.example.jsonl`: one example JSONL row showing the expected schema

Each JSONL row uses `object_global_id` from `object_meta.jsonl`:

```json
{"pair_id":"pair_000001","db_dir":"spatial_db_split_k8s","obj_a_id":104,"obj_b_id":107,"is_same_instance":true,"split":"dev","notes":"same bed seen from two headings"}
```

Use `python -m spatial_rag.object_instance_pair_mining --db_dir ... --output_dir ...` to export candidate pairs and source images for annotation.
