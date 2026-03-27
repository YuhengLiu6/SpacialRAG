import json

from spatial_rag.object_crop_prompt_probe import collect_jobs


def test_collect_jobs_reads_crop_and_latest_query_context(tmp_path):
    query_dir = tmp_path / "query_01_sampled_attempt_01"
    query_dir.mkdir(parents=True)
    crop_path = query_dir / "query_object_crop.jpg"
    crop_path.write_bytes(b"fake-jpg")

    older = query_dir / "query_20260312_100000.json"
    newer = query_dir / "query_20260312_100100.json"
    older.write_text(json.dumps({"query": {"yolo_label_hint": "old"}}), encoding="utf-8")
    newer.write_text(
        json.dumps(
            {
                "query": {"yolo_label_hint": "chair", "yolo_confidence_hint": 0.75},
                "selected_detection": {"label": "chair"},
            }
        ),
        encoding="utf-8",
    )

    jobs = collect_jobs(tmp_path)

    assert len(jobs) == 1
    assert jobs[0]["crop_path"] == str(crop_path)
    assert jobs[0]["query_json_path"] == str(newer)
    assert jobs[0]["yolo_label"] == "chair"
    assert jobs[0]["yolo_confidence"] == 0.75
