import json

import numpy as np

from spatial_rag.object_instance_eval import (
    PairScoreRecord,
    _prepare_pair_scores,
    build_graph_context_strings,
    export_pair_artifacts,
    load_object_pair_ground_truth,
    summarize_similarity_metrics,
)


def _meta_row(entry_id, x, z, orientation, room_function="resting", view_type="living room"):
    return {
        "id": entry_id,
        "x": float(x),
        "y": float(z),
        "world_position": [float(x), 0.0, float(z)],
        "orientation": int(orientation),
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "parse_status": "ok",
        "frame_text_short": f"short {entry_id}",
        "frame_text_long": f"long {entry_id}",
        "room_function": room_function,
        "view_type": view_type,
    }


def _object_row(object_global_id, entry_id, label, description, orientation, distance):
    return {
        "object_global_id": object_global_id,
        "entry_id": entry_id,
        "file_name": f"images/view_{entry_id:05d}.jpg",
        "label": label,
        "description": description,
        "long_form_open_description": f"{description} long",
        "laterality": "center",
        "distance_bin": "middle",
        "verticality": "middle",
        "distance_from_camera_m": distance,
        "object_orientation_deg": orientation,
        "location_relative_to_other_objects": "",
        "parse_status": "ok",
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _make_eval_db(tmp_path):
    db_dir = tmp_path / "toy_db"
    image_dir = db_dir / "images"
    image_dir.mkdir(parents=True)
    for entry_id in range(4):
        (image_dir / f"view_{entry_id:05d}.jpg").write_bytes(f"img-{entry_id}".encode("utf-8"))

    meta_rows = [
        _meta_row(0, 0.0, 0.0, 0),
        _meta_row(1, 0.0, 0.0, 90),
        _meta_row(2, 0.0, 3.0, 0, room_function="bedroom", view_type="bedroom"),
        _meta_row(3, 0.0, 3.0, 90, room_function="bedroom", view_type="bedroom"),
    ]
    object_rows = [
        _object_row(10, 0, "chair", "red chair near table", 0.0, 1.0),
        _object_row(20, 1, "chair", "red chair side view", 45.0, 1.1),
        _object_row(30, 2, "chair", "chair in mirror reflection", 180.0, 2.5),
        _object_row(40, 0, "table", "small wooden table", 20.0, 1.2),
        _object_row(50, 2, "lamp", "floor lamp by bed", 200.0, 2.0),
    ]
    _write_jsonl(db_dir / "meta.jsonl", meta_rows)
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    (db_dir / "build_report.json").write_text(
        json.dumps({"scan_angles": [0, 90], "random_config": {"scan_angles": [0, 90]}}),
        encoding="utf-8",
    )

    short_emb = np.asarray(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.8],
        ],
        dtype=np.float32,
    )
    long_emb = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-0.95, 0.05],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(db_dir / "object_text_emb_short.npy", short_emb)
    np.save(db_dir / "object_text_emb_long.npy", long_emb)
    return db_dir


def test_load_ground_truth_dedupes_reversed_pairs_and_filters_split(tmp_path):
    gt_path = tmp_path / "pairs.jsonl"
    rows = [
        {"pair_id": "pair_1", "db_dir": "toy_db", "obj_a_id": 20, "obj_b_id": 10, "is_same_instance": True, "split": "dev"},
        {"pair_id": "pair_2", "db_dir": "toy_db", "obj_a_id": 10, "obj_b_id": 20, "is_same_instance": True, "split": "dev"},
        {"pair_id": "pair_3", "db_dir": "toy_db", "obj_a_id": 10, "obj_b_id": 30, "is_same_instance": False, "split": "test"},
    ]
    _write_jsonl(gt_path, rows)

    records = load_object_pair_ground_truth(str(gt_path), split="dev")

    assert len(records) == 1
    assert records[0].obj_a_id == 10
    assert records[0].obj_b_id == 20


def test_prepare_pair_scores_maps_row_embeddings_by_object_global_id(tmp_path, monkeypatch):
    db_dir = _make_eval_db(tmp_path)
    gt_path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        gt_path,
        [
            {"pair_id": "pair_pos", "db_dir": db_dir.name, "obj_a_id": 20, "obj_b_id": 10, "is_same_instance": True, "split": "dev"},
            {"pair_id": "pair_neg", "db_dir": db_dir.name, "obj_a_id": 10, "obj_b_id": 30, "is_same_instance": False, "split": "dev"},
        ],
    )

    def _fake_embed(context_by_obj_id, embedder=None):
        out = np.zeros((max(context_by_obj_id) + 1, 2), dtype=np.float32)
        out[10] = [1.0, 0.0]
        out[20] = [0.9, 0.1]
        out[30] = [-1.0, 0.0]
        out[40] = [0.0, 1.0]
        out[50] = [0.0, 0.8]
        return out

    monkeypatch.setattr("spatial_rag.object_instance_eval.embed_graph_contexts", _fake_embed)

    pair_gt, pair_scores, graph_context = _prepare_pair_scores(
        db_dir=str(db_dir),
        gt_pairs_path=str(gt_path),
        split="dev",
        text_modes=("short", "long", "graph"),
        use_long_descriptions=True,
    )

    assert len(pair_gt) == 2
    assert 10 in graph_context
    scores_by_id = {row.pair_id: row for row in pair_scores}
    assert scores_by_id["pair_pos"].short_cosine > 0.99
    assert scores_by_id["pair_pos"].long_cosine > 0.99
    assert scores_by_id["pair_neg"].short_cosine < -0.99
    assert scores_by_id["pair_neg"].graph_cosine < -0.99


def test_build_graph_context_strings_is_deterministic_and_includes_same_node_and_direction(tmp_path):
    db_dir = _make_eval_db(tmp_path)

    first = build_graph_context_strings(str(db_dir))
    second = build_graph_context_strings(str(db_dir))

    assert first == second
    context = first[10]
    assert "place=place_00000" in context
    assert "same_node:" in context
    assert "table: small wooden table long" in context
    assert "north: chair: chair in mirror reflection" in context
    assert "room_function=resting" in context


def test_export_pair_artifacts_writes_pair_manifests_and_copies_images(tmp_path):
    db_dir = _make_eval_db(tmp_path)
    pair_scores = [
        PairScoreRecord(
            pair_id="pair_pos",
            db_dir=db_dir.name,
            obj_a_id=10,
            obj_b_id=20,
            is_same_instance=True,
            split="dev",
            short_cosine=0.99,
            long_cosine=0.995,
            graph_cosine=0.98,
        ),
        PairScoreRecord(
            pair_id="pair_neg",
            db_dir=db_dir.name,
            obj_a_id=10,
            obj_b_id=30,
            is_same_instance=False,
            split="dev",
            short_cosine=-1.0,
            long_cosine=-0.99,
            graph_cosine=-1.0,
        ),
    ]

    summary = export_pair_artifacts(
        pair_scores=pair_scores,
        db_dir=str(db_dir),
        output_dir=str(tmp_path / "artifacts"),
        graph_context_by_obj_id={10: "ctx-a", 20: "ctx-b", 30: "ctx-c"},
        max_examples_per_bucket=1,
    )

    pair_dir = tmp_path / "artifacts" / "examples" / "positives_high_similarity" / "pair_pos"
    manifest = json.loads((pair_dir / "pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["graph_context_a"] == "ctx-a"
    assert (pair_dir / "a_view_00000.jpg").exists()
    assert (pair_dir / "b_view_00001.jpg").exists()
    assert summary["bucket_counts"]["positives_high_similarity"] == 1


def test_summarize_similarity_metrics_reports_all_representations():
    pair_scores = [
        PairScoreRecord("p1", "toy_db", 10, 20, True, "dev", 0.9, 0.95, 0.92),
        PairScoreRecord("p2", "toy_db", 10, 30, False, "dev", -0.8, -0.7, -0.85),
        PairScoreRecord("p3", "toy_db", 20, 30, False, "dev", -0.7, -0.6, -0.8),
    ]

    metrics = summarize_similarity_metrics(pair_scores)

    assert metrics["num_pairs"] == 3
    assert metrics["short"]["positive_mean_cosine"] == 0.9
    assert metrics["graph"]["roc_auc"] == 1.0
    assert metrics["long"]["retrieval"]["num_anchors"] == 2
