import csv
import json
from pathlib import Path

import numpy as np
import pytest

from spatial_rag.occlusion_scoring import compute_reweighted_detection_score_from_penalty
from spatial_rag.score_threshold_analysis import (
    _compute_score_stats,
    _curve_points,
    run_score_threshold_analysis,
)
from spatial_rag.spatial_db_builder import _write_jsonl


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_score_db(tmp_path: Path, *, filtered: bool) -> Path:
    db_dir = tmp_path / ("filtered_score_db" if filtered else "canonical_score_db")
    db_dir.mkdir(parents=True, exist_ok=True)

    object_rows = [
        {
            "object_global_id": 0,
            "label": "chair",
            "geometry_source": "mask_depth",
            "detector_confidence": 0.2,
            "occlusion_level": "fully visible",
            "occlusion_penalty_p_o": 0.0,
            "reweighted_detection_score_r": 0.2,
        },
        {
            "object_global_id": 1,
            "label": "table",
            "geometry_source": "mask_depth",
            "detector_confidence": 0.9,
            "occlusion_level": "fully visible",
            "occlusion_penalty_p_o": 0.0,
            "reweighted_detection_score_r": 0.9,
        },
        {
            "object_global_id": 2,
            "label": "lamp",
            "geometry_source": "vlm_fallback",
            "detector_confidence": None,
            "occlusion_level": "uncertain",
            "occlusion_penalty_p_o": 0.35,
            "reweighted_detection_score_r": 0.65,
        },
    ]
    _write_jsonl(db_dir / "object_meta.jsonl", object_rows)
    _write_csv(
        db_dir / "object_r_scores.csv",
        ["object_global_id", "reweighted_detection_score_r"],
        [
            {
                "object_global_id": row["object_global_id"],
                "reweighted_detection_score_r": row["reweighted_detection_score_r"],
            }
            for row in object_rows
        ],
    )

    build_report = {
        "output_dir": str(db_dir),
        "r_threshold": 0.5 if filtered else None,
        "r_threshold_enabled": bool(filtered),
        "object_config": {
            "r_threshold": 0.5 if filtered else None,
            "r_threshold_enabled": bool(filtered),
            "occlusion_reweight": {"w1": 1.7, "w2": 0.8, "b": -0.1},
        },
    }
    (db_dir / "build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    return db_dir


def test_curve_points_match_reweight_formula_samples():
    penalties, scores = _curve_points(
        c_det=0.8,
        w1=1.0,
        w2=1.0,
        b=0.0,
        po_min=0.0,
        po_max=0.5,
        num_points=6,
    )
    assert len(penalties) == 6
    assert len(scores) == 6
    for penalty, score in zip(penalties, scores):
        expected = compute_reweighted_detection_score_from_penalty(0.8, float(penalty), w1=1.0, w2=1.0, b=0.0)
        assert score == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_compute_score_stats_and_threshold_rows_are_consistent(tmp_path):
    db_dir = _make_score_db(tmp_path, filtered=False)
    report = run_score_threshold_analysis(
        db_dir=str(db_dir),
        output_dir=str(tmp_path / "analysis"),
        candidate_thresholds=[0.4, 0.8],
    )

    assert report["reweight"] == {"w1": 1.7, "w2": 0.8, "b": -0.1}
    assert report["count"] == 3
    assert report["mean"] == pytest.approx(_compute_score_stats(np.array([0.2, 0.9, 0.65], dtype=float))["mean"])
    assert Path(report["sigmoid_po_curve_path"]).exists()
    assert Path(report["object_r_score_histogram_path"]).exists()
    assert Path(report["threshold_candidates_csv_path"]).exists()
    assert Path(report["summary_md_path"]).exists()

    rows = list(csv.DictReader(Path(report["threshold_candidates_csv_path"]).open("r", encoding="utf-8", newline="")))
    assert rows == [
        {
            "threshold": "0.4",
            "num_below": "1",
            "num_at_or_above": "2",
            "drop_rate": "0.3333333333333333",
            "keep_rate": "0.6666666666666666",
        },
        {
            "threshold": "0.8",
            "num_below": "2",
            "num_at_or_above": "1",
            "drop_rate": "0.6666666666666666",
            "keep_rate": "0.3333333333333333",
        },
    ]


def test_filtered_db_still_runs_and_writes_warning(tmp_path):
    db_dir = _make_score_db(tmp_path, filtered=True)
    report = run_score_threshold_analysis(
        db_dir=str(db_dir),
        output_dir=str(tmp_path / "analysis_filtered"),
    )

    assert report["warnings"]
    stats_payload = json.loads((Path(report["output_dir"]) / "score_stats.json").read_text(encoding="utf-8"))
    assert stats_payload["warnings"]
    summary_text = (Path(report["output_dir"]) / "summary.md").read_text(encoding="utf-8")
    assert "already be threshold-filtered" in summary_text
