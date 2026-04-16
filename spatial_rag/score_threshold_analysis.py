from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spatial_rag.config import OCCLUSION_REWEIGHT_B, OCCLUSION_REWEIGHT_W1, OCCLUSION_REWEIGHT_W2
from spatial_rag.occlusion_scoring import (
    OCCLUSION_LEVEL_TO_PENALTY,
    compute_reweighted_detection_score_from_penalty,
)


DEFAULT_C_DET = 0.8
DEFAULT_PO_MIN = 0.0
DEFAULT_PO_MAX = 0.6
DEFAULT_NUM_POINTS = 200
DEFAULT_HIST_BINS = 30
DEFAULT_CANDIDATE_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_float_list(value: Any) -> List[float]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    out: List[float] = []
    for item in parts:
        parsed = _safe_float(item)
        if parsed is None:
            raise ValueError(f"Invalid numeric value: {item!r}")
        out.append(float(parsed))
    return out


def _infer_reweight_defaults(db_root: Path) -> Dict[str, float]:
    build_report = _load_json(db_root / "build_report.json")
    occlusion = dict((build_report.get("object_config") or {}).get("occlusion_reweight") or {})
    return {
        "w1": float(_safe_float(occlusion.get("w1")) if _safe_float(occlusion.get("w1")) is not None else OCCLUSION_REWEIGHT_W1),
        "w2": float(_safe_float(occlusion.get("w2")) if _safe_float(occlusion.get("w2")) is not None else OCCLUSION_REWEIGHT_W2),
        "b": float(_safe_float(occlusion.get("b")) if _safe_float(occlusion.get("b")) is not None else OCCLUSION_REWEIGHT_B),
    }


def _is_filtered_db(build_report: Mapping[str, Any]) -> bool:
    if bool(build_report.get("r_threshold_enabled")):
        return True
    object_cfg = dict(build_report.get("object_config") or {})
    return bool(object_cfg.get("r_threshold_enabled"))


def _load_score_rows(db_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    object_rows = _load_jsonl(db_root / "object_meta.jsonl")
    if object_rows:
        enriched_rows: List[Dict[str, Any]] = []
        valid_count = 0
        for row in object_rows:
            score = _safe_float(row.get("reweighted_detection_score_r"))
            if score is not None:
                valid_count += 1
            enriched_rows.append(
                {
                    "object_global_id": row.get("object_global_id"),
                    "label": row.get("label"),
                    "geometry_source": row.get("geometry_source"),
                    "detector_confidence": row.get("detector_confidence"),
                    "reweighted_detection_score_r": score,
                }
            )
        if valid_count > 0:
            return enriched_rows, "object_meta.jsonl"

    score_rows = _load_csv_rows(db_root / "object_r_scores.csv")
    if score_rows:
        fallback_rows: List[Dict[str, Any]] = []
        for row in score_rows:
            fallback_rows.append(
                {
                    "object_global_id": row.get("object_global_id"),
                    "label": None,
                    "geometry_source": None,
                    "detector_confidence": None,
                    "reweighted_detection_score_r": _safe_float(row.get("reweighted_detection_score_r")),
                }
            )
        return fallback_rows, "object_r_scores.csv"

    raise FileNotFoundError(
        f"Could not load score rows from {db_root / 'object_meta.jsonl'} or {db_root / 'object_r_scores.csv'}"
    )


def _valid_scores(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = [_safe_float(row.get("reweighted_detection_score_r")) for row in rows]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        raise ValueError("No valid reweighted_detection_score_r values found")
    return np.asarray(clean, dtype=np.float32)


def _compute_score_stats(scores: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Expected at least one score")
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _threshold_candidate_rows(scores: np.ndarray, thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    rows: List[Dict[str, Any]] = []
    total = max(int(values.size), 1)
    for threshold in thresholds:
        below = int(np.count_nonzero(values < float(threshold)))
        at_or_above = int(values.size - below)
        rows.append(
            {
                "threshold": float(threshold),
                "num_below": below,
                "num_at_or_above": at_or_above,
                "drop_rate": float(below / total),
                "keep_rate": float(at_or_above / total),
            }
        )
    return rows


def _curve_points(
    *,
    c_det: float,
    w1: float,
    w2: float,
    b: float,
    po_min: float,
    po_max: float,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    penalties = np.linspace(float(po_min), float(po_max), int(num_points), dtype=np.float32)
    scores = np.asarray(
        [
            compute_reweighted_detection_score_from_penalty(
                c_det=float(c_det),
                occlusion_penalty_p_o=float(value),
                w1=float(w1),
                w2=float(w2),
                b=float(b),
            )
            for value in penalties
        ],
        dtype=np.float32,
    )
    return penalties, scores


def _plot_sigmoid_curve(
    path: Path,
    *,
    c_det: float,
    w1: float,
    w2: float,
    b: float,
    po_min: float,
    po_max: float,
    num_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    penalties, scores = _curve_points(
        c_det=c_det,
        w1=w1,
        w2=w2,
        b=b,
        po_min=po_min,
        po_max=po_max,
        num_points=num_points,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(penalties, scores, color="#1f77b4", linewidth=2.2, label="r(p_o)")
    for label, penalty in sorted(OCCLUSION_LEVEL_TO_PENALTY.items(), key=lambda item: float(item[1])):
        ax.axvline(float(penalty), color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            float(penalty),
            0.02,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#555555",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_title(f"Reweighted Score vs Occlusion Penalty (c_det={float(c_det):.2f})")
    ax.set_xlabel("occlusion_penalty_p_o")
    ax.set_ylabel("reweighted_detection_score_r")
    ax.set_xlim(float(po_min), float(po_max))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_score_histogram(
    path: Path,
    *,
    scores: np.ndarray,
    stats: Mapping[str, Any],
    candidate_thresholds: Sequence[float],
    bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(np.asarray(scores, dtype=np.float32), bins=int(bins), color="#4c78a8", edgecolor="white", alpha=0.85)
    stat_lines = [
        ("mean", float(stats["mean"]), "#d62728"),
        ("median", float(stats["median"]), "#2ca02c"),
        ("p10", float(stats["p10"]), "#9467bd"),
        ("p25", float(stats["p25"]), "#8c564b"),
        ("p75", float(stats["p75"]), "#8c564b"),
        ("p90", float(stats["p90"]), "#9467bd"),
    ]
    for name, value, color in stat_lines:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.3, alpha=0.85, label=name)
    for threshold in candidate_thresholds:
        ax.axvline(float(threshold), color="#ff7f0e", linestyle=":", linewidth=1.8, alpha=0.95, label=f"t={threshold:g}")
    handles, labels = ax.get_legend_handles_labels()
    dedup_handles = []
    dedup_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    ax.legend(dedup_handles, dedup_labels, loc="best", fontsize=8)
    ax.set_title("Distribution of Reweighted Detection Scores")
    ax.set_xlabel("reweighted_detection_score_r")
    ax.set_ylabel("object count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_markdown(
    *,
    db_root: Path,
    score_source: str,
    stats: Mapping[str, Any],
    thresholds: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    curve_path: Path,
    hist_path: Path,
) -> str:
    lines = [
        "# Score Threshold Analysis",
        "",
        f"- DB: `{db_root}`",
        f"- Score source: `{score_source}`",
        f"- Count: `{int(stats['count'])}`",
        f"- Mean / Median: `{float(stats['mean']):.4f}` / `{float(stats['median']):.4f}`",
        f"- p10 / p25 / p75 / p90: `{float(stats['p10']):.4f}` / `{float(stats['p25']):.4f}` / `{float(stats['p75']):.4f}` / `{float(stats['p90']):.4f}`",
        f"- Sigmoid curve: `{curve_path.name}`",
        f"- Score histogram: `{hist_path.name}`",
        "",
    ]
    if warnings:
        lines.append("## Warning")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(
        [
            "## Candidate Thresholds",
            "",
            "| threshold | num_below | num_at_or_above | drop_rate | keep_rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in thresholds:
        lines.append(
            f"| {float(row['threshold']):g} | {int(row['num_below'])} | {int(row['num_at_or_above'])} | "
            f"{float(row['drop_rate']):.4f} | {float(row['keep_rate']):.4f} |"
        )
    return "\n".join(lines).strip() + "\n"


def run_score_threshold_analysis(
    *,
    db_dir: str,
    output_dir: Optional[str] = None,
    c_det: float = DEFAULT_C_DET,
    w1: Optional[float] = None,
    w2: Optional[float] = None,
    b: Optional[float] = None,
    po_min: float = DEFAULT_PO_MIN,
    po_max: float = DEFAULT_PO_MAX,
    num_points: int = DEFAULT_NUM_POINTS,
    bins: int = DEFAULT_HIST_BINS,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
) -> Dict[str, Any]:
    db_root = Path(db_dir).expanduser().resolve()
    if not db_root.exists():
        raise FileNotFoundError(f"DB directory does not exist: {db_root}")

    build_report = _load_json(db_root / "build_report.json")
    inferred = _infer_reweight_defaults(db_root)
    resolved_w1 = float(inferred["w1"] if w1 is None else w1)
    resolved_w2 = float(inferred["w2"] if w2 is None else w2)
    resolved_b = float(inferred["b"] if b is None else b)
    thresholds = [float(value) for value in list(candidate_thresholds)]
    if not thresholds:
        thresholds = list(DEFAULT_CANDIDATE_THRESHOLDS)

    rows, score_source = _load_score_rows(db_root)
    scores = _valid_scores(rows)
    stats = _compute_score_stats(scores)
    threshold_rows = _threshold_candidate_rows(scores, thresholds)

    warnings: List[str] = []
    if _is_filtered_db(build_report):
        warnings.append(
            "This DB appears to already be threshold-filtered; threshold-selection statistics may be biased. "
            "A canonical no-threshold DB is recommended."
        )

    out_root = Path(output_dir).expanduser().resolve() if output_dir else (db_root / "score_threshold_analysis")
    out_root.mkdir(parents=True, exist_ok=True)
    curve_path = out_root / "sigmoid_po_curve.png"
    hist_path = out_root / "object_r_score_histogram.png"
    stats_path = out_root / "score_stats.json"
    threshold_csv_path = out_root / "threshold_candidates.csv"
    summary_md_path = out_root / "summary.md"

    _plot_sigmoid_curve(
        curve_path,
        c_det=float(c_det),
        w1=resolved_w1,
        w2=resolved_w2,
        b=resolved_b,
        po_min=float(po_min),
        po_max=float(po_max),
        num_points=int(num_points),
    )
    _plot_score_histogram(
        hist_path,
        scores=scores,
        stats=stats,
        candidate_thresholds=thresholds,
        bins=int(bins),
    )
    _write_csv_rows(
        threshold_csv_path,
        ("threshold", "num_below", "num_at_or_above", "drop_rate", "keep_rate"),
        threshold_rows,
    )

    report = {
        "db_dir": str(db_root),
        "output_dir": str(out_root),
        "score_source": score_source,
        "c_det": float(c_det),
        "reweight": {"w1": resolved_w1, "w2": resolved_w2, "b": resolved_b},
        "po_range": {"min": float(po_min), "max": float(po_max), "num_points": int(num_points)},
        "histogram_bins": int(bins),
        "candidate_thresholds": [float(value) for value in thresholds],
        "warnings": list(warnings),
        "sigmoid_po_curve_path": str(curve_path),
        "object_r_score_histogram_path": str(hist_path),
        "threshold_candidates_csv_path": str(threshold_csv_path),
        "summary_md_path": str(summary_md_path),
        **stats,
    }
    _write_json(stats_path, report)
    summary_md_path.write_text(
        _summary_markdown(
            db_root=db_root,
            score_source=score_source,
            stats=stats,
            thresholds=threshold_rows,
            warnings=warnings,
            curve_path=curve_path,
            hist_path=hist_path,
        ),
        encoding="utf-8",
    )
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot score-threshold diagnostics for an existing spatial DB.")
    parser.add_argument("--db_dir", type=str, required=True, help="Spatial DB directory to analyze.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional analysis output directory.")
    parser.add_argument("--c_det", type=float, default=DEFAULT_C_DET, help="Fixed detector confidence for the p_o -> r curve.")
    parser.add_argument("--w1", type=float, default=None, help="Override reweight formula w1.")
    parser.add_argument("--w2", type=float, default=None, help="Override reweight formula w2.")
    parser.add_argument("--b", type=float, default=None, help="Override reweight formula bias.")
    parser.add_argument("--po_min", type=float, default=DEFAULT_PO_MIN, help="Minimum p_o for the sigmoid curve.")
    parser.add_argument("--po_max", type=float, default=DEFAULT_PO_MAX, help="Maximum p_o for the sigmoid curve.")
    parser.add_argument("--num_points", type=int, default=DEFAULT_NUM_POINTS, help="Number of samples in the sigmoid curve.")
    parser.add_argument("--bins", type=int, default=DEFAULT_HIST_BINS, help="Histogram bin count.")
    parser.add_argument(
        "--candidate_thresholds",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated candidate thresholds to mark in the histogram and summary table.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = run_score_threshold_analysis(
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        c_det=float(args.c_det),
        w1=args.w1,
        w2=args.w2,
        b=args.b,
        po_min=float(args.po_min),
        po_max=float(args.po_max),
        num_points=int(args.num_points),
        bins=int(args.bins),
        candidate_thresholds=_parse_float_list(args.candidate_thresholds),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


if __name__ == "__main__":
    main()
