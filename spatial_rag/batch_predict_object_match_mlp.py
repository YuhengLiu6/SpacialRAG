from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from spatial_rag.object_match_mlp import OBJECT_MATCH_FEATURE_SET_ORDER
from spatial_rag.predict_object_match_mlp import predict


DEFAULT_MODELS_ROOT = "runs"
DEFAULT_OUTPUT_DIR = "runs/object_match_mlp_batch_predictions"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one object-match query against every discovered object-match MLP checkpoint."
    )
    parser.add_argument(
        "--models_root",
        type=str,
        default=DEFAULT_MODELS_ROOT,
        help="Directory to recursively search for model.pt checkpoints.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths. If omitted, model.pt files are discovered under --models_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-model prediction JSON plus summary JSON/CSV.",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=None,
        help="Spatial DB directory. Defaults to each checkpoint's saved db_dir.",
    )
    parser.add_argument(
        "--obj_id",
        "--anchor_obj_id",
        dest="anchor_obj_id",
        type=int,
        required=True,
        help="Anchor object_global_id to search for in the target view.",
    )
    parser.add_argument(
        "--view",
        "--candidate_entry_id",
        dest="view",
        type=str,
        required=True,
        help="Target view/entry id, e.g. 26, view_00026, or an image filename ending in an id.",
    )
    parser.add_argument(
        "--candidate_obj_ids",
        type=str,
        default="",
        help="Optional comma-separated object_global_id list to rank instead of all objects in the view.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of ranked candidates to keep per model.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Defaults to mps, cuda, then cpu.")
    parser.add_argument(
        "--no_graph",
        action="store_true",
        help="Skip graph/context embeddings for a faster but less faithful smoke test.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on the first checkpoint error instead of recording the error and continuing.",
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "model"


def _checkpoint_label(checkpoint: Path) -> str:
    parent = checkpoint.parent.name
    if parent in OBJECT_MATCH_FEATURE_SET_ORDER:
        return parent
    if checkpoint.name == "model.pt":
        return parent
    return checkpoint.stem


def _discover_checkpoints(models_root: str | Path, explicit: Sequence[str] | None) -> List[Path]:
    if explicit:
        checkpoints = [Path(path) for path in explicit]
    else:
        root = Path(models_root)
        if not root.exists():
            raise FileNotFoundError(f"Missing models_root: {root}")
        checkpoints = sorted(root.rglob("model.pt"))

    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(str(path) for path in missing)}")
    if not checkpoints:
        raise ValueError(f"No model.pt checkpoints found under {models_root}")
    return _sort_checkpoints(checkpoints)


def _sort_checkpoints(checkpoints: Sequence[Path]) -> List[Path]:
    feature_order = {name: idx for idx, name in enumerate(OBJECT_MATCH_FEATURE_SET_ORDER)}

    def sort_key(path: Path) -> tuple[int, str]:
        label = _checkpoint_label(path)
        return (feature_order.get(label, len(feature_order)), str(path))

    return sorted((Path(path) for path in checkpoints), key=sort_key)


def _prediction_args(args: argparse.Namespace, checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint),
        db_dir=args.db_dir,
        anchor_obj_id=int(args.anchor_obj_id),
        view=str(args.view),
        candidate_obj_ids=str(args.candidate_obj_ids or ""),
        top_k=int(args.top_k),
        device=args.device,
        no_graph=bool(args.no_graph),
        json=True,
        output_json="",
    )


def _summary_row(
    *,
    label: str,
    checkpoint: Path,
    payload: Mapping[str, Any] | None = None,
    output_json: Path | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    top_candidate = {}
    if payload and payload.get("top_candidates"):
        top_candidate = dict(payload["top_candidates"][0])
    pred_label = None
    if payload and not payload.get("pred_is_none"):
        pred_label = top_candidate.get("label")
    return {
        "model": label,
        "checkpoint": str(checkpoint),
        "status": "error" if error else "ok",
        "feature_set": None if payload is None else payload.get("feature_set"),
        "input_dim": None if payload is None else payload.get("input_dim"),
        "query_label": None if payload is None else dict(payload.get("anchor") or {}).get("label"),
        "pred_is_none": None if payload is None else payload.get("pred_is_none"),
        "pred_obj_id": None if payload is None else payload.get("pred_obj_id"),
        "pred_label": pred_label,
        "pred_prob": None if payload is None else payload.get("pred_prob"),
        "none_prob": None if payload is None else payload.get("none_prob"),
        "top1_obj_id": top_candidate.get("object_global_id"),
        "top1_prob": top_candidate.get("prob"),
        "top1_label": top_candidate.get("label"),
        "candidate_count": None if payload is None else payload.get("candidate_count"),
        "output_json": None if output_json is None else str(output_json),
        "error": error,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "model",
        "checkpoint",
        "status",
        "feature_set",
        "input_dim",
        "query_label",
        "pred_is_none",
        "pred_obj_id",
        "pred_label",
        "pred_prob",
        "none_prob",
        "top1_obj_id",
        "top1_prob",
        "top1_label",
        "candidate_count",
        "output_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoints = _discover_checkpoints(args.models_root, args.checkpoints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    used_labels: Dict[str, int] = {}

    for checkpoint in checkpoints:
        label = _safe_name(_checkpoint_label(checkpoint))
        used_labels[label] = used_labels.get(label, 0) + 1
        output_label = label if used_labels[label] == 1 else f"{label}_{used_labels[label]}"
        output_json = output_dir / f"{output_label}.json"
        try:
            payload = predict(_prediction_args(args, checkpoint))
            output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            rows.append(
                _summary_row(
                    label=output_label,
                    checkpoint=checkpoint,
                    payload=payload,
                    output_json=output_json,
                )
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append(_summary_row(label=output_label, checkpoint=checkpoint, error=str(exc)))

    summary = {
        "query": {
            "anchor_obj_id": int(args.anchor_obj_id),
            "view": str(args.view),
            "candidate_obj_ids": str(args.candidate_obj_ids or ""),
            "top_k": int(args.top_k),
            "db_dir": args.db_dir,
            "no_graph": bool(args.no_graph),
        },
        "models_root": str(args.models_root),
        "num_models": int(len(checkpoints)),
        "num_ok": int(sum(1 for row in rows if row["status"] == "ok")),
        "num_error": int(sum(1 for row in rows if row["status"] == "error")),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_csv(output_dir / "summary.csv", rows)
    return summary


def _print_summary(summary: Mapping[str, Any]) -> None:
    print(
        f"models={summary['num_models']} ok={summary['num_ok']} "
        f"errors={summary['num_error']}"
    )

    table_rows: List[Dict[str, str]] = []
    for row in summary["rows"]:
        if row["status"] != "ok":
            table_rows.append(
                {
                    "model": str(row["model"]),
                    "status": "error",
                    "dim": "",
                    "query_label": "",
                    "prediction": "",
                    "pred_prob": "",
                    "none_prob": "",
                    "top1": "",
                    "error": str(row.get("error") or ""),
                }
            )
            continue
        if row["pred_is_none"]:
            prediction = "NONE"
        else:
            pred_label = str(row.get("pred_label") or "")
            prediction = f"obj_{row['pred_obj_id']}:{pred_label}" if pred_label else f"obj_{row['pred_obj_id']}"
        top1 = "" if row["top1_obj_id"] is None else f"obj_{row['top1_obj_id']}:{row['top1_prob']:.6f}"
        table_rows.append(
            {
                "model": str(row["model"]),
                "status": "ok",
                "dim": str(row["input_dim"]),
                "query_label": str(row.get("query_label") or ""),
                "prediction": prediction,
                "pred_prob": f"{row['pred_prob']:.6f}",
                "none_prob": f"{row['none_prob']:.6f}",
                "top1": top1,
                "error": "",
            }
        )

    headers = {
        "model": "model",
        "status": "status",
        "dim": "dim",
        "query_label": "query_label",
        "prediction": "prediction",
        "pred_prob": "pred_prob",
        "none_prob": "none_prob",
        "top1": "top1",
        "error": "error",
    }
    columns = ["model", "status", "dim", "query_label", "prediction", "pred_prob", "none_prob", "top1", "error"]
    widths = {
        col: max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in columns
    }

    def fmt(row: Mapping[str, str]) -> str:
        return (
            f"{row['model']:<{widths['model']}}  "
            f"{row['status']:<{widths['status']}}  "
            f"{row['dim']:>{widths['dim']}}  "
            f"{row['query_label']:<{widths['query_label']}}  "
            f"{row['prediction']:<{widths['prediction']}}  "
            f"{row['pred_prob']:>{widths['pred_prob']}}  "
            f"{row['none_prob']:>{widths['none_prob']}}  "
            f"{row['top1']:<{widths['top1']}}  "
            f"{row['error']:<{widths['error']}}"
        ).rstrip()

    header_row = {col: headers[col] for col in columns}
    print(fmt(header_row))
    print(
        f"{'-' * widths['model']}  "
        f"{'-' * widths['status']}  "
        f"{'-' * widths['dim']}  "
        f"{'-' * widths['query_label']}  "
        f"{'-' * widths['prediction']}  "
        f"{'-' * widths['pred_prob']}  "
        f"{'-' * widths['none_prob']}  "
        f"{'-' * widths['top1']}  "
        f"{'-' * widths['error']}".rstrip()
    )
    for row in table_rows:
        print(fmt(row))


def main() -> None:
    summary = run_batch(_parse_args())
    _print_summary(summary)


if __name__ == "__main__":
    main()
