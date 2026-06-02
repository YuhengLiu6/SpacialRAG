from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from spatial_rag.object_instance_eval import build_graph_context_strings, embed_graph_contexts
from spatial_rag.object_match_mlp import (
    OBJECT_MATCH_FEATURE_NAMES,
    ObjectFeatureStore,
    ObjectMatchMLP,
    PositionStats,
    object_match_feature_indices,
)


DEFAULT_CHECKPOINT = "runs/object_match_mlp_validated_no_cabinet_80_20/model.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a saved object-match MLP for one anchor object and one target view. "
            "The model ranks objects in the target view, plus a final none option."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to a trained object-match MLP checkpoint.",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=None,
        help="Spatial DB directory. Defaults to the db_dir saved in the checkpoint.",
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
        help="Target view/entry id, e.g. 58, view_00058, or an image filename ending in an id.",
    )
    parser.add_argument(
        "--candidate_obj_ids",
        type=str,
        default="",
        help="Optional comma-separated object_global_id list to rank instead of all objects in the view.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of ranked candidates to print.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Defaults to mps, cuda, then cpu.")
    parser.add_argument(
        "--no_graph",
        action="store_true",
        help="Skip graph/context embeddings for a faster but less faithful smoke test.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full prediction payload as JSON instead of a compact table.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to write the prediction payload as JSON.",
    )
    return parser.parse_args()


def _choose_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _parse_entry_id(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("View cannot be empty")
    if text.isdigit():
        return int(text)
    stem = Path(text).stem
    groups = re.findall(r"\d+", stem or text)
    if not groups:
        raise ValueError(f"Could not parse an entry id from view={value!r}")
    return int(groups[-1])


def _parse_candidate_obj_ids(value: str) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _row_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "object_global_id": int(row.get("object_global_id")),
        "entry_id": row.get("entry_id"),
        "file_name": row.get("file_name"),
        "label": row.get("label"),
        "description": row.get("description"),
        "long_form_open_description": row.get("long_form_open_description"),
    }


def _candidate_ids_for_entry(
    store: ObjectFeatureStore,
    entry_id: int,
    explicit_candidate_obj_ids: Sequence[int],
) -> List[int]:
    if explicit_candidate_obj_ids:
        missing = [int(obj_id) for obj_id in explicit_candidate_obj_ids if int(obj_id) not in store.object_by_id]
        if missing:
            raise KeyError(f"Candidate object ids not found in DB: {missing}")
        return [int(obj_id) for obj_id in explicit_candidate_obj_ids]

    return [
        int(obj_id)
        for obj_id in store.entry_to_object_ids.get(int(entry_id), [])
        if str(store.object_by_id[int(obj_id)].get("label") or "").lower() != "none"
    ]


def _build_sparse_graph_embeddings(db_dir: str | Path, obj_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    contexts = build_graph_context_strings(str(db_dir))
    selected_contexts = {
        int(obj_id): contexts[int(obj_id)]
        for obj_id in sorted(set(int(value) for value in obj_ids))
        if int(obj_id) in contexts
    }
    if not selected_contexts:
        return {}
    dense = embed_graph_contexts(selected_contexts)
    return {
        int(obj_id): np.asarray(dense[int(obj_id)], dtype=np.float32)
        for obj_id in selected_contexts
        if int(obj_id) < int(dense.shape[0])
    }


def _build_model(checkpoint: Mapping[str, Any], device: torch.device) -> ObjectMatchMLP:
    config = dict(checkpoint.get("config") or {})
    model = ObjectMatchMLP(
        input_dim=int(config.get("input_dim", 10)),
        hidden=int(config.get("hidden", 64)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _feature_names_from_checkpoint(checkpoint: Mapping[str, Any]) -> List[str]:
    config = dict(checkpoint.get("config") or {})
    names = config.get("feature_names")
    input_dim = int(config.get("input_dim", len(OBJECT_MATCH_FEATURE_NAMES)))
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return list(OBJECT_MATCH_FEATURE_NAMES[:input_dim])


def predict(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint = _load_checkpoint(args.checkpoint)
    config = dict(checkpoint.get("config") or {})
    db_dir = Path(args.db_dir or config.get("db_dir") or "spatial_db_origin")
    if not db_dir.exists():
        raise FileNotFoundError(f"Missing db_dir: {db_dir}")

    candidate_entry_id = _parse_entry_id(args.view)
    base_store = ObjectFeatureStore.from_db(db_dir, build_graph_embeddings=False)
    if int(args.anchor_obj_id) not in base_store.object_by_id:
        raise KeyError(f"Anchor object id not found in DB: {args.anchor_obj_id}")

    candidate_ids = _candidate_ids_for_entry(
        base_store,
        candidate_entry_id,
        _parse_candidate_obj_ids(args.candidate_obj_ids),
    )
    if not candidate_ids:
        raise ValueError(f"No candidate objects found for entry/view id {candidate_entry_id}")

    store = base_store
    use_graph = not bool(args.no_graph)
    graph_embedding_count = 0
    if use_graph:
        with contextlib.redirect_stdout(sys.stderr):
            graph_embeddings = _build_sparse_graph_embeddings(
                db_dir,
                [int(args.anchor_obj_id), *candidate_ids],
            )
        graph_embedding_count = len(graph_embeddings)
        store = ObjectFeatureStore.from_db(db_dir, graph_embeddings_by_obj_id=graph_embeddings)

    stats = PositionStats.from_dict(checkpoint["position_stats"])
    features = np.vstack(
        [
            store.build_pair_feature(int(args.anchor_obj_id), int(candidate_obj_id), stats)
            for candidate_obj_id in candidate_ids
        ]
    ).astype(np.float32)
    feature_names = _feature_names_from_checkpoint(checkpoint)
    if len(feature_names) != int(config.get("input_dim", len(feature_names))):
        raise ValueError(
            f"Checkpoint feature_names length {len(feature_names)} does not match "
            f"input_dim={config.get('input_dim')}"
        )
    if tuple(feature_names) != tuple(OBJECT_MATCH_FEATURE_NAMES):
        features = features[:, object_match_feature_indices(feature_names)].astype(np.float32, copy=False)

    device = _choose_device(args.device)
    model = _build_model(checkpoint, device)
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        valid_mask = torch.ones((1, len(candidate_ids)), dtype=torch.bool, device=device)
        logits = model(feature_tensor, valid_mask=valid_mask)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    none_prob = float(probs[-1])
    candidate_rows: List[Dict[str, Any]] = []
    for obj_id, prob in sorted(zip(candidate_ids, probs[:-1]), key=lambda item: item[1], reverse=True):
        row = store.object_by_id[int(obj_id)]
        candidate_rows.append(
            {
                "object_global_id": int(obj_id),
                "prob": float(prob),
                "entry_id": row.get("entry_id"),
                "file_name": row.get("file_name"),
                "label": row.get("label"),
                "description": row.get("description"),
            }
        )

    best_candidate = candidate_rows[0]
    pred_is_none = none_prob >= float(best_candidate["prob"])
    anchor_row = store.object_by_id[int(args.anchor_obj_id)]
    return {
        "checkpoint": str(args.checkpoint),
        "db_dir": str(db_dir),
        "device": str(device),
        "feature_set": config.get("feature_set"),
        "feature_names": list(feature_names),
        "input_dim": int(len(feature_names)),
        "use_graph": bool(use_graph),
        "graph_embedding_count": int(graph_embedding_count),
        "anchor_obj_id": int(args.anchor_obj_id),
        "anchor": _row_summary(anchor_row),
        "candidate_entry_id": int(candidate_entry_id),
        "candidate_count": int(len(candidate_rows)),
        "pred_is_none": bool(pred_is_none),
        "pred_obj_id": None if pred_is_none else int(best_candidate["object_global_id"]),
        "pred_prob": float(none_prob if pred_is_none else best_candidate["prob"]),
        "none_prob": float(none_prob),
        "top_candidates": candidate_rows[: max(1, int(args.top_k))],
    }


def _print_table(payload: Mapping[str, Any]) -> None:
    anchor = dict(payload.get("anchor") or {})
    print(
        "anchor_obj_id={anchor_obj_id} target_entry_id={candidate_entry_id} "
        "query_label={query_label} candidates={candidate_count} use_graph={use_graph}".format(
            query_label=anchor.get("label") or "",
            **payload,
        )
    )
    if payload["pred_is_none"]:
        print(f"prediction=NONE prob={payload['pred_prob']:.6f}")
    else:
        pred_label = ""
        if payload.get("top_candidates"):
            pred_label = str(payload["top_candidates"][0].get("label") or "")
        pred_text = f"obj_{payload['pred_obj_id']}:{pred_label}" if pred_label else f"obj_{payload['pred_obj_id']}"
        print(f"prediction={pred_text} prob={payload['pred_prob']:.6f}")
    print(f"none_prob={payload['none_prob']:.6f}")
    print("")
    rows = []
    for rank, row in enumerate(payload["top_candidates"], start=1):
        rows.append(
            {
                "rank": str(rank),
                "obj_id": str(row["object_global_id"]),
                "prob": f"{row['prob']:.6f}",
                "label": str(row.get("label") or ""),
                "description": str(row.get("description") or ""),
            }
        )
    headers = {
        "rank": "rank",
        "obj_id": "obj_id",
        "prob": "prob",
        "label": "label",
        "description": "description",
    }
    columns = ["rank", "obj_id", "prob", "label", "description"]
    widths = {
        col: max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in columns
    }

    def fmt(row: Mapping[str, str]) -> str:
        return (
            f"{row['rank']:>{widths['rank']}}  "
            f"{row['obj_id']:>{widths['obj_id']}}  "
            f"{row['prob']:>{widths['prob']}}  "
            f"{row['label']:<{widths['label']}}  "
            f"{row['description']:<{widths['description']}}"
        ).rstrip()

    print(fmt(headers))
    print(
        f"{'-' * widths['rank']}  "
        f"{'-' * widths['obj_id']}  "
        f"{'-' * widths['prob']}  "
        f"{'-' * widths['label']}  "
        f"{'-' * widths['description']}"
    )
    for row in rows:
        print(
            fmt(row)
        )


def main() -> None:
    args = _parse_args()
    payload = predict(args)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        _print_table(payload)


if __name__ == "__main__":
    main()
