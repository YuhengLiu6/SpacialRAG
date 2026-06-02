from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from spatial_rag.object_match_mlp import (
    OBJECT_MATCH_ALL_FEATURE_SET_ORDER,
    OBJECT_MATCH_FEATURE_SET_REMOVED_GROUPS,
    OBJECT_MATCH_FEATURE_NAMES,
    OBJECT_MATCH_LEAVE_ONE_OUT_FEATURE_SET_ORDER,
    ObjectFeatureStore,
    ObjectMatchMLP,
    ObjectMatchTensorDataset,
    ObjectMatchTensorSample,
    build_match_sample_specs,
    build_tensor_samples,
    collate_object_match_samples,
    evaluate_object_match_model,
    fit_position_stats,
    load_match_pair_records,
    object_match_feature_set_metadata,
    resolve_object_match_feature_names,
    save_predictions_csv,
    select_object_match_feature_columns,
    train_object_match_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an object-to-object matching MLP.")
    parser.add_argument("--db_dir", type=str, default="spatial_db_origin", help="Spatial DB directory")
    parser.add_argument(
        "--gt_pairs",
        type=str,
        default="evaluation/object_instance_pairs.jsonl",
        help="JSONL file with manual object pair annotations",
    )
    parser.add_argument("--output_dir", type=str, default="runs/object_match_mlp", help="Output directory")
    parser.add_argument("--candidates_per_sample", type=int, default=16, help="Max candidates per O1/V2 sample")
    parser.add_argument("--train_split", type=str, default="train", help="Split name for training rows")
    parser.add_argument("--val_split", type=str, default="dev", help="Split name for validation rows")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablation_suite",
        type=str,
        choices=("none", "leave_one_out", "all_feature_sets"),
        default="none",
        help=(
            "Optional feature ablation suite. 'leave_one_out' trains full plus six No-* feature sets; "
            "'all_feature_sets' adds Only-* and focused combination feature sets."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to mps, cuda, then cpu.",
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


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_ablation_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature_set",
        "experiment_type",
        "included_groups",
        "removed_group",
        "aliases",
        "feature_names",
        "input_dim",
        "train_accuracy",
        "val_accuracy",
        "val_match_accuracy",
        "val_none_accuracy",
        "delta_val_accuracy_vs_full",
        "num_train_pairs",
        "num_val_pairs",
        "num_train_samples",
        "num_val_samples",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["included_groups"] = json.dumps(list(row["included_groups"]), ensure_ascii=True)
            csv_row["aliases"] = json.dumps(list(row["aliases"]), ensure_ascii=True)
            csv_row["feature_names"] = json.dumps(list(row["feature_names"]), ensure_ascii=True)
            writer.writerow(csv_row)


def _feature_sets_for_suite(ablation_suite: str) -> Sequence[str]:
    suite = str(ablation_suite)
    if suite == "leave_one_out":
        return OBJECT_MATCH_LEAVE_ONE_OUT_FEATURE_SET_ORDER
    if suite == "all_feature_sets":
        return OBJECT_MATCH_ALL_FEATURE_SET_ORDER
    raise ValueError(f"Unsupported ablation suite: {ablation_suite!r}")


def _make_data_loader(
    samples: Sequence[ObjectMatchTensorSample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(0 if seed is None else seed))
    return DataLoader(
        ObjectMatchTensorDataset(samples),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        collate_fn=collate_object_match_samples,
    )


def _run_training_experiment(
    *,
    args: argparse.Namespace,
    gt_path: Path,
    output_dir: Path,
    device: torch.device,
    position_stats: Any,
    train_pairs_count: int,
    val_pairs_count: int,
    train_samples: Sequence[ObjectMatchTensorSample],
    val_samples: Sequence[ObjectMatchTensorSample],
    feature_set: str,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    _set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = _make_data_loader(
        train_samples,
        batch_size=int(args.batch_size),
        shuffle=True,
        seed=int(args.seed),
    )
    val_loader = (
        _make_data_loader(
            val_samples,
            batch_size=int(args.batch_size),
            shuffle=False,
        )
        if val_samples
        else None
    )

    input_dim = len(tuple(feature_names))
    model = ObjectMatchMLP(
        input_dim=input_dim,
        hidden=int(args.hidden),
        dropout=float(args.dropout),
    )
    train_result = train_object_match_model(
        model,
        train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        val_loader=val_loader,
        weight_decay=float(args.weight_decay),
    )

    train_eval = evaluate_object_match_model(model, train_loader, device=device)
    val_eval = evaluate_object_match_model(model, val_loader, device=device) if val_loader is not None else None
    feature_set_meta = object_match_feature_set_metadata(feature_set)
    removed_group = OBJECT_MATCH_FEATURE_SET_REMOVED_GROUPS.get(feature_set)

    config = {
        "db_dir": str(args.db_dir),
        "gt_pairs": str(gt_path),
        "output_dir": str(output_dir),
        "ablation_suite": str(args.ablation_suite),
        "feature_set": str(feature_set),
        "experiment_type": feature_set_meta["experiment_type"],
        "included_groups": list(feature_set_meta["included_groups"]),
        "removed_group": removed_group,
        "aliases": list(feature_set_meta["aliases"]),
        "feature_names": list(feature_names),
        "input_dim": int(input_dim),
        "candidates_per_sample": int(args.candidates_per_sample),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden": int(args.hidden),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "device": str(device),
        "num_train_pairs": int(train_pairs_count),
        "num_val_pairs": int(val_pairs_count),
        "num_train_samples": int(len(train_samples)),
        "num_val_samples": int(len(val_samples)),
    }
    metrics = {
        "history": train_result["history"],
        "train": {key: value for key, value in train_eval.items() if key != "predictions"},
        "val": (
            {key: value for key, value in val_eval.items() if key != "predictions"}
            if val_eval is not None
            else None
        ),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "position_stats": position_stats.to_dict(),
        },
        output_dir / "model.pt",
    )
    _write_json(output_dir / "feature_stats.json", position_stats.to_dict())
    _write_json(output_dir / "train_config.json", config)
    _write_json(output_dir / "metrics.json", metrics)
    if val_eval is not None:
        save_predictions_csv(output_dir / "val_predictions.csv", val_eval["predictions"])

    val_metrics = metrics["val"] or {}
    return {
        "feature_set": str(feature_set),
        "experiment_type": feature_set_meta["experiment_type"],
        "included_groups": list(feature_set_meta["included_groups"]),
        "removed_group": removed_group,
        "aliases": list(feature_set_meta["aliases"]),
        "feature_names": list(feature_names),
        "input_dim": int(input_dim),
        "train_accuracy": metrics["train"]["accuracy"],
        "val_accuracy": val_metrics.get("accuracy"),
        "val_match_accuracy": val_metrics.get("match_accuracy"),
        "val_none_accuracy": val_metrics.get("none_accuracy"),
        "delta_val_accuracy_vs_full": None,
        "num_train_pairs": int(train_pairs_count),
        "num_val_pairs": int(val_pairs_count),
        "num_train_samples": int(len(train_samples)),
        "num_val_samples": int(len(val_samples)),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.device)

    gt_path = Path(args.gt_pairs)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Missing pair annotation file: {gt_path}. "
            "Create evaluation/object_instance_pairs.jsonl before training."
        )

    feature_store = ObjectFeatureStore.from_db(args.db_dir)
    train_pairs = load_match_pair_records(gt_path, db_dir=args.db_dir, split=args.train_split)
    val_pairs = load_match_pair_records(gt_path, db_dir=args.db_dir, split=args.val_split)
    if not train_pairs:
        raise ValueError(f"No training pair records matched split={args.train_split!r} and db_dir={args.db_dir!r}")

    train_specs = build_match_sample_specs(
        train_pairs,
        feature_store,
        candidates_per_sample=args.candidates_per_sample,
        seed=args.seed,
    )
    val_specs = build_match_sample_specs(
        val_pairs,
        feature_store,
        candidates_per_sample=args.candidates_per_sample,
        seed=args.seed + 1,
    )
    if not train_specs:
        raise ValueError("No train samples could be built. Check pair labels, entry_id coverage, and split names.")

    position_stats = fit_position_stats(feature_store, train_specs)
    train_samples = build_tensor_samples(feature_store, train_specs, position_stats)
    val_samples = build_tensor_samples(feature_store, val_specs, position_stats) if val_specs else []
    if not train_samples:
        raise ValueError("No train tensor samples could be built. Check position and embedding coverage.")

    if args.ablation_suite != "none":
        summary_rows: List[Dict[str, Any]] = []
        for feature_set in _feature_sets_for_suite(args.ablation_suite):
            feature_names = resolve_object_match_feature_names(feature_set)
            selected_train_samples = select_object_match_feature_columns(train_samples, feature_names)
            selected_val_samples = select_object_match_feature_columns(val_samples, feature_names) if val_samples else []
            summary_rows.append(
                _run_training_experiment(
                    args=args,
                    gt_path=gt_path,
                    output_dir=output_dir / feature_set,
                    device=device,
                    position_stats=position_stats,
                    train_pairs_count=len(train_pairs),
                    val_pairs_count=len(val_pairs),
                    train_samples=selected_train_samples,
                    val_samples=selected_val_samples,
                    feature_set=feature_set,
                    feature_names=feature_names,
                )
            )

        full_val_accuracy = next(
            (row["val_accuracy"] for row in summary_rows if row["feature_set"] == "full"),
            None,
        )
        if full_val_accuracy is not None:
            for row in summary_rows:
                if row["val_accuracy"] is not None:
                    row["delta_val_accuracy_vs_full"] = float(row["val_accuracy"]) - float(full_val_accuracy)

        summary = {
            "ablation_suite": str(args.ablation_suite),
            "baseline_feature_set": "full",
            "rows": summary_rows,
        }
        _write_json(output_dir / "ablation_summary.json", summary)
        _write_ablation_summary_csv(output_dir / "ablation_summary.csv", summary_rows)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    feature_set = "full"
    feature_names = tuple(OBJECT_MATCH_FEATURE_NAMES)
    summary_row = _run_training_experiment(
        args=args,
        gt_path=gt_path,
        output_dir=output_dir,
        device=device,
        position_stats=position_stats,
        train_pairs_count=len(train_pairs),
        val_pairs_count=len(val_pairs),
        train_samples=train_samples,
        val_samples=val_samples,
        feature_set=feature_set,
        feature_names=feature_names,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_accuracy": summary_row["train_accuracy"],
                "val_accuracy": summary_row["val_accuracy"],
                "num_train_samples": len(train_samples),
                "num_val_samples": len(val_samples),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
