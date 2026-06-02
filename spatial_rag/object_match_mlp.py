from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from spatial_rag.object_instance_eval import (
    PairGTRecord,
    build_graph_context_strings,
    embed_graph_contexts,
    load_object_pair_ground_truth,
)


OBJECT_MATCH_FEATURE_NAMES = (
    "p1_x_norm",
    "p1_y_norm",
    "p1_z_norm",
    "pi_x_norm",
    "pi_y_norm",
    "pi_z_norm",
    "desc_cos",
    "neighborhood_cos",
    "dinov3_cos",
    "dinov3_valid",
)
OBJECT_MATCH_FEATURE_DIM = len(OBJECT_MATCH_FEATURE_NAMES)
OBJECT_MATCH_FEATURE_GROUPS = {
    "p_o": ("p1_x_norm", "p1_y_norm", "p1_z_norm"),
    "p_i": ("pi_x_norm", "pi_y_norm", "pi_z_norm"),
    "cos_e": ("desc_cos",),
    "cos_n": ("neighborhood_cos",),
    "cos_d": ("dinov3_cos",),
    "v_i": ("dinov3_valid",),
}
OBJECT_MATCH_LEAVE_ONE_OUT_FEATURE_SET_ORDER = (
    "full",
    "no_p_o",
    "no_p_i",
    "no_cos_e",
    "no_cos_n",
    "no_cos_d",
    "no_v_i",
)
OBJECT_MATCH_ALL_FEATURE_SET_ORDER = (
    *OBJECT_MATCH_LEAVE_ONE_OUT_FEATURE_SET_ORDER,
    "only_p_o",
    "only_p_i",
    "only_position",
    "only_cos_e",
    "only_cos_n",
    "only_cos_d",
    "only_cosine",
    "only_v_i",
    "position_cos_e",
    "position_cos_n",
    "position_cos_d",
    "position_v_i",
    "cosine_v_i",
    "cos_e_cos_n",
    "cos_e_cos_d",
    "cos_n_cos_d",
)
OBJECT_MATCH_FEATURE_SET_ORDER = OBJECT_MATCH_ALL_FEATURE_SET_ORDER
_ALL_FEATURE_GROUPS = ("p_o", "p_i", "cos_e", "cos_n", "cos_d", "v_i")
OBJECT_MATCH_FEATURE_SET_DEFINITIONS = {
    "full": {
        "groups": _ALL_FEATURE_GROUPS,
        "experiment_type": "baseline",
        "removed_group": None,
        "aliases": (),
    },
    "no_p_o": {
        "groups": ("p_i", "cos_e", "cos_n", "cos_d", "v_i"),
        "experiment_type": "leave_one_out",
        "removed_group": "p_o",
        "aliases": (),
    },
    "no_p_i": {
        "groups": ("p_o", "cos_e", "cos_n", "cos_d", "v_i"),
        "experiment_type": "leave_one_out",
        "removed_group": "p_i",
        "aliases": (),
    },
    "no_cos_e": {
        "groups": ("p_o", "p_i", "cos_n", "cos_d", "v_i"),
        "experiment_type": "leave_one_out",
        "removed_group": "cos_e",
        "aliases": (),
    },
    "no_cos_n": {
        "groups": ("p_o", "p_i", "cos_e", "cos_d", "v_i"),
        "experiment_type": "leave_one_out",
        "removed_group": "cos_n",
        "aliases": (),
    },
    "no_cos_d": {
        "groups": ("p_o", "p_i", "cos_e", "cos_n", "v_i"),
        "experiment_type": "leave_one_out",
        "removed_group": "cos_d",
        "aliases": (),
    },
    "no_v_i": {
        "groups": ("p_o", "p_i", "cos_e", "cos_n", "cos_d"),
        "experiment_type": "leave_one_out",
        "removed_group": "v_i",
        "aliases": ("position_cosine", "p_o_p_i_cos_e_cos_n_cos_d"),
    },
    "only_p_o": {
        "groups": ("p_o",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "only_p_i": {
        "groups": ("p_i",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "only_position": {
        "groups": ("p_o", "p_i"),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": ("p_o_p_i",),
    },
    "only_cos_e": {
        "groups": ("cos_e",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "only_cos_n": {
        "groups": ("cos_n",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "only_cos_d": {
        "groups": ("cos_d",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "only_cosine": {
        "groups": ("cos_e", "cos_n", "cos_d"),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": ("cos_e_cos_n_cos_d",),
    },
    "only_v_i": {
        "groups": ("v_i",),
        "experiment_type": "only_group",
        "removed_group": None,
        "aliases": (),
    },
    "position_cos_e": {
        "groups": ("p_o", "p_i", "cos_e"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": ("p_o_p_i_cos_e",),
    },
    "position_cos_n": {
        "groups": ("p_o", "p_i", "cos_n"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": ("p_o_p_i_cos_n",),
    },
    "position_cos_d": {
        "groups": ("p_o", "p_i", "cos_d"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": ("p_o_p_i_cos_d",),
    },
    "position_v_i": {
        "groups": ("p_o", "p_i", "v_i"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": ("p_o_p_i_v_i",),
    },
    "cosine_v_i": {
        "groups": ("cos_e", "cos_n", "cos_d", "v_i"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": ("cos_e_cos_n_cos_d_v_i",),
    },
    "cos_e_cos_n": {
        "groups": ("cos_e", "cos_n"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": (),
    },
    "cos_e_cos_d": {
        "groups": ("cos_e", "cos_d"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": (),
    },
    "cos_n_cos_d": {
        "groups": ("cos_n", "cos_d"),
        "experiment_type": "focused_combo",
        "removed_group": None,
        "aliases": (),
    },
}
OBJECT_MATCH_FEATURE_SET_REMOVED_GROUPS = {
    name: definition["removed_group"]
    for name, definition in OBJECT_MATCH_FEATURE_SET_DEFINITIONS.items()
}


@dataclass(frozen=True)
class PositionStats:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositionStats":
        return cls(
            mean=tuple(float(v) for v in payload["mean"][:3]),
            std=tuple(float(v) for v in payload["std"][:3]),
        )


@dataclass(frozen=True)
class ObjectMatchSampleSpec:
    anchor_obj_id: int
    candidate_entry_id: int
    candidate_obj_ids: Tuple[int, ...]
    target_index: int
    target_obj_id: Optional[int]
    split: str

    @property
    def target_is_none(self) -> bool:
        return self.target_obj_id is None


@dataclass(frozen=True)
class ObjectMatchTensorSample:
    spec: ObjectMatchSampleSpec
    features: np.ndarray


@dataclass(frozen=True)
class ObjectMatchPrediction:
    anchor_obj_id: int
    candidate_entry_id: int
    candidate_obj_ids: Tuple[int, ...]
    target_index: int
    target_obj_id: Optional[int]
    pred_index: int
    pred_obj_id: Optional[int]
    pred_is_none: bool
    hit: bool
    pred_prob: float
    none_prob: float
    candidate_probs: Tuple[float, ...]


def resolve_object_match_feature_names(feature_set: str) -> Tuple[str, ...]:
    key = str(feature_set).strip()
    if key not in OBJECT_MATCH_FEATURE_SET_DEFINITIONS:
        valid = ", ".join(OBJECT_MATCH_FEATURE_SET_ORDER)
        raise ValueError(f"Unknown object match feature set {feature_set!r}. Valid feature sets: {valid}")
    names: List[str] = []
    for group in resolve_object_match_feature_groups(key):
        names.extend(OBJECT_MATCH_FEATURE_GROUPS[group])
    return tuple(names)


def resolve_object_match_feature_groups(feature_set: str) -> Tuple[str, ...]:
    key = str(feature_set).strip()
    if key not in OBJECT_MATCH_FEATURE_SET_DEFINITIONS:
        valid = ", ".join(OBJECT_MATCH_FEATURE_SET_ORDER)
        raise ValueError(f"Unknown object match feature set {feature_set!r}. Valid feature sets: {valid}")
    return tuple(str(group) for group in OBJECT_MATCH_FEATURE_SET_DEFINITIONS[key]["groups"])


def object_match_feature_set_metadata(feature_set: str) -> Dict[str, Any]:
    key = str(feature_set).strip()
    if key not in OBJECT_MATCH_FEATURE_SET_DEFINITIONS:
        valid = ", ".join(OBJECT_MATCH_FEATURE_SET_ORDER)
        raise ValueError(f"Unknown object match feature set {feature_set!r}. Valid feature sets: {valid}")
    definition = OBJECT_MATCH_FEATURE_SET_DEFINITIONS[key]
    return {
        "feature_set": key,
        "experiment_type": str(definition["experiment_type"]),
        "included_groups": list(resolve_object_match_feature_groups(key)),
        "removed_group": definition["removed_group"],
        "aliases": list(definition["aliases"]),
    }


def object_match_feature_indices(feature_names: Sequence[str]) -> Tuple[int, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("At least one object match feature must be selected")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate object match feature names are not allowed: {names}")

    index_by_name = {name: idx for idx, name in enumerate(OBJECT_MATCH_FEATURE_NAMES)}
    missing = [name for name in names if name not in index_by_name]
    if missing:
        valid = ", ".join(OBJECT_MATCH_FEATURE_NAMES)
        raise ValueError(f"Unknown object match feature name(s) {missing}. Valid features: {valid}")
    return tuple(index_by_name[name] for name in names)


def select_object_match_feature_columns(
    samples: Sequence[ObjectMatchTensorSample],
    feature_names: Sequence[str],
) -> List[ObjectMatchTensorSample]:
    indices = object_match_feature_indices(feature_names)
    selected: List[ObjectMatchTensorSample] = []
    for sample in samples:
        arr = np.asarray(sample.features, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected sample features to be 2D, got shape {arr.shape}")
        if int(arr.shape[1]) != OBJECT_MATCH_FEATURE_DIM:
            raise ValueError(
                f"Can only select columns from full {OBJECT_MATCH_FEATURE_DIM}-D features, "
                f"got shape {arr.shape}"
            )
        selected.append(
            ObjectMatchTensorSample(
                spec=sample.spec,
                features=arr[:, indices].astype(np.float32, copy=False),
            )
        )
    return selected


class ObjectFeatureStore:
    def __init__(
        self,
        *,
        object_rows: Sequence[Mapping[str, Any]],
        desc_embeddings: Mapping[int, np.ndarray],
        graph_embeddings: Mapping[int, np.ndarray],
        dinov3_embeddings: Mapping[int, np.ndarray],
    ) -> None:
        self.object_rows = [dict(row) for row in object_rows]
        self.object_by_id = {
            int(row.get("object_global_id", idx)): dict(row)
            for idx, row in enumerate(self.object_rows)
        }
        self.entry_to_object_ids: Dict[int, List[int]] = {}
        for obj_id, row in self.object_by_id.items():
            entry_id = _safe_int(row.get("entry_id"))
            if entry_id is None:
                continue
            self.entry_to_object_ids.setdefault(int(entry_id), []).append(int(obj_id))
        for ids in self.entry_to_object_ids.values():
            ids.sort()

        self.positions = {
            int(obj_id): pos
            for obj_id, row in self.object_by_id.items()
            if (pos := _object_position(row)) is not None
        }
        self.desc_embeddings = _normalize_embedding_map(desc_embeddings)
        self.graph_embeddings = _normalize_embedding_map(graph_embeddings)
        self.dinov3_embeddings = _normalize_embedding_map(dinov3_embeddings)

    @classmethod
    def from_db(
        cls,
        db_dir: str | Path,
        *,
        graph_embeddings_by_obj_id: Optional[Mapping[int, np.ndarray] | np.ndarray] = None,
        build_graph_embeddings: bool = True,
    ) -> "ObjectFeatureStore":
        root = Path(db_dir)
        object_rows = _load_jsonl(root / "object_meta.jsonl")
        desc_arr = np.load(root / "object_text_emb_long.npy", allow_pickle=False).astype(np.float32)
        if desc_arr.ndim != 2 or desc_arr.shape[0] != len(object_rows):
            raise ValueError(
                "object_text_emb_long.npy must be 2D and row-aligned with object_meta.jsonl"
            )
        desc_embeddings = {
            int(row.get("object_global_id", idx)): np.asarray(desc_arr[idx], dtype=np.float32)
            for idx, row in enumerate(object_rows)
        }

        if graph_embeddings_by_obj_id is None:
            if not build_graph_embeddings:
                graph_embeddings = {}
            else:
                contexts = build_graph_context_strings(str(root))
                graph_dense = embed_graph_contexts(contexts)
                graph_embeddings = _dense_by_object_id(graph_dense, object_rows)
        else:
            graph_embeddings = _coerce_embedding_by_obj_id(graph_embeddings_by_obj_id, object_rows)

        dinov3_embeddings: Dict[int, np.ndarray] = {}
        dino_path = root / "object_dinov3_emb.npy"
        if dino_path.exists():
            dino_arr = np.load(dino_path, allow_pickle=False).astype(np.float32)
            if dino_arr.ndim != 2:
                raise ValueError(f"object_dinov3_emb.npy must be 2D, got {dino_arr.shape}")
            for idx, row in enumerate(object_rows):
                obj_id = int(row.get("object_global_id", idx))
                sidecar_idx = _safe_int(row.get("dinov3_embedding_row_index"))
                if sidecar_idx is None:
                    continue
                if 0 <= int(sidecar_idx) < int(dino_arr.shape[0]):
                    dinov3_embeddings[obj_id] = np.asarray(dino_arr[int(sidecar_idx)], dtype=np.float32)

        return cls(
            object_rows=object_rows,
            desc_embeddings=desc_embeddings,
            graph_embeddings=graph_embeddings,
            dinov3_embeddings=dinov3_embeddings,
        )

    def entry_id_for_object(self, obj_id: int) -> Optional[int]:
        row = self.object_by_id.get(int(obj_id))
        if row is None:
            return None
        return _safe_int(row.get("entry_id"))

    def build_pair_feature(
        self,
        anchor_obj_id: int,
        candidate_obj_id: int,
        position_stats: PositionStats,
    ) -> np.ndarray:
        anchor = int(anchor_obj_id)
        candidate = int(candidate_obj_id)
        p1 = self.positions.get(anchor)
        pi = self.positions.get(candidate)
        if p1 is None:
            raise KeyError(f"Object {anchor} has no usable position")
        if pi is None:
            raise KeyError(f"Object {candidate} has no usable position")

        p1_norm = _normalize_position(p1, position_stats)
        pi_norm = _normalize_position(pi, position_stats)
        desc_cos = _cosine_from_map(self.desc_embeddings, anchor, candidate, missing_value=0.0)
        graph_cos = _cosine_from_map(self.graph_embeddings, anchor, candidate, missing_value=0.0)
        dino_valid = float(anchor in self.dinov3_embeddings and candidate in self.dinov3_embeddings)
        dino_cos = (
            _cosine_from_map(self.dinov3_embeddings, anchor, candidate, missing_value=0.0)
            if dino_valid
            else 0.0
        )
        return np.asarray(
            [
                *p1_norm.tolist(),
                *pi_norm.tolist(),
                float(desc_cos),
                float(graph_cos),
                float(dino_cos),
                float(dino_valid),
            ],
            dtype=np.float32,
        )


class ObjectMatchTensorDataset(Dataset):
    def __init__(self, samples: Sequence[ObjectMatchTensorSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ObjectMatchTensorSample:
        return self.samples[int(index)]


class ObjectMatchMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = OBJECT_MATCH_FEATURE_DIM,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )
        self.none_logit = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, M, D], got {tuple(features.shape)}")
        batch, num_candidates, dim = features.shape
        if int(dim) != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {dim}")

        scores = self.scorer(features.reshape(batch * num_candidates, dim)).reshape(batch, num_candidates)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.bool(), -1e9)
        none = self.none_logit.reshape(1, 1).expand(batch, 1)
        return torch.cat([scores, none], dim=-1)


def load_match_pair_records(
    gt_pairs_path: str | Path,
    *,
    db_dir: str | Path,
    split: Optional[str],
) -> List[PairGTRecord]:
    records = load_object_pair_ground_truth(str(gt_pairs_path), split=split)
    db_key = _normalize_db_key(str(db_dir))
    return [record for record in records if _normalize_db_key(record.db_dir) == db_key]


def build_match_sample_specs(
    pair_records: Sequence[PairGTRecord],
    feature_store: ObjectFeatureStore,
    *,
    candidates_per_sample: int = 16,
    seed: int = 0,
) -> List[ObjectMatchSampleSpec]:
    rng = random.Random(int(seed))
    grouped: Dict[Tuple[int, int, str], Dict[int, bool]] = {}

    for record in pair_records:
        for anchor, candidate in (
            (int(record.obj_a_id), int(record.obj_b_id)),
            (int(record.obj_b_id), int(record.obj_a_id)),
        ):
            anchor_entry = feature_store.entry_id_for_object(anchor)
            candidate_entry = feature_store.entry_id_for_object(candidate)
            if anchor_entry is None or candidate_entry is None:
                continue
            if int(anchor_entry) == int(candidate_entry):
                continue
            key = (anchor, int(candidate_entry), str(record.split))
            existing = grouped.setdefault(key, {})
            previous = existing.get(candidate)
            label = bool(record.is_same_instance)
            if previous is not None and bool(previous) != label:
                raise ValueError(f"Conflicting labels for anchor={anchor}, candidate={candidate}")
            existing[candidate] = label

    specs: List[ObjectMatchSampleSpec] = []
    for (anchor, candidate_entry_id, split), labels_by_candidate in sorted(grouped.items()):
        positives = sorted(obj_id for obj_id, is_pos in labels_by_candidate.items() if bool(is_pos))
        negatives = sorted(obj_id for obj_id, is_pos in labels_by_candidate.items() if not bool(is_pos))
        if positives:
            for target_obj_id in positives:
                candidates = [target_obj_id]
                candidates.extend(_sample_ids(negatives, max(0, int(candidates_per_sample) - 1), rng))
                candidates, target_index = _shuffle_candidates(candidates, target_obj_id, rng)
                specs.append(
                    ObjectMatchSampleSpec(
                        anchor_obj_id=int(anchor),
                        candidate_entry_id=int(candidate_entry_id),
                        candidate_obj_ids=tuple(int(v) for v in candidates),
                        target_index=int(target_index),
                        target_obj_id=int(target_obj_id),
                        split=split,
                    )
                )
        elif negatives:
            candidates = _sample_ids(negatives, int(candidates_per_sample), rng)
            specs.append(
                ObjectMatchSampleSpec(
                    anchor_obj_id=int(anchor),
                    candidate_entry_id=int(candidate_entry_id),
                    candidate_obj_ids=tuple(int(v) for v in candidates),
                    target_index=len(candidates),
                    target_obj_id=None,
                    split=split,
                )
            )
    return specs


def fit_position_stats(
    feature_store: ObjectFeatureStore,
    sample_specs: Sequence[ObjectMatchSampleSpec],
) -> PositionStats:
    obj_ids: set[int] = set()
    for spec in sample_specs:
        obj_ids.add(int(spec.anchor_obj_id))
        obj_ids.update(int(obj_id) for obj_id in spec.candidate_obj_ids)
    positions = [
        feature_store.positions[int(obj_id)]
        for obj_id in sorted(obj_ids)
        if int(obj_id) in feature_store.positions
    ]
    if not positions:
        raise ValueError("Cannot fit position stats without at least one object position")
    arr = np.vstack(positions).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.maximum(std, 1e-6)
    return PositionStats(
        mean=tuple(float(v) for v in mean.tolist()),
        std=tuple(float(v) for v in std.tolist()),
    )


def build_tensor_samples(
    feature_store: ObjectFeatureStore,
    sample_specs: Sequence[ObjectMatchSampleSpec],
    position_stats: PositionStats,
) -> List[ObjectMatchTensorSample]:
    tensor_samples: List[ObjectMatchTensorSample] = []
    for spec in sample_specs:
        features = [
            feature_store.build_pair_feature(
                anchor_obj_id=spec.anchor_obj_id,
                candidate_obj_id=candidate_obj_id,
                position_stats=position_stats,
            )
            for candidate_obj_id in spec.candidate_obj_ids
        ]
        if not features:
            continue
        tensor_samples.append(
            ObjectMatchTensorSample(
                spec=spec,
                features=np.vstack(features).astype(np.float32),
            )
        )
    return tensor_samples


def collate_object_match_samples(samples: Sequence[ObjectMatchTensorSample]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_candidates = max(int(sample.features.shape[0]) for sample in samples)
    feature_dims = {int(sample.features.shape[1]) for sample in samples}
    if len(feature_dims) != 1:
        raise ValueError(f"All samples in a batch must have the same feature dim, got {sorted(feature_dims)}")
    feature_dim = int(next(iter(feature_dims)))
    batch_size = len(samples)
    features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    targets = torch.zeros((batch_size,), dtype=torch.long)
    specs: List[ObjectMatchSampleSpec] = []

    for row_idx, sample in enumerate(samples):
        count = int(sample.features.shape[0])
        features[row_idx, :count, :] = torch.from_numpy(sample.features.astype(np.float32))
        valid_mask[row_idx, :count] = True
        if sample.spec.target_is_none:
            targets[row_idx] = max_candidates
        else:
            targets[row_idx] = int(sample.spec.target_index)
        specs.append(sample.spec)

    return {
        "features": features,
        "valid_mask": valid_mask,
        "targets": targets,
        "specs": specs,
    }


def object_match_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets.long())


@torch.no_grad()
def evaluate_object_match_model(
    model: ObjectMatchMLP,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    none_total = 0
    none_correct = 0
    match_total = 0
    match_correct = 0
    predictions: List[ObjectMatchPrediction] = []

    for batch in data_loader:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        targets = batch["targets"].to(device)
        specs = batch["specs"]
        logits = model(features, valid_mask=valid_mask)
        loss = object_match_loss(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1)

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        correct += int((pred_indices == targets).sum().item())

        none_index = int(logits.shape[-1] - 1)
        for idx, spec in enumerate(specs):
            pred_index = int(pred_indices[idx].item())
            target_index = int(targets[idx].item())
            pred_is_none = pred_index == none_index
            hit = pred_index == target_index
            if spec.target_is_none:
                none_total += 1
                none_correct += int(hit)
            else:
                match_total += 1
                match_correct += int(hit)
            pred_obj_id = None if pred_is_none else int(spec.candidate_obj_ids[pred_index])
            predictions.append(
                ObjectMatchPrediction(
                    anchor_obj_id=int(spec.anchor_obj_id),
                    candidate_entry_id=int(spec.candidate_entry_id),
                    candidate_obj_ids=tuple(int(v) for v in spec.candidate_obj_ids),
                    target_index=int(spec.target_index),
                    target_obj_id=spec.target_obj_id,
                    pred_index=int(pred_index),
                    pred_obj_id=pred_obj_id,
                    pred_is_none=bool(pred_is_none),
                    hit=bool(hit),
                    pred_prob=float(probs[idx, pred_index].item()),
                    none_prob=float(probs[idx, none_index].item()),
                    candidate_probs=tuple(float(v) for v in probs[idx, : len(spec.candidate_obj_ids)].cpu().tolist()),
                )
            )

    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": float(correct / max(total, 1)),
        "none_accuracy": float(none_correct / none_total) if none_total else None,
        "match_accuracy": float(match_correct / match_total) if match_total else None,
        "num_samples": int(total),
        "num_none_samples": int(none_total),
        "num_match_samples": int(match_total),
        "predictions": predictions,
    }


def train_object_match_model(
    model: ObjectMatchMLP,
    train_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    epochs: int,
    lr: float,
    val_loader: Optional[Iterable[Mapping[str, Any]]] = None,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad()
            logits = model(features, valid_mask=valid_mask)
            loss = object_match_loss(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            correct += int((torch.argmax(logits.detach(), dim=-1) == targets).sum().item())

        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(total, 1)),
            "train_accuracy": float(correct / max(total, 1)),
        }
        if val_loader is not None:
            val_metrics = evaluate_object_match_model(model, val_loader, device=device)
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_none_accuracy": val_metrics["none_accuracy"],
                    "val_match_accuracy": val_metrics["match_accuracy"],
                }
            )
        history.append(row)
    return {"history": history}


def save_predictions_csv(path: str | Path, predictions: Sequence[ObjectMatchPrediction]) -> None:
    import csv

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_obj_id",
        "candidate_entry_id",
        "candidate_obj_ids",
        "target_index",
        "target_obj_id",
        "pred_index",
        "pred_obj_id",
        "pred_is_none",
        "hit",
        "pred_prob",
        "none_prob",
        "candidate_probs",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pred in predictions:
            row = asdict(pred)
            row["candidate_obj_ids"] = json.dumps(list(pred.candidate_obj_ids), ensure_ascii=True)
            row["candidate_probs"] = json.dumps(list(pred.candidate_probs), ensure_ascii=True)
            writer.writerow(row)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize_db_key(db_dir: str) -> str:
    text = str(db_dir or "").strip()
    if not text:
        return ""
    return Path(text).name or Path(text).as_posix()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _object_position(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    keys = ("estimated_global_x", "estimated_global_y", "estimated_global_z")
    values = [_safe_float(row.get(key)) for key in keys]
    if all(value is not None for value in values):
        return np.asarray(values, dtype=np.float32)

    keys = ("projected_x", "projected_y", "projected_z")
    values = [_safe_float(row.get(key)) for key in keys]
    if all(value is not None for value in values):
        return np.asarray(values, dtype=np.float32)

    world = row.get("world_position")
    if isinstance(world, (list, tuple)) and len(world) >= 3:
        values = [_safe_float(world[0]), _safe_float(world[1]), _safe_float(world[2])]
        if all(value is not None for value in values):
            return np.asarray(values, dtype=np.float32)
    return None


def _normalize_position(position: np.ndarray, stats: PositionStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return (np.asarray(position, dtype=np.float32) - mean) / np.maximum(std, 1e-6)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _normalize_embedding_map(embeddings: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {int(key): _l2_normalize(np.asarray(value, dtype=np.float32)) for key, value in embeddings.items()}


def _cosine_from_map(
    embeddings: Mapping[int, np.ndarray],
    obj_a_id: int,
    obj_b_id: int,
    *,
    missing_value: float,
) -> float:
    a = embeddings.get(int(obj_a_id))
    b = embeddings.get(int(obj_b_id))
    if a is None or b is None:
        return float(missing_value)
    return float(np.dot(a, b))


def _dense_by_object_id(arr: np.ndarray, object_rows: Sequence[Mapping[str, Any]]) -> Dict[int, np.ndarray]:
    dense = np.asarray(arr, dtype=np.float32)
    if dense.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got {dense.shape}")
    out: Dict[int, np.ndarray] = {}
    for idx, row in enumerate(object_rows):
        obj_id = int(row.get("object_global_id", idx))
        if 0 <= obj_id < int(dense.shape[0]):
            out[obj_id] = np.asarray(dense[obj_id], dtype=np.float32)
    return out


def _coerce_embedding_by_obj_id(
    value: Mapping[int, np.ndarray] | np.ndarray,
    object_rows: Sequence[Mapping[str, Any]],
) -> Dict[int, np.ndarray]:
    if isinstance(value, np.ndarray):
        return _dense_by_object_id(value, object_rows)
    return {int(key): np.asarray(vec, dtype=np.float32) for key, vec in value.items()}


def _sample_ids(ids: Sequence[int], limit: int, rng: random.Random) -> List[int]:
    values = list(int(v) for v in ids)
    if int(limit) <= 0 or len(values) <= int(limit):
        return values
    shuffled = values[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[: int(limit)])


def _shuffle_candidates(
    candidates: Sequence[int],
    target_obj_id: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    values = list(int(v) for v in candidates)
    rng.shuffle(values)
    return values, values.index(int(target_obj_id))
