from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict

from spatial_rag.config import (
    OCCLUSION_REWEIGHT_B,
    OCCLUSION_REWEIGHT_EPS,
    OCCLUSION_REWEIGHT_W1,
    OCCLUSION_REWEIGHT_W2,
)

OCCLUSION_SCORE_FORMULA_VERSION = "occlusion_reweight_v1"
OCCLUSION_LEVEL_TO_PENALTY: Dict[str, float] = {
    "fully visible": 0.0,
    "slightly occluded": 0.1,
    "moderately occluded": 0.25,
    "heavily occluded": 0.5,
    "uncertain": 0.35,
}


def normalize_occlusion_level(level: Any, default: str = "uncertain") -> str:
    token = str(level or "").strip().lower()
    if token in OCCLUSION_LEVEL_TO_PENALTY:
        return token
    fallback = str(default or "").strip().lower()
    if fallback in OCCLUSION_LEVEL_TO_PENALTY:
        return fallback
    raise ValueError(f"Unsupported occlusion level: {level!r}")


def map_occlusion_level_to_penalty(level: str) -> float:
    token = str(level or "").strip().lower()
    if token not in OCCLUSION_LEVEL_TO_PENALTY:
        raise ValueError(f"Unsupported occlusion level: {level!r}")
    return float(OCCLUSION_LEVEL_TO_PENALTY[token])


def clamp_probability(probability: float, eps: float = OCCLUSION_REWEIGHT_EPS) -> float:
    p = float(probability)
    margin = max(float(eps), 1e-12)
    return min(max(p, margin), 1.0 - margin)


def logit(probability: float, eps: float = OCCLUSION_REWEIGHT_EPS) -> float:
    p = clamp_probability(probability, eps=eps)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    value = float(z)
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def compute_reweighted_detection_score(
    c_det: float,
    occlusion_level: str,
    w1: float = OCCLUSION_REWEIGHT_W1,
    w2: float = OCCLUSION_REWEIGHT_W2,
    b: float = OCCLUSION_REWEIGHT_B,
    eps: float = OCCLUSION_REWEIGHT_EPS,
) -> float:
    normalized_level = normalize_occlusion_level(occlusion_level, default="uncertain")
    penalty = map_occlusion_level_to_penalty(normalized_level)
    z = float(w1) * logit(float(c_det), eps=eps) - float(w2) * penalty + float(b)
    return float(sigmoid(z))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect occlusion penalty and reweighted detection score.")
    parser.add_argument("--c_det", type=float, required=True, help="YOLO detector confidence in [0, 1].")
    parser.add_argument(
        "--occlusion_level",
        type=str,
        required=True,
        choices=tuple(OCCLUSION_LEVEL_TO_PENALTY.keys()),
        help="Occlusion enum value.",
    )
    parser.add_argument("--w1", type=float, default=OCCLUSION_REWEIGHT_W1, help="Weight for logit(c_det).")
    parser.add_argument("--w2", type=float, default=OCCLUSION_REWEIGHT_W2, help="Weight for p(o).")
    parser.add_argument("--b", type=float, default=OCCLUSION_REWEIGHT_B, help="Bias term.")
    parser.add_argument("--eps", type=float, default=OCCLUSION_REWEIGHT_EPS, help="Probability clamp epsilon.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    normalized_level = normalize_occlusion_level(args.occlusion_level)
    penalty = map_occlusion_level_to_penalty(normalized_level)
    clipped_confidence = clamp_probability(args.c_det, eps=args.eps)
    payload = {
        "formula_version": OCCLUSION_SCORE_FORMULA_VERSION,
        "c_det": float(args.c_det),
        "clipped_c_det": float(clipped_confidence),
        "occlusion_level": normalized_level,
        "occlusion_penalty_p_o": float(penalty),
        "logit_c_det": float(logit(args.c_det, eps=args.eps)),
        "reweighted_detection_score_r": float(
            compute_reweighted_detection_score(
                args.c_det,
                normalized_level,
                w1=args.w1,
                w2=args.w2,
                b=args.b,
                eps=args.eps,
            )
        ),
        "weights": {
            "w1": float(args.w1),
            "w2": float(args.w2),
            "b": float(args.b),
            "eps": float(args.eps),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
