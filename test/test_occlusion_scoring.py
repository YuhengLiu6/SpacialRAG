import math

from spatial_rag.occlusion_scoring import (
    OCCLUSION_LEVEL_TO_PENALTY,
    clamp_probability,
    compute_reweighted_detection_score,
    logit,
    map_occlusion_level_to_penalty,
    normalize_occlusion_level,
    sigmoid,
)


def test_occlusion_penalty_mapping_matches_spec():
    assert OCCLUSION_LEVEL_TO_PENALTY == {
        "fully visible": 0.0,
        "slightly occluded": 0.1,
        "moderately occluded": 0.25,
        "heavily occluded": 0.5,
        "uncertain": 0.35,
    }
    assert map_occlusion_level_to_penalty("fully visible") == 0.0
    assert map_occlusion_level_to_penalty("slightly occluded") == 0.1
    assert map_occlusion_level_to_penalty("moderately occluded") == 0.25
    assert map_occlusion_level_to_penalty("heavily occluded") == 0.5
    assert map_occlusion_level_to_penalty("uncertain") == 0.35


def test_occlusion_level_normalization_defaults_to_uncertain():
    assert normalize_occlusion_level("Fully Visible") == "fully visible"
    assert normalize_occlusion_level("not-a-valid-level") == "uncertain"
    assert normalize_occlusion_level(None) == "uncertain"


def test_probability_helpers_match_expected_math():
    assert math.isclose(clamp_probability(0.0, eps=1e-6), 1e-6)
    assert math.isclose(clamp_probability(1.0, eps=1e-6), 1.0 - 1e-6)
    assert math.isclose(sigmoid(0.0), 0.5)
    assert math.isclose(logit(0.5), 0.0)


def test_reweighted_detection_score_matches_formula():
    c_det = 0.8
    penalty = 0.25
    expected = 1.0 / (1.0 + math.exp(-(math.log(c_det / (1.0 - c_det)) - penalty)))
    observed = compute_reweighted_detection_score(c_det, "moderately occluded")
    assert math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_reweighted_detection_score_is_monotonic_in_occlusion_and_confidence():
    visible = compute_reweighted_detection_score(0.8, "fully visible")
    uncertain = compute_reweighted_detection_score(0.8, "uncertain")
    heavy = compute_reweighted_detection_score(0.8, "heavily occluded")
    low_conf = compute_reweighted_detection_score(0.4, "slightly occluded")
    high_conf = compute_reweighted_detection_score(0.9, "slightly occluded")

    assert visible > uncertain > heavy
    assert high_conf > low_conf
