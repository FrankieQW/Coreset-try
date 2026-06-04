import numpy as np
from pathlib import Path

from aloha_baseline.coreset import (
    compute_predictive_diversity_scores,
    minmax_normalize,
    select_top_with_temporal_suppression,
)


def test_minmax_normalize_returns_zeros_for_constant_values():
    values = np.array([3.0, 3.0, 3.0], dtype=np.float32)

    normalized = minmax_normalize(values)

    np.testing.assert_array_equal(normalized, np.zeros_like(values))


def test_predictive_diversity_scores_favor_changed_samples():
    vision = np.zeros((5, 3), dtype=np.float32)
    state = np.zeros((5, 2), dtype=np.float32)
    action = np.zeros((5, 2), dtype=np.float32)
    vision[3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    state[3] = np.array([2.0, 0.0], dtype=np.float32)
    action[3] = np.array([0.0, 2.0], dtype=np.float32)

    scores, parts = compute_predictive_diversity_scores(vision, state, action)

    assert scores[3] == scores.max()
    assert parts["vision_delta"][3] > parts["vision_delta"][1]
    assert parts["state_delta"][3] > parts["state_delta"][1]
    assert parts["action_delta"][3] > parts["action_delta"][1]


def test_temporal_suppression_prefers_non_adjacent_high_scores():
    scores = np.array([0.10, 0.90, 0.89, 0.20, 0.80], dtype=np.float32)

    selected = select_top_with_temporal_suppression(scores, count=2, window=1)

    assert selected.tolist() == [1, 4]


def test_coreset_module_does_not_import_baseline_module():
    source = Path("aloha_baseline/coreset.py").read_text(encoding="utf-8")

    assert "aloha_baseline.baseline" not in source
