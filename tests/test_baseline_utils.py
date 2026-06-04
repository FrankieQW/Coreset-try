import numpy as np
import pytest

from aloha_baseline.baseline import (
    assert_pyav_available,
    build_language_features,
    select_action_arm,
    select_episode_subset,
)


def test_select_episode_subset_uses_at_least_one_episode_and_is_reproducible():
    episodes = list(range(7))

    first = select_episode_subset(episodes, fraction=0.10, seed=123)
    second = select_episode_subset(episodes, fraction=0.10, seed=123)

    assert len(first) == 1
    assert first == second
    assert first[0] in episodes


def test_select_episode_subset_samples_ten_percent_for_aloha_size():
    episodes = list(range(50))

    subset = select_episode_subset(episodes, fraction=0.10, seed=42)

    assert len(subset) == 5
    assert len(set(subset)) == 5


def test_select_action_arm_extracts_left_or_right_seven_dof():
    actions = np.arange(28, dtype=np.float32).reshape(2, 14)

    np.testing.assert_array_equal(select_action_arm(actions, "left"), actions[:, :7])
    np.testing.assert_array_equal(select_action_arm(actions, "right"), actions[:, 7:])


def test_build_language_features_is_constant_per_instruction():
    features = build_language_features("Transfer the cube", rows=3, dim=16)

    assert features.shape == (3, 16)
    np.testing.assert_array_equal(features[0], features[1])
    assert np.linalg.norm(features[0]) > 0


def test_assert_pyav_available_raises_clear_install_message(monkeypatch):
    def fake_import_module(name):
        raise ImportError("No module named av")

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="conda install -c conda-forge av"):
        assert_pyav_available()
