import numpy as np
import pytest

from mem_storage_machine.hopfield import add_noise, recall, train


def test_hebb_weights_sum_both_memories_and_remove_self_connections():
    memories = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
    expected = [[0, 0, 0, -0.5], [0, 0, -0.5, 0],
                [0, -0.5, 0, 0], [-0.5, 0, 0, 0]]
    np.testing.assert_allclose(train(memories), expected)


def test_recall_uses_new_state_within_the_same_sweep():
    weights = np.array([[0, 0.5], [0.5, 0]])
    cue = np.array([1, -1])
    np.testing.assert_array_equal(recall(weights, cue, max_sweeps=1), [-1, -1])
    np.testing.assert_array_equal(cue, [1, -1])


def test_zero_field_retains_previous_bit():
    cue = np.array([-1, 1])
    np.testing.assert_array_equal(recall(np.zeros((2, 2)), cue), [-1, 1])


def test_recall_respects_limit_and_can_continue_to_fixed_point():
    # 節點 0 只聽節點 1；節點 1 受節點 2 的較大權重影響。
    weights = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
    cue = np.array([-1, -1, 1])
    np.testing.assert_array_equal(recall(weights, cue, max_sweeps=1), [-1, 1, 1])
    np.testing.assert_array_equal(recall(weights, cue), [1, 1, 1])


@pytest.mark.bonus
def test_seeded_noise_flips_distinct_pixels_and_preserves_training_copy():
    original = np.array([[1, 1, -1, -1]] * 4)
    image = original.copy()
    noisy = add_noise(image, flips=3, seed=17)
    assert noisy.shape == original.shape
    assert set(np.unique(noisy)) <= {-1, 1}
    assert np.count_nonzero(noisy != original) == 3
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(add_noise(image, flips=3, seed=17), noisy)
    weights = train(original.reshape(1, -1))
    np.testing.assert_array_equal(recall(weights, noisy.ravel()), original.ravel())
