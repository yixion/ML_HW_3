import numpy as np

from mem_storage_machine.som import bmu, step, train


def test_bmu_uses_every_feature_and_returns_row_col():
    weights = np.array([[[0, 0, 10], [0, 0, 7], [9, 9, 9]],
                        [[8, 2, 4], [4, 8, 4], [0, 0, 3]]])
    assert tuple(bmu(weights, np.array([0, 0, 4]))) == (1, 2)


def test_gaussian_update_uses_grid_distance_and_learning_rate():
    initial = np.array([[[0., 0.], [2., 0.]], [[0., 3.], [2., 3.]]])
    sample = np.array([0., 1.])
    side = np.exp(-0.5)
    diagonal = np.exp(-1.)
    expected = [[[0., 0.5], [2. - side, 0.5 * side]],
                [[0., 3. - side], [2. - diagonal, 3. - diagonal]]]
    np.testing.assert_allclose(step(initial.copy(), sample, rate=0.5, radius=1.), expected)


def test_iterations_count_updates_and_cycle_through_samples():
    features = np.array([[2.], [8.]])
    initial = np.zeros((1, 1, 1))
    history = train(features, initial, iterations=3, rate=0.5, radius=1.)
    assert len(history) == 4
    np.testing.assert_allclose(history[0], [[[0.]]])
    np.testing.assert_allclose(history[1], [[[1.]]])
    values = [np.asarray(frame).item() for frame in history]
    assert 1 < values[2] < 8
    assert min(2, values[2]) <= values[3] <= max(2, values[2])
    assert values[3] != values[2] or np.isclose(values[2], 2)
    np.testing.assert_array_equal(initial, [[[0.]]])
