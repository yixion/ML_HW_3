import numpy as np
import pytest

from mem_storage_machine.workflows import run_hopfield, run_som


@pytest.mark.parametrize(("name", "shape"), [
    ("Basic", (12, 9)), pytest.param("Bonus", (10, 10), marks=pytest.mark.bonus),
])
def test_hopfield_pairs_and_recovers_every_synthetic_cue(tmp_path, make_pair, name, shape):
    targets, cues = make_pair(tmp_path, name, shape)
    cases = run_hopfield(tmp_path / f"{name}_Training.txt", tmp_path / f"{name}_Testing.txt")
    assert len(cases) == len(targets)
    for case, target, cue in zip(cases, targets, cues, strict=True):
        np.testing.assert_array_equal(case.target, target)
        np.testing.assert_array_equal(case.cue, cue)
        np.testing.assert_array_equal(case.recalled, target)


def test_recall_follows_cue_even_when_paired_reference_differs(tmp_path, make_pair):
    targets, _ = make_pair(tmp_path)
    path = tmp_path / "Basic_Testing.txt"
    blocks = path.read_text(encoding="utf-8").rstrip("\n").split("\n\n")
    path.write_text("\n\n".join(reversed(blocks)), encoding="utf-8")
    cases = run_hopfield(tmp_path / "Basic_Training.txt", path)
    for index, case in enumerate(cases):
        np.testing.assert_array_equal(case.target, targets[index])
        np.testing.assert_array_equal(case.recalled, targets[1 - index])
    assert len(cases) == 2


def test_som_trains_on_features_and_maps_every_sample(toy_data):
    path = toy_data / "som/cloud.txt"
    config = dict(iterations=8, grid_shape=(2, 3), seed=27)
    run = run_som(path, **config)
    np.testing.assert_allclose(run.features, [[0, 0], [1, 2], [8, 9], [9, 8]])
    assert list(run.labels) == ["A", "A", "B", "B"]
    assert np.shape(run.history) == (9, 2, 3, 2)
    assert np.isfinite(run.history).all()
    assert not np.allclose(run.history[0], run.history[-1])
    assert len(run.winners) == len(run.features)
    for sample, winner in zip(run.features, run.winners, strict=True):
        distances = np.linalg.norm(run.history[-1] - sample, axis=-1)
        row, col = winner
        assert 0 <= row < distances.shape[0] and 0 <= col < distances.shape[1]
        assert distances[tuple(winner)] == pytest.approx(distances.min())
    path.write_text("0 0 9\n1 2 8\n8 9 7\n9 8 6\n", encoding="utf-8")
    relabeled = run_som(path, **config)
    np.testing.assert_allclose(relabeled.history, run.history)
    np.testing.assert_array_equal(relabeled.winners, run.winners)


def test_hopfield_reports_unpaired_files(tmp_path, make_pair):
    make_pair(tmp_path)
    testing = tmp_path / "Basic_Testing.txt"
    testing.write_text(testing.read_text(encoding="utf-8").split("\n\n")[0], encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        run_hopfield(tmp_path / "Basic_Training.txt", testing)
    assert str(exc.value) == "[DATA_PAIR_COUNT] 訓練圖有 2 張，測試圖有 1 張。"


def test_som_reports_invalid_iteration_count(toy_data):
    with pytest.raises(ValueError) as exc:
        run_som(toy_data / "som/cloud.txt", iterations=0)
    assert str(exc.value) == "[SOM_ITERATIONS] 迭代次數必須為正整數。"
