"""逐張圖、逐個正式附件走與 GUI 相同的流程。"""
import numpy as np
import pytest

from mem_storage_machine.data import load_patterns, load_samples
from mem_storage_machine.hopfield import recall, train
from mem_storage_machine.workflows import run_hopfield, run_som

pytestmark = pytest.mark.course


# 壓縮檔內的原始檔名、非空樣本列數、特徵欄數（已扣除標籤）。
SOM_DATASETS = [
    ("123.txt", 4, 2),
    ("2Ccircle1.txt", 240, 2),
    ("2Circle1.txt", 240, 2),
    ("2Circle2.txt", 260, 2),
    ("2CloseS.txt", 200, 2),
    ("2CloseS2.txt", 200, 2),
    ("2CloseS3.txt", 1000, 2),
    ("2cring.txt", 400, 2),
    ("2CS.txt", 372, 2),
    ("2Hcircle1.txt", 120, 2),
    ("2ring.txt", 300, 2),
    ("4satellite-6.txt", 3772, 4),
    ("5CloseS1.txt", 180, 2),
    ("8OX.TXT", 45, 8),
    ("C10D.TXT", 200, 10),
    ("C3D.TXT", 200, 3),
    ("IRIS.TXT", 150, 4),
    ("Number.txt", 4, 25),
    ("perceptron1.txt", 4, 2),
    ("perceptron2.txt", 4, 2),
    ("perceptron3.txt", 4, 3),
    ("perceptron4.txt", 8, 2),
    ("wine.txt", 178, 13),
    ("xor.txt", 4, 2),
]


def test_course_files_present(course_dir):
    expected = {f"hopfield/{name}_{kind}.txt" for name in ("Basic", "Bonus")
                for kind in ("Training", "Testing")}
    expected.update(f"som/{name}" for name, _, _ in SOM_DATASETS)
    actual = {p.relative_to(course_dir).as_posix() for p in course_dir.rglob("*")
              if p.is_file() and p.suffix.lower() == ".txt"}
    assert expected <= actual, sorted(expected - actual)


@pytest.mark.parametrize(("name", "shape", "changed_pixels"), [
    ("Basic", (3, 12, 9), [15, 22, 32]),
    pytest.param("Bonus", (15, 10, 10), [11, 12, 13, 11, 15, 8, 16, 10, 13, 16, 15, 17, 8, 4, 13],
                 marks=pytest.mark.bonus),
])
def test_course_hopfield_pairs(course_dir, name, shape, changed_pixels, record_property):
    train_path = course_dir / "hopfield" / f"{name}_Training.txt"
    test_path = course_dir / "hopfield" / f"{name}_Testing.txt"
    assert train_path.is_file() and test_path.is_file(), f"缺少 {name} 訓練檔或測試檔"
    targets, cues = load_patterns(train_path), load_patterns(test_path)
    assert targets.shape == cues.shape == shape
    np.testing.assert_array_equal(np.count_nonzero(targets != cues, axis=(1, 2)), changed_pixels)
    weights = train(targets.reshape(len(targets), -1))
    cases = run_hopfield(train_path, test_path)
    exact = 0
    for case, target, cue in zip(cases, targets, cues, strict=True):
        np.testing.assert_array_equal(case.target, target)
        np.testing.assert_array_equal(case.cue, cue)
        np.testing.assert_array_equal(case.recalled, recall(weights, cue.ravel()).reshape(cue.shape))
        exact += int(np.array_equal(case.recalled, target))
    record_property("exact_recalls", f"{exact}/{len(cases)}")


@pytest.mark.parametrize(("name", "samples", "dimensions"), SOM_DATASETS,
                         ids=[name for name, _, _ in SOM_DATASETS])
def test_course_som_maps_every_file(course_dir, name, samples, dimensions, record_property):
    path = course_dir / "som" / name
    features, labels = load_samples(path)
    raw = np.loadtxt(path, dtype=str, comments=None)
    assert features.shape == (samples, dimensions)
    np.testing.assert_allclose(features, raw[:, :-1].astype(float))
    assert list(labels) == raw[:, -1].tolist()

    run = run_som(path, iterations=12, seed=11)
    np.testing.assert_array_equal(run.features, features)
    assert list(run.labels) == list(labels)
    assert len(run.history) == 13 and np.isfinite(run.history).all()
    assert len(run.winners) == samples
    final = run.history[-1]
    assert final.shape == (4, 4, dimensions)
    for sample, winner in zip(features, run.winners, strict=True):
        distances = np.linalg.norm(final - sample, axis=-1)
        row, col = winner
        assert 0 <= row < distances.shape[0] and 0 <= col < distances.shape[1]
        np.testing.assert_allclose(distances[row, col], distances.min())
    record_property("samples", samples)
