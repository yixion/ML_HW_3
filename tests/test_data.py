import numpy as np

from mem_storage_machine.data import load_patterns, load_samples


def test_patterns_preserve_blank_pixels_rows_and_block_order(tmp_path):
    path = tmp_path / "pictures.txt"
    # 中間的三個空格是一整列白色像素；完全空行才是分隔。
    path.write_bytes(b" 11\r\n   \r\n1  \r\n\r\n11 \r\n  1\r\n 1 ")
    np.testing.assert_array_equal(load_patterns(path), [
        [[-1, 1, 1], [-1, -1, -1], [1, -1, -1]],
        [[1, 1, -1], [-1, -1, 1], [-1, 1, -1]],
    ])


def test_som_loader_splits_last_column_from_all_features(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text("-1.5  0\t4  10\n2  3.25  5  20\n", encoding="utf-8")
    features, labels = load_samples(path)
    np.testing.assert_allclose(features, [[-1.5, 0, 4], [2, 3.25, 5]])
    assert list(labels) == ["10", "20"]
