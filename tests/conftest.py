from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--require-gui", action="store_true", help="將 Tk 無法啟動視為驗收失敗")
    parser.addoption("--course-data", metavar="DIR", help="驗收正式資料根目錄，缺檔即失敗")
    parser.addoption("--submission", metavar="ZIP", help="驗收實際繳交的 ZIP")


@pytest.fixture
def course_dir(request):
    supplied = request.config.getoption("--course-data")
    if supplied is not None:
        return Path(supplied).resolve()
    return request.config.rootpath / "data"


@pytest.fixture
def submission(request):
    supplied = request.config.getoption("--submission")
    if supplied is None:
        pytest.skip("繳交包驗收請指定 --submission")
    return Path(supplied).resolve()


@pytest.fixture
def make_pair():
    """以黑像素 1、白像素空格寫檔；正反兩張圖可可靠地修復一個翻轉。"""
    def write(directory, name="Basic", shape=(12, 9)):
        directory.mkdir(parents=True, exist_ok=True)
        rows, cols = np.indices(shape)
        target = np.where((rows + cols) % 2 == 0, 1, -1)
        targets = np.stack([target, -target])
        cues = targets.copy()
        cues[0, 0, 0] *= -1
        cues[1, -1, -1] *= -1
        for suffix, images in [("Training", targets), ("Testing", cues)]:
            blocks = ["\n".join("".join("1" if p == 1 else " " for p in row)
                                for row in img) for img in images]
            (directory / f"{name}_{suffix}.txt").write_text("\n\n".join(blocks) + "\n",
                                                         encoding="utf-8")
        return targets, cues
    return write


@pytest.fixture
def toy_data(tmp_path, make_pair):
    make_pair(tmp_path / "hopfield")
    make_pair(tmp_path / "hopfield", "Bonus", (10, 10))
    directory = tmp_path / "som"
    directory.mkdir()
    (directory / "cloud.txt").write_text("0 0 A\n1 2 A\n8 9 B\n9 8 B\n", encoding="utf-8")
    (directory / "line.TXT").write_text("2 1 X\n3 2 Y\n", encoding="utf-8")
    return tmp_path
