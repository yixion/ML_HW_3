"""課程資料轉換。圖案保留原始列、欄方向。"""
from pathlib import Path

import numpy as np

def load_patterns(path: Path) -> np.ndarray:
    """回傳 (圖數, 高, 寬) 的雙極像素。"""
    patterns = []
    pattern = []
    f = open(path, 'r')
    line = f.readline()
    while line:
        if line == "\n":
            patterns.append(pattern)
            pattern = []
        else:
            data = []
            for i in line:
                if i == ' ':
                    data.append(-1)
                elif i == '1':
                    data.append(1)
            pattern.append(data)
        line = f.readline()
    patterns.append(pattern)
    patterns = np.array(patterns)
    print(patterns.shape)
    #raise NotImplementedError("T01 load_patterns")
    return patterns


def load_samples(path: Path) -> tuple[np.ndarray, list[str]]:
    """回傳數值特徵矩陣與最右欄標籤。"""
    raise NotImplementedError("T05 load_samples")


if __name__ == "__main__":
    path = Path("/home/xion/ML_HW_3/data/hopfield/Basic_Training.txt")
    images = load_patterns(path)
    images = images.reshape(len(images),-1)
    print(images.shape)
