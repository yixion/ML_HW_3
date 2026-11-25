"""SOM 格點以 (row, col) 表示，權重最後一軸為特徵。"""
import numpy as np


def bmu(weights: np.ndarray, sample: np.ndarray) -> tuple[int, int]:
    """回傳距離樣本最近的格點座標。"""
    raise NotImplementedError("T06 bmu")


def step(weights: np.ndarray, sample: np.ndarray, *, rate: float, radius: float) -> np.ndarray:
    """執行一次高斯鄰域更新，回傳新的權重。"""
    raise NotImplementedError("T06 step")


def train(features: np.ndarray, initial: np.ndarray, *, iterations: int,
          rate: float = 0.5, radius: float = 1.0) -> list[np.ndarray]:
    """回傳初始快照及每次更新後的快照。"""
    raise NotImplementedError("T07 train")
