"""兩個 GUI 分頁共用的應用層入口。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RecallCase:
    target: np.ndarray
    cue: np.ndarray
    recalled: np.ndarray


@dataclass
class SomRun:
    features: np.ndarray
    labels: list[str]
    history: list[np.ndarray]
    winners: list[tuple[int, int]]


def run_hopfield(training: Path, testing: Path) -> list[RecallCase]:
    """逐組回傳訓練目標、對應測試圖與模型輸出，三者均為二維圖。"""
    raise NotImplementedError("T03 run_hopfield")


def run_som(path: Path, *, iterations: int, grid_shape: tuple[int, int] = (4, 4),
            seed: int = 0) -> SomRun:
    """使用特徵訓練，按原始順序回傳每筆樣本的得勝格點。"""
    raise NotImplementedError("T07 run_som")
