"""兩個 GUI 分頁共用的應用層入口。"""
from dataclasses import dataclass
from pathlib import Path
from .data import load_patterns, load_samples
from . import hopfield, som

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
    print("========runing hopfield=================")
    train_dataset = load_patterns(training)
    train_flat = train_dataset.reshape(len(train_dataset), -1)
    test_dataset = load_patterns(testing)
    test_flat = test_dataset.reshape(len(test_dataset),-1)
    if len(test_flat) != len(train_flat):
        raise ValueError(f'[DATA_PAIR_COUNT] 訓練圖有 {len(train_flat)} 張，測試圖有 {len(test_flat)} 張。')
    weight = hopfield.train(train_flat)
    recall_cases = []
    for i in range(len(test_dataset)):
        print(f'recall {i}-th cue')
        case = RecallCase(target = train_dataset[i], 
                          cue = test_dataset[i],
                          recalled=hopfield.recall(weight, test_flat[i]).reshape(train_dataset.shape[1],train_dataset.shape[2])
                          )
        recall_cases.append(case)
    print("finished ~!")
    return recall_cases

    raise NotImplementedError("T03 run_hopfield")


def run_som(path: Path, *, iterations: int, grid_shape: tuple[int, int] = (4, 4),
            seed: int = 0) -> SomRun:
    """使用特徵訓練，按原始順序回傳每筆樣本的得勝格點。"""
    if iterations <=0:
        raise ValueError("[SOM_ITERATIONS] 迭代次數必須為正整數。")
    features, labels = load_samples(path)
    rng = np.random.default_rng(seed)  
    initial = rng.random((grid_shape[0],grid_shape[1], features.shape[1]))
    history = som.train(features, initial, iterations=iterations)
    winners = [som.bmu(history[-1],sample) for sample in features]
    return SomRun(features,labels, history, winners)
    raise NotImplementedError("T07 run_som")

