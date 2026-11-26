"""SOM 格點以 (row, col) 表示，權重最後一軸為特徵。"""
import numpy as np


def bmu(weights: np.ndarray, sample: np.ndarray) -> tuple[int, int]:
    """回傳距離樣本最近的格點座標。"""
    min_d = -1
    for r in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            d = 0
            for w in range(sample.shape[0]):
                d += abs(sample[w] - weights[r][c][w])
            if min_d == -1:
                min_d = d
                c_p = (r,c)
            elif min_d > d:
                min_d = d
                c_p = (r,c)
    return c_p



def step(weights: np.ndarray, sample: np.ndarray, *, rate: float, radius: float) -> np.ndarray:
    """執行一次高斯鄰域更新，回傳新的權重。"""
    new_weights = np.array(weights)
    winner = bmu(weights, sample)
    # Gaussian influence
    rows, cols = np.indices((weights.shape[0],weights.shape[1]))
    distance = (rows-winner[0])**2 + (cols-winner[1])**2
    pi = np.exp(-(distance/2*(radius**2)))
    for r in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            weights[r][c] = weights[r][c] + rate * pi[r][c] * (sample-weights[r][c])
    return new_weights


def train(features: np.ndarray, initial: np.ndarray, *, iterations: int,
          rate: float = 0.5, radius: float = 1.0) -> list[np.ndarray]:
    """回傳初始快照及每次更新後的快照。"""
    result = []
    weights = np.array(initial)
    result.append(weights)
    for n in range(iterations):
        r = radius*np.exp(-(n/143))
        l_r = rate*np.exp(-(n/143))
        weights = step(weights,features[n%len(features)], rate = l_r, radius = r)
        result.append(weights)
    return result
