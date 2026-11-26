"""Hopfield 在模型內一律使用展平的雙極向量。"""
import numpy as np
# test library
#from data import load_patterns
#from pathlib import Path

def train(patterns: np.ndarray) -> np.ndarray:
    """輸入 (樣本數, 像素數)，回傳權重矩陣。"""
    #raise NotImplementedError("T02 train")
    p = patterns.shape[1]
    n = patterns.shape[0] 
    weight = np.zeros((p,p))
    delta_w = (patterns.T @  patterns - n*np.identity(p))/p
    weight = weight + delta_w
    return weight

def recall(weights: np.ndarray, cue: np.ndarray, *, max_sweeps: int = 30) -> np.ndarray:
    """按索引非同步更新，狀態穩定或達到上限時回傳向量。"""
    input_x = np.array(cue)
    # calculate threadhold for recalling
#    bias = []
#    for j in range(weights.shape[0]):
#        cur_neuron = weights[j]
#        b = np.sum(cur_neuron)
#        bias.append(b)
    for epoch in range(max_sweeps):
        # non-simultaneously update
        for j in range(weights.shape[0]):
            cur_neuron = weights[j]
            a = weights[j] @ input_x
            if a > 0:
                input_x[j] = 1
            elif a < 0:
                input_x[j] = -1
            else:
                input_x[j] = input_x[j]
    return input_x


def add_noise(pattern: np.ndarray, *, flips: int, seed: int) -> np.ndarray:
    """複製圖案，選 flips 個不同位置翻轉正負號。"""
    noise_pattern = pattern.copy()
    rng = np.random.default_rng(seed)
    arr = rng.integers(0,pattern.shape[0]*pattern.shape[1],size=flips)
    noise_pattern = noise_pattern.reshape(1,-1)
    for i in arr:
        noise_pattern[0][i] = -noise_pattern[0][i]
    return noise_pattern.reshape(pattern.shape[0],pattern.shape[1])


if __name__ == "__main__":
    path = Path("/home/xion/ML_HW_3/data/hopfield/Basic_Training.txt")
    patterns = load_patterns(path)
    train(patterns)
