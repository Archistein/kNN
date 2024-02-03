import numpy as np
import numpy.typing as npt


class kNN:
    
    def __init__(self, k: int) -> None:
        self.k = k
    
    def fit(self, X: npt.NDArray, Y: npt.NDArray):
        self.train_X = X
        self.train_Y = Y

    def predict(self, x: npt.NDArray) -> float:
        neighbors = np.argpartition([euclidean(x, t) for t in self.train_X], self.k)[:self.k]
        values, counts = np.unique(self.train_Y[neighbors], return_counts=True)
        
        return values[np.argmax(counts)]


def euclidean(x1: npt.NDArray, x2: npt.NDArray) -> float:
    if len(x1) != len(x2):
        raise ValueError('Dimension mismatch')

    return np.sqrt(np.sum((np.array(x2) - np.array(x1)) ** 2))