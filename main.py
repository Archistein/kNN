from kNN import kNN
import numpy as np
import numpy.typing as npt
import pandas as pd


def split_data_by_column(sample: pd.DataFrame, col: str) -> tuple[npt.NDArray, npt.NDArray]:
    X = np.array(sample.drop(col, axis=1))
    Y = np.array(sample[col])

    return X, Y


def main() -> None:
    raw_data = pd.read_csv('dataset/iris.data', 
                        header=None,
                        names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    raw_data['class'] = raw_data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    train = raw_data.sample(frac=0.6, random_state=1)
    test = raw_data[~raw_data.isin(train)].dropna()

    train_X, train_Y = split_data_by_column(train, 'class')
    test_X, test_Y = split_data_by_column(test, 'class')

    k = 8

    print('kNN method on Iris dataset')
    print(f'k = {k}')

    clf = kNN(k)
    clf.fit(train_X, train_Y)

    acc = np.sum([clf.predict(test_X[i]) == test_Y[i] for i in range(len(test_X))]) / len(test_X)

    print(f'Accuracy = {acc:.3}')


if __name__ == '__main__':
    main()