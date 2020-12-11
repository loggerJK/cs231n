from os import kill
from matplotlib.pyplot import axis
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                # 데이터 로드
                ith_te = X[i]
                jth_tr = self.X_train[j]

                # L2 distance 계산
                value = ith_te - jth_tr
                data = np.sqrt((np.sum(np.square(value))))
                dists[i, j] = data

        return dists

    def compute_distances_one_loop(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            value = self.X_train - X[i]
            value = np.square(value)
            value = np.sum(value, axis=1)
            value = np.sqrt(value)
            value = np.reshape(value, [1, num_train])
            dists[i] = value

        return dists

    def compute_distances_no_loops(self, X):
        """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        value = []
        value.append(np.sum(np.square(X), axis=1).reshape([num_test, 1]))
        value.append(np.sum(np.square(self.X_train), axis=1).reshape([1, num_train]))
        value.append(np.matmul(X, self.X_train.T) * (-2))
        dists = np.sqrt(np.sum(value))
        return dists

    def predict_labels(self, dists, k=1):
        """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # i번째 test example
            # A list of length k he k nearest nestoring the labels of tighbors to
            # the ith test point.
            closest_y = []

            # 방문할 인덱스를 순서대로 담고 있는 배열 선언
            idx = np.argsort(dists[i])

            # k번만큼 방문하여 closest_y에 해당 example의 label을 추가한다
            for j in range(k):
                index = idx[j]
                closest_y.append(self.y_train[index])

            dict = {}

            # label의 빈도 조사
            for j in range(len(closest_y)):
                if closest_y[j] in dict:
                    dict[closest_y[j]] += dict[closest_y[j]]
                else:
                    dict[closest_y[j]] = 1

            # 가장 작은 빈도가 나온 label을 고른다
            # 최대 빈도는 2k를 넘을 수 없다
            max = 0
            max_label = 0
            for j in list(dict.keys()):
                # 기존보다 큰 라벨을 발견한다면
                if max < dict[j]:
                    max = dict[j]
                    max_label = j

            # max_label 저장
            y_pred[i] = max_label

        return y_pred

