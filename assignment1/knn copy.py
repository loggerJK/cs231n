# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import print_function
from IPython import get_ipython

# %% [markdown]
# # k-Nearest Neighbor (kNN) exercise
#
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*
#
# The kNN classifier consists of two stages:
#
# - During training, the classifier takes the training data and simply remembers it
# - During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
# - The value of k is cross-validated
#
# In this exercise you will implement these steps and understand the basic Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, vectorized code.

# %%
# !git clone https://github.com/loggerJK/cs231n.git
# !cp -r ./cs231n/assignment1/cs231n/ ./
# !cd cs231n/datasets/ && ./get_datasets.sh


# %%
# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# get_ipython().run_line_magic("matplotlib", "inline")
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")


# %%
# Load the raw CIFAR-10 data.
cifar10_dir = "cs231n/datasets/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print("Training data shape: ", X_train.shape)
print("Training labels shape: ", y_train.shape)
print("Test data shape: ", X_test.shape)
print("Test labels shape: ", y_test.shape)


# %%
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]


# %%
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# %%
from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# %%
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)


# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


# %%
np.array(X_train_folds[1:-1]).shape


# %%

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
pass
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# 하이퍼 파라미터 = K
# 총 k개의 모델을 테스트한다
for k in k_choices:
    # num_folds만큼 K-fold Cross-Validation을 실행한다
    accuracy = 0
    for fold_num in range(num_folds):
        # validation set 설정
        X_tr = np.array(X_train_folds[:fold_num] + X_train_folds[fold_num + 1 :])
        y_tr = np.array(
            np.array(y_train_folds[:fold_num] + y_train_folds[fold_num + 1 :])
        )
        X_tr = X_tr.reshape(X_tr.shape[0] * X_tr.shape[1], X_tr.shape[2])
        y_tr = y_tr.flatten()
        X_val = X_train_folds[fold_num]
        y_val = y_train_folds[fold_num].flatten()
        # print("fold_num = ", fold_num)
        # print("X_tr = ", X_tr.shape)
        # print("y_tr = ", y_tr.shape)
        # print("X_val = ", X_val.shape)
        # print("y_val = ", y_val.shape)

        # classifier 로딩
        classifier = KNearestNeighbor()
        # train데이터 입력
        classifier.train(X_tr, y_tr)
        # Validation Set으로 Predict
        y_pred = classifier.predict(X_val, k=k, num_loops=0)
        # Accuracy를 계산한다
        num_correct = np.sum(y_pred == y_val)
        accuracy += float(num_correct) / (X_val.shape[0])
        num_correct = np.sum(y_pred == y_val)
        accuracy += float(num_correct) / (X_val.shape[0])

    # Average acurracy를 구한다
    avg_acc = float(accuracy) / num_folds
    k_to_accuracies[k] = avg_acc


# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print("k = %d, accuracy = %f" % (k, accuracy))


# %%
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title("Cross-validation on k")
plt.xlabel("k")
plt.ylabel("Cross-validation accuracy")
plt.show()


# %%
# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 1

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print("Got %d / %d correct => accuracy: %f" % (num_correct, num_test, accuracy))

