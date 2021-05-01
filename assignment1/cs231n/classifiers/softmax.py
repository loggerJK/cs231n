import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_class = W.shape[1]
    num_train = X.shape[0]

    score = X.dot(W)
    softmax = np.zeros_like(score)
    loss = 0
    # single datia X[i]마다 루프를 돈다
    for i in range(num_train):
        # scalar
        exp_sum = np.sum(np.exp(score[i]))

        s_y_i = np.exp(score[i][y[i]]) / exp_sum
        L_i = -np.log(s_y_i)
        loss += L_i

        for j in range(num_class):
            if j != y[i]:
                softmax[i][j] = np.exp(score[i][j]) / exp_sum
                dW[:, j] += softmax[i][j] * X[i]
            else:
                softmax[i][j] = s_y_i
                dW[:, y[i]] += (s_y_i - 1) * X[i]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    score = X.dot(W)
    score -= np.max(score)
    softmax = np.exp(score)
    exp_sum = np.sum(softmax, axis=1)

    loss += -(np.sum(np.log(softmax[np.arange(num_train), y] / exp_sum)))

    softmax[np.arange(num_train), y] -= 1

    dW = X.T.dot(softmax)

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

