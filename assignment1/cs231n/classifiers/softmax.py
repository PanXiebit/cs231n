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
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        score_i = X[i].dot(W)  # [1,10]
        ##为避免数值的不稳定行
        score_i -= np.max(score_i)
        loss += np.log(np.sum(np.exp(score_i))) - score_i[y[i]]  # (1,)

        ##微分法求梯度
        coef = (np.exp(score_i) / np.sum(np.exp(score_i))).reshape(1,10)  #(1,10)
        # for k in range(num_class):
        #     if k == y[i]:
        #         dW[:, k] += X[i].T * (coef - 1)  # (3073,1)*(1,)
        #     else:
        #         dW[:, k] += X[i].T * coef
        coef_yi = np.zeros_like(coef)
        coef_yi[0,y[i]] = 1
        dW += X[i].T.reshape(3073,1)*(coef-coef_yi)
        

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
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
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = X.dot(W)  ##(N,3073)*(3073,10) = (N,10)
    score -= np.max(score, axis=1, keepdims=True)  # [N,10]
    exp_score = np.exp(score)  # [N,10]
    sum_score = np.sum(exp_score, axis=1, keepdims=True)  # numpy 广播机制，很关键！
    coef = exp_score / sum_score
    loss = np.sum(-np.log(coef[range(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    coef_yi = np.zeros_like(coef)
    coef_yi[range(num_train), y] = 1
    dW = X.T.dot(coef - coef_yi)
    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


