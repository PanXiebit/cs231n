import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero [N,D]
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):#样本L_i
        scores = X[i].dot(W)  #score function,[1,3072]*[3072,10] = [1,10],一个样本对应每一类的得分
        correct_class_score = scores[y[i]]  # y[i]表示第i个样本的真实标签,score[y[i]]表示真实标签对应的得分
        for j in xrange(num_classes):
            if j == y[i]:
                continue     # 如果满足，则跳出该for循环。即公式里面，求和时j!=y_{i}
            margin = scores[j] - correct_class_score + 1  # note delta = 1 不正确分类的超出的边界
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T

            # Right now the loss is a sum over all training examples, but we want it
            # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)  # L2 正则化项
    dW += reg * 2 * W # 梯度为什么乘以正则项系数，乘以2为啥？

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)
    correct_class_scores = []
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    margins = scores - correct_class_scores + 1
    margins[margins < 0] = 0
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)  # 所有的超过边界的值的和
    loss /= num_train
    loss += reg * np.sum(W ** 2)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # 矩阵对矩阵求导，可以转换为矩阵L对列向量Wj，Wyj求导，再转换为行向量Li对列向量Wj，Wyj求导
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] = -incorrect_counts
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW


