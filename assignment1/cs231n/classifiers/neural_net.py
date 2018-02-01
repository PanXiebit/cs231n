from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients. 这里说y可能不存在对吧。。
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        scores_1 = X.dot(W1) + b1 # (N,H)+(H,)=(N,H)
        input_2 = np.maximum(scores_1, 0)         ## relu  (N,H)
        scores = input_2.dot(W2) + b2  # (N,C)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores  ##如果没有标签就只返回score，不计算loss，gradient了

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)  # (N,C)
        coef = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N,C)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = np.sum(-np.log(coef[range(N), y]))  # (N,C)->(N,1)->(1,)
        loss /= N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))    ###这里多乘了一个reg，影响好大。。。！！！
        loss_reg = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  ## loss_reg
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        coef[range(N),y] -= 1
        coef /= N                  #这一步忘了。。

        grads['W2'] = input_2.T.dot(coef)          #(N,H).T.dot(N,C)=(H,C)
        grads['W2'] += reg * 2 * W2
        grads['b2'] = np.sum(coef,axis=0,keepdims=True).reshape(-1,)  #(N,C)->(1,C)

        dhidden = coef.dot(W2.T)   #(N,C).*(C,H)=(N,H)
        dhidden[input_2 <= 0] =0
        grads['W1'] = X.T.dot(dhidden)   #(D,N).*(N,H) = (D,H)
        grads['W1'] += reg * 2 * W1
        grads['b1'] = np.sum(dhidden,axis=0).reshape(-1,) #(H,)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads, loss_reg

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)  ##跑完整个数据需要多少次

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            batch_index = np.random.choice(num_train,batch_size,replace=True)   ##这里是可重复的
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads, loss_reg = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.
            # 随机梯度下降                                                          #
            #########################################################################
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            ##随机梯度下降的原理：每迭代一次，参数W，b沿着使loss减小的方向变化
            ##这个方向就是loss关于参数的梯度dW,db,其中减小的步长是learning_rate
            ##在数据量很大的情况下，采用随机梯度下降，就是得到loss的原始数据是从所有数据中随机提取的

            if verbose and it % 100 == 0:   ##每迭代100次记录下loss
                print('iteration %d / %d: loss %f, loss_reg %f' % (it, num_iters, loss, loss_reg))
                ##我想看下归一化之前权重的值具体是多少,并且看下reg怎么权衡loss_data和loss_reg的大小的
                print("W1: "),print(self.params['W1'][:1,:5]),print(np.sum(self.params['W1']*self.params['W1']))
                print("b1: "),print(self.params['b1'][:5])
                print("W2: "),print(self.params['W2'][:1,:5]),print(np.sum(self.params['W2']*self.params['W2']))
                print("b2: "),print(self.params['b2'][:5])

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:   ##每跑完一整个数据，记录一次。
                # Check accuracy
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)               

                # Decay learning rate
                learning_rate *= learning_rate_decay   ##学习率衰减

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        scores_1 = X.dot(self.params['W1']) + self.params['b1'] # (1,H)+(H,)=(1,H)
        input_2 = np.maximum(scores_1, 0)         ## relu  (1,H)
        scores = input_2.dot(self.params['W2']) + self.params['b2'] 
        y_pred = np.argmax(scores,axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred


