from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        ##rand()是(0,1)上的随机数  randn()是标准正态分布
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)   
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(hidden_dim,)
        self.params['b2'] = np.zeros(num_classes,)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        X_vec = X.reshape(X.shape[0],-1)   #(N,D)
        Z = X_vec.dot(self.params['W1'])+self.params['b1']  #(N,H)
        A = np.maximum(Z,0)  #(N,H)
        scores = A.dot(self.params['W2'])+self.params['b2']
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #loss
        N = X.shape[0]
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        shift_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)   #(N,C)
        loss = np.sum(-np.log(shift_scores[range(N),y]))
        loss /= N
        loss += 0.5 * self.reg * np.sum(self.params['W1']**2) + 0.5 * self.reg * np.sum(self.params['W2']**2)
        #gradient
        shift_scores[range(N),y] -= 1
        shift_scores /= N     #(N,C)  这一步按理说是求梯度时都应该做的，只不过提前了
        grads['b2'] = np.sum(shift_scores, axis=0, keepdims=True).reshape(-1,)  #(C,)
        grads['W2'] = A.T.dot(shift_scores)                      #(H,C)
        grads['W2'] += self.reg * self.params['W2']
        
        hidden = shift_scores.dot(self.params['W2'].T)   #(N,H)
        hidden[Z <= 0] = 0
        grads['b1'] = np.sum(hidden, axis=0,keepdims=True).reshape(-1,)   #(H,)
        grads['W1'] = X_vec.T.dot(hidden)
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, ##需要修改的参数hidden_dims,dropout, use_batchnorm, reg
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer. ##一个list储存每个隐藏层的神经元数量
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.           ## integer 
        - use_batchnorm: Whether or not the network should use batch normalization.    ## boolean 
        - reg: Scalar giving L2 regularization strength.        ## L2正则化
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.                       ## 权重初始化
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.          ## 计算都使用 float64
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0   ## boolean 判断是否使用dropout
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {} 

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        layers_dims = [input_dim] + hidden_dims + [num_classes] ## list+list 输入层\隐藏层\输出层的神经元的个数
        
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(layers_dims[i],layers_dims[i+1])
            self.params["b"+str(i+1)] = np.zeros(layers_dims[i+1])
            if self.use_batchnorm and i < self.num_layers-1 :  ## 最后一层是得到score,没有激活函数,也不需要BN
                self.params["gamma"+str(i+1)] = np.ones((1,layers_dims[i+1]))  ##(1,D)当前隐藏层神经元数目,与归一化的均值和方差一致
                self.params["beta"+str(i+1)] = np.zeros((1,layers_dims[i+1]))
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward  
        # pass of the second batch normalization layer, etc. 
        # elf.bn_params是个list,里面的元素是dict,这样就解决了不知道bn参数对应哪一层的问题
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)] 

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'   ## mode ='train' or 'test'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out, z, z_bn, cache_aff, cache_bn ,cache_relu, cache_drop = {}, {}, {}, {}, {}, {}, {}
        out[0] = X.reshape(X.shape[0],-1)
        for i in range(self.num_layers-1):
            w,b = self.params['W'+ str(i+1)], self.params['b'+ str(i+1)]
            if self.use_batchnorm:
                gamma, beta = self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)]
                z[i+1],cache_aff[i] = affine_forward(out[i], w, b)
                z_bn[i+1],cache_bn[i] = batchnorm_forward(z[i+1], gamma, beta, self.bn_params[i])
                ## bn_params中的元素是bn_param,一个dict,里面存储的有 mode, eps, momentum, running_mean, running_var
                ## 其中eps和momentum在batchnorm_forward是定值,mode需要根据外部mode来确定, 
                ## bn_params[i]中的running值刚开始有初始值,但会对应更新,以便test时使用
                out[i+1],cache_relu[i] = relu_forward(z_bn[i+1]) 
                if self.use_dropout:
                    out[i+1], cache_drop[i] = dropout_forward(out[i+1],self.dropout_param)                    
            else:
                out[i+1],cache_relu[i] = affine_relu_forward(out[i],w,b)
                if self.use_dropout:
                    out[i+1], cache_drop[i] = dropout_forward(out[i+1],self.dropout_param)
            
        w, b = self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)]
        scores,cache = affine_forward(out[self.num_layers-1],w,b)  ##最后一层全连接层       
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss_data, dscores = softmax_loss(scores, y)
        loss_reg = 0
        for i in range(self.num_layers ):
            loss_reg += np.sum(0.5 * self.reg * self.params["W"+str(i+1)]**2)
        loss = loss_data + loss_reg
        
        dout, ddrop, drelu, dbn, dgamma, dbeta = {}, {}, {}, {}, {}, {}
        h = self.num_layers-1
        dout[h],grads['W'+str(h+1)],grads['b'+str(h+1)] = affine_backward(dscores, cache) 
        for i in range(h):
            if self.use_batchnorm:
                if self.use_dropout:
                    ddrop[h-i-1] = dropout_backward(dout[h-i], cache_drop[h-i-1])
                    dout[h-i] = ddrop[h-i-1]
                drelu[h-i-1] = relu_backward(dout[h-i], cache_relu[h-i-1])
                dbn[h-i-1],grads['gamma'+str(h-i)], grads['beta'+str(h-i)] = batchnorm_backward(drelu[h-i-1],cache_bn[h-i-1])
                dout[h-i-1],grads['W'+str(h-i)], grads['b'+str(h-i)] = affine_backward(dbn[h-i-1],cache_aff[h-i-1])
            else:
                if self.use_dropout:
                    ddrop[h-i-1] = dropout_backward(dout[h-i], cache_drop[h-i-1])
                    dout[h-i] = ddrop[h-i-1]
                dout[h-i-1],grads['W'+str(h-i)], grads['b'+str(h-i)] = affine_relu_backward(dout[h-i],cache_relu[h-i-1])
        for i in range(self.num_layers):
            grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
