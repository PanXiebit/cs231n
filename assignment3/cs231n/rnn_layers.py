from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward1(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    temp = prev_h.dot(Wh) + x.dot(Wx) + b  #(N,H)
    next_h = np.tanh(temp)
    cache = (x, prev_h, temp, Wh, Wx)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    输入:
    - x: 外部输入数据, 维度 (N, D).
    - prev_h: 上一个时刻的隐藏状态, 维度 (N, H)
    - Wx: x对应的权值矩阵, 维度 (D, H)
    - Wh: 隐藏状态h对应的权值矩阵, 维度 (H, H)
    - b: 偏差值 shape (H,)

    输出:
    - next_h: 下一个时态的隐藏状态, 维度 (N, H)
    - cache: 计算梯度反向传播时需要用到的变量.
    """
    temp1 = np.dot(x,Wx)
    temp2 = np.dot(prev_h,Wh)
    cache=(x,prev_h,Wx,Wh,temp1+temp2+b)
    next_h = np.tanh(temp1+temp2+b)
    return next_h, cache

def rnn_step_backward1(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state (N,H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, temp, Wh, Wx = cache
    N,H = temp.shape
    N,D = x.shape
    
    dtemp = np.ones((N,H)) - np.square(np.tanh(temp))
    assert(dtemp.shape==(N,H))
    dtemp *= dnext_h        # (N, H)
    assert(dtemp.shape==(N,H))
    dx = dtemp.dot(Wx.T)    # (N, D)
    print(dx.shape)
    assert(dx.shape == (N,D))
    dprev_h = dtemp.dot(Wh.T)  # (N, H)
    dWx = x.T.dot(dtemp)     # (D, H)
    dWh = prev_h.T.dot(dtemp)
    db = np.sum(dtemp,axis=0).T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db

def rnn_step_backward(dnext_h, cache):
    """
    输入:
    - dnext_h: 下一层传来的梯度
    - cache: 前向传播存下来的值

    输出
    - dx: 输入x的梯度, 维度(N, D)
    - dprev_h: 上一层隐藏状态的梯度, 维度(N, H)
    - dWx: 权值矩阵Wxh的梯度, 维度(D, H)
    - dWh: 权值矩阵Whh的梯度, 维度(H, H)
    - db: 偏差值b的梯度，维度（H,）
    """
    x=cache[0]
    h=cache[1]
    Wx=cache[2]
    Wh=cache[3]
    # cache[4]对应着公式中的a
    cacheD=cache[4]
    N,H=h.shape
    # 计算激活函数的导数
    temp = np.ones((N,H))-np.square(np.tanh(cacheD))
    delta = np.multiply(temp,dnext_h)
    # 计算x的梯度
    tempx = np.dot(Wx,delta.T)
    dx=tempx.T
    # h的提督
    temph = np.dot(Wh,delta.T)
    dprev_h=temph.T
    # Wxh的梯度
    dWx = np.dot(x.T,delta)
    # Whh
    dWh = np.dot(h.T,delta)
    # b的梯度
    tempb = np.sum(delta,axis=0)
    db=tempb.T
    return dx, dprev_h, dWx, dWh, db

def rnn_forward1(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T, H))
    h2 = np.zeros((N, T, H))
    temp = np.zeros((N, T, H))
    prev_h = h0
    for t in range(T):
        temp_h, cache_temp = rnn_step_forward(x[:,t,:], prev_h, Wx, Wh, b)
        h2[:,t,:] = prev_h
        prev_h = temp_h
        h[:,t,:] = temp_h 
        temp[:,t,:] = cache_temp[2]
    cache = (x, h2, temp, Wx, Wh)
      
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache

def rnn_forward(x, h0, Wx, Wh, b):
    """

    输入:
    - x: 整个序列的输入数据, 维度 (N, T, D).
    - h0: 初始隐藏层, 维度 (N, H)
    - Wx: 权值矩阵Wxh, 维度 (D, H)
    - Wh: 权值矩阵Whh, 维度 (H, H)
    - b: 偏差值，维度 (H,)

    输出:
    - h: 整个序列的隐藏状态, 维度 (N, T, H).
    - cache: 反向传播时需要的变量
    """
    N,T,D=x.shape
    N,H=h0.shape
    prev_h=h0
    # 之前公式中对应的a
    h1=np.empty([N,T,H])
    # 隐藏状态h的序列
    h2=np.empty([N,T,H])
    # 滞后h一个时间点
    h3=np.empty([N,T,H])
    for i in range(0, T):
        #单步前向传播
        temp_h, cache_temp = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        #记录下需要的变量
        h3[:,i,:]=prev_h
        prev_h=temp_h
        h2[:,i,:]=temp_h
        h1[:,i,:]=cache_temp[4]
    cache=(x,h3,Wx,Wh,h1)
    return h2, cache

def rnn_backward1(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    ## 损失函数关于每一个隐藏层的梯度, 维度 (N, T, H)
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)     
    - cache : (x, h2, temp, Wx, Wh)
    
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # initialize
    x = cache[0]
    N,T,D = x.shape
    N,T,H = dh.shape
    
    assert(dh.shape==(N,T,H))
    assert(cache[0][:,0,:].shape==(N,D))   # x
    assert(cache[1][:,0,:].shape==(N,H))   # prev_h
    assert(cache[2][:,0,:].shape==(N,H))   # temp
    assert(cache[3].shape==(D,H))    # Wx
    assert(cache[4].shape==(H,H))    # Wh
    
    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros((H,))
    dhnow = np.zeros((N,H))
    
    for k in range(0, T):
        i = T-1-k
        #除了上一层传来的梯度，我们每一层都有输出，对应的误差函数也会传入梯度
        dhnow += dh[:,i,:]  #(N,H) 
        assert(dhnow.shape==(N,H))
        ######## (N,T,D),(N,T,H),(N,T,H),(D,H),(H,H)
        cacheT = (cache[0][:,i,:],cache[1][:,i,:],cache[2][:,i,:],cache[3],cache[4])
        #单步反向传播
        dx_temp, dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dhnow, cacheT)
        print(dx_temp.shape)
        dhnow = dprev_h
        dx[:,i,:] = dx_temp        
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp

    dh0 = dhnow

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db

def rnn_backward(dh, cache):
    """
    输入:
    - dh: 损失函数关于每一个隐藏层的梯度, 维度 (N, T, H)
    - cache: 前向传播时存的变量
    输出
    - dx: 每一层输入x的梯度, 维度(N, T, D)
    - dh0: 初始隐藏状态的梯度, 维度(N, H)
    - dWx: 权值矩阵Wxh的梯度, 维度(D, H)
    - dWh: 权值矩阵Whh的梯度, 维度(H, H)
    - db: 偏差值b的梯度，维度（H,）
    """
    x=cache[0]
    N,T,D=x.shape
    N,T,H=dh.shape
    #初始化
    dWx=np.zeros((D,H))
    dWh=np.zeros((H,H))
    db=np.zeros(H)
    dout=dh
    dx=np.empty([N,T,D])
    dh=np.empty([N,T,H])
    #当前时刻隐藏状态对应的梯度
    hnow=np.zeros([N,H])

    for k in range(0, T):
        i=T-1-k
        #我们要注意，除了上一层传来的梯度，我们每一层都有输出，对应的误差函数也会传入梯度
        hnow += dout[:,i,:]
        cacheT=(cache[0][:,i,:],cache[1][:,i,:],cache[2],cache[3],cache[4][:,i,:])
        #单步反向传播
        dx_temp, dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(hnow, cacheT)
        hnow = dprev_h
        dx[:,i,:] = dx_temp
        #将每一层共享的参数对应的梯度相加
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp

    dh0=hnow
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.　### 预训练好的词向量

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x,:] ##numpy的广播机制，将x的每一个元素带进去，然后取对应的词向量
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache    # x.shape=(N, T)
    dW=np.zeros_like(W) # W.shape=(V, D)
    # 在x指定的位置将dout加到dW上
    np.add.at(dW, x, dout) # dout.shape(N, T, D)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])  ###　大于０的加符号，小于０的不加。。为啥？
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b  # (N, 4H)
    
    # compute gate
    ai = a[:, :H]
    af = a[:, H:2*H]
    ao = a[:, 2*H:3*H]
    ag = a[:, 3*H:]
    gate_i = sigmoid(ai)        # update gate
    gate_f = sigmoid(af)        # forget gate
    gate_o = sigmoid(ao)        # output gate
    gate_g = np.tanh(ag)        # c_tilde
    
    next_c = gate_i * gate_g + gate_f * prev_c   # new cell state (N, H)
    next_h = gate_o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b, next_c, ai, af, ao, ag, gate_i, gate_f, gate_o, gate_g)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    N, H = dnext_h.shape
    # unroll cache
    x, prev_h, prev_c, Wx, Wh, b, next_c, ai, af, ao, ag, gate_i, gate_f, gate_o, gate_g = cache
    
    dgate_o = dnext_h * np.tanh(next_c)
    # 状态c的梯度要累加!!!!!!
    dnext_c += dnext_h * gate_o * (1 - np.tanh(next_c)**2)
    
    dgate_i = dnext_c * gate_g
    dgate_f = dnext_c * prev_c
    dgate_g = dnext_c * gate_i
    dprev_c = dnext_c * gate_f    # dprev_c  (N, H)
       
    dai = gate_i * (1 - gate_i) * dgate_i
    daf = gate_f * (1 - gate_f) * dgate_f
    dao = gate_o * (1 - gate_o) * dgate_o
    dag = (1 - gate_g**2) * dgate_g
    
    da = np.hstack((dai, daf, dao, dag)) # (N ,4H)
    assert(da.shape == (N, 4*H))
    
    dx = da.dot(Wx.T)         # dx (N, D)
    dWx = x.T.dot(da)         # dWx  (D, 4H)
    dprev_h = da.dot(Wh.T)    # dprev_h  (N, H)
    dWh = prev_h.T.dot(da)    # dWh (H, 4H)    
    db = np.sum(da, axis=0)   # db (1, 4H)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T, H))
    c = np.zeros((N, T, H))
    cache = []
    next_h = h0
    next_c = np.zeros((N ,H))
    for t in range(T):
        next_h, next_c, cache_t = lstm_step_forward(x[:,t,:], next_h, next_c, Wx, Wh, b) # 权重参数共享
        h[:, t, :] = next_h
        c[:, t, :] = next_c
        cache.append(cache_t)    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)  ###　损失函数得到的梯度
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    
    x, prev_h, prev_c, Wx, Wh, b, next_c, ai, af, ao, ag, gate_i, gate_f, gate_o, gate_g = cache[0]
    N, T, H = dh.shape
    N, D = x.shape
    
    # initilizate
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    
    # 隐藏层迭代的
    dnext_h = np.zeros((N, H))
    dnext_c = np.zeros_like(dnext_h)
    
    for k in range(T):
        t = T-1-k
        # 隐藏层h的梯度不仅来源于损失函数计算得到的dh[:,t,:]，而且有上一层回流的梯度dnext_h
        # 记忆细胞也是这样，但在step backward中已经考虑到了两种回流
        dnext_h += dh[:,t,:]
        dx_t, dnext_h, dnext_c, dWx_t, dWh_t, db_t = lstm_step_backward(dnext_h, dnext_c, cache[t])
        dx[:,t,:] = dx_t
        # 参数是共享的
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        
    dh0 = dnext_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)  ##　时间t时刻对应的minibatch个词向量(N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
