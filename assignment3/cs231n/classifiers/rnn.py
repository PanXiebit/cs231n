from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.　　### 图像的特征向量
        - wordvec_dim: Dimension W of word vectors.　　　###词向量维度
        - hidden_dim: Dimension H for the hidden state of the RNN.　　###隐藏层
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.　###神经元状态rnn or lstm
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}  ###idx_to_word
        self.params = {}

        vocab_size = len(word_to_idx)  ## 词典大小

        ### ３个特殊字符的索引
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors　初始化词向量
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)  ## (V, W)
        self.params['W_embed'] /= 100  ##???

        # Initialize CNN -> hidden state projection parameters  根据图像生成的特征向量到第一个隐藏层的矩阵初始化
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)  ### 使得生成的初始权重的方差为0.2?
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN　　初始化RNN参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]  ###　为什么lstm的隐藏层维度要是rnn的４倍....？？
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights 从隐藏层到输出词之间的矩阵初始化
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)  # (H,V)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)　## 输入是图像的特征向量(N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V  ## 真实标签是标题对应在字典中的索引(N, T)

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        ## 这里将captions分成了两个部分，captions_in是除了最后一个词外的所有词，是输入到RNN/LSTM的输入；
        ## captions_out是除了第一个词外的所有词，是RNN/LSTM期望得到的输出。
        captions_in = captions[:, :-1]  # (N, T)
        captions_out = captions[:, 1:]  # 真实值

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        # 从图像特征到初始隐藏状态的权值矩阵和偏差值 
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        # 词嵌入矩阵
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        # RNN/LSTM参数
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        # 每一隐藏层到输出的权值矩阵和偏差
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        
        ## step1: 图像特征到隐藏层：全连接层
        h0 = np.dot(features, W_proj) + b_proj  # (N,H)
        
        ## step2: 将输入序列转换为词向量
        embed_out, cache_embed = word_embedding_forward(captions_in, W_embed)  # (N ,T, W)
        
        ## step3: 将图像特征和词向量作为输入，通过RNN,得到隐藏层的输出
        if self.cell_type == 'rnn':
            hidden_out, cache_hidden = rnn_forward(embed_out, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            hidden_out, cache_hidden = lstm_forward(embed_out, h0, Wx, Wh, b)
        else:
            raise ValueError('%s not implemented' % (self.cell_type))
        
        ## step4: 将隐藏层的输出作为输入，通过affine-softmax层得到输出，词典中每个词的得分
        score_vocab, cache_vocab = temporal_affine_forward(hidden_out, W_vocab, b_vocab)  # (N, T, V)
        
        # 用softmax函数计算损失，真实值为captions_out, 用mask忽视所有向量中<NULL>词汇
        loss, dscore_vocab = temporal_softmax_loss(score_vocab, captions_out, mask, verbose=False)

        ## 反向传播计算梯度
        grads = dict.fromkeys(self.params)
        
        # step4 backward
        dhidden_out, dW_vocab, db_vocab = temporal_affine_backward(dscore_vocab, cache_vocab)
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab

        # Backward into step 3
        if self.cell_type == 'rnn':
            dembed_out, dh0, dWx, dWh, db = rnn_backward(dhidden_out, cache_hidden)
        elif self.cell_type == 'lstm':
            dembed_out, dh0, dWx, dWh, db = lstm_backward(dhidden_out, cache_hidden)
        else:
            raise ValueError('%s not implemented' % (self.cell_type))
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db

        # Backward into step 2
        dW_embed = word_embedding_backward(dembed_out, cache_embed)
        grads['W_embed'] = dW_embed
        
        ## aBackward into step 1
        dW_proj = features.T.dot(dh0)
        db_proj = np.sum(dh0,axis=0)
        grads['W_proj'] = dW_proj
        grads['b_proj'] = db_proj
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        
        ## step1: 图像特征到隐藏层：全连接层
        N, D = features.shape
        H, V = W_vocab.shape
        V, W = W_embed.shape
        
        h0 = np.dot(features, W_proj) + b_proj  # (N,H)
        assert(h0.shape==(N,H))
        
        captions[:,0] = self._start 
        
        prev_h = h0 # Previous hidden state
        prev_c = np.zeros(h0.shape)  # lstm: memory cell state
        # Current word (start word)
        x = self._start * np.ones((N, 1), dtype=np.int32)  # (N,1) 测试：sample时输入
        
        for t in range(max_length):
            embed_out, _ = word_embedding_forward(x, W_embed)  # (N ,1, W) embedded word vector
            assert(embed_out.shape == (N,1,W))
            if self.cell_type == 'rnn':
                # Run a step of rnn   # Remove single-dimensional entries from the shape of an array.
                hidden_out, _ = rnn_step_forward(np.squeeze(embed_out), prev_h, Wx, Wh, b)   ## (N,H)                
            elif self.cell_type == 'lstm':
                # Run a step of lstm
                hidden_out, _, _ = lstm_step_forward(np.squeeze(embed_out), prev_h, prev_c, Wx, Wh, b)
            else:
                raise ValueError('%s not implemented' % (self.cell_type))
        
        # Compute the score distrib over the dictionary      
        score_vocab, cache_vocab = temporal_affine_forward(hidden_out[:, np.newaxis, :], W_vocab, b_vocab)  # (N, 1, V)
        # Squeeze unecessari dimension and get the best word idx
        idx_best = np.squeeze(np.argmax(score_vocab, axis=2))
        # Put it in the captions
        captions[:, t] = idx_best
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions