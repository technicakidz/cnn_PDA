import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-1, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """

    # Store network parameters
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    stride = 1
    pad = (filter_size - 1) / 2
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size

    # conv layer output volume dimensions
    H2 = (1 + (H + 2 * pad - HH) / stride) / 2
    W2 = (1 + (W + 2 * pad - WW) / stride) / 2


    # Weights initialization for convolutionnal layer
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)

    # Weights initialization for hidden affine layer
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(H2*W2*num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    # Weights initialization for output affine layer
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     



 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """

    # Get weights and bias
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # initialize scores
    scores = None

    ## FORWARD PASS
    
    # Forward into conv-relu-pool layer
    out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # Forward into hidden affine-relu layer
    out_hidden, cache_hidden = affine_relu_forward(out_conv, W2, b2)
    # Forward into output affine layer
    scores, cache_scores = affine_forward(out_hidden, W3, b3)

    if y is None:
      return scores
    
    loss, grads = 0, {}
    

    ## BACKWARD PASS
    
    # Compute loss and grad on the scores
    data_loss, dscores = softmax_loss(scores, y)

    # Add regularization loss (penalize large weights)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = data_loss + reg_loss
    
    # Backpropagate into the affine layer
    daffine, dw3, db3 = affine_backward(dscores, cache_scores)
    dw3 += self.reg * W3

    # Backpropagate the hidden layer
    dhidden, dw2, db2 = affine_relu_backward(daffine, cache_hidden)
    dw2 += self.reg * W2

    # Backpropagate the conv layer
    dconv, dw1, db1 = conv_relu_pool_backward(dhidden, cache_conv)
    dw1 += self.reg * W1

    # Store grads in grads
    grads.update({'W1': dw1,
                  'b1': db1,
                  'W2': dw2,
                  'b2': db2,
                  'W3': dw3,
                  'b3': db3})
    
    return loss, grads
  





















  

class ConvNetDrop(object):
  """
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  """

  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-1, reg=0.0,
               dtype=np.float32, dropout=0.5, seed=123):
    """
    Initialize a new network, using dropout.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    - dropout: probability for a unit of being droped
    - seed: value used to initialize randomization for dropout
    """

    # Store network parameters
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    stride = 1
    pad = (filter_size - 1) / 2
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size

    # conv layer output volume dimensions
    H2 = (1 + (H + 2 * pad - HH) / stride) / 2
    W2 = (1 + (W + 2 * pad - WW) / stride) / 2


    # Weights initialization for convolutionnal layer
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, HH, WW))
    self.params['b1'] = np.zeros(num_filters)

    # Weights initialization for hidden affine layer
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(H2*W2*num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    # Weights initialization for output affine layer
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    

    # dropout params
    self.dropout_param = {'mode': 'train', 'p': dropout, 'seed': seed}

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """

    # Get mode for dropout
    mode = 'test' if y is None else 'train'
    self.dropout_param['mode'] = mode

    # Get weights and bias
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # initialize scores
    scores = None

    ## FORWARD PASS
    
    # Forward into conv-relu layer
    out_conv, cache_conv = conv_relu_forward(X, W1, b1, conv_param)
    # Forward into first dropout layer
    out_drop1, cache_drop1 = dropout_forward(out_conv, self.dropout_param)
    # Forward into pool layer
    out_pool, cache_pool = max_pool_forward_fast(out_drop1, pool_param)
    # Forward into hidden affine-relu layer
    out_hidden, cache_hidden = affine_relu_forward(out_pool, W2, b2)
    # Forward into the second dropout layer
    out_drop2, cache_drop2 = dropout_forward(out_hidden, self.dropout_param)
    # Forward into output affine layer
    scores, cache_scores = affine_forward(out_drop2, W3, b3)

    if y is None:
      return scores
    
    loss, grads = 0, {}
    

    ## BACKWARD PASS
    
    # Compute loss and grad on the scores
    data_loss, dscores = softmax_loss(scores, y)

       
    # Backpropagate into the affine layer
    daffine, dw3, db3 = affine_backward(dscores, cache_scores)
    # Backpropagate into the second dropout layer
    ddrop2 = dropout_backward(daffine, cache_drop2)
    # Backpropagate the hidden layer
    dhidden, dw2, db2 = affine_relu_backward(ddrop2, cache_hidden)    
    # Backpropagate into the pool layer
    dpool = max_pool_backward_fast(dhidden, cache_pool)
    # Backpropagate inro the first dropout layer
    ddrop1 = dropout_backward(dpool, cache_drop1)
    # Backpropagate the conv-relu layer
    dconv, dw1, db1 = conv_relu_backward(ddrop1, cache_conv)
    
    
    # Add regularization loss (penalize large weights)
    dw3 += self.reg * W3
    dw2 += self.reg * W2
    dw1 += self.reg * W1

    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = data_loss + reg_loss


    # Store grads in grads
    grads.update({'W1': dw1,
                  'b1': db1,
                  'W2': dw2,
                  'b2': db2,
                  'W3': dw3,
                  'b3': db3})
    
    return loss, grads