import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  """
  print("Affine forward")
  print(x.shape)
  print(w.shape)
  print(b.shape)
  """
  
  xr = x.reshape(x.shape[0], w.shape[0])
  out = xr.dot(w) + b


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M) (gradient sur les scores)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  
  dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
  db = np.sum(dout, axis=0)
  dx = dout.dot(w.T).reshape(x.shape)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  
  out = np.maximum(x,0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  
  dx = dout
  dx[cache < 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, {}
  if mode == 'train':
        
    sample_mean = np.mean(x,axis=0)
    sample_var = np.var(x,axis=0)

    num = x - sample_mean
    denom = np.sqrt(sample_var + eps)
    xnorm = num / denom
    out = gamma*xnorm + beta

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    cache['N'] = N
    cache['x'] = x
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['num'] = num
    cache['denom'] = denom
    cache['xnorm'] = xnorm
    cache['gamma'] = gamma
    cache['eps'] = eps

    

  elif mode == 'test':
    
    
    xnorm = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma*xnorm + beta

    
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache





def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  
  dgamma = np.sum(cache['xnorm'] * dout, 0)
  dbeta = np.sum(dout, 0)

  sample_mean = cache['sample_mean']
  gamma = cache['gamma']
  N = cache['N']
  num = cache['num']
  denom = cache['denom']
  x = cache['x']

  dnorm = dout * gamma
  dnorm2 = np.sum(-(num * dnorm) / np.square(denom), 0)
  dsqrt = dnorm2 / (2 * denom)
  dsample_var = (2 * x - 2 * sample_mean) * dsqrt

  dsample_mean = np.sum(-dnorm / denom, 0)

  dx = dnorm / denom + dsample_var / N + dsample_mean / N

  return dx, dgamma, dbeta





def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta




def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  #out = None

  if mode == 'train':
    
    mask = (np.random.rand(*x.shape) < (1-p)) / (1-p)
    out = x*mask
    
  elif mode == 'test':
    
    out = x
   

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache




def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    
    dx = dout*mask
    
  elif mode == 'test':
    dx = dout
  return dx





def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  S = conv_param['stride']
  P = conv_param['pad']

  # Number of possible positions vertically
  Hp = 1 + (H + 2 * P - HH) / S

  # Number of possible positions horizontally
  Wp = 1 + (W + 2 * P - WW) / S
  
  # Output initialization
  out = np.zeros((N, F, Hp, Wp))
  xpad = np.pad(x, [(0,0), (0,0), (P,P), (P,P)], 'constant')

  for n in xrange(N): # for each image
      for f in xrange(F): # for each filter
        for i in xrange(Hp): # for each vertical possible position
          ix = i*S
          for j in xrange(Wp): # for each horizontal possible position
            jx = j*S

            window = xpad[n, :, ix:ix+HH, jx:jx+WW] # windows of the input on which the filter will be applied
            out[n,f,i,j] =  np.sum(window * w[f]) + b[f]


  cache = (x, w, b, conv_param)
  return out, cache






def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for n in xrange(N): # ith example
    for f in xrange(F): # jth filter
      # Convolve this filter over windows
      for k in xrange(Hp):
        hs = k * stride
        for l in xrange(Wp):
          ws = l * stride

          # Window we applies the respective jth filter over (C, HH, WW)
          window = padded[n, :, hs:hs+HH, ws:ws+WW]

          # Compute gradient of out[n, f, k, l] = np.sum(window*w[f]) + b[f]
          db[f] += dout[n, f, k, l]
          dw[f] += window*dout[n, f, k, l]
          padded_dx[n, :, hs:hs+HH, ws:ws+WW] += w[f] * dout[n, f, k, l]

  # "Unpad"
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]

  return dx, dw, db




def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # Get data dimemsions
  N, C, H, W = x.shape

  # Get pool params
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']


  # Nombre de positions possibles en hauteur
  Hp = (H - pool_height)/stride + 1

  # Nombre de positions possibles en largeur
  Wp = (W - pool_width)/stride + 1
  
  # Initialisation de l'output
  out = np.zeros((N, C, Hp, Wp))

  for n in xrange(N): # for each image
      for c in xrange(C): # for each filter
        for i in xrange(Hp): # for each vertical possible position
          ix = i*stride
          for j in xrange(Wp): # for each horizontal possible position
            jx = j*stride

            window = x[n, c, ix:ix+pool_height, jx:jx+pool_width]
            out[n,c,i,j] =  np.amax(window)


  cache = (x, pool_param)
  return out, cache




def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  
  # Get cache
  x, pool_param = cache
  # Get data dimemsions
  N, C, H, W = x.shape

  # Get pool params
  stride = pool_param['stride']
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']

  dx = np.zeros_like(x)

  # Nombre de positions possibles en hauteur
  Hp = (H - pool_height)/stride + 1

  # Nombre de positions possibles en largeur
  Wp = (W - pool_width)/stride + 1

  for n in xrange(N): # for each image
      for c in xrange(C): # for each filter
        for i in xrange(Hp): # for each vertical possible position
          ix = i*stride
          for j in xrange(Wp): # for each horizontal possible position
            jx = j*stride

            window = x[n, c, ix:ix+pool_height, jx:jx+pool_width]
            m = np.max(window)
            dx[n,c,ix:ix+pool_height,jx:jx+pool_width] += (window == m) * dout[n, c, i, j]

  return dx





def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  N, C, H, W = x.shape
  x_reshaped = x.transpose(0,2,3,1).reshape(N*H*W, C)
  out_tmp, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
  out = out_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)


  return out, cache





def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape
  dout_reshaped = dout.transpose(0,2,3,1).reshape(N*H*W, C)
  dx_tmp, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
  dx = dx_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  return dx, dgamma, dbeta



  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
