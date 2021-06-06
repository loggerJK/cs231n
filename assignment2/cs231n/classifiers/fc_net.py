from cs231n.layers import affine_forward, relu_forward, softmax_loss
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

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.b
        """
        self.params = {}
        self.reg = reg
        W1 = np.random.normal(
            loc=0, scale=weight_scale, size=(input_dim, hidden_dim)
        )  # (input_dim, hidden_dim)
        b1 = np.zeros(shape=hidden_dim)
        W2 = np.random.normal(
            loc=0, scale=weight_scale, size=(hidden_dim, num_classes)
        )  # (hidden_dim, num_classes)
        b2 = np.zeros(shape=num_classes)

        self.params["W1"] = W1
        self.params["W2"] = W2
        self.params["b1"] = b1
        self.params["b2"] = b2

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        pass
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
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        fc_1, cache_fc_1 = affine_forward(X, W1, b1)  # (N, H)
        relu_1, cache_relu_1 = relu_forward(fc_1)  # (N, H)
        fc_2, cache_fc_2 = affine_forward(relu_1, W2, b2)  # (N, C)
        import copy
        scores = copy.deepcopy(fc_2)

        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, d_scores = softmax_loss(scores, y)

        d_relu_1, d_W2, d_b2 = affine_backward(d_scores, cache_fc_2)
        d_fc_1 = relu_backward(d_relu_1, cache_relu_1)
        dx, d_W1, d_b1 = affine_backward(d_fc_1, cache_fc_1)

        grads['W1'] = d_W1
        grads['W2'] = d_W2
        grads['b1'] = d_b1
        grads['b2'] = d_b2

        loss += 0.5 * self.reg * \
            (np.sum(np.square(self.params['W1'])) +
             np.sum(np.square(self.params['W2'])))

        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
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
        pass
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

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=0,
        use_batchnorm=False,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

# weight and bias initialization
        for i in range(1, self.num_layers):  # 1 , L-1
            if i == 1:
                W = np.random.normal(loc=0, scale=weight_scale,
                                     size=(input_dim, hidden_dims[i-1]))
            else:
                W = np.random.normal(loc=0, scale=weight_scale, size=(
                    hidden_dims[i-2], hidden_dims[i-1]))
            b = np.zeros(shape=hidden_dims[i-1])
            self.params['W' + str(i)] = W
            self.params['b' + str(i)] = b

        self.params['W' + str(self.num_layers)] = np.random.normal(
            loc=0, scale=weight_scale, size=(hidden_dims[self.num_layers-2], num_classes))
        self.params['b' + str(self.num_layers)
                    ] = np.zeros(shape=num_classes)

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
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"}
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        fc = []
        relu = []
        cache_fc = []
        cache_relu = []
        fc.append(0)
        relu.append(X)
        cache_fc.append(0)
        cache_relu.append(0)                       # 맨 처음 trian data X를 집어넣어준다
        # 0으로 모든 리스트를 초기화해준다

        for i in range(1, self.num_layers):  # 1부터 L-1까지
            fc_i, cache_fc_i = affine_forward(
                relu[i-1], self.params['W' + str(i)], self.params['b' + str(i)])
            fc.append(fc_i)
            cache_fc.append(cache_fc_i)
            relu_i, cache_relu_i = relu_forward(fc[i])
            relu.append(relu_i)
            cache_relu.append(cache_relu_i)

        # 마지막 L번째 layer : affine & softmax
        fc_L, cache_fc_L = affine_forward(
            relu[-1], self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        fc.append(fc_L)
        cache_fc.append(cache_fc_L)

        # (N,C)

        scores = fc[self.num_layers]
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
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0,  {}
        loss, d_scores = softmax_loss(scores, y)

        dreluL, dWL, dbL = affine_backward(
            d_scores, cache_fc[self.num_layers])

        dfc = []
        drelu = []
        # 맨 마지막 Layer
        dfc.append(d_scores)
        drelu.append(dreluL)
        grads['W' + str(self.num_layers)] = dWL
        grads['b' + str(self.num_layers)] = dbL

        for i in range(self.num_layers-1, 0, -1):  # N-1, 1
            d_fc = relu_backward(drelu[-1], cache_relu[i])
            dfc.append(d_fc)
            dx, dw, db = affine_backward(
                dfc[-1], cache_fc[i])
            drelu.append(dx)
            grads['W' + str(i)] = dw
            grads['b' + str(i)] = db
            # print('db = ', db)

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
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
