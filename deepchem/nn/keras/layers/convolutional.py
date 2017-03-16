from deepchem.nn.keras.utils.np_utils import conv_output_length
from deepchem.nn.keras import activations, initializations, regularizers, constraints
from deepchem.nn.keras.engine import Layer, InputSpec
from deepchem.nn.keras import backend as K

class Convolution1D(Layer):
  '''Convolution operator for filtering neighborhoods of one-dimensional inputs.
  When using this layer as the first layer in a model,
  either provide the keyword argument `input_dim`
  (int, e.g. 128 for sequences of 128-dimensional vectors),
  or `input_shape` (tuple of integers, e.g. (10, 128) for sequences
  of 10 vectors of 128-dimensional vectors).

  # Example

  ```python
      # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
      # with 64 output filters
      model = Sequential()
      model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
      # now model.output_shape == (None, 10, 64)

      # add a new conv1d on top
      model.add(Convolution1D(32, 3, border_mode='same'))
      # now model.output_shape == (None, 10, 32)
  ```

  # Arguments
      nb_filter: Number of convolution kernels to use
          (dimensionality of the output).
      filter_length: The extension (spatial or temporal) of each filter.
      init: name of initialization function for the weights of the layer
          (see [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      activation: name of activation function to use
          (see [activations](../activations.md)),
          or alternatively, elementwise Theano function.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: a(x) = x).
      weights: list of numpy arrays to set as initial weights.
      border_mode: 'valid' or 'same'.
      subsample_length: factor by which to subsample output.
      W_regularizer: instance of [WeightRegularizer](../regularizers.md)
          (eg. L1 or L2 regularization), applied to the main weights matrix.
      b_regularizer: instance of [WeightRegularizer](../regularizers.md),
          applied to the bias.
      activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
          applied to the network output.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the main weights matrix.
      b_constraint: instance of the [constraints](../constraints.md) module,
          applied to the bias.
      bias: whether to include a bias
          (i.e. make the layer affine rather than linear).
      input_dim: Number of channels/dimensions in the input.
          Either this argument or the keyword argument `input_shape`must be
          provided when using this layer as the first layer in a model.
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).

  # Input shape
      3D tensor with shape: `(samples, steps, input_dim)`.

  # Output shape
      3D tensor with shape: `(samples, new_steps, nb_filter)`.
      `steps` value might have changed due to padding.
  '''
  def __init__(self, nb_filter, filter_length,
               init='uniform', activation='linear', weights=None,
               border_mode='valid', subsample_length=1,
               W_regularizer=None, b_regularizer=None, activity_regularizer=None,
               W_constraint=None, b_constraint=None,
               bias=True, input_dim=None, input_length=None, **kwargs):

    if border_mode not in {'valid', 'same'}:
      raise Exception('Invalid border mode for Convolution1D:', border_mode)
    self.nb_filter = nb_filter
    self.filter_length = filter_length
    self.init = initializations.get(init, dim_ordering='th')
    self.activation = activations.get(activation)
    assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
    self.border_mode = border_mode
    self.subsample_length = subsample_length

    self.subsample = (subsample_length, 1)

    self.W_regularizer = regularizers.get(W_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    self.input_spec = [InputSpec(ndim=3)]
    self.initial_weights = weights
    self.input_dim = input_dim
    self.input_length = input_length
    if self.input_dim:
      kwargs['input_shape'] = (self.input_length, self.input_dim)
    super(Convolution1D, self).__init__(**kwargs)

  def build(self, input_shape):
    input_dim = input_shape[2]
    self.W_shape = (self.filter_length, 1, input_dim, self.nb_filter)
    self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
    if self.bias:
      self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
      self.trainable_weights = [self.W, self.b]
    else:
      self.trainable_weights = [self.W]
    self.regularizers = []

    if self.W_regularizer:
      self.W_regularizer.set_param(self.W)
      self.regularizers.append(self.W_regularizer)

    if self.bias and self.b_regularizer:
      self.b_regularizer.set_param(self.b)
      self.regularizers.append(self.b_regularizer)

    if self.activity_regularizer:
      self.activity_regularizer.set_layer(self)
      self.regularizers.append(self.activity_regularizer)

    self.constraints = {}
    if self.W_constraint:
      self.constraints[self.W] = self.W_constraint
    if self.bias and self.b_constraint:
      self.constraints[self.b] = self.b_constraint

    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights

  def get_output_shape_for(self, input_shape):
    length = conv_output_length(input_shape[1],
                                self.filter_length,
                                self.border_mode,
                                self.subsample[0])
    return (input_shape[0], length, self.nb_filter)

  def call(self, x, mask=None):
    x = K.expand_dims(x, 2)  # add a dummy dimension
    output = K.conv2d(x, self.W, strides=self.subsample,
                      border_mode=self.border_mode,
                      dim_ordering='tf')
    output = K.squeeze(output, 2)  # remove the dummy dimension
    if self.bias:
      output += K.reshape(self.b, (1, 1, self.nb_filter))
    output = self.activation(output)
    return output

  def get_config(self):
    config = {'nb_filter': self.nb_filter,
              'filter_length': self.filter_length,
              'init': self.init.__name__,
              'activation': self.activation.__name__,
              'border_mode': self.border_mode,
              'subsample_length': self.subsample_length,
              'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
              'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
              'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
              'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
              'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
              'bias': self.bias,
              'input_dim': self.input_dim,
              'input_length': self.input_length}
    base_config = super(Convolution1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))