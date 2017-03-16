import warnings

from deepchem.nn.keras import backend as K


def to_list(x):
  '''This normalizes a list/tensor into a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.
  '''
  if type(x) is list:
    return x
  return [x]


class Layer(object):
  '''Abstract base layer class.

  # Properties
      name: string, must be unique within a model.
      input_spec: list of InputSpec class instances
          each entry describes one required input:
              - ndim
              - dtype
          A layer with `n` input tensors must have
          an `input_spec` of length `n`.
      trainable: boolean, whether the layer weights
          will be updated during training.
      uses_learning_phase: whether any operation
          of the layer uses `K.in_training_phase()`
          or `K.in_test_phase()`.
      input_shape: shape tuple. Provided for convenience,
          but note that there may be cases in which this
          attribute is ill-defined (e.g. a shared layer
          with multiple input shapes), in which case
          requesting `input_shape` will raise an Exception.
          Prefer using `layer.get_input_shape_for(input_shape)`,
          or `layer.get_input_shape_at(node_index)`.
      output_shape: shape tuple. See above.
      inbound_nodes: list of nodes.
      outbound_nodes: list of nodes.
      supports_masking: boolean
      input, output: input/output tensor(s). Note that if the layer is used
          more than once (shared layer), this is ill-defined
          and will raise an exception. In such cases, use
          `layer.get_input_at(node_index)`.
      input_mask, output_mask: same as above, for masks.

      trainable_weights: list of variables.
      non_trainable_weights: list of variables.
      regularizers: list of regularizers.
      constraints: dict mapping weights to constraints.

  # Methods
      call(x, mask=None): where the layer's logic lives.
      __call__(x, mask=None): wrapper around the layer logic (`call`).
          if x is a Keras tensor:
              - connect current layer with last layer from tensor:
                  `self.add_inbound_node(last_layer)`
              - add layer to tensor history
          if layer is not built:
              - build from x._keras_shape
      get_weights()
      set_weights(weights)
      get_config()
      count_params()
      get_output_shape_for(input_shape)
      compute_mask(x, mask)
      get_input_at(node_index)
      get_output_at(node_index)
      get_input_shape_at(node_index)
      get_output_shape_at(node_index)
      get_input_mask_at(node_index)
      get_output_mask_at(node_index)

  # Class Methods
      from_config(config)

  # Internal methods:
      build(input_shape)
      add_inbound_node(layer, index=0)
      create_input_layer()
      assert_input_compatibility()
  '''

  def __init__(self, **kwargs):
    # these properties should have been set
    # by the child class, as appropriate.
    if not hasattr(self, 'input_spec'):
      self.input_spec = None
    if not hasattr(self, 'supports_masking'):
      self.supports_masking = False
    if not hasattr(self, 'uses_learning_phase'):
      self.uses_learning_phase = False

    # these lists will be filled via successive calls
    # to self.add_inbound_node()
    self.inbound_nodes = []
    self.outbound_nodes = []

    # these properties will be set upon call of self.build(),
    # which itself will be called upon self.add_inbound_node if necessary.
    if not hasattr(self, 'trainable_weights'):
      self.trainable_weights = []
    if not hasattr(self, 'non_trainable_weights'):
      self.non_trainable_weights = []
    if not hasattr(self, 'regularizers'):
      self.regularizers = []
    if not hasattr(self, 'constraints'):
      self.constraints = {}  # dict {tensor: constraint instance}
    self.built = False

    # these properties should be set by the user via keyword arguments.
    # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {'input_shape',
                      'batch_input_shape',
                      'input_dtype',
                      'name',
                      'trainable',
                      'create_input_layer'}
    for kwarg in kwargs.keys():
      assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg

    name = kwargs.get('name')
    if not name:
      prefix = self.__class__.__name__.lower()
      name = prefix + '_' + str(K.get_uid(prefix))
    self.name = name

    self.trainable = kwargs.get('trainable', True)
    if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
      # in this case we will create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        batch_input_shape = (None,) + tuple(kwargs['input_shape'])
      self.batch_input_shape = batch_input_shape
      input_dtype = kwargs.get('input_dtype', K.floatx())
      self.input_dtype = input_dtype
      if 'create_input_layer' in kwargs:
        self.create_input_layer(batch_input_shape, input_dtype)

  @property
  def trainable_weights(self):
    trainable = getattr(self, 'trainable', True)
    if trainable:
      return self._trainable_weights
    else:
      return []

  @trainable_weights.setter
  def trainable_weights(self, weights):
    self._trainable_weights = weights

  @property
  def non_trainable_weights(self):
    trainable = getattr(self, 'trainable', True)
    if not trainable:
      return self._trainable_weights + self._non_trainable_weights
    else:
      return self._non_trainable_weights

  @non_trainable_weights.setter
  def non_trainable_weights(self, weights):
    self._non_trainable_weights = weights

  def create_input_layer(self, batch_input_shape,
                         input_dtype=None, name=None):
    if not name:
      prefix = self.__class__.__name__.lower() + '_input_'
      name = prefix + str(K.get_uid(prefix))
    if not input_dtype:
      input_dtype = K.floatx()

    self.batch_input_shape = batch_input_shape
    self.input_dtype = input_dtype

    # instantiate the input layer
    x = Input(batch_shape=batch_input_shape,
              dtype=input_dtype, name=name)
    # this will build the current layer
    # and create the node connecting the current layer
    # to the input layer we just created.
    self(x)

  def assert_input_compatibility(self, input):
    '''This checks that the tensor(s) `input`
    verify the input assumptions of the layer
    (if any). If not, exceptions are raised.
    '''
    if not self.input_spec:
      return True
    assert type(self.input_spec) is list, ('input_spec must be a list of ' +
                                           'InputSpec instances. Found: ' +
                                           str(self.input_spec))
    inputs = to_list(input)
    if len(self.input_spec) > 1:
      if len(inputs) != len(self.input_spec):
        raise Exception('Layer ' + self.name + ' expects ' +
                        str(len(self.input_spec)) + ' inputs, '
                                                    'but it received ' + str(len(inputs)) +
                        ' input tensors. Input received: ' +
                        str(input))
    for input_index, (x, spec) in enumerate(zip(inputs, self.input_spec)):
      if spec is None:
        continue

      # check ndim
      if spec.ndim is not None:
        if type(spec.ndim) is str:
          int_ndim = spec.ndim[:spec.ndim.find('+')]
          ndim = int(int_ndim)
          if K.ndim(x) < ndim:
            raise Exception('Input ' + str(input_index) +
                            ' is incompatible with layer ' +
                            self.name + ': expected ndim >= ' +
                            str(ndim) + ', found ndim=' +
                            str(K.ndim(x)))
        else:
          if K.ndim(x) != spec.ndim:
            raise Exception('Input ' + str(input_index) +
                            ' is incompatible with layer ' +
                            self.name + ': expected ndim=' +
                            str(spec.ndim) + ', found ndim=' +
                            str(K.ndim(x)))
      if spec.dtype is not None:
        if K.dtype(x) != spec.dtype:
          raise Exception('Input ' + str(input_index) +
                          ' is incompatible with layer ' +
                          self.name + ': expected dtype=' +
                          str(spec.dtype) + ', found dtype=' +
                          str(K.dtype(x)))
      if spec.shape is not None:
        if hasattr(x, '_keras_shape'):
          x_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
          # tensorflow shape inference
          x_shape = K.int_shape(x)
        else:
          continue
        for spec_dim, dim in zip(spec.shape, x_shape):
          if spec_dim is not None:
            if spec_dim != dim:
              raise Exception('Input ' + str(input_index) +
                              ' is incompatible with layer ' +
                              self.name + ': expected shape=' +
                              str(spec.shape) + ', found shape=' +
                              str(x_shape))

  def call(self, x, mask=None):
    '''This is where the layer's logic lives.

    # Arguments
        x: input tensor, or list/tuple of input tensors.
        mask: a masking tensor (or list of tensors). Used mainly in RNNs.

    # Returns:
        A tensor or list/tuple of tensors.
    '''
    return x

  def __call__(self, x, mask=None):
    '''Wrapper around self.call(), for handling
    internal Keras references.

    If a Keras tensor is passed:
        - we call self.add_inbound_node()
        - if necessary, we `build` the layer to match
            the _keras_shape of the input(s)
        - we update the _keras_shape of every input tensor with
            its new shape (obtained via self.get_output_shape_for).
            This is done as part of add_inbound_node().
        - we update the _keras_history of the output tensor(s)
            with the current layer.
            This is done as part of add_inbound_node().

    # Arguments
        x: can be a tensor or list/tuple of tensors.
        mask: tensor or list/tuple of tensors.
    '''
    if not self.built:
      # raise exceptions in case the input is not compatible
      # with the input_spec specified in the layer constructor
      self.assert_input_compatibility(x)

      # collect input shapes to build layer
      input_shapes = []
      for x_elem in to_list(x):
        if hasattr(x_elem, '_keras_shape'):
          input_shapes.append(x_elem._keras_shape)
        elif hasattr(K, 'int_shape'):
          input_shapes.append(K.int_shape(x_elem))
        else:
          raise Exception('You tried to call layer "' + self.name +
                          '". This layer has no information'
                          ' about its expected input shape, '
                          'and thus cannot be built. '
                          'You can build it manually via: '
                          '`layer.build(batch_input_shape)`')
      if len(input_shapes) == 1:
        self.build(input_shapes[0])
      else:
        self.build(input_shapes)
      self.built = True

    # raise exceptions in case the input is not compatible
    # with the input_spec set at build time
    self.assert_input_compatibility(x)
    # build and connect layer
    input_added = False
    input_tensors = to_list(x)

    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for input_tensor in input_tensors:
      if hasattr(input_tensor, '_keras_history') and input_tensor._keras_history:
        # this is a Keras tensor
        previous_layer, node_index, tensor_index = input_tensor._keras_history
        inbound_layers.append(previous_layer)
        node_indices.append(node_index)
        tensor_indices.append(tensor_index)
      else:
        inbound_layers = None
        break
    if inbound_layers:
      # this will call layer.build() if necessary
      self.add_inbound_node(inbound_layers, node_indices, tensor_indices)
      input_added = True

    # get the output tensor to be returned
    if input_added:
      # output was already computed when calling self.add_inbound_node
      outputs = self.inbound_nodes[-1].output_tensors
      # if single output tensor: return it,
      # else return a list (at least 2 elements)
      if len(outputs) == 1:
        return outputs[0]
      else:
        return outputs
    else:
      # this case appears if the input was not a Keras tensor
      return self.call(x, mask)

  def add_inbound_node(self, inbound_layers,
                       node_indices=None, tensor_indices=None):
    '''
    # Arguments:
        inbound_layers: can be a layer instance
            or a list/tuple of layer instances.
        node_indices: integer (or list of integers).
            The input layer might have a number of
            parallel output streams;
            this is the index of the stream (in the input layer)
            where to connect the current layer.
        tensor_indices: integer or list of integers.
            The output of the inbound node might be a list/tuple
            of tensor, and we might only be interested in one specific entry.
            This index allows you to specify the index of the entry in the output list
            (if applicable). "None" means that we take all outputs (as a list).
    '''
    inbound_layers = to_list(inbound_layers)
    if not node_indices:
      node_indices = [0 for _ in range(len(inbound_layers))]
    else:
      node_indices = to_list(node_indices)
      assert len(node_indices) == len(inbound_layers)
    if not tensor_indices:
      tensor_indices = [0 for _ in range(len(inbound_layers))]
    else:
      tensor_indices = to_list(tensor_indices)

    if not self.built:
      # collect input_shapes for call to build()
      input_shapes = []
      for layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
        input_shapes.append(layer.inbound_nodes[node_index].output_shapes[tensor_index])
      # call build()
      if len(input_shapes) == 1:
        self.build(input_shape=input_shapes[0])
      else:
        self.build(input_shape=input_shapes)
      self.built = True
    # creating the node automatically updates self.inbound_nodes
    # as well as outbound_nodes on inbound layers.
    Node.create_node(self, inbound_layers, node_indices, tensor_indices)

  def get_output_shape_for(self, input_shape):
    '''Computes the output shape of the layer given
    an input shape (assumes that the layer will be built
    to match that input shape).

    # Arguments
        input_shape: shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.
    '''
    return input_shape

  def compute_mask(self, input, input_mask=None):
    '''Computes an output masking tensor, given an input tensor
    (or list thereof) and an input mask (or list thereof).

    # Arguments
        input: tensor or list of tensors.
        input_mask: tensor or list of tensors.

    # Returns
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    '''
    if not hasattr(self, 'supports_masking') or not self.supports_masking:
      if input_mask is not None:
        if type(input_mask) is list:
          if any(input_mask):
            raise Exception('Layer ' + self.name + ' does not support masking, ' +
                            'but was passed an input_mask: ' + str(input_mask))
        else:
          raise Exception('Layer ' + self.name + ' does not support masking, ' +
                          'but was passed an input_mask: ' + str(input_mask))
      # masking not explicitly supported: return None as mask
      return None
    # if masking is explictly supported, by default
    # carry over the input mask
    return input_mask

  def build(self, input_shape):
    '''Creates the layer weights.
    Must be implemented on all layers that have weights.

    # Arguments
        input_shape: Keras tensor (future input to layer)
            or list/tuple of Keras tensors to reference
            for weight shape computations.
    '''
    self.built = True

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    '''Retrieves an attribute (e.g. input_tensors) from a node.

    # Arguments
        node_index: integer index of the node from which
            to retrieve the attribute
        attr: exact node attribute name
        attr_name: human-readable attribute name, for error messages
    '''
    if not self.inbound_nodes:
      raise Exception('The layer has never been called ' +
                      'and thus has no defined ' + attr_name + '.')
    if not len(self.inbound_nodes) > node_index:
      raise Exception('Asked to get ' + attr_name +
                      ' at node ' + str(node_index) +
                      ', but the layer has only ' +
                      str(len(self.inbound_nodes)) + ' inbound nodes.')
    values = getattr(self.inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_shape_at(self, node_index):
    '''Retrieves the input shape(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    '''Retrieves the output shape(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    '''Retrieves the input tensor(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    '''Retrieves the output tensor(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'output_tensors',
                                             'output')

  def get_input_mask_at(self, node_index):
    '''Retrieves the input mask tensor(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'input_masks',
                                             'input mask')

  def get_output_mask_at(self, node_index):
    '''Retrieves the output mask tensor(s) of a layer at a given node.
    '''
    return self._get_node_attribute_at_index(node_index,
                                             'output_masks',
                                             'output mask')

  @property
  def input(self):
    '''Retrieves the input tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    '''
    if len(self.inbound_nodes) > 1:
      raise Exception('Layer ' + self.name +
                      ' has multiple inbound nodes, ' +
                      'hence the notion of "layer input" '
                      'is ill-defined. '
                      'Use `get_input_at(node_index)` instead.')
    elif not self.inbound_nodes:
      raise Exception('Layer ' + self.name +
                      ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors',
                                             'input')

  def set_input(self, input_tensor, shape=None):
    if len(self.inbound_nodes) > 1:
      raise Exception('Cannot `set_input` for layer ' + self.name +
                      ' because it has more than one inbound connection.')
    if len(self.inbound_nodes) == 1:
      # check that the inbound node is an Input node
      if self.inbound_nodes[0].inbound_layers:
        warnings.warn('You are manually setting the input for layer ' +
                      self.name + ' but it is not an Input layer. '
                                  'This will cause part of your model '
                                  'to be disconnected.')
    if self.outbound_nodes:
      warnings.warn('You are manually setting the input for layer ' +
                    self.name + ' but it has ' +
                    str(len(self.outbound_nodes)) +
                    ' outbound layers. '
                    'This will cause part of your model '
                    'to be disconnected.')
    if hasattr(K, 'int_shape'):
      # auto-infered shape takes priority
      shape = K.int_shape(input_tensor)
    elif not shape:
      raise Exception('`set_input` needs to know the shape '
                      'of the `input_tensor` it receives, but '
                      'Keras was not able to infer it automatically.'
                      ' Specify it via: '
                      '`model.set_input(input_tensor, shape)`')
    # reset layer connections
    self.inbound_nodes = []
    self.outbound_nodes = []
    input_shape = tuple(shape)
    self.build(input_shape=input_shape)

    # set Keras tensor metadata
    input_tensor._uses_learning_phase = False
    input_tensor._keras_history = (None, 0, 0)
    input_tensor._keras_shape = input_shape

    output_tensors = to_list(self.call(input_tensor))
    output_shapes = to_list(self.get_output_shape_for(input_shape))
    output_masks = to_list(self.compute_mask(input_tensor, None))

    for i, output_tensor in enumerate(output_tensors):
      output_tensor._keras_history = (self, 0, i)
      output_tensor._keras_shape = output_shapes[i]
      output_tensor._uses_learning_phase = self.uses_learning_phase

    # create node
    Node(self,
         inbound_layers=[],
         node_indices=[],
         tensor_indices=[],
         input_tensors=[input_tensor],
         output_tensors=output_tensors,
         input_masks=[None],
         output_masks=output_masks,
         input_shapes=[input_shape],
         output_shapes=output_shapes)

  @property
  def output(self):
    '''Retrieves the output tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    '''
    if len(self.inbound_nodes) != 1:
      raise Exception('Layer ' + self.name +
                      ' has multiple inbound nodes, ' +
                      'hence the notion of "layer output" '
                      'is ill-defined. '
                      'Use `get_output_at(node_index)` instead.')
    return self._get_node_attribute_at_index(0, 'output_tensors',
                                             'output')

  @property
  def input_mask(self):
    '''Retrieves the input mask tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    '''
    if len(self.inbound_nodes) != 1:
      raise Exception('Layer ' + self.name +
                      ' has multiple inbound nodes, ' +
                      'hence the notion of "layer input mask" '
                      'is ill-defined. '
                      'Use `get_input_mask_at(node_index)` instead.')
    return self._get_node_attribute_at_index(0, 'input_masks',
                                             'input mask')

  @property
  def output_mask(self):
    '''Retrieves the output mask tensor(s) of a layer (only applicable if
    the layer has exactly one inbound node, i.e. if it is connected
    to one incoming layer).
    '''
    if len(self.inbound_nodes) != 1:
      raise Exception('Layer ' + self.name +
                      ' has multiple inbound nodes, ' +
                      'hence the notion of "layer output mask" '
                      'is ill-defined. '
                      'Use `get_output_mask_at(node_index)` instead.')
    return self._get_node_attribute_at_index(0, 'output_masks',
                                             'output mask')

  @property
  def input_shape(self):
    '''Retrieves the input shape tuple(s) of a layer. Only applicable
    if the layer has one inbound node,
    or if all inbound nodes have the same input shape.
    '''
    if not self.inbound_nodes:
      raise Exception('The layer has never been called ' +
                      'and thus has no defined input shape.')
    all_input_shapes = set([str(node.input_shapes) for node in self.inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self.inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return input_shapes[0]
      else:
        return input_shapes
    else:
      raise Exception('The layer "' + str(self.name) +
                      ' has multiple inbound nodes, ' +
                      'with different input shapes. Hence ' +
                      'the notion of "input shape" is ' +
                      'ill-defined for the layer. ' +
                      'Use `get_input_shape_at(node_index)` instead.')

  @property
  def output_shape(self):
    '''Retrieves the output shape tuple(s) of a layer. Only applicable
    if the layer has one inbound node,
    or if all inbound nodes have the same output shape.
    '''
    if not self.inbound_nodes:
      raise Exception('The layer has never been called ' +
                      'and thus has no defined output shape.')
    all_output_shapes = set([str(node.output_shapes) for node in self.inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self.inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return output_shapes[0]
      else:
        return output_shapes
    else:
      raise Exception('The layer "' + str(self.name) +
                      ' has multiple inbound nodes, ' +
                      'with different output shapes. Hence ' +
                      'the notion of "output shape" is ' +
                      'ill-defined for the layer. ' +
                      'Use `get_output_shape_at(node_index)` instead.')

  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights

  def set_weights(self, weights):
    '''Sets the weights of the layer, from Numpy arrays.

    # Arguments
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).
    '''
    params = self.weights
    if len(params) != len(weights):
      raise Exception('You called `set_weights(weights)` on layer "' + self.name +
                      '" with a  weight list of length ' + str(len(weights)) +
                      ', but the layer was expecting ' + str(len(params)) +
                      ' weights. Provided weights: ' + str(weights)[:50] + '...')
    if not params:
      return
    weight_value_tuples = []
    param_values = K.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise Exception('Layer weight shape ' +
                        str(pv.shape) +
                        ' not compatible with '
                        'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    K.batch_set_value(weight_value_tuples)

  def get_weights(self):
    '''Returns the current weights of the layer,
    as a list of numpy arrays.
    '''
    params = self.weights
    return K.batch_get_value(params)

  def get_config(self):
    '''Returns a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by Container (one layer of abstraction above).
    '''
    config = {'name': self.name,
              'trainable': self.trainable}
    if hasattr(self, 'batch_input_shape'):
      config['batch_input_shape'] = self.batch_input_shape
    if hasattr(self, 'input_dtype'):
      config['input_dtype'] = self.input_dtype
    return config

  @classmethod
  def from_config(cls, config):
    '''This method is the reverse of get_config,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Container), nor weights (handled by `set_weights`).

    # Arguments
        config: a Python dictionary, typically the
            output of get_config.
    '''
    return cls(**config)

  def count_params(self):
    '''Returns the total number of floats (or ints)
    composing the weights of the layer.
    '''
    if not self.built:
      if self.__class__.__name__ in {'Sequential', 'Graph'}:
        self.build()
      else:
        raise Exception('You tried to call `count_params` on ' +
                        self.name + ', but the layer isn\'t built. '
                                    'You can build it manually via: `' +
                        self.name + '.build(batch_input_shape)`.')
    return sum([K.count_params(p) for p in self.trainable_weights])


def Input(shape=None, batch_shape=None,
          name=None, dtype=K.floatx(), sparse=False,
          tensor=None):
  '''`Input()` is used to instantiate a Keras tensor.
  A Keras tensor is a tensor object from the underlying backend
  (Theano or TensorFlow), which we augment with certain
  attributes that allow us to build a Keras model
  just by knowing the inputs and outputs of the model.

  For instance, if a, b and c and Keras tensors,
  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  The added Keras attributes are:
      ._keras_shape: integer shape tuple propagated
          via Keras-side shape inference.
      ._keras_history: last layer applied to the tensor.
          the entire layer graph is retrievable from that layer,
          recursively.

  # Arguments
      shape: a shape tuple (integer), not including the batch size.
          For instance, `shape=(32,)` indicates that the expected input
          will be batches of 32-dimensional vectors.
      batch_shape: a shape tuple (integer), including the batch size.
          For instance, `batch_shape=(10, 32)` indicates that
          the expected input will be batches of 10 32-dimensional vectors.
          `batch_shape=(None, 32)` indicates batches of an arbitrary number
          of 32-dimensional vectors.
      name: An optional name string for the layer.
          Should be unique in a model (do not reuse the same name twice).
          It will be autogenerated if it isn't provided.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
      sparse: a boolean specifying whether this will be a sparse tensor

  # Example usage

      ```python
      # this is a logistic regression in Keras
      a = Input(shape=(32,))
      b = Dense(16, activation='softmax')(a)
      model = Model(input=a, output=b)
      ```
  '''
  if not batch_shape and tensor is None:
    assert shape, ('Please provide to Input either a `shape`' +
                   ' or a `batch_shape` argument. Note that ' +
                   '`shape` does not include the batch '
                   'dimension.')
  if shape and not batch_shape:
    batch_shape = (None,) + tuple(shape)
  input_layer = InputLayer(batch_input_shape=batch_shape,
                           name=name, input_dtype=dtype,
                           sparse=sparse,
                           input_tensor=tensor)
  # return tensor including _keras_shape and _keras_history
  # note that in this case train_output and test_output are the same pointer.
  outputs = input_layer.inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs


class Node(object):
  '''A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer.inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer.outbound_nodes`.

  # Attributes
      outbound_layer: the layer that takes
          `input_tensors` and turns them into `output_tensors`.
      inbound_layers: a list of layers, the same length as `input_tensors`,
          the layers from where `input_tensors` originate.
      node_indices: a list of integers, the same length as `inbound_layers`.
          `node_indices[i]` is the origin node of `input_tensors[i]`
          (necessary since each inbound layer might have several nodes,
          e.g. if the layer is being shared with a different data stream).
      tensor_indices: a list of integers, the same length as `inbound_layers`.
          `tensor_indices[i]` is the index of `input_tensors[i]` within the
          output of the inbound layer (necessary since each inbound layer might
          have multiple tensor outputs, with each one being
          independently manipulable).
      input_tensors: list of input tensors.
      output_tensors: list of output tensors.
      input_masks: list of input masks (a mask can be a tensor, or None).
      output_masks: list of output masks (a mask can be a tensor, or None).
      input_shapes: list of input shape tuples.
      output_shapes: list of output shape tuples.

  `node_indices` and `tensor_indices` are basically fine-grained coordinates
  describing the origin of the `input_tensors`, verifying the following:

  `input_tensors[i] == inbound_layers[i].inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

  A node from layer A to layer B is added to:
      A.outbound_nodes
      B.inbound_nodes
  '''

  def __init__(self, outbound_layer,
               inbound_layers, node_indices, tensor_indices,
               input_tensors, output_tensors,
               input_masks, output_masks,
               input_shapes, output_shapes):
    # layer instance (NOT a list).
    # this is the layer that takes a list of input tensors
    # and turns them into a list of output tensors.
    # the current node will be added to the inbound_nodes of outbound_layer
    self.outbound_layer = outbound_layer

    # the following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.
    self.inbound_layers = inbound_layers  # list of layer instances
    self.node_indices = node_indices  # list of integers, 1:1 mapping with inbound_layers
    self.tensor_indices = tensor_indices  # list of integers, 1:1 mapping with inbound_layers

    # tensor inputs and outputs of outbound_layer
    self.input_tensors = input_tensors  # list of tensors. 1:1 mapping with inbound_layers
    self.output_tensors = output_tensors  # list of tensors, created by outbound_layer.call()

    # input and output masks
    self.input_masks = input_masks  # list of tensors, 1:1 mapping with input_tensor
    self.output_masks = output_masks  # list of tensors, created by outbound_layer.compute_mask()

    # input and output shapes
    self.input_shapes = input_shapes  # list of shape tuples, shapes of input_tensors
    self.output_shapes = output_shapes  # list of shape tuples, shapes of output_tensors

    # add nodes to all layers involved.
    for layer in inbound_layers:
      if layer is not None:
        layer.outbound_nodes.append(self)
    outbound_layer.inbound_nodes.append(self)

  @classmethod
  def create_node(cls, outbound_layer,
                  inbound_layers, node_indices=None, tensor_indices=None):
    if not node_indices:
      node_indices = [0 for _ in range(len(inbound_layers))]
    else:
      assert len(node_indices) == len(inbound_layers)
    if not tensor_indices:
      tensor_indices = [0 for _ in range(len(inbound_layers))]

    input_tensors = []
    input_masks = []
    input_shapes = []

    for inbound_layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
      inbound_node = inbound_layer.inbound_nodes[node_index]
      input_tensors.append(inbound_node.output_tensors[tensor_index])
      input_masks.append(inbound_node.output_masks[tensor_index])
      input_shapes.append(inbound_node.output_shapes[tensor_index])

    assert len(input_shapes) == len(input_tensors) == len(input_masks)

    if len(input_tensors) == 1:
      output_tensors = to_list(outbound_layer.call(input_tensors[0], mask=input_masks[0]))
      output_masks = to_list(outbound_layer.compute_mask(input_tensors[0], input_masks[0]))
      # TODO: try to auto-infer shape if exception is raised by get_output_shape_for
      output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes[0]))
    else:
      output_tensors = to_list(outbound_layer.call(input_tensors, mask=input_masks))
      output_masks = to_list(outbound_layer.compute_mask(input_tensors, input_masks))
      output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes))

    if not output_tensors or output_tensors[0] is None:
      raise Exception('The `call` method of layer "' +
                      outbound_layer.name +
                      '" should return a tensor. Found: ' +
                      str(output_tensors[0]))
    if len(output_tensors) != len(output_shapes):
      raise Exception('The `get_output_shape_for` method of layer "' +
                      outbound_layer.name +
                      '"" should return one shape tuple per '
                      'output tensor of the layer. Found: ' +
                      str(output_shapes))
    if len(output_tensors) != len(output_masks):
      raise Exception('The `compute_mask` method of layer "' +
                      outbound_layer.name +
                      '" should return one mask tensor per '
                      'output tensor of the layer. Found: ' +
                      str(output_masks))

    for i in range(len(output_tensors)):
      output_tensors[i]._keras_shape = output_shapes[i]
      output_tensors[i]._uses_learning_phase = any(
        [x._uses_learning_phase for x in input_tensors]) or outbound_layer.uses_learning_phase
      output_tensors[i]._keras_history = (outbound_layer, len(outbound_layer.inbound_nodes), i)

    return cls(outbound_layer,
               inbound_layers, node_indices, tensor_indices,
               input_tensors, output_tensors,
               input_masks, output_masks,
               input_shapes, output_shapes)

  def get_config(self):
    inbound_names = []
    for layer in self.inbound_layers:
      if layer:
        inbound_names.append(layer.name)
      else:
        inbound_names.append(None)
    return {'outbound_layer': self.outbound_layer.name if self.outbound_layer else None,
            'inbound_layers': inbound_names,
            'node_indices': self.node_indices,
            'tensor_indices': self.tensor_indices}


class InputLayer(Layer):
  '''TODO: dosctring
  '''

  def __init__(self, input_shape=None, batch_input_shape=None,
               input_dtype=None, input_tensor=None, sparse=False, name=None):
    self.input_spec = None
    self.supports_masking = False
    self.uses_learning_phase = False
    self.trainable = False
    self.built = True
    self.trainable_weights = []
    self.non_trainable_weights = []

    self.inbound_nodes = []
    self.outbound_nodes = []

    self.trainable_weights = []
    self.non_trainable_weights = []
    self.regularizers = []
    self.constraints = {}

    self.sparse = sparse

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(K.get_uid(prefix))
    self.name = name

    if input_shape and batch_input_shape:
      raise ValueError('Only provide the input_shape OR '
                       'batch_input_shape argument to '
                       'InputLayer, not both at the same time.')
    if input_tensor is not None:
      # attempt automatic input shape inference
      try:
        batch_input_shape = K.int_shape(input_tensor)
      except:
        if not input_shape and not batch_input_shape:
          raise ValueError('InputLayer was provided an input_tensor argument, '
                           'but its input shape cannot be automatically inferred. '
                           'You should pass an input_shape or batch_input_shape '
                           'argument.')
    if not batch_input_shape:
      if not input_shape:
        raise ValueError('An Input layer should be passed either '
                         'a `batch_input_shape` or an `input_shape`.')
      else:
        batch_input_shape = (None,) + tuple(input_shape)
    else:
      batch_input_shape = tuple(batch_input_shape)

    if not input_dtype:
      if input_tensor is None:
        input_dtype = K.floatx()
      else:
        input_dtype = K.dtype(input_tensor)

    self.batch_input_shape = batch_input_shape
    self.input_dtype = input_dtype

    if input_tensor is None:
      input_tensor = K.placeholder(shape=batch_input_shape,
                                   dtype=input_dtype,
                                   sparse=self.sparse,
                                   name=self.name)
    else:
      input_tensor._keras_shape = batch_input_shape
    # create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history
    input_tensor._uses_learning_phase = False
    input_tensor._keras_history = (self, 0, 0)
    Node(self,
         inbound_layers=[],
         node_indices=[],
         tensor_indices=[],
         input_tensors=[input_tensor],
         output_tensors=[input_tensor],
         input_masks=[None],
         output_masks=[None],
         input_shapes=[batch_input_shape],
         output_shapes=[batch_input_shape])

  def get_config(self):
    config = {'batch_input_shape': self.batch_input_shape,
              'input_dtype': self.input_dtype,
              'sparse': self.sparse,
              'name': self.name}
    return config


class InputSpec(object):
  '''This specifies the ndim, dtype and shape of every input to a layer.
  Every layer should expose (if appropriate) an `input_spec` attribute:
  a list of instances of InputSpec (one per input tensor).

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.
  '''

  def __init__(self, dtype=None, shape=None, ndim=None):
    if type(ndim) is str:
      assert '+' in ndim, 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
      int_ndim = ndim[:ndim.find('+')]
      assert int_ndim.isdigit(), 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
    if shape is not None:
      self.ndim = len(shape)
    else:
      self.ndim = ndim
    self.dtype = dtype
    self.shape = shape
