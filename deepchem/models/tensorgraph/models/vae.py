import tensorflow as tf
from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, CombineMeanStd, Weights, Dense, L2Loss, KLDivergenceLoss, Add, \
  TensorWrapper, ReduceSum

import numpy as np


class VaeModel(TensorGraph):

  def __init__(self,
               n_features,
               encoder_layers=[512, 512, 521],
               decoder_layers=[512, 512, 512],
               kl_annealing_start_step=500,
               kl_annealing_stop_step=1000,
               **kwargs):
    self.n_features = n_features
    self.encoder_layers = encoder_layers
    self.decoder_layers = decoder_layers
    self.kl_annealing_start_step = kl_annealing_start_step
    self.kl_annealing_stop_step = kl_annealing_stop_step
    super(VaeModel, self).__init__(**kwargs)

    self.build_graph()

  def build_graph(self):
    features = Feature(shape=(None, self.n_features))
    last_layer = features
    for layer_size in self.encoder_layers:
      last_layer = Dense(
          in_layers=last_layer,
          activation_fn=tf.nn.elu,
          out_channels=layer_size)

    self.mean = Dense(in_layers=last_layer, activation_fn=None, out_channels=1)
    self.std = Dense(in_layers=last_layer, activation_fn=None, out_channels=1)

    readout = CombineMeanStd([self.mean, self.std], training_only=True)
    last_layer = readout
    for layer_size in self.decoder_layers:
      last_layer = Dense(
          in_layers=readout, activation_fn=tf.nn.elu, out_channels=layer_size)

    self.reconstruction = Dense(
        in_layers=last_layer, activation_fn=None, out_channels=self.n_features)
    weights = Weights(shape=(None, self.n_features))
    reproduction_loss = L2Loss(
        in_layers=[features, self.reconstruction, weights])
    reproduction_loss = ReduceSum(in_layers=reproduction_loss, axis=0)
    global_step = TensorWrapper(self._get_tf("GlobalStep"))
    kl_loss = KLDivergenceLoss(
        in_layers=[self.mean, self.std, global_step],
        annealing_start_step=self.kl_annealing_start_step,
        annealing_stop_step=self.kl_annealing_stop_step)
    loss = Add(in_layers=[kl_loss, reproduction_loss], weights=[0.5, 1])

    self.add_output(self.mean)
    self.add_output(self.reconstruction)
    self.set_loss(loss)

  def save_kwargs(self):
    return {
        "n_features": self.n_features,
        "encoder_layers": self.encoder_layers,
        "decoder_layers": self.decoder_layers,
        "kl_annealing_start_step": self.kl_annealing_start_step,
        "kl_annealing_stop_step": self.kl_annealing_stop_step,
    }

  @classmethod
  def load_from_dir(cls, model_dir):
    kwargs_file = os.path.join(model_dir, 'kwargs.json')
    if not os.path.exists(kwargs_file):
      return VaeModel.load_from_dir(model_dir)
    kwargs = json.loads(open(kwargs_file).read())
    kwargs['model_dir'] = model_dir
    model = cls(**kwargs)
    model.restore()
    return model

  def save(self):
    d = {
        "tensorboard": self.tensorboard,
        "tensorboard_log_frequency": self.tensorboard_log_frequency,
        "batch_size": self.batch_size,
        "random_seed": self.random_seed,
        "use_queue": self.use_queue,
        "model_dir": self.model_dir
    }
    d.update(self.save_kwargs())
    kwargs_path = os.path.join(self.model_dir, 'kwargs.json')
    with open(kwargs_path, 'w') as fout:
      fout.write(json.dumps(d))
