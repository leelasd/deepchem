from unittest import TestCase

from math import e
import numpy as np
import deepchem
from deepchem.models.tensorgraph.models.vae import VaeModel


class TestVaeModel(TestCase):

  def create_dataset(self, n_features=3, n_samples=100):
    X = np.random.random(size=(n_samples, n_features))
    w = np.ones(shape=(n_samples, n_features))
    return deepchem.data.NumpyDataset(X, X, w)

  def test_vae_model(self):
    n_features = 3
    model = VaeModel(
        n_features=n_features,
        use_queue=False,
        kl_annealing_start_step=0,
        kl_annealing_stop_step=0)
    ds = self.create_dataset(n_features)

    model.fit(ds, nb_epoch=1000)

    means = model.predict(ds, outputs=model.mean)
    reconstructions = model.predict(ds, outputs=model.reconstruction)
    model.save()

    model = VaeModel.load_from_dir(model.model_dir)

    m2 = model.predict(ds, outputs=model.mean)
    r2 = model.predict(ds, outputs=model.reconstruction)
    self.assertTrue(np.all(means == m2))
    self.assertTrue(np.all(reconstructions == r2))
