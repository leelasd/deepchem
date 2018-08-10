from unittest import TestCase

import numpy as np
from deepchem.data import NumpyDataset
from deepchem.models.tensorgraph.models.vae import VaeModel
from deepchem.feat.graph_features import VaeConvMolFeaturizer, ConvMolFeaturizer
from deepchem.molnet import load_bace_classification


class TestVaeConvMolFeaturizer(TestCase):
  def get_mols(self,
               featurizer='Raw'):
    data_points = 10
    tasks, all_dataset, transformers = load_bace_classification(featurizer)
    train, valid, test = all_dataset
    return train.X[:data_points]

  def create_dataset(self, n_features=3, n_samples=100):
    X = np.random.random(size=(n_samples, n_features))
    w = np.ones(shape=(n_samples, n_features))
    return NumpyDataset(X, X, w)

  def create_vae(self):
    n_features = 75
    model = VaeModel(
      n_features=n_features,
      latent_size=5,
      use_queue=False,
      kl_annealing_start_step=0,
      kl_annealing_stop_step=0)
    ds = self.create_dataset(n_features)
    model.fit(ds, nb_epoch=1)
    model.save()
    return model

  def test_featurizer(self):
    vae_model = self.create_vae()
    featurizer = VaeConvMolFeaturizer(vae_model)
    mols = self.get_mols()
    vae_features = featurizer.featurize(mols)
    conv_features = ConvMolFeaturizer().featurize(mols)
    for vae_mol, conv_mol in zip(vae_features, conv_features):
      self.assertEqual(vae_mol.atom_features.shape[1], vae_model.latent_size)
      self.assertEqual(vae_mol.atom_features.shape[0], conv_mol.atom_features.shape[0])
