"""
qm7 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem as dc
import scipy.io


def load_qm7_from_mat(featurizer=None, split='stratified'):
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "qm7.mat")

  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir +
              ' http://www.quantum-machine.org/data/qm7.mat')

  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)
  print(len(dataset))

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }

  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  qm7_tasks = np.arange(y.shape[0])
  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7b_from_mat(featurizer=None, split='stratified'):
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "qm7b.mat")

  if not os.path.exists(dataset_file):
    os.system('wget -P ' + data_dir +
              ' http://www.quantum-machine.org/data/qm7b.mat')
  dataset = scipy.io.loadmat(dataset_file)

  X = dataset['X']
  y = dataset['T']
  w = np.ones_like(y)
  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids=None)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  qm7_tasks = np.arange(y.shape[1])
  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_qm7(featurizer=None, split='random'):
  """Load qm7 datasets."""
  # Featurize qm7 dataset
  print("About to featurize qm7 dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "gdb7.sdf")

  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/gdb7.tar.gz '
    )
    os.system('tar -zxvf ' + os.path.join(data_dir, 'gdb7.tar.gz') + ' -C ' +
              data_dir)

  qm7_tasks = ["u0_atom"]
  if featurizer is None:
    featurizer = dc.feat.CoulombMatrixEig(23)
  loader = dc.data.SDFLoader(
      tasks=qm7_tasks,
      smiles_field="smiles",
      mol_field="mol",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'stratified': dc.splits.SingletaskStratifiedSplitter(task_number=0)
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

  transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset)
  ]

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  return qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers
