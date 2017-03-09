"""
hiv dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc


def load_hiv(featurizer='ECFP', split='index'):
  """Load hiv datasets. Does not do train/test split"""
  # Featurize hiv dataset
  print("About to featurize hiv dataset.")
  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"

  dataset_file = os.path.join(data_dir, "HIV.csv")
  if not os.path.exists(dataset_file):
    os.system(
        'wget -P ' + data_dir +
        ' http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv'
    )

  hiv_tasks = ["HIV_active"]
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'Raw':
    featurizer = dc.feat.RawFeaturizer()

  loader = dc.data.CSVLoader(
      tasks=hiv_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)
  # Initialize transformers 
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'butina': dc.splits.ButinaSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  return hiv_tasks, (train, valid, test), transformers
