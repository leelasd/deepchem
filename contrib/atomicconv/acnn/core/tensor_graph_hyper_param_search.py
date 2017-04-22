from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import deepchem as dc
import json
import numpy as np
import tensorflow as tf
from deepchem.models.tensorgraph.models.atomic_conv import atomic_conv_model

sys.path.append("../../models")
from deepchem.models.tensorgraph.layers import Layer, Feature, Label, L2LossLayer, AtomicConvolution, Transpose, Dense
from deepchem.models import TensorGraph

import numpy as np
import tensorflow as tf
import itertools
import time

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "datasets")
train_dir = os.path.join(data_dir, "random_train")
test_dir = os.path.join(data_dir, "random_test")
model_dir = os.path.join(base_dir, "random_model")

train_dataset = dc.data.DiskDataset(train_dir)
test_dataset = dc.data.DiskDataset(test_dir)
pdbbind_tasks = ["-logKd/Ki"]
transformers = []

y_train = train_dataset.y
y_train *= -1 * 2.479 / 4.184
train_dataset = dc.data.DiskDataset.from_numpy(
  train_dataset.X,
  y_train,
  train_dataset.w,
  train_dataset.ids,
  tasks=pdbbind_tasks)

y_test = test_dataset.y
y_test *= -1 * 2.479 / 4.184
test_dataset = dc.data.DiskDataset.from_numpy(
  test_dataset.X,
  y_test,
  test_dataset.w,
  test_dataset.ids,
  tasks=pdbbind_tasks)

batch_size = 24
radial1 = [[
  1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
  9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
], [float(x) for x in range(1, 13)],
  [float(x) * 2 for x in range(1, 7)]
]
radial2 = [
  [0.0, 4.0, 8.0],
  [0.0, 2.0, 4.0, 6.0, 8.0],
  [0.0, 16.0],
  [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
]
radial3 = [
  [0.2],
  [0.4],
  [0.6]
]
layer_sizes = [
  [32, 32, 16],
  [32, 32, 32, 16],
  [16, 16, 16, 16],
  [20, 20, 20, 16],
  [10, 10, 10, 10, 10],
  [48, 48],
  [32, 32, 32, 32, 32]
]

learning_rates = [
  0.001,
  0.005,
  0.01,
  0.015,
  0.02,
  0.025
]

def params():
  for values in itertools.product(radial1, radial2, radial3, layer_sizes, learning_rates):
    d = {
      "radial": [values[0], values[1], values[2]],
      "layer_sizes": values[3],
      "learning_rate": values[4]
    }
    yield d
metric = [
  dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
  dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
for param in params():
  tg, feed_dict_generator, label = atomic_conv_model(**param)
  tg.fit_generator(feed_dict_generator(train_dataset, batch_size, epochs=10))

  test_evaluator = dc.utils.evaluate.GeneratorEvaluator(
    tg, feed_dict_generator(test_dataset, batch_size), transformers, [label])
  test_scores = test_evaluator.compute_model_performance(metric)
  param.update(test_scores)
  print("Results")
  print(param)
  with open('hyper_results.txt', 'a') as fout:
    fout.write(json.dumps(param))
