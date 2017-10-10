import numpy as np
import os

import tensorflow as tf

import deepchem as dc
import pyanitools as pya
import sqlite3
import sys
import arrow
import json
import shutil


# import app

def save_result(exp_id, train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae, epochs, timestamp):
  conn = sqlite3.connect('exps.db')
  c = conn.cursor()

  # Insert a row of data
  c.execute("""
  INSERT INTO result (exp_id, train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae, epochs, timestamp) 
  VALUES (?, ?, ?, ?,?, ?, ?, ?, ?)""",
            (exp_id, train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae, epochs, timestamp))

  # Save (commit) the changes
  conn.commit()

  # We can also close the connection if we are done with it.
  # Just be sure any changes have been committed or they will be lost.
  conn.close()


def get_experiment():
  """'
  TODO(LESWING) thread safe
  Returns
  -------

  """
  conn = sqlite3.connect('exps.db')
  c = conn.cursor()

  c.execute("""
  SELECT oid, model_folder, num_epochs, kwargs_json, status FROM experiment
  WHERE status = 'READY'
  ORDER BY oid
  LIMIT 1
  """)
  l = c.fetchone()
  c.execute("""
  UPDATE experiment SET status = 'RUNNING' WHERE oid=?
  """, (l[0],))

  conn.commit()
  conn.close()
  return l


def set_exp_finished(oid):
  conn = sqlite3.connect('exps.db')
  c = conn.cursor()

  c.execute("""
  UPDATE experiment SET status = 'FINISHED' WHERE oid=?
  """, (oid,))

  conn.commit()
  conn.close()


def convert_species_to_atomic_nums(s):
  PERIODIC_TABLE = {"H": 1, "C": 6, "N": 7, "O": 8}
  res = []
  for k in s:
    res.append(PERIODIC_TABLE[k])
  return np.array(res, dtype=np.float32)


def data_dir_f(kwargs):
  if "data_length" not in kwargs:
    return "/home/leswing/ANI-1/datasets"
  return "/home/leswing/ANI-1/datasets_%s" % kwargs['data_length']


def load_roiterberg_ANI(data_dir, data_length=None, mode="atomization"):
  """
  Load the ANI dataset.

  Parameters
  ----------
  data_dir : str
    Directory to store data in
  mode: str
    Accepted modes are "relative", "atomization", or "absolute". These settings are used
    to adjust the dynamic range of the model, with absolute having the greatest and relative
    having the lowest. Note that for atomization we approximate the single atom energy
    using a different level of theory


  Returns
  -------
  tuples
    Elements returned are 3-tuple (a,b,c) where and b are the train and test datasets, respectively,
    and c is an array of indices denoting the group of each

  """

  # Number of conformations in each file increases exponentially.
  # Start with a smaller dataset before continuing. Use all of them
  # for production
  base_dir = json.loads(open('paths.json').read())['raw_dir']
  all_dir = os.path.join(data_dir, "all")
  test_dir = os.path.join(data_dir, "test")
  fold_dir = os.path.join(data_dir, "fold")
  if os.path.isdir(fold_dir) and os.path.isdir(test_dir):
    return dc.data.DiskDataset(fold_dir), dc.data.DiskDataset(test_dir), 598110

  hdf5files = [
    'ani_gdb_s01.h5',
    'ani_gdb_s02.h5',
    'ani_gdb_s03.h5',
    'ani_gdb_s04.h5',
    'ani_gdb_s05.h5',
    'ani_gdb_s06.h5',
    'ani_gdb_s07.h5',
    'ani_gdb_s08.h5'
  ]
  if data_length is not None:
    hdf5files = hdf5files[:data_length]

  hdf5files = [os.path.join(base_dir, f) for f in hdf5files]

  groups = []

  def shard_generator():

    shard_size = 4096 * 64

    row_idx = 0
    group_idx = 0

    X_cache = []
    y_cache = []
    w_cache = []
    ids_cache = []

    for hdf5file in hdf5files:
      adl = pya.anidataloader(hdf5file)
      for data in adl:

        # Extract the data
        P = data['path']
        R = data['coordinates']
        E = data['energies']
        S = data['species']
        smi = data['smiles']

        if len(S) > 23:
          print("skipping:", smi, "due to atom count.")
          continue

        # Print the data
        print("Processing: ", P)
        print("  Smiles:      ", "".join(smi))
        print("  Symbols:     ", S)
        print("  Coordinates: ", R.shape)
        print("  Energies:    ", E.shape)

        Z_padded = np.zeros((23,), dtype=np.float32)
        nonpadded = convert_species_to_atomic_nums(S)
        Z_padded[:nonpadded.shape[0]] = nonpadded

        if mode == "relative":
          offset = np.amin(E)
        elif mode == "atomization":

          # self-interaction energies taken from
          # https://github.com/isayev/ANI1_dataset README
          atomizationEnergies = {
            0: 0,
            1: -0.500607632585,
            6: -37.8302333826,
            7: -54.5680045287,
            8: -75.0362229210
          }

          offset = 0

          for z in nonpadded:
            offset += atomizationEnergies[z]
        elif mode == "absolute":
          offset = 0
        else:
          raise Exception("Unsupported mode: ", mode)

        for k in range(len(E)):
          R_padded = np.zeros((23, 3), dtype=np.float32)
          R_padded[:R[k].shape[0], :R[k].shape[1]] = R[k]

          X = np.concatenate([np.expand_dims(Z_padded, 1), R_padded], axis=1)

          y = E[k] - offset

          if len(X_cache) == shard_size:
            yield np.array(X_cache), np.array(y_cache), np.array(
              w_cache), np.array(ids_cache)

            X_cache = []
            y_cache = []
            w_cache = []
            ids_cache = []

          X_cache.append(X)
          y_cache.append(np.array(y).reshape((1,)))
          w_cache.append(np.array(1).reshape((1,)))
          ids_cache.append(row_idx)
          row_idx += 1
          groups.append(group_idx)

        group_idx += 1

    # flush once more at the end
    if len(X_cache) > 0:
      yield np.array(X_cache), np.array(y_cache), np.array(w_cache), np.array(
        ids_cache)

  tasks = ["ani"]
  dataset = dc.data.DiskDataset.create_dataset(
    shard_generator(), tasks=tasks, data_dir=all_dir)

  print("Number of groups", np.amax(groups))
  splitter = dc.splits.RandomGroupSplitter(groups)

  train_dataset, test_dataset = splitter.train_test_split(
    dataset, train_dir=fold_dir, test_dir=test_dir, frac_train=.8)

  return train_dataset, test_dataset, groups


def broadcast(dataset, metadata):
  new_metadata = []

  for (_, _, _, ids) in dataset.itershards():
    for idx in ids:
      new_metadata.append(metadata[idx])

  return new_metadata


def main(model_dir, exp_id, num_epochs, kwargs):
  if 'data_length' not in kwargs:
    data_length = None
  else:
    data_length = kwargs['data_length']
  data_dir = data_dir_f(kwargs)

  fold_dir = os.path.join(data_dir, "fold")
  train_dir = os.path.join(fold_dir, "train")
  valid_dir = os.path.join(fold_dir, "valid")

  max_atoms = 23
  batch_size = json.loads(open('paths.json').read())['batch_size']
  atom_number_cases = [1, 6, 7, 8]
  if 'activation' in kwargs:
    activation = kwargs['activation']
  else:
    activation = 'tanh'

  if 'batch_norms' in kwargs:
    batch_norms = kwargs['batch_norms']
  else:
    batch_norms = [False, False, False]

  if 'dropouts' in kwargs:
    dropouts = kwargs['dropouts']
  else:
    dropouts = [0.0, 0.0, 0.0, 0.0]

  if 'layer_structures' in kwargs:
    layer_structures = kwargs['layer_structures']
  else:
    layer_structures = [128, 128, 64]

  if 'learning_rate' in kwargs:
    learning_rate = kwargs['learning_rate']
  else:
    learning_rate = 0.001

  if os.path.exists(os.path.join(model_dir, "save_pickle.npz")):
    model = dc.models.ANIRegression.load_numpy(model_dir, batch_size)
    print("loaded from dir")
  else:
    model = dc.models.ANIRegression(
      1,
      max_atoms,
      layer_structures=layer_structures,
      atom_number_cases=atom_number_cases,
      batch_size=batch_size,
      learning_rate=learning_rate,
      use_queue=True,
      model_dir=model_dir,
      mode="regression",
      activation=activation,
      batch_norms=batch_norms,
      dropouts=dropouts)
    model.build()

  metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
  ]

  train_valid_dataset, test_dataset, all_groups = load_roiterberg_ANI(data_dir,
                                                                      data_length,
                                                                      mode="atomization")

  if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
    train_dataset, valid_dataset = dc.data.DiskDataset(train_dir), dc.data.DiskDataset(valid_dir)
  else:
    splitter = dc.splits.RandomGroupSplitter(
      broadcast(train_valid_dataset, all_groups))

    print("Performing 1-fold split...")
    train_dataset, valid_dataset = splitter.train_test_split(
      train_valid_dataset, train_dir=train_dir, test_dir=valid_dir)
    train_dataset.reshard(10000000000000)

  transformers = [
    dc.trans.NormalizationTransformer(
      transform_y=True, dataset=train_dataset)
  ]

  print("Total training set shape: ", train_dataset.get_shape())

  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  for i in range(num_epochs):
    model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

    print("Saving model...")
    model.save_numpy()
    print("Done.")

    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, metric, transformers)
    valid_scores = model.evaluate(valid_dataset, metric, transformers)
    test_scores = model.evaluate(test_dataset, metric, transformers)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)

    print("Test scores")
    print(test_scores)

    train_r2, train_mae = train_scores['pearson_r2_score'], train_scores['mean_absolute_error']
    valid_r2, valid_mae = valid_scores['pearson_r2_score'], valid_scores['mean_absolute_error']
    test_r2, test_mae = test_scores['pearson_r2_score'], test_scores['mean_absolute_error']
    nb_epochs = 10 * i
    timestamp = arrow.utcnow().float_timestamp
    save_result(exp_id, train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae, nb_epochs, timestamp)


def save_test():
  exp_id = 1
  train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae = 0, 0, 0, 0, 0, 0
  epochs = 10
  timestamp = 100
  save_result(exp_id, train_r2, valid_r2, test_r2, train_mae, valid_mae, test_mae, epochs, timestamp)


if __name__ == "__main__":
  oid, model_folder, num_epochs, kwargs_json, status = get_experiment()
  model_dir = "%s/%s" % (json.loads(open('paths.json').read())['models_dir'], model_folder)
  kwargs_json = json.loads(kwargs_json)
  main(model_dir, oid, num_epochs, kwargs_json)
  set_exp_finished(oid)
