"""
PDBBind dataset loader.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
import shutil
import time
import re
from rdkit import Chem
import deepchem as dc


def load_pdbbind_labels(labels_file):
  """Loads pdbbind labels as dataframe"""
  # Some complexes have labels but no PDB files. Filter these manually
  missing_pdbs = ["1d2v", "1jou", "1s8j", "1cam", "4mlt", "4o7d"]
  contents = []
  with open(labels_file) as f:
    for line in f:
      if line.startswith("#"):
        continue
      else:
        # Some of the ligand-names are of form (FMN ox). Use regex
        # to merge into form (FMN-ox)
        p = re.compile('\(([^\)\s]*) ([^\)\s]*)\)')
        line = p.sub('(\\1-\\2)', line)
        elts = line.split()
        # Filter if missing PDB files
        if elts[0] in missing_pdbs:
          continue
        contents.append(elts)
  contents_df = pd.DataFrame(
      contents,
      columns=("PDB code", "resolution", "release year", "-logKd/Ki", "Kd/Ki",
               "ignore-this-field", "reference", "ligand name"))
  return contents_df


def compute_pdbbind_features(grid_featurizer, pdb_subdir, pdb_code):
  """Compute features for a given complex"""
  protein_file = os.path.join(pdb_subdir, "%s_protein.pdb" % pdb_code)
  ligand_file = os.path.join(pdb_subdir, "%s_ligand.sdf" % pdb_code)
  features = grid_featurizer.featurize_complexes([ligand_file], [protein_file])
  features = np.squeeze(features)
  return features


def featurize_pdbbind(data_dir=None, feat="grid", subset="core"):
  """Featurizes pdbbind according to provided featurization"""
  tasks = ["-logKd/Ki"]
  current_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(current_dir, "%s_%s" % (subset, feat))
  if os.path.exists(data_dir):
    return dc.data.DiskDataset(data_dir), tasks
  pdbbind_dir = os.path.join(current_dir, "v2015")

  # Load PDBBind dataset
  if subset == "core":
    labels_file = os.path.join(pdbbind_dir, "INDEX_core_data.2013")
  elif subset == "refined":
    labels_file = os.path.join(pdbbind_dir, "INDEX_refined_data.2015")
  elif subset == "full":
    labels_file = os.path.join(pdbbind_dir, "INDEX_general_PL_data.2015")
  else:
    raise ValueError("Only core, refined, and full subsets supported.")
  print("About to load contents.")
  if not os.path.exists(labels_file):
    raise ValueError("Run get_pdbbind.sh to download dataset.")
  contents_df = load_pdbbind_labels(labels_file)
  ids = contents_df["PDB code"].values
  y = np.array([float(val) for val in contents_df["-logKd/Ki"].values])

  # Define featurizers
  if feat == "grid":
    featurizer = dc.feat.GridFeaturizer(
        voxel_width=16.0,
        feature_types="voxel_combined",
        # TODO(rbharath, enf): Figure out why pi_stack is slow and cation_pi
        # causes segfaults.
        #voxel_feature_types=["ecfp", "splif", "hbond", "pi_stack", "cation_pi",
        #"salt_bridge"], ecfp_power=9, splif_power=9,
        voxel_feature_types=["ecfp", "splif", "hbond", "salt_bridge"],
        ecfp_power=9,
        splif_power=9,
        parallel=True,
        flatten=True)
  elif feat == "coord":
    neighbor_cutoff = 4
    max_num_neighbors = 10
    featurizer = dc.feat.NeighborListComplexAtomicCoordinates(max_num_neighbors,
                                                              neighbor_cutoff)
  else:
    raise ValueError("feat not defined.")

  # Featurize Dataset
  features = []
  feature_len = None
  y_inds = []
  missing_pdbs = []
  time1 = time.time()
  for ind, pdb_code in enumerate(ids):
    print("Processing complex %d, %s" % (ind, str(pdb_code)))
    pdb_subdir = os.path.join(pdbbind_dir, pdb_code)
    if not os.path.exists(pdb_subdir):
      print("%s is missing!" % pdb_subdir)
      missing_pdbs.append(pdb_subdir)
      continue
    computed_feature = compute_pdbbind_features(featurizer, pdb_subdir,
                                                pdb_code)
    if feature_len is None:
      feature_len = len(computed_feature)
    if len(computed_feature) != feature_len:
      print("Featurization failed for %s!" % pdb_code)
      continue
    y_inds.append(ind)
    features.append(computed_feature)
  time2 = time.time()
  print("TIMING: PDBBind Featurization took %0.3f s" % (time2 - time1))
  print("missing_pdbs")
  print(missing_pdbs)
  y = y[y_inds]
  X = np.vstack(features)
  w = np.ones_like(y)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids, data_dir=data_dir)
  return dataset, tasks


def load_pdbbind_grid(split="index", featurizer="grid", subset="refined"):
  """Load PDBBind datasets. Does not do train/test split"""
  dataset, tasks = featurize_pdbbind(feat=featurizer, subset=subset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = []
  for transformer in transformers:
    train = transformer.transform(train)
  for transformer in transformers:
    valid = transformer.transform(valid)
  for transformer in transformers:
    test = transformer.transform(test)

  return tasks, (train, valid, test), transformers


if __name__ == "__main__":
  load_pdbbind_grid()
