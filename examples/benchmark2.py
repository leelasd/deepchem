#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:53:27 2016

@author: Michael Wu

Benchmark test:

Giving classification performances of: 
    Random forest(rf), MultitaskDNN(tf), 
    RobustMultitaskDNN(tf_robust),
    Logistic regression(logreg), IRV(irv)
    Graph convolution(graphconv)                 
on datasets: muv, pcba, tox21, sider, toxcast, clintox, hiv

Giving regression performances of:
    MultitaskDNN(tf_regression),
    Random forest(rf_regression),
    Graph convolution regression(graphconvreg)
on datasets: delaney(ESOL), nci, kaggle, pdbbind, 
             qm7, qm7b, qm9, chembl, sampl(FreeSolv)

time estimation listed in README file

Total time of running a benchmark test(for one splitting function): 20h
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
import numpy as np
import shutil
import time
import deepchem as dc
import tensorflow as tf
import argparse
from keras import backend as K
import csv

np.random.seed(123)

parser = argparse.ArgumentParser(
    description='Deepchem benchmark: ' +
    'giving performances of different learning models on datasets')
parser.add_argument(
    '-s',
    action='append',
    dest='splitter_args',
    default=[],
    help='Choice of splitting function: index, random, scaffold')
parser.add_argument(
    '-m',
    action='append',
    dest='model_args',
    default=[],
    help='Choice of model: tf, tf_robust, logreg, rf, irv, graphconv, ' +
    'tf_regression, rf_regression, graphconvreg')
parser.add_argument(
    '-d',
    action='append',
    dest='dataset_args',
    default=[],
    help='Choice of dataset: tox21, sider, muv, toxcast, pcba, ' +
    'kaggle, delaney, nci, pdbbind, chembl, sampl, qm7, qm7b, qm9, clintox, hiv')
parser.add_argument(
    '-t',
    action='store_true',
    dest='test',
    default=False,
    help='Evalute performance on test set')
args = parser.parse_args()
#Datasets and models used in the benchmark test
splitters = args.splitter_args
models = args.model_args
datasets = args.dataset_args
test = args.test

if len(splitters) == 0:
  splitters = ['index', 'random', 'scaffold']
if len(models) == 0:
  models = [
      'tf', 'tf_robust', 'logreg', 'graphconv', 'tf_regression', 'graphconvreg'
  ]
  #irv, rf, rf_regression should be assigned manually
if len(datasets) == 0:
  datasets = [
      'tox21', 'sider', 'muv', 'toxcast', 'pcba', 'clintox', 'hiv', 'sampl',
      'delaney', 'nci', 'kaggle', 'pdbbind', 'chembl', 'qm7b', 'qm9'
  ]

for split in splitters:
  for dataset in datasets:
    if dataset in [
        'tox21', 'sider', 'muv', 'toxcast', 'pcba', 'clintox', 'hiv'
    ]:
      for model in models:
        if model in ['tf', 'tf_robust', 'logreg', 'graphconv', 'rf', 'irv']:
          dc.molnet.run_benchmark([dataset], str(model), split=split, test=test)
    else:
      for model in models:
        if model in ['tf_regression', 'rf_regression', 'graphconvreg']:
          dc.molnet.run_benchmark([dataset], str(model), split=split, test=test)
