#!/usr/bin/env python
# coding: utf-8
#
# Filename:   minhash_jaccard.py
# Author:     Peinan ZHANG
# Created at: 2018-09-27

import numpy as np

def jaccard(m1: np.array, m2: np.array):
  return np.float(np.count_nonzero(m1==m2)) / np.float(m1.shape[0])


def jaccard_m(m1: np.array, m2: np.array):
  assert len(m1.shape) == len(m2.shape) == 2
  assert m1.shape[1] == m2.shape[1]
  assert m2.shape[0] == 1

  return np.count_nonzero(m1==m2, axis=1).astype(np.float) \
           / np.float(m1.shape[1])
