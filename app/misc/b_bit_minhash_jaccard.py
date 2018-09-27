#!/usr/bin/env python
# coding: utf-8
#
# Filename:   b_bit_minhash_jaccard.py
# Author:     Peinan ZHANG
# Created at: 2018-09-27

import numpy as np

def jaccard(bm1: np.array, bm2: np.array, b=1 , r=0.0):
  """
  前提として、bm1 と bm2 のそれぞれの r と b は同じ
  """
  intersection = np.count_nonzero(bm1==bm2)
  raw_est = float(intersection) / float(bm1.size)
  a = calc_a(r, b)
  c = calc_c(a, r)

  print(f'inters: {intersection}, raw_est: {raw_est}, a: {a}, c: {c}')

  return (raw_est - c) / (1 - c)

def calc_a(r, b):
  if r == 0.0:
    return 1.0 / (1 << b)

  return r * (1 - r) ** (2 ** b - 1) / (1 - (1 - r) ** (2 * b))

def calc_c(a, r):
  if r == 0.0:
    return a

  div = 1 / (r + r)
  c = (a * r + a * r) * div

  return c
