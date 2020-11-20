#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: misc.py
# =====================================

import random
import time
from contextlib import contextmanager

import numpy as np
import tensorflow as tf


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def judge_is_nan(list_of_np_or_tensor):
    for m in list_of_np_or_tensor:
        if hasattr(m, 'numpy'):
            if np.any(np.isnan(m.numpy())):
                print(list_of_np_or_tensor)
                raise ValueError
        else:
            if np.any(np.isnan(m)):
                print(list_of_np_or_tensor)
                raise ValueError


def flatvars(var_list):
    if type(var_list[0]) is np.ndarray:
        return np.concatenate([np.reshape(v, [-1]) for v in var_list], axis=0)
    else:
        return tf.concat([tf.reshape(v, [-1]) for v in var_list], axis=0)


def get_shapes(var_list):
    if type(var_list[0]) is np.ndarray:
        return [list(var.shape) for var in var_list]
    else:
        return [var.get_shape().as_list() for var in var_list]


def unflatvars(flatvars, shapes):
    start = 0
    unflat_vars = []
    for shape in shapes:
        sz = int(np.prod(shape))
        if type(flatvars) is np.ndarray:
            unflat_vars.append(np.reshape(flatvars[start:start + sz], shape))
        else:
            unflat_vars.append(tf.reshape(flatvars[start:start + sz], shape))
        start += sz
    return unflat_vars


def colorize(string, color='green', bold=False, highlight=False):
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38
    )
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    def has_units_processed(self):
        return len(self._units_processed) > 0

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))

    @property
    def mean_units_processed(self):
        if not self._units_processed:
            return 0.0
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = float(sum(self._samples))
        if not time_total:
            return 0.0
        return float(sum(self._units_processed)) / time_total



