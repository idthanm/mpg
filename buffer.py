#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: buffer.py
# =====================================

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, args, buffer_id):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self.args = args
        self.buffer_id = buffer_id
        self._storage = []
        self._maxsize = self.args.max_buffer_size
        self._next_idx = 0
        self.replay_starts = self.args.replay_starts
        self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0
        logger.info('Buffer initialized')

    def get_stats(self):
        self.stats.update(dict(storage=len(self._storage)))
        return self.stats

    def __len__(self):
        return len(self._storage)

    def add(self, obs_ego_next, obs_others_next, veh_num_next, done, ref_index, weight):
        data = (obs_ego_next, obs_others_next, veh_num_next, done, ref_index)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_ego_next, obses_other_next, vehs_num_next, dones, ref_indexs = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_ego_next, obs_other_next, veh_num_next, done, ref_index = data
            obses_ego_next.append(np.array(obs_ego_next, copy=False))
            obses_other_next.append(np.array(obs_other_next, copy=False))
            vehs_num_next.append(veh_num_next)
            dones.append(done)
            ref_indexs.append(ref_index)
        obses_others_next = np.concatenate(([obses_other_next[i] for i in range(len(obses_other_next))]), axis=0)
        # print(vehs_mode_next.shape, obses_others_next.shape, np.sum(np.array(vehs_num_next)))

        return np.array(obses_ego_next), np.array(obses_others_next), np.array(vehs_num_next), \
               np.array(dones), np.array(ref_indexs)

    def sample_idxes(self, batch_size):
        return np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)], dtype=np.int32)

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes)) + [idxes,]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for trans in batch:
            self.add(*trans, 0)

    def replay(self):
        if len(self._storage) < self.replay_starts:
            return None
        if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
            logger.info('Buffer info: {}'.format(self.get_stats()))

        self.replay_times += 1
        return self.sample(self.replay_batch_size)
