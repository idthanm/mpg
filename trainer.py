#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: trainer.py
# =====================================

import logging

import ray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer(object):
    def __init__(self, policy_cls, worker_cls, learner_cls, buffer_cls, optimizer_cls, evaluator_cls, args):
        self.args = args
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.evaluator = evaluator_cls(policy_cls, self.args.env_id, self.args)
            self.local_worker = worker_cls(policy_cls, learner_cls, self.args.env_id, self.args, 0)
            self.optimizer = optimizer_cls(self.local_worker, self.evaluator, self.args)


    def load_weights(self, load_dir, iteration):
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.local_worker.load_weights(load_dir, iteration)
        else:
            self.local_worker.load_weights(load_dir, iteration)
            self.sync_remote_workers()

    def load_ppc_params(self, load_dir):
        if self.args.optimizer_type.startswith('SingleProcess'):
            self.local_worker.load_ppc_params(load_dir)

    def sync_remote_workers(self):
        weights = ray.put(self.local_worker.get_weights())

    def train(self):
        logger.info('training beginning')
        while self.optimizer.num_sampled_steps < self.args.max_sampled_steps \
                or self.optimizer.iteration < self.args.max_iter:
            self.optimizer.step()
        self.optimizer.stop()
