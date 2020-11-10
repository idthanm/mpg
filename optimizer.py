#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: optimizer.py
# =====================================

import logging
import os

import numpy as np
import ray
import tensorflow as tf

from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AllReduceOptimizer(object):
    def __init__(self, workers, evaluator, args):
        self.args = args
        self.evaluator = evaluator
        self.workers = workers
        self.local_worker = self.workers['local_worker']
        self.sync_remote_workers()
        self.num_sampled_steps = 0
        self.num_updates = 0
        self.iteration = 0
        self.log_dir = self.args.log_dir
        self.model_dir = self.args.model_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir + '/optimizer')
        self.stats = {}
        self.sampling_timer = TimerStat()
        self.optimizing_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.grad_apply_timer = TimerStat()

        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(iteration=self.iteration,
                               num_sampled_steps=self.num_sampled_steps,
                               num_updates=self.num_updates,
                               sampling_time=self.sampling_timer.mean,
                               optimizing_time=self.optimizing_timer.mean,
                               grad_apply_time=self.grad_apply_timer.mean,
                               grad_time=self.grad_timer.mean
                               )
                          )
        return self.stats

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.iteration))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.sampling_timer:
            for worker in self.workers['remote_workers']:
                batch_data, epinfos = ray.get(worker.sample.remote())
                worker.put_data_into_learner.remote(batch_data, epinfos)
        with self.optimizing_timer:
            all_stats = [[] for _ in range(self.args.num_workers)]
            for i in range(self.args.epoch):
                for worker in self.workers['remote_workers']:
                    worker.shuffle.remote()
                for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                    with self.grad_timer:
                        mb_grads = ray.get([worker.compute_gradient_over_ith_minibatch.remote(mb_index)
                                            for worker in self.workers['remote_workers']])
                    worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers['remote_workers']])
                    final_grads = np.array(mb_grads).mean(axis=0)
                    with self.grad_apply_timer:
                        self.local_worker.apply_gradients(self.iteration, final_grads)
                    self.sync_remote_workers()
                    for worker_index in range(self.args.num_workers):
                        all_stats[worker_index] = worker_stats

        # deal with stats
        reduced_stats_for_all_workers = []
        for worker_index in range(self.args.num_workers):
            allstats4thisworker = all_stats[worker_index]
            reduced_stats_for_this_worker = {}
            for key in allstats4thisworker[0].keys():
                value_list = list(map(lambda x: x[key], allstats4thisworker))
                reduced_stats_for_this_worker.update({key: sum(value_list) / len(value_list)})
            reduced_stats_for_all_workers.append(reduced_stats_for_this_worker)
        all_reduced_stats = {}
        for key in reduced_stats_for_all_workers[0].keys():
            value_list = list(map(lambda x: x[key], reduced_stats_for_all_workers))
            all_reduced_stats.update({key: sum(value_list) / len(value_list)})

        # log
        if self.iteration % self.args.log_interval == 0:
            with self.writer.as_default():
                for key, val in all_reduced_stats.items():
                    tf.summary.scalar('optimizer/learner_stats/{}'.format(key), val, step=self.iteration)
                for key, val in self.stats.items():
                    tf.summary.scalar('optimizer/{}'.format(key), val, step=self.iteration)
                self.writer.flush()
        # evaluate
        if self.iteration % self.args.eval_interval == 0:
            self.evaluator.set_weights.remote(self.local_worker.get_weights())
            self.evaluator.set_ppc_params.remote(self.workers['remote_workers'][0].get_ppc_params.remote())
            self.evaluator.run_evaluation.remote(self.iteration)
        # save
        if self.iteration % self.args.save_interval == 0:
            self.workers['local_worker'].save_weights(self.model_dir, self.iteration)
            self.workers['remote_workers'][0].save_ppc_params.remote(self.args.model_dir)
        self.iteration += 1
        self.num_sampled_steps += self.args.sample_batch_size * len(self.workers['remote_workers'])
        self.num_updates += self.iteration * self.args.epoch * int(self.args.sample_batch_size / self.args.mini_batch_size)
        self.get_stats()

    def sync_remote_workers(self):
        weights = ray.put(self.local_worker.get_weights())
        for e in self.workers['remote_workers']:
            e.set_weights.remote(weights)

    def stop(self):
        pass
