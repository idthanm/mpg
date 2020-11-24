#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: optimizer.py
# =====================================

import logging
import os

import tensorflow as tf

from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleProcessOptimizer(object):
    def __init__(self, worker, evaluator, args):
        self.args = args
        self.evaluator = evaluator
        self.worker = worker
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
        self.step_timer = TimerStat()
        self.sampling_timer = TimerStat()
        self.learning_timer = TimerStat()
        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(iteration=self.iteration,
                               num_sampled_steps=self.num_sampled_steps,
                               num_updates=self.num_updates,
                               step_timer=self.step_timer.mean,
                               sampling_time=self.sampling_timer.mean,
                               learning_time=self.learning_timer.mean
                               )
                          )
        return self.stats

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.iteration))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.step_timer:
            with self.sampling_timer:
                self.worker.sample_and_process()
            all_stats = []
            lrnow = 3e-4*(1.-self.iteration/488)
            with self.learning_timer:
                for i in range(self.args.epoch):
                    for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                        mb_grads = self.worker.compute_gradient_over_ith_minibatch(mb_index)
                        worker_stats = self.worker.get_stats()
                        self.worker.apply_grads_all(mb_grads, lrnow)
                        all_stats.append(worker_stats.copy())

        all_reduced_stats = {}
        for key in all_stats[0].keys():
            value_list = list(map(lambda x: x[key], all_stats))
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
            self.evaluator.set_weights(self.worker.get_weights())
            self.evaluator.set_ppc_params(self.worker.get_ppc_params())
            self.evaluator.run_evaluation(self.iteration)
        # save
        if self.iteration % self.args.save_interval == 0:
            self.worker.save_weights(self.model_dir, self.iteration)
            self.worker.save_ppc_params(self.args.model_dir)
        self.iteration += 1
        self.num_sampled_steps += self.args.sample_batch_size
        self.num_updates += self.iteration * self.args.epoch * int(
            self.args.sample_batch_size / self.args.mini_batch_size)
        self.get_stats()

    def stop(self):
        pass

