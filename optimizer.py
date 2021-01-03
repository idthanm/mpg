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

from utils.misc import TimerStat, flatvars, unflatvars, get_shapes, judge_is_nan

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
            with self.learning_timer:
                for i in range(self.args.epoch):
                    for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                        mb_grads = self.worker.compute_gradient_over_ith_minibatch(mb_index)
                        # apply grad
                        try:
                            judge_is_nan(mb_grads)
                        except ValueError:
                            mb_grads = [tf.zeros_like(grad) for grad in mb_grads]
                            logger.info('Grad is nan!, zero it')
                        self.worker.apply_grads_all(mb_grads)
                        worker_stats = self.worker.get_stats()
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
        self.step_timer = TimerStat()
        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(iteration=self.iteration,
                               num_sampled_steps=self.num_sampled_steps,
                               num_updates=self.num_updates,
                               step_timer=self.step_timer.mean,
                               )
                          )
        return self.stats

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.iteration))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.step_timer:
            for worker in self.workers['remote_workers']:
                worker.sample_and_process.remote()
            all_stats = [[] for _ in range(self.args.num_workers)]
            for i in range(self.args.epoch):
                for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                    mb_grads = ray.get([worker.compute_gradient_over_ith_minibatch.remote(mb_index)
                                        for worker in self.workers['remote_workers']])
                    worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers['remote_workers']])
                    final_grads = np.array(mb_grads).mean(axis=0).tolist()
                    try:
                        judge_is_nan(final_grads)
                    except ValueError:
                        final_grads = [tf.zeros_like(grad) for grad in final_grads]
                        logger.info('Grad is nan!, zero it')
                    self.local_worker.apply_grads_all(final_grads)
                    self.sync_remote_workers()
                    for worker_index in range(self.args.num_workers):
                        all_stats[worker_index].append(worker_stats[worker_index].copy())

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


class SingleProcessTRPOOptimizer(object):
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
        self.search_result = 1
        self.step_timer = TimerStat()
        self.value_optimizing_timer = TimerStat()
        self.direction_computing_timer = TimerStat()
        self.sample_and_po_timer = TimerStat()
        self.line_search_timer = TimerStat()

        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(iteration=self.iteration,
                               num_sampled_steps=self.num_sampled_steps,
                               step_time=self.step_timer.mean,
                               sample_and_po_time=self.sample_and_po_timer.mean,
                               value_optimizing_time=self.value_optimizing_timer.mean,
                               direction_computing_time=self.direction_computing_timer.mean,
                               line_search_time=self.line_search_timer.mean,
                               search_result=self.search_result
                               )
                          )
        return self.stats

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.iteration))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.step_timer:
            with self.sample_and_po_timer:
                self.worker.sample_and_process()
                flat_g = self.worker.prepare_for_policy_update()
                worker_stats_before = self.worker.get_stats()

            def compute_fvp(vec):
                return self.worker.compute_fvp(vec).numpy() + self.args.cg_damping * vec

            with self.direction_computing_timer:
                # the following is to solve the update direction: Hessian^-1 * obj_grad by construct Ax=b
                # Ax=b -> min J(x) = 1/2 * (x^T)Ax - (b^T)x, then solve it by cg method
                # 1. Given x_1=0, k=1,
                # 2. g_k = J'(x_k) = Ax_k-b (g_{k+1}-g_k = A(x_{k+1}-x_k) = lambda*Ad_k)
                # 3. d_k = -g_k + beta_{k-1}*d_{k-1}, beta_0 = 0, beta_{k-1} = ||g_k||^2/||g_{k-1}||^2
                # 4. x_{k+1} = x_k + lambda*d_k, lambda_k=-(g_k)^T d_k/(d_k)^T A(d_k) = (g_k)^T g_k/(d_k)^T A(d_k)
                # in our problem, A is fisher matrix, which is the Hessian matrix of KL divergence, denoted by H,
                # b is the obj grad, i.e. mean_flat_g
                d = flat_g.copy()  # conjugate gradient direction
                g = -flat_g.copy()  # local gradient
                x = np.zeros_like(flat_g)  # x_1
                gdotg = g.dot(g)
                print("{:10s} {:10s} {:10s}".format("cg_iter", "residual norm", "soln norm"))
                for i in range(self.args.cg_iters):
                    print("{:<10d} {:10.3g} {:10.3g}".format(i, gdotg, np.linalg.norm(x)))
                    Hd = compute_fvp(d)  # H*d
                    lmda = gdotg / d.dot(Hd)
                    x += lmda * d
                    g += lmda * Hd
                    newgdotg = g.dot(g)
                    beta = newgdotg / gdotg
                    d = -g + beta * d
                    gdotg = newgdotg
                    if gdotg < self.args.residual_tol:
                        break
                stepdir = x

            with self.line_search_timer:
                # with the direction, line search is used to determine the stepsize
                if not np.isfinite(stepdir).all():
                    stepdir = np.zeros_like(stepdir)
                    logger.info('stepdir is nan!, zero it')
                shs = .5 * stepdir.dot(compute_fvp(stepdir))
                lm = np.sqrt(shs / self.args.max_kl)
                fullstep = stepdir / lm
                expectedimprove = flat_g.dot(fullstep)
                pggainbefore = worker_stats_before['pg_gain']
                stepsize = 1.0
                thbefore = self.worker.policy_with_value.policy.get_weights()
                self.search_result = 1
                for _ in range(10):
                    thnew_flat = flatvars(thbefore) + fullstep * stepsize
                    thnew = unflatvars(thnew_flat, get_shapes(self.worker.policy_with_value.policy.trainable_weights))
                    self.worker.policy_with_value.policy.set_weights(thnew)
                    self.worker.prepare_for_policy_update()
                    worker_stats = self.worker.get_stats()
                    pggain, kl = worker_stats['pg_gain'], worker_stats['kl']
                    improve = pggain - pggainbefore
                    logger.info("Expected: {:.3f} Actual: {:.3f}".format(expectedimprove, improve))
                    if not np.isfinite(list(worker_stats.values())).all():
                        logger.info("Got non-finite value of losses -- bad!")
                    elif kl > self.args.max_kl * 1.5:
                        logger.info("Violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.info("Surrogate didn't improve. shrinking step.")
                    else:
                        logger.info("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    self.search_result = 0
                    logger.info("couldn't compute a good step")
                    self.worker.policy_with_value.policy.set_weights(thbefore)
                    self.worker.prepare_for_policy_update()

            all_stats = []
            with self.value_optimizing_timer:
                for i in range(self.args.v_iter):
                    for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                        v_mb_grads = self.worker.value_gradient_for_ith_mb(mb_index)
                        all_stats.append(self.worker.get_stats().copy())
                        self.worker.apply_v_gradients(v_mb_grads)

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
        self.get_stats()

    def stop(self):
        pass


class TRPOOptimizer(object):
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
        self.search_result = 1
        self.step_timer = TimerStat()
        self.value_optimizing_timer = TimerStat()
        self.direction_computing_timer = TimerStat()
        self.sample_and_po_timer = TimerStat()
        self.line_search_timer = TimerStat()

        logger.info('Optimizer initialized')

    def get_stats(self):
        self.stats.update(dict(iteration=self.iteration,
                               num_sampled_steps=self.num_sampled_steps,
                               step_time=self.step_timer.mean,
                               sample_and_po_time=self.sample_and_po_timer.mean,
                               value_optimizing_time=self.value_optimizing_timer.mean,
                               direction_computing_time=self.direction_computing_timer.mean,
                               line_search_time=self.line_search_timer.mean,
                               search_result=self.search_result
                               )
                          )
        return self.stats

    def step(self):
        logger.info('begin the {}-th optimizing step'.format(self.iteration))
        logger.info('sampling {} in total'.format(self.num_sampled_steps))
        with self.step_timer:
            with self.sample_and_po_timer:
                for worker in self.workers['remote_workers']:
                    worker.sample_and_process.remote()
                flat_gs = ray.get([worker.prepare_for_policy_update.remote() for worker in self.workers['remote_workers']])
                worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers['remote_workers']])
                mean_stats_before = {}
                for key in worker_stats[0].keys():
                    value_list = list(map(lambda x: x[key], worker_stats))
                    mean_stats_before.update({key: sum(value_list) / len(value_list)})
                mean_flat_g = np.array(flat_gs).mean(axis=0)

            def compute_fvp(vec):
                return np.array(ray.get([worker.compute_fvp.remote(vec)
                                         for i, worker in enumerate(self.workers['remote_workers'])])).mean(axis=0) \
                       + self.args.cg_damping * vec

            with self.direction_computing_timer:
                # the following is to solve the update direction: Hessian^-1 * obj_grad by construct Ax=b
                # Ax=b -> min J(x) = 1/2 * (x^T)Ax - (b^T)x, then solve it by cg method
                # 1. Given x_1=0, k=1,
                # 2. g_k = J'(x_k) = Ax_k-b (g_{k+1}-g_k = A(x_{k+1}-x_k) = lambda*Ad_k)
                # 3. d_k = -g_k + beta_{k-1}*d_{k-1}, beta_0 = 0, beta_{k-1} = ||g_k||^2/||g_{k-1}||^2
                # 4. x_{k+1} = x_k + lambda*d_k, lambda_k=-(g_k)^T d_k/(d_k)^T A(d_k) = (g_k)^T g_k/(d_k)^T A(d_k)
                # in our problem, A is fisher matrix, which is the Hessian matrix of KL divergence, denoted by H,
                # b is the obj grad, i.e. mean_flat_g
                d = mean_flat_g.copy()  # conjugate gradient direction
                g = -mean_flat_g.copy()  # local gradient
                x = np.zeros_like(mean_flat_g)  # x_1
                gdotg = g.dot(g)
                print("{:10s} {:10s} {:10s}".format("cg_iter", "residual norm", "soln norm"))
                for i in range(self.args.cg_iters):
                    print("{:<10d} {:10.3g} {:10.3g}".format(i, gdotg, np.linalg.norm(x)))
                    Hd = compute_fvp(d)  # H*d
                    lmda = gdotg / d.dot(Hd)
                    x += lmda * d
                    g += lmda * Hd
                    newgdotg = g.dot(g)
                    beta = newgdotg / gdotg
                    d = -g + beta * d
                    gdotg = newgdotg
                    if gdotg < self.args.residual_tol:
                        break
                stepdir = x

            with self.line_search_timer:
                # with the direction, line search is used to determine the stepsize
                if not np.isfinite(stepdir).all():
                    stepdir = np.zeros_like(stepdir)
                    logger.info('stepdir is nan!, zero it')
                shs = .5 * stepdir.dot(compute_fvp(stepdir))
                lm = np.sqrt(shs / self.args.max_kl)
                fullstep = stepdir / lm
                expectedimprove = mean_flat_g.dot(fullstep)
                pggainbefore = mean_stats_before['pg_gain']
                stepsize = 1.0
                thbefore = self.local_worker.policy_with_value.policy.get_weights()
                self.search_result = 1
                for _ in range(10):
                    thnew_flat = flatvars(thbefore) - fullstep * stepsize
                    thnew = unflatvars(thnew_flat, get_shapes(self.local_worker.policy_with_value.policy.trainable_weights))
                    self.local_worker.policy_with_value.policy.set_weights(thnew)
                    self.sync_remote_workers()
                    ray.get([worker.prepare_for_policy_update.remote() for worker in self.workers['remote_workers']])
                    worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers['remote_workers']])
                    mean_stats = {}
                    for key in worker_stats[0].keys():
                        value_list = list(map(lambda x: x[key], worker_stats))
                        mean_stats.update({key: sum(value_list) / len(value_list)})
                    pggain, kl = mean_stats['pg_gain'], mean_stats['kl']
                    improve = pggain - pggainbefore
                    logger.info("Expected: {:.3f} Actual: {:.3f}".format(expectedimprove, improve))
                    if not np.isfinite(list(mean_stats.values())).all():
                        logger.info("Got non-finite value of losses -- bad!")
                    elif kl > self.args.max_kl * 1.5:
                        logger.info("Violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.info("Surrogate didn't improve. shrinking step.")
                    else:
                        logger.info("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    self.search_result = 0
                    logger.info("couldn't compute a good step")
                    self.local_worker.policy_with_value.policy.set_weights(thbefore)
                    ray.get([worker.prepare_for_policy_update.remote() for worker in self.workers['remote_workers']])

                self.sync_remote_workers()

            all_stats = [[] for _ in range(self.args.num_workers)]
            with self.value_optimizing_timer:
                for i in range(self.args.v_iter):
                    for mb_index in range(int(self.args.sample_batch_size / self.args.mini_batch_size)):
                        v_mb_grads = ray.get([worker.value_gradient_for_ith_mb.remote(mb_index)
                                              for worker in self.workers['remote_workers']])
                        v_final_grads = np.array(v_mb_grads).mean(axis=0).tolist()
                        worker_stats = ray.get([worker.get_stats.remote() for worker in self.workers['remote_workers']])
                        for worker_index in range(self.args.num_workers):
                            all_stats[worker_index].append(worker_stats[worker_index].copy())
                        self.local_worker.apply_v_gradients(v_final_grads)
                        self.sync_remote_workers()

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
        self.get_stats()

    def sync_remote_workers(self):
        weights = ray.put(self.local_worker.get_weights())
        for e in self.workers['remote_workers']:
            e.set_weights.remote(weights)

    def stop(self):
        pass

