#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging
import os
from collections import deque

import gym
import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat, safemean
from utils.monitor import Monitor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OnPolicyWorker(object):
    """
    Act as both actor and learner
    """
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, learner_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env = gym.make(env_id)
        self.env = Monitor(env)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.learner = learner_cls(self.policy_with_value, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.log_dir = self.args.log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.stats = {}
        self.sampling_timer = TimerStat()
        self.processing_timer = TimerStat()
        self.epinfobuf = deque(maxlen=100)
        logger.info('Worker initialized')

    def get_stats(self):
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_grads_sepe(self, grads):
        self.policy_with_value.apply_grads_sepe(grads)

    def apply_grads_all(self, grads):
        self.policy_with_value.apply_grads_all(grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample_and_process(self):
        with self.sampling_timer:
            epinfos = []
            batch_data = []
            for _ in range(self.sample_batch_size):
                processed_obs = self.preprocessor.process_obs(self.obs)
                processed_obs_tensor = self.tf.constant(processed_obs[np.newaxis, :])
                action, logp = self.policy_with_value.compute_action(processed_obs_tensor)
                action, logp = action.numpy()[0], logp.numpy()[0]

                obs_tp1, reward, self.done, info = self.env.step(action)
                processed_rew = self.preprocessor.process_rew(reward, self.done)
                self.obs = self.env.reset() if self.done else obs_tp1.copy()
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

                batch_data.append((processed_obs.copy(), action, processed_rew, self.done, logp))

        with self.processing_timer:
            data = self.learner.get_batch_data(batch_data)
        ev = 1. - np.var(data['batch_tdlambda_returns']-data['batch_values'])/np.var(data['batch_tdlambda_returns'])
        self.epinfobuf.extend(epinfos)
        self.stats.update(explained_variance=ev,
                          eprewmean=safemean([epinfo['r'] for epinfo in self.epinfobuf]),
                          eplenmean=safemean([epinfo['l'] for epinfo in self.epinfobuf]))
        self.stats.update(dict(worker_sampling_time=self.sampling_timer.mean,
                               worker_processing_time=self.processing_timer.mean))
        if self.args.reward_preprocess_type == 'normalize':
            self.stats.update(dict(ret_rms_var=self.preprocessor.ret_rms.var,
                                   ret_rms_mean=self.preprocessor.ret_rms.mean))

    def compute_gradient_over_ith_minibatch(self, i):
        grad = self.learner.compute_gradient_over_ith_minibatch(i)
        learner_stats = self.learner.get_stats()
        self.stats.update(learner_stats)
        return grad


def debug_worker():
    from train_script import built_PPO_parser
    from policy import PolicyWithValue
    from learners.ppo import PPOLearner
    env_id = 'Pendulum-v0'
    worker_id = 0
    args = built_PPO_parser()
    worker = OnPolicyWorker(PolicyWithValue, PPOLearner, env_id, args, worker_id)
    for _ in range(10):
        worker.sample_and_process()


if __name__ == '__main__':
    debug_worker()

