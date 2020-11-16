#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================
# @Time    : 2020/11/9
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: trpo.py
# =====================================
import logging
import os
from collections import deque

import gym
import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat, safemean, judge_is_nan, flatvars
from utils.monitor import Monitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TRPOWorker(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, learner_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env = gym.make(env_id)
        self.env = Monitor(env)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.old_policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.log_dir = self.args.log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.batch_data = None
        self.g_grad_timer = TimerStat()
        self.v_grad_timer = TimerStat()
        self.epinfobuf = deque(maxlen=100)

        self.stats = {}
        logger.info('TRPO Worker initialized')

    def get_stats(self):
        return self.stats

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_logps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32),
               }

        return tmp

    def get_batch_data(self, batch_data, epinfos):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns, batch_values = self.compute_advantage()
        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_values=batch_values))
        # self.subsampling()
        self.shuffle()
        self.epinfobuf.extend(epinfos)
        self.stats.update(eprewmean=safemean([epinfo['r'] for epinfo in self.epinfobuf]),
                          eplenmean=safemean([epinfo['l'] for epinfo in self.epinfobuf]))
        self.old_policy_with_value.set_weights(self.policy_with_value.get_weights())

    def subsampling(self):
        sampling_index = np.arange(0, self.batch_data_count(), self.args.subsampling)
        for key, val in self.batch_data.items():
            self.batch_data.update({'sub_'+key: val[sampling_index]})

    def batch_data_count(self):
        return len(self.batch_data['batch_obs'])

    def shuffle(self):
        permutation = np.random.permutation(self.batch_data_count())
        for key, val in self.batch_data.items():
            self.batch_data[key] = val[permutation]

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def apply_gradients(self, iteration, grads):
        self.policy_with_value.apply_gradients(iteration, grads)

    def apply_v_gradients(self, iteration, v_grads):
        self.policy_with_value.value_optimizer.apply_gradients(zip(v_grads,
                                                                   self.policy_with_value.value.trainable_weights))

    def apply_policy_gradients(self, iteration, policy_grads):
        self.policy_with_value.policy_optimizer.apply_gradients(zip(policy_grads,
                                                                    self.policy_with_value.policy.trainable_weights))

    def sample(self):
        batch_data = []
        epinfos = []
        for _ in range(self.sample_batch_size):
            judge_is_nan(self.obs)
            processed_obs = self.preprocessor.process_obs(self.obs)
            action, logp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
            # print(action, logp)
            judge_is_nan(action)
            judge_is_nan(logp)
            obs_tp1, reward, self.done, info = self.env.step(action[0].numpy())
            processed_rew = self.preprocessor.process_rew(reward, self.done)

            batch_data.append((self.obs, action[0].numpy(), reward, obs_tp1, self.done, logp[0].numpy()))
            self.obs = self.env.reset() if self.done else obs_tp1.copy()
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
        return batch_data, epinfos

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])
        processed_obses = self.preprocessor.np_process_obses(self.batch_data['batch_obs'])
        processed_rewards = self.preprocessor.np_process_rewards(self.batch_data['batch_rewards'])
        batch_values = self.policy_with_value.value(processed_obses).numpy()[:, 0]  # len = n_steps + 1
        batch_advs = np.zeros_like(self.batch_data['batch_rewards'])
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t + 1]
            delta = processed_rewards[t] + self.args.gamma * batch_values[t + 1] * nextnonterminal - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns, batch_values

    @tf.function
    def fisher_vector_product(self, subsampling_obs, vector):  # vector is flatten vars
        with self.tf.GradientTape() as tape1:
            with self.tf.GradientTape() as tape2:
                processed_obses = self.preprocessor.tf_process_obses(subsampling_obs)
                kl = self.policy_with_value.compute_kl(processed_obses, self.old_policy_with_value)
            kl_grads = tape2.gradient(kl, self.policy_with_value.policy.trainable_weights)
            flat_kl_grads = flatvars(kl_grads)
            total = self.tf.reduce_mean(flat_kl_grads*vector)
        fvp = tape1.gradient(total, self.policy_with_value.policy.trainable_weights)
        flat_fvp = flatvars(fvp)
        return flat_fvp

    def compute_fvp(self, vector):
        batch_obs = self.batch_data['batch_obs']
        subsampling_index = np.arange(0, len(batch_obs), self.args.subsampling)
        subsampling_obs = batch_obs[subsampling_index]
        return self.fisher_vector_product(subsampling_obs, vector)

    @tf.function
    def compute_g(self, batch_obs, batch_actions, batch_neglogps, batch_advs):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(batch_obs)
            kl = self.policy_with_value.compute_kl(processed_obses, self.old_policy_with_value)
            policy_entropy = self.policy_with_value.compute_entropy(processed_obses)
            current_neglogp = -self.policy_with_value.compute_logps(processed_obses, batch_actions)
            ratio = self.tf.exp(batch_neglogps - current_neglogp)
            surr_loss = -self.tf.reduce_mean(ratio * batch_advs)
            ent_bonus = self.args.ent_coef * policy_entropy
            pg_loss = surr_loss - ent_bonus
        g = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
        flat_g = flatvars(g)
        return pg_loss, flat_g, surr_loss, ent_bonus, policy_entropy, kl

    def prepare_for_policy_update(self):
        batch_obs = self.batch_data['batch_obs']
        batch_advs = self.batch_data['batch_advs']
        batch_tdlambda_returns = self.batch_data['batch_tdlambda_returns']
        batch_actions = self.batch_data['batch_actions']
        batch_neglogps = -self.batch_data['batch_logps']
        with self.v_grad_timer:
            v_loss, value_gradient, value_mean = self.value_forward_and_backward(batch_obs, batch_tdlambda_returns)
            value_gradient, value_gradient_norm = self.tf.clip_by_global_norm(value_gradient, self.args.gradient_clip_norm)
        with self.g_grad_timer:
            pg_loss, flat_g, surr_loss, ent_bonus, policy_entropy, kl =\
                self.compute_g(batch_obs, batch_actions, batch_neglogps, batch_advs)

        self.stats.update(dict(
            v_timer=self.v_grad_timer.mean,
            pg_time=self.g_grad_timer.mean,
            v_loss=v_loss.numpy(),
            policy_loss=pg_loss.numpy(),
            surr_loss=surr_loss.numpy(),
            ent_bonus=ent_bonus.numpy(),
            policy_entropy=policy_entropy.numpy(),
            kl=kl.numpy(),
            value_mean=value_mean.numpy(),
            target_mean=np.mean(batch_tdlambda_returns),
            value_gradient_norm=value_gradient_norm.numpy(),
        ))
        if self.args.reward_preprocess_type == 'normalize':
            self.stats.update(dict(ret_rms_var=self.preprocessor.ret_rms.var,
                                   ret_rms_mean=self.preprocessor.ret_rms.mean))
        return flat_g

    @tf.function
    def value_forward_and_backward(self, mb_obs, target):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            v_pred = self.policy_with_value.compute_vf(processed_obses)
            v_loss = .5 * self.tf.reduce_mean(self.tf.square(v_pred - target))
            value_mean = self.tf.reduce_mean(v_pred)
        value_gradient = tape.gradient(v_loss, self.policy_with_value.value.trainable_weights)
        return v_loss, value_gradient, value_mean

    def value_gradient_for_ith_mb(self, i):
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        v_loss, value_gradient, value_mean = self.value_forward_and_backward(mb_obs, mb_tdlambda_returns)
        value_gradient, value_gradient_norm = self.tf.clip_by_global_norm(value_gradient, self.args.gradient_clip_norm)

        return list(map(lambda x: x.numpy(), value_gradient))
