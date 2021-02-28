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
from utils.misc import TimerStat, safemean, flatvars
from utils.monitor import Monitor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TRPOWorker(object):
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
        self.sampling_timer = TimerStat()
        self.processing_timer = TimerStat()
        self.epinfobuf = deque(maxlen=100)
        self.permutation = None
        self.policy_shapes = [var.get_shape().as_list() for var in self.policy_with_value.policy.trainable_weights]

        self.stats = {}
        logger.info('TRPO Worker initialized')

    def get_stats(self):
        return self.stats

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_logps': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               }

        return tmp

    def get_batch_data(self, batch_data):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns, batch_values = self.compute_advantage()
        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_values=batch_values))

        self.old_policy_with_value.set_weights(self.policy_with_value.get_weights())
        return self.batch_data

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

    @tf.function
    def apply_v_gradients(self, v_grads):
        self.policy_with_value.value_optimizer.apply_gradients(zip(v_grads,
                                                                   self.policy_with_value.value.trainable_weights))

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
            data = self.get_batch_data(batch_data)
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

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])
        batch_obs = self.batch_data['batch_obs']
        batch_rewards = self.batch_data['batch_rewards']
        batch_obs_tensor = self.tf.constant(batch_obs)
        batch_values = self.policy_with_value.compute_vf(batch_obs_tensor).numpy()
        batch_advs = np.zeros_like(self.batch_data['batch_rewards'])
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1. - self.batch_data['batch_dones'][t]
            delta = batch_rewards[t] + self.args.gamma * batch_values[t + 1] * nextnonterminal - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values

        return batch_advs, batch_tdlambda_returns, batch_values

    @tf.function
    def fisher_vector_product(self, subsampling_obs, vector):  # vector is flatten vars
        with self.tf.GradientTape() as outter_tape:
            with self.tf.GradientTape() as inner_tape:
                kl = self.policy_with_value.compute_kl(subsampling_obs,
                                                       self.old_policy_with_value.policy(subsampling_obs))
            kl_grads = inner_tape.gradient(kl, self.policy_with_value.policy.trainable_weights)
            start = 0
            tangents = []
            for shape in self.policy_shapes:
                sz = int(np.prod(shape))
                tangents.append(self.tf.reshape(vector[start:start + sz], shape))
                start += sz
            gvp = self.tf.add_n([self.tf.reduce_sum(g * tangent) for (g, tangent) in zip(kl_grads, tangents)])
        fvp = outter_tape.gradient(gvp, self.policy_with_value.policy.trainable_weights)
        flat_fvp = flatvars(fvp)
        return flat_fvp

    def compute_fvp(self, vector):
        batch_obs = self.batch_data['batch_obs']
        subsampling_index = np.arange(0, len(batch_obs), self.args.subsampling)
        subsampling_obs = self.tf.constant(batch_obs[subsampling_index])
        vector = self.tf.constant(vector)
        return self.fisher_vector_product(subsampling_obs, vector)

    @tf.function
    def compute_g(self, batch_obses, batch_actions, batch_logps, batch_advs):
        with self.tf.GradientTape() as tape:
            kl = self.policy_with_value.compute_kl(batch_obses,
                                                   self.old_policy_with_value.policy(batch_obses))
            policy_entropy = self.policy_with_value.compute_entropy(batch_obses)
            current_logp = self.policy_with_value.compute_logps(batch_obses, batch_actions)
            ratio = self.tf.exp(current_logp - batch_logps)
            surr_gain = self.tf.reduce_mean(ratio * batch_advs)
            ent_bonus = self.args.ent_coef * policy_entropy
            pg_gain = surr_gain + ent_bonus
        g = tape.gradient(pg_gain, self.policy_with_value.policy.trainable_weights)
        flat_g = flatvars(g)
        return pg_gain, flat_g, surr_gain, ent_bonus, policy_entropy, kl

    def prepare_for_policy_update(self):
        batch_obs = self.tf.constant(self.batch_data['batch_obs'])
        batch_advs = self.tf.constant(self.batch_data['batch_advs'])
        batch_tdlambda_returns = self.tf.constant(self.batch_data['batch_tdlambda_returns'])
        batch_actions = self.tf.constant(self.batch_data['batch_actions'])
        batch_logps = self.tf.constant(self.batch_data['batch_logps'])

        with self.g_grad_timer:
            pg_gain, flat_g, surr_gain, ent_bonus, policy_entropy, kl =\
                self.compute_g(batch_obs, batch_actions, batch_logps, batch_advs)

        self.stats.update(dict(
            pg_time=self.g_grad_timer.mean,
            pg_gain=pg_gain.numpy(),
            surr_gain=surr_gain.numpy(),
            ent_bonus=ent_bonus.numpy(),
            policy_entropy=policy_entropy.numpy(),
            kl=kl.numpy(),
            target_mean=np.mean(batch_tdlambda_returns),
        ))

        return flat_g.numpy()

    @tf.function
    def value_forward_and_backward(self, batch_obses, target):
        with self.tf.GradientTape() as tape:
            v_pred = self.policy_with_value.compute_vf(batch_obses)
            v_loss = .5 * self.tf.reduce_mean(self.tf.square(v_pred - target))
            value_mean = self.tf.reduce_mean(v_pred)
        value_gradient = tape.gradient(v_loss, self.policy_with_value.value.trainable_weights)
        value_gradient, value_gradient_norm = self.tf.clip_by_global_norm(value_gradient, self.args.gradient_clip_norm)
        return v_loss, value_gradient, value_mean, value_gradient_norm

    def value_gradient_for_ith_mb(self, i):
        if i == 0:
            self.permutation = np.arange(len(self.batch_data['batch_obs']))
            np.random.shuffle(self.permutation)
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mbinds = self.permutation[start_idx:end_idx]
        mb_obs = self.tf.constant(self.batch_data['batch_obs'][mbinds])
        mb_tdlambda_returns = self.tf.constant(self.batch_data['batch_tdlambda_returns'][mbinds])
        with self.v_grad_timer:
            v_loss, value_gradient, value_mean, value_gradient_norm \
                = self.value_forward_and_backward(mb_obs, mb_tdlambda_returns)
        self.stats.update(dict(
            v_timer=self.v_grad_timer.mean,
            v_loss=v_loss.numpy(),
            value_mean=value_mean.numpy(),
            value_gradient_norm=value_gradient_norm.numpy(),
        ))

        return value_gradient
