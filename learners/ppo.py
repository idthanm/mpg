#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/9
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ppo.py
# =====================================

import logging

import numpy as np
import tensorflow as tf

from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PPOLearner(tf.Module):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_with_value, args):
        super().__init__()
        self.args = args
        self.policy_with_value = policy_with_value
        self.batch_data = None
        self.mb_learning_timer = TimerStat()
        self.stats = {}
        self.permutation = None

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
        return self.batch_data

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
    def get_grads(self, mb_obses, mb_actions, mb_logps, mb_advs, target, mb_oldvs):
        mb_advs = (mb_advs - self.tf.reduce_mean(mb_advs)) / (self.tf.keras.backend.std(mb_advs) + 1e-8)
        with self.tf.GradientTape() as tape:
            v_pred = self.policy_with_value.compute_vf(mb_obses)
            vpredclipped = mb_oldvs + self.tf.clip_by_value(v_pred - mb_oldvs,
                                                            -self.args.ppo_loss_clip,
                                                            self.args.ppo_loss_clip)
            v_loss1 = self.tf.square(v_pred - target)
            v_loss2 = self.tf.square(vpredclipped - target)
            v_loss = .5 * self.tf.reduce_mean(self.tf.maximum(v_loss1, v_loss2))

            current_logp = self.policy_with_value.compute_logps(mb_obses, mb_actions)
            ratio = self.tf.exp(current_logp - mb_logps)
            pg_loss1 = ratio * mb_advs
            pg_loss2 = mb_advs * self.tf.clip_by_value(ratio, 1 - self.args.ppo_loss_clip, 1 + self.args.ppo_loss_clip)
            pg_loss = -self.tf.reduce_mean(self.tf.minimum(pg_loss1, pg_loss2))

            policy_entropy = self.policy_with_value.compute_entropy(mb_obses)
            ent_bonus = self.args.ent_coef * policy_entropy

            value_mean = self.tf.reduce_mean(v_pred)
            approxkl = .5 * self.tf.reduce_mean(self.tf.square(current_logp - mb_logps))
            clipfrac = self.tf.reduce_mean(self.tf.cast(
                self.tf.greater(self.tf.abs(ratio - 1.0), self.args.ppo_loss_clip), self.tf.float32))

            total_loss = v_loss + pg_loss - ent_bonus

        grads = tape.gradient(total_loss, self.policy_with_value.trainable_variables)
        grad, grad_norm = self.tf.clip_by_global_norm(grads, self.args.gradient_clip_norm)
        return grad, grad_norm, pg_loss, ent_bonus, policy_entropy, clipfrac, v_loss, value_mean, approxkl

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        if i == 0:
            self.permutation = np.arange(self.args.sample_batch_size)
            np.random.shuffle(self.permutation)
        with self.mb_learning_timer:
            start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
            mbinds = self.permutation[start_idx:end_idx]
            mb_obs = self.tf.constant(self.batch_data['batch_obs'][mbinds])
            mb_advs = self.tf.constant(self.batch_data['batch_advs'][mbinds])
            mb_tdlambda_returns = self.tf.constant(self.batch_data['batch_tdlambda_returns'][mbinds])
            mb_actions = self.tf.constant(self.batch_data['batch_actions'][mbinds])
            mb_logps = self.tf.constant(self.batch_data['batch_logps'][mbinds])
            mb_oldvs = self.tf.constant(self.batch_data['batch_values'][mbinds])

            grad, grad_norm, pg_loss, ent_bonus, policy_entropy, clipfrac, v_loss, value_mean, approxkl = \
                self.get_grads(mb_obs, mb_actions, mb_logps, mb_advs, mb_tdlambda_returns, mb_oldvs)

        self.stats = dict(
            mb_learning_time=self.mb_learning_timer.mean,
            v_loss=v_loss.numpy(),
            policy_loss=pg_loss.numpy(),
            ent_bonus=ent_bonus.numpy(),
            policy_entropy=policy_entropy.numpy(),
            value_mean=value_mean.numpy(),
            target_mean=np.mean(mb_tdlambda_returns),
            grad_norm=grad_norm.numpy(),
            clipfrac=clipfrac.numpy(),
            approxkl=approxkl.numpy()
        )

        return grad
