#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/09
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

from gym import spaces

from model import MLPNet, MLPNetDSAC, PPONet
import tensorflow as tf

NAME2MODELCLS = dict([('MLP', MLPNet), ('DSAC', MLPNetDSAC), ('PPO', PPONet)])


class PolicyWithValue(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.args = args
        assert isinstance(obs_space, spaces.Box)
        assert isinstance(act_space, spaces.Box)
        obs_dim = obs_space.shape[0] if args.obs_dim is None else self.args.obs_dim
        act_dim = act_space.shape[0] if args.act_dim is None else self.args.act_dim
        n_hiddens, n_units = self.args.num_hidden_layers, self.args.num_hidden_units
        value_model_cls = NAME2MODELCLS[self.args.value_model_cls]
        policy_model_cls = NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        self.value = value_model_cls(obs_dim, n_hiddens, n_units, 1, name='value')
        self.models = (self.policy, self.value,)
        policy_lr_schedule = self.tf.keras.optimizers.schedules.PolynomialDecay(*self.args.policy_lr_schedule)
        value_lr_schedule = self.tf.keras.optimizers.schedules.PolynomialDecay(*self.args.value_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule)
        self.value_optimizer = self.tf.keras.optimizers.Adam(value_lr_schedule)
        self.optimizers = (self.policy_optimizer, self.value_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    # @property
    # def trainable_weights(self):
    #     return self.tf.nest.flatten([model.trainable_weights for model in self.models])

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        value_weights_len = len(self.value.trainable_weights)
        value_grad, policy_grad = grads[:value_weights_len], grads[value_weights_len:]
        self.value_optimizer.apply_gradients(zip(value_grad, self.value.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            actions = act_dist.sample()
            logps = act_dist.log_prob(actions)
            return actions, logps

    @tf.function
    def compute_logps(self, obs, actions):
        with self.tf.name_scope('compute_logps') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            if self.args.action_range is not None:
                act_dist = (
                    self.tfp.distributions.TransformedDistribution(
                        distribution=act_dist,
                        bijector=self.tfb.Affine(scale_identity_multiplier=(1. + 1e-6))
                    ))
            return act_dist.log_prob(actions)

    @tf.function
    def compute_entropy(self, obs):
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            try:
                entropy = self.tf.reduce_mean(act_dist.entropy())
            except NotImplementedError:
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                entropy = -self.tf.reduce_mean(logps)
            finally:
                return entropy

    @tf.function
    def compute_kl(self, obs, other_out):  # KL(other||ego)
        with self.tf.name_scope('compute_entropy') as scope:
            logits = self.policy(obs)
            act_dist = self._logits2dist(logits)
            other_act_dist = self._logits2dist(self.tf.stop_gradient(other_out))
            try:
                kl = self.tf.reduce_mean(other_act_dist.kl_divergence(act_dist))
            except NotImplementedError:
                other_actions = other_act_dist.sample()
                other_logps = other_act_dist.log_prob(other_actions)
                logps = self.compute_logps(obs, other_actions)
                kl = self.tf.reduce_mean(other_logps - logps)
            finally:
                return kl

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    @tf.function
    def compute_vf(self, obs):
        with self.tf.name_scope('compute_value') as scope:
            return self.value(obs)


def test_logps():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors

    import numpy as np
    mean, log_std = np.array([[0.]], np.float32), np.array([[4]], np.float32)
    base_dist = tfd.MultivariateNormalDiag(mean, tf.exp(log_std))
    act_dist = (
            tfp.distributions.TransformedDistribution(
                distribution=base_dist,
                bijector=tfb.Chain(
                    [tfb.Affine(scale_identity_multiplier=2.+1e-6),
                     tfb.Tanh()])))
    actions = act_dist.sample()
    log_pis = act_dist.log_prob(np.array([[1.5]]))
    logps = tf.reduce_sum(log_pis, axis=-1)
    print(actions, logps)


def testMultivariateNormalDiag():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    import numpy as np
    mean, log_std = np.array([[0.1]]), np.array([[0.1]])

    dist = tfd.MultivariateNormalDiag(mean, tf.exp(log_std))
    print(dist.sample())


if __name__ == "__main__":
    test_logps()
