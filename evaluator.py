#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

import gym
import numpy as np

from preprocessor import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id,
                            training_task=self.args.training_task,
                            num_future_data=self.args.num_future_data)
        self.policy_with_value = policy_cls(self.env.observation_space, self.env.action_space, self.args)
        self.iteration = 0
        self.log_dir = self.args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(self.env.observation_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir + '/evaluator')
        self.stats = {}

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self):
        reward_list = []
        reward_info_dict_list = []
        done = 0
        obs = self.env.reset()
        self.env.render()
        while not done:
            processed_obs = self.preprocessor.tf_process_obses(obs)
            action, neglogp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
            obs, reward, done, info = self.env.step(action[0].numpy())
            reward_info_dict_list.append(dict(punish_steer=info['punish_steer'],
                                              punish_a_x=info['punish_a_x'],
                                              punish_yaw_rate=info['punish_yaw_rate'],
                                              devi_v=info['devi_v'],
                                              devi_y=info['devi_y'],
                                              devi_phi=info['devi_phi'],
                                              veh2road=info['veh2road'],
                                              veh2veh=info['veh2veh'],
                                              rew_alpha_f=info['rew_alpha_f'],
                                              rew_alpha_r=info['rew_alpha_r'],
                                              rew_r=info['rew_r']
                                              ))
            self.env.render()
            reward_list.append(reward)
        self.env.close()
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in ['punish_steer', 'punish_a_x', 'punish_yaw_rate', 'devi_v', 'devi_y',
                    'devi_phi', 'veh2road', 'veh2veh', 'rew_alpha_f', 'rew_alpha_r', 'rew_r']:
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        return episode_return, episode_len, info_dict

    def run_n_episode(self, n):
        list_of_return = []
        list_of_len = []
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            episode_return, episode_len, info_dict = self.run_an_episode()
            list_of_return.append(episode_return)
            list_of_len.append(episode_len)
            list_of_info_dict.append(info_dict)
        average_return = sum(list_of_return) / len(list_of_return)
        average_len = sum(list_of_len) / len(list_of_len)
        n_info_dict = dict()
        for key in ['punish_steer', 'punish_a_x', 'punish_yaw_rate', 'devi_v', 'devi_y',
                    'devi_phi', 'veh2road', 'veh2veh', 'rew_alpha_f', 'rew_alpha_r', 'rew_r']:
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return average_return, average_len, n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        self.iteration = iteration
        average_return, average_len, n_info_dict = self.run_n_episode(5)
        n_info_dict.update(dict(average_return=average_return,
                                average_len=average_len))
        logger.info(n_info_dict)
        with self.writer.as_default():
            for key, val in n_info_dict.items():
                self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)

            self.writer.flush()


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)


if __name__ == '__main__':
    model_dir = './results/mixed_pg/experiment-2020-04-22-14-01-37/models'
    # model_dir = './results/mixed_pg/experiment-2020-04-22-15-02-12/models'
    print(test_trained_model(model_dir, model_dir, 20))
