#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2

sns.set(style="darkgrid")
ENV2ID = {'ant': 'Ant-v2',
          'halfcheetah': 'HalfCheetah-v2',
          'humanoid': 'Humanoid-v2',
          'inverted': 'InvertedDoublePendulum-v2',
          'walker': 'Walker2d-v2'}
METHOD2IDX = {'ppo': '7', 'trpo': '8'}


def save_eval_results_of_all_alg_n_runs_all_env(dirs_dict_for_plot=None):
    tag2plot = ['average_return_with_max1', 'average_return_with_max3', 'average_return_with_max5', 'true_v_mean', 'eval_v_mean']
    for alg in ['ppo', 'trpo']:
        for env in ['ant', 'halfcheetah', 'humanoid', 'inverted', 'walker']:
            data2plot_dir = './{alg}_{env}1/results/{alg_upper}/data2plot'.format(alg=alg, env=env, alg_upper=alg.upper())
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
                eval_file = os.path.join(eval_dir,
                                         [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
                eval_summarys = tf.data.TFRecordDataset([eval_file])
                data_in_one_run_of_one_alg_one_env = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg_one_env.update({'iteration': []})
                for eval_summary in eval_summarys:
                    event = event_pb2.Event.FromString(eval_summary.numpy())
                    for v in event.summary.value:
                        t = tf.make_ndarray(v.tensor)
                        for tag in tag2plot:
                            if tag in v.tag:
                                data_in_one_run_of_one_alg_one_env[tag].append(float(t))
                                data_in_one_run_of_one_alg_one_env['iteration'].append(int(event.step))
                len1, len2 = len(data_in_one_run_of_one_alg_one_env['iteration']), len(data_in_one_run_of_one_alg_one_env[tag2plot[0]])
                period = int(len1 / len2)
                if alg == 'ppo':
                    freq = 80
                else:
                    assert alg == 'trpo'
                    freq = 20
                data_in_one_run_of_one_alg_one_env['iteration'] = [
                    data_in_one_run_of_one_alg_one_env['iteration'][i * period] * freq for i in range(len2)]

                # -------------------------------save------------------------------------
                np.save(
                    './' + ENV2ID[env] + '-run' + str(num_run) + '/method_' + METHOD2IDX[alg] + '/result/iteration.npy',
                    np.array(data_in_one_run_of_one_alg_one_env['iteration']))
                np.save('./' + ENV2ID[env] + '-run' + str(num_run) + '/method_' + METHOD2IDX[
                    alg] + '/result/average_return_with_diff_base.npy',
                        np.array([data_in_one_run_of_one_alg_one_env['average_return_with_max1'],
                                  data_in_one_run_of_one_alg_one_env['average_return_with_max3'],
                                  data_in_one_run_of_one_alg_one_env['average_return_with_max5']]))
                np.save(
                    './' + ENV2ID[env] + '-run' + str(num_run) + '/method_' + METHOD2IDX[alg] + '/result/evaluated_Q_mean.npy',
                    np.array(data_in_one_run_of_one_alg_one_env['eval_v_mean']))
                np.save(
                    './' + ENV2ID[env] + '-run' + str(num_run) + '/method_' + METHOD2IDX[
                        alg] + '/result/true_gamma_return_mean.npy',
                    np.array(data_in_one_run_of_one_alg_one_env['true_v_mean']))
                # -------------------------------save------------------------------------


if __name__ == "__main__":
    save_eval_results_of_all_alg_n_runs_all_env()
