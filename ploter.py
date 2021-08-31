#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.font_manager as fm
china_font = fm.FontProperties(fname='/home/idlaber/programs/anaconda3/envs/torch_gpu/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf',size=18)
sns.set(style="darkgrid")

palette = [(1.0, 0.48627450980392156, 0.0),
                    (0.9098039215686274, 0.0, 0.043137254901960784),
                    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                    (0.6235294117647059, 0.2823529411764706, 0.0),]

WINDOWSIZE = 15
reward_scale = 1.

def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_opt_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['learner_stats/scalar/obj_loss',
                'learner_stats/scalar/obj_v_loss',
                'learner_stats/scalar/pg_loss',
                'learner_stats/scalar/punish_factor',
                'learner_stats/scalar/real_punish_term',
                'learner_stats/scalar/veh2road4real',
                'learner_stats/scalar/veh2veh4real',
                ]
    alg_list = ['CrossroadEnd2endPiIntegrate-v0']
    task_list = ['DPIE']
    palette = "bright"
    lbs = ['动态排列不变编码', '固定排序状态编码']
    # lbs = ['动态排列不变编码', '固定排序状态编码']
    dir_str = './results/{}/data2plot_{}'
    df_list = []
    for alg in alg_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir + '/logs/optimizer'
                opt_file = os.path.join(opt_dir,
                                         [file_name for file_name in os.listdir(opt_dir) if
                                          file_name.startswith('events')][0])
                opt_summarys = tf.data.TFRecordDataset([opt_file])
                data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg.update({'iteration': []})
                for opt_summary in opt_summarys:
                    event = event_pb2.Event.FromString(opt_summary.numpy())
                    for v in event.summary.value:
                        t = tf.make_ndarray(v.tensor)
                        for tag in tag2plot:
                            if tag in v.tag:
                                data_in_one_run_of_one_alg[tag].append(float(t) * reward_scale)
                                data_in_one_run_of_one_alg['iteration'].append(int(event.step))
                len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                period = int(len1 / len2)
                data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                           i in range(len2)]

                data_in_one_run_of_one_alg = {key: val[2:] for key, val in data_in_one_run_of_one_alg.items()}
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (10, 8)
    axes_size = [0.12, 0.12, 0.87, 0.87]
    fontsize = 25

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.13, 0.10, 0.86, 0.86])
    sns.lineplot(x="iteration", y="learner_stats/scalar/obj_loss_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette,)
    plt.ylim(0, 100)
    # handles, labels = ax1.get_legend_handles_labels()
    # labels = lbs
    # ax1.legend(handles=handles, labels=labels, prop=china_font, loc='upper right', frameon=False, fontsize=fontsize)
    ax1.set_ylabel('$J_{\\rm track}$', fontsize=fontsize)
    ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_track.pdf')

    f2 = plt.figure(2, figsize=figsize)
    ax2 = f2.add_axes([0.15, 0.12, 0.85, 0.86])
    sns.lineplot(x="iteration", y="learner_stats/scalar/obj_v_loss_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    plt.ylim(0, 1000)
    ax2.set_ylabel('$J_{\\rm critic}$', fontsize=fontsize)
    ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_critic.pdf')

    f3 = plt.figure(3, figsize=figsize)
    ax3 = f3.add_axes([0.12, 0.12, 0.87, 0.86])
    sns.lineplot(x="iteration", y="learner_stats/scalar/real_punish_term_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax3.set_ylabel('$J_{\\rm penalty}$', fontsize=fontsize)
    ax3.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_penalty.pdf')

    f4 = plt.figure(4, figsize=figsize)
    ax4 = f4.add_axes([0.11, 0.12, 0.86, 0.86])
    sns.lineplot(x="iteration", y="learner_stats/scalar/veh2veh4real", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax4.set_ylabel('Ego-to-veh Penalty', fontsize=fontsize)
    ax4.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    # plt.ylim(0, 3)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./ego2veh_penalty.pdf')
    plt.show()


if __name__ == "__main__":
    plot_opt_results_of_all_alg_n_runs()