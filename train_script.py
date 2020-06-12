#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import ray

from mixed_pg_learner import MixedPGLearner
from off_policy.buffer import PrioritizedReplayBuffer, ReplayBuffer
from off_policy.offpolicy_mb_learner import OffPolicyMBLearner
from off_policy.offpolicy_optimizer import OffPolicyAsyncOptimizer
from optimizer import AllReduceOptimizer
from policy import PolicyWithQs
from trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# logging.getLogger().setLevel(logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NAME2LEARNERCLS = dict([('Offpolicy_MB', OffPolicyMBLearner), ('Mixed_PG', MixedPGLearner)])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('priority', PrioritizedReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('AllReduce', AllReduceOptimizer), ('OffPolicyAsync', OffPolicyAsyncOptimizer)])


def built_offpolicy_mb_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", default='Offpolicy_MB')
    parser.add_argument("--env_id", default='CrossroadEnd2end-v0')
    parser.add_argument('--off_policy', default=True, action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_learners', type=int, default=3)
    parser.add_argument('--num_buffers', type=int, default=2)

    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')

    parser.add_argument('--max_sampled_steps', type=int, default=1000000)
    parser.add_argument('--max_updated_steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=5)

    parser.add_argument('--training_task', type=str, default='left')

    parser.add_argument('--num_future_data', type=int, default=5)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--model_based', default=True)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[5])
    parser.add_argument('--num_rollout_list_for_q_estimation', type=list, default=[5])
    parser.add_argument('--deriv_interval_policy', default=True)

    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=1500)
    parser.add_argument('--replay_batch_size', type=int, default=64)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)

    parser.add_argument("--policy_lr_schedule", type=list,
                        default=[3e-4, 20000, 3e-6])
    parser.add_argument("--value_lr_schedule", type=list,
                        default=[8e-4, 20000, 8e-6])
    parser.add_argument("--gradient_clip_norm", type=float, default=3)
    parser.add_argument("--deterministic_policy", default=True, action='store_true')

    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument('--Q_num', type=int, default=1)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='normalize')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument("--obs_scale_factor", type=list, default=[0.2, 1., 2., 1., 2.4] +
                                                                 [1.]*num_future_data)
    parser.add_argument("--reward_preprocess_type", type=str, default='normalize')
    parser.add_argument("--reward_scale_factor", type=float, default=0.01)

    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/toyota/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)

    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    # parser.add_argument("--model_load_dir", type=str, default='./results/mixed_pg/experiment-2020-04-22-13-30-57/models')
    # parser.add_argument("--model_load_ite", type=int, default=160)
    # parser.add_argument("--ppc_load_dir", type=str, default='./results/mixed_pg/experiment-2020-04-22-13-30-57/models')

    return parser.parse_args()


def built_mixedpg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_name", default='Mixed_PG')
    parser.add_argument("--env_id", default='CrossroadEnd2end-v0')
    parser.add_argument('--off_policy', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--max_sampled_steps', type=int, default=1000000)
    parser.add_argument('--max_updated_steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=5)

    parser.add_argument('--training_task', type=str, default='left')

    parser.add_argument('--num_future_data', type=int, default=5)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--model_based', default=True)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[5])
    parser.add_argument('--num_rollout_list_for_q_estimation', type=list, default=[5])
    parser.add_argument('--deriv_interval_policy', default=True)

    parser.add_argument("--policy_lr_schedule", type=list,
                        default=[3e-4, 20000, 3e-6])
    parser.add_argument("--value_lr_schedule", type=list,
                        default=[8e-4, 20000, 8e-6])
    parser.add_argument("--gradient_clip_norm", type=float, default=3)
    parser.add_argument("--deterministic_policy", default=True, action='store_true')

    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument('--Q_num', type=int, default=1)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='normalize')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument("--obs_scale_factor", type=list, default=[0.2, 1., 2., 1., 2.4] +
                                                                 [1.]*num_future_data)
    parser.add_argument("--reward_preprocess_type", type=str, default='normalize')
    parser.add_argument("--reward_scale_factor", type=float, default=0.01)

    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--optimizer_type', type=str, default='AllReduce')

    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/toyota/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)

    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    # parser.add_argument("--model_load_dir", type=str, default='./results/mixed_pg/experiment-2020-04-22-13-30-57/models')
    # parser.add_argument("--model_load_ite", type=int, default=160)
    # parser.add_argument("--ppc_load_dir", type=str, default='./results/mixed_pg/experiment-2020-04-22-13-30-57/models')

    return parser.parse_args()


def built_parser(alg_name):
    if alg_name == 'Offpolicy_MB':
        return built_offpolicy_mb_parser()
    elif alg_name == 'Mixed_PG':
        return built_mixedpg_parser()


def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    ray.init(redis_max_memory=1024*1024*1024, object_store_memory=1024*1024*1024)
    os.makedirs(args.result_dir)
    with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    trainer = Trainer(policy_cls=PolicyWithQs,
                      learner_cls=NAME2LEARNERCLS[args.alg_name],
                      buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                      optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                      args=args)
    if args.model_load_dir is not None:
        logger.info('loading model')
        trainer.load_weights(args.model_load_dir, args.model_load_ite)
    if args.ppc_load_dir is not None:
        logger.info('loading ppc parameter')
        trainer.load_ppc_params(args.ppc_load_dir)

    trainer.train()


if __name__ == '__main__':
    main('Offpolicy_MB')
