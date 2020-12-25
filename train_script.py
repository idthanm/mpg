#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import ray

from evaluator import Evaluator
from learners.ppo import PPOLearner
from learners.trpo import TRPOWorker
from optimizer import AllReduceOptimizer, TRPOOptimizer, SingleProcessOptimizer, SingleProcessTRPOOptimizer
from policy import PolicyWithValue
from tester import Tester
from trainer import Trainer
from worker import OnPolicyWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NAME2WORKERCLS = dict([('OnPolicyWorker', OnPolicyWorker), ('TRPOWorker', TRPOWorker)])
NAME2LEARNERCLS = dict([('PPO', PPOLearner), ('TRPO', None)])
NAME2BUFFERCLS = dict([('None', None),])
NAME2OPTIMIZERCLS = dict([('AllReduce', AllReduceOptimizer),
                          ('TRPOOptimizer', TRPOOptimizer),
                          ('SingleProcess', SingleProcessOptimizer),
                          ('SingleProcessTRPOOptimizer', SingleProcessTRPOOptimizer)])
NAME2POLICIES = dict([('PolicyWithValue', PolicyWithValue)])
NAME2EVALUATORS = dict([('Evaluator', Evaluator)])

def built_PPO_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/PPO/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=70))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithValue')
    parser.add_argument('--worker_type', type=str, default='OnPolicyWorker')
    parser.add_argument('--optimizer_type', type=str, default='SingleProcess')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--off_policy', type=str, default=False)

    # env
    parser.add_argument("--env_id", default='Ant-v2')
    #Humanoid-v2 Ant-v2 HalfCheetah-v2 Walker2d-v2 InvertedDoublePendulum-v2 Pendulum-v0
    env_id = parser.parse_args().env_id
    action_range = 0.4 if env_id == 'Humanoid-v2' else 1.
    parser.add_argument("--action_range", type=float, default=None)

    # learner
    parser.add_argument("--alg_name", default='PPO')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--ppo_loss_clip", type=float, default=0.2)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--ent_coef", type=float, default=0.0)

    # worker
    parser.add_argument('--sample_batch_size', type=int, default=2048)

    # tester and evaluator
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_log_interval", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--eval_render", type=bool, default=False)

    # policy and model
    parser.add_argument("--value_model_cls", type=str, default='MLP')
    parser.add_argument("--policy_model_cls", type=str, default='PPO')
    parser.add_argument("--policy_lr_schedule", type=list, default=[3e-4, 320*488, 0.])
    parser.add_argument("--value_lr_schedule", type=list, default=[3e-4, 320*488, 0.])
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=64)
    parser.add_argument('--hidden_activation', type=str, default='tanh')
    parser.add_argument("--policy_out_activation", type=str, default='linear')

    # preprocessor
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='normalize')
    parser.add_argument("--obs_scale", type=list, default=None)
    parser.add_argument("--reward_preprocess_type", type=str, default='normalize')
    parser.add_argument("--reward_scale", type=float, default=None)
    parser.add_argument("--reward_shift", type=float, default=None)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=488)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/PPO/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)
    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    return parser.parse_args()

def built_TRPO_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/TRPO/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=70))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithValue')
    parser.add_argument('--worker_type', type=str, default='TRPOWorker')
    parser.add_argument('--optimizer_type', type=str, default='SingleProcessTRPOOptimizer')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--off_policy', type=str, default=False)

    # env
    parser.add_argument("--env_id", default='Ant-v2')
    # Humanoid-v2 Ant-v2 HalfCheetah-v2 Walker2d-v2 InvertedDoublePendulum-v2, Pendulum-v0
    env_id = parser.parse_args().env_id
    action_range = 0.4 if env_id == 'Humanoid-v2' else 1.
    parser.add_argument("--action_range", type=float, default=None)

    # learner
    parser.add_argument("--alg_name", default='TRPO')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.98)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.5)
    parser.add_argument("--v_iter", type=int, default=5)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--ent_coef", type=float, default=0.)
    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--cg_damping", type=float, default=0.1)
    parser.add_argument("--max_kl", type=float, default=0.001)
    parser.add_argument("--residual_tol", type=float, default=1e-10)
    parser.add_argument("--subsampling", type=int, default=5)

    # worker
    parser.add_argument('--sample_batch_size', type=int, default=1024)

    # tester and evaluator
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_log_interval", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--eval_render", type=bool, default=False)

    # policy and model
    parser.add_argument("--value_model_cls", type=str, default='MLP')
    parser.add_argument("--policy_model_cls", type=str, default='PPO')
    parser.add_argument("--policy_lr_schedule", type=list, default=[1e-3, 1000, 1e-3])
    parser.add_argument("--value_lr_schedule", type=list, default=[1e-3, 1000, 1e-3])
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=32)
    parser.add_argument('--hidden_activation', type=str, default='tanh')
    parser.add_argument("--policy_out_activation", type=str, default='linear')

    # preprocessor
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='normalize')
    parser.add_argument("--obs_scale", type=list, default=None)
    parser.add_argument("--reward_preprocess_type", type=str, default='normalize')
    parser.add_argument("--reward_scale", type=float, default=None)
    parser.add_argument("--reward_shift", type=float, default=None)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/TRPO/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)
    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    return parser.parse_args()

def built_PPO_parser_for_DSAC():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/PPO/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=70))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithValue')
    parser.add_argument('--worker_type', type=str, default='OnPolicyWorker')
    parser.add_argument('--optimizer_type', type=str, default='SingleProcess')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--off_policy', type=str, default=False)

    # env
    parser.add_argument("--env_id", default='Ant-v2')
    #Humanoid-v2 Ant-v2 HalfCheetah-v2 Walker2d-v2 InvertedDoublePendulum-v2 Pendulum-v0
    env_id = parser.parse_args().env_id
    action_range = 0.4 if env_id == 'Humanoid-v2' else 1.
    parser.add_argument("--action_range", type=float, default=action_range)

    # learner
    parser.add_argument("--alg_name", default='PPO')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--gradient_clip_norm", type=float, default=100000000.)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--ppo_loss_clip", type=float, default=0.2)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--ent_coef", type=float, default=0.0)

    # worker
    parser.add_argument('--sample_batch_size', type=int, default=2048)

    # tester and evaluator
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_log_interval", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--eval_render", type=bool, default=False)

    max_inner_iter = 500000 if env_id == 'InvertedDoublePendulum-v2' else 3000000
    epoch = parser.parse_args().epoch
    batch_size = parser.parse_args().sample_batch_size
    mb_size = parser.parse_args().mini_batch_size
    inner_iter_per_iter = epoch * int(batch_size / mb_size)
    max_iter = int(max_inner_iter / inner_iter_per_iter)
    eval_num = 150
    eval_interval = save_interval = int(int(max_inner_iter / eval_num) / inner_iter_per_iter)

    # policy and model
    parser.add_argument("--value_model_cls", type=str, default='MLP')
    parser.add_argument("--policy_model_cls", type=str, default='DSAC')
    parser.add_argument("--policy_lr_schedule", type=list, default=[5e-5, max_inner_iter, 1e-6])
    parser.add_argument("--value_lr_schedule", type=list, default=[8e-5, max_inner_iter, 1e-6])
    parser.add_argument('--num_hidden_layers', type=int, default=5)
    parser.add_argument('--num_hidden_units', type=int, default=256)
    parser.add_argument('--hidden_activation', type=str, default='elu')
    parser.add_argument("--policy_out_activation", type=str, default='linear')

    # preprocessor
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default=None)
    parser.add_argument("--obs_scale", type=list, default=None)
    parser.add_argument("--reward_preprocess_type", type=str, default='scale')
    parser.add_argument("--reward_scale", type=float, default=0.2)
    parser.add_argument("--reward_shift", type=float, default=0.)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=max_iter)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=eval_interval)
    parser.add_argument("--save_interval", type=int, default=save_interval)
    parser.add_argument("--log_interval", type=int, default=1)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/PPO/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)
    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    return parser.parse_args()

def built_TRPO_parser_for_DSAC():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/TRPO/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[0],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=70))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithValue')
    parser.add_argument('--worker_type', type=str, default='TRPOWorker')
    parser.add_argument('--optimizer_type', type=str, default='SingleProcessTRPOOptimizer')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='None')
    parser.add_argument('--off_policy', type=str, default=False)

    # env
    parser.add_argument("--env_id", default='Ant-v2')
    # Humanoid-v2 Ant-v2 HalfCheetah-v2 Walker2d-v2 InvertedDoublePendulum-v2, Pendulum-v0
    env_id = parser.parse_args().env_id
    action_range = 0.4 if env_id == 'Humanoid-v2' else 1.
    parser.add_argument("--action_range", type=float, default=action_range)

    # learner
    parser.add_argument("--alg_name", default='TRPO')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.98)
    parser.add_argument("--gradient_clip_norm", type=float, default=100000000.)
    parser.add_argument("--v_iter", type=int, default=5)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--ent_coef", type=float, default=0.)
    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--cg_damping", type=float, default=0.1)
    parser.add_argument("--max_kl", type=float, default=0.01)
    parser.add_argument("--residual_tol", type=float, default=1e-10)
    parser.add_argument("--subsampling", type=int, default=5)

    # worker
    parser.add_argument('--sample_batch_size', type=int, default=1024)

    # tester and evaluator
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_log_interval", type=int, default=1)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--eval_render", type=bool, default=False)

    max_inner_iter = 500000 if env_id == 'InvertedDoublePendulum-v2' else 3000000
    epoch = parser.parse_args().v_iter
    batch_size = parser.parse_args().sample_batch_size
    mb_size = parser.parse_args().mini_batch_size
    inner_iter_per_iter = epoch * int(batch_size / mb_size)
    max_iter = int(max_inner_iter / inner_iter_per_iter)
    eval_num = 150
    eval_interval = save_interval = int(int(max_inner_iter / eval_num) / inner_iter_per_iter)

    # policy and model
    parser.add_argument("--value_model_cls", type=str, default='MLP')
    parser.add_argument("--policy_model_cls", type=str, default='DSAC')
    parser.add_argument("--policy_lr_schedule", type=list, default=[1e-3, max_inner_iter, 1e-3])
    parser.add_argument("--value_lr_schedule", type=list, default=[1e-3, max_inner_iter, 1e-3])
    parser.add_argument('--num_hidden_layers', type=int, default=5)
    parser.add_argument('--num_hidden_units', type=int, default=256)
    parser.add_argument('--hidden_activation', type=str, default='elu')
    parser.add_argument("--policy_out_activation", type=str, default='linear')

    # preprocessor
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default=None)
    parser.add_argument("--obs_scale", type=list, default=None)
    parser.add_argument("--reward_preprocess_type", type=str, default='scale')
    parser.add_argument("--reward_scale", type=float, default=0.2)
    parser.add_argument("--reward_shift", type=float, default=0.)

    # Optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=max_iter)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=eval_interval)
    parser.add_argument("--save_interval", type=int, default=save_interval)
    parser.add_argument("--log_interval", type=int, default=1)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/TRPO/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)
    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    return parser.parse_args()


def built_parser(alg_name):
    if alg_name == 'PPO':
        return built_PPO_parser()
    elif alg_name == 'TRPO':
        return built_TRPO_parser()

def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=5120*1024*1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=NAME2POLICIES[args.policy_type],
                          worker_cls=NAME2WORKERCLS[args.worker_type],
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                          evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        if args.ppc_load_dir is not None:
            logger.info('loading ppc parameter')
            trainer.load_ppc_params(args.ppc_load_dir)
        trainer.train()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(policy_cls=NAME2POLICIES[args.policy_type],
                        evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('PPO')
