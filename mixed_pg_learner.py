import numpy as np
import gym
from gym.envs.user_defined.toyota_env.dynamics_and_models import EnvironmentModel
from preprocessor import Preprocessor
import time
import logging
from collections import deque
from utils.misc import safemean

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# logger.setLevel(logging.INFO)

# def judge_is_nan(list_of_np_or_tensor):
#     for m in list_of_np_or_tensor:
#         if hasattr(m, 'numpy'):
#             if np.any(np.isnan(m.numpy())):
#                 print(list_of_np_or_tensor)
#                 raise ValueError
#         else:
#             if np.any(np.isnan(m)):
#                 print(list_of_np_or_tensor)
#                 raise ValueError
#
#
# def judge_less_than(list_of_np_or_tensor, thres=0.001):
#     for m in list_of_np_or_tensor:
#         if hasattr(m, 'numpy'):
#             assert not np.all(m.numpy() < thres)
#         else:
#             assert not np.all(m < thres)


class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    @property
    def mean(self):
        return np.mean(self._samples)


class MixedPGLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args, obs_space, act_space):
        self.args = args
        self.batch_size = self.args.batch_size
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.all_data = {}
        self.policy_for_rollout = policy_cls(obs_space, act_space, self.args)
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update
        self.num_rollout_list_for_q_estimation = self.args.num_rollout_list_for_q_estimation

        self.model = EnvironmentModel(task=self.args.training_task,
                                      num_future_data=self.args.num_future_data)
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.stats = {}
        self.reduced_num_minibatch = 1
        self.w_list_old = 1/len(self.num_rollout_list_for_policy_update)*np.ones(len(self.num_rollout_list_for_policy_update))
        assert self.args.mini_batch_size % self.reduced_num_minibatch == 0
        self.epinfobuf = deque(maxlen=100)

    def get_stats(self):
        return self.stats

    def get_batch_data(self, batch_data, epinfos):
        self.batch_data = self.post_processing(batch_data)
        self.epinfobuf.extend(epinfos)
        eprewmean = safemean([epinfo['r'] for epinfo in self.epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in self.epinfobuf])
        self.stats.update(dict(eprewmean=eprewmean,
                               eplenmean=eplenmean))
        batch_advs, batch_tdlambda_returns = self.compute_advantage()

        batch_mc = self.compute_mc_estimate()

        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_mc=batch_mc
                                    ))
        self.shuffle()

        # print(self.batch_data['batch_obs'].shape)  # batch_size * obs_dim
        # print(self.batch_data['batch_actions'].shape)  # batch_size * act_dim
        # print(self.batch_data['batch_advs'].shape)  # batch_size,
        # print(self.batch_data['batch_tdlambda_returns'].shape)  # batch_size,

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_neglogps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32)}
        return tmp

    def shuffle(self):
        permutation = np.random.permutation(self.batch_size)
        for key, val in self.batch_data.items():
            self.batch_data[key] = val[permutation]

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def compute_mc_estimate(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])  # (n_step, )
        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        mc = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastmc = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t]
            mc[t] = lastmc = processed_batch_rewards[t] + self.args.gamma * nextnonterminal * lastmc
        return mc

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])  # (n_step, )
        # print(self.batch_data['batch_rewards'].shape)
        # print(self.batch_data['batch_obs'].shape)
        # print(self.batch_data['batch_actions'].shape)
        # print(self.batch_data['batch_obs_tp1'].shape)
        # print(self.batch_data['batch_dones'].shape)
        # print(self.batch_data['batch_neglogps'].shape)

        processed_batch_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_batch_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()
        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        batch_values = \
            self.policy_with_value.compute_Q_target(processed_batch_obs, self.batch_data['batch_actions']).numpy()[:, 0]
        act_tp1, _ = self.policy_with_value.compute_action(processed_batch_obs_tp1)
        batch_values_tp1 = \
            self.policy_with_value.compute_Q_target(processed_batch_obs_tp1, act_tp1.numpy()).numpy()[:, 0]

        batch_advs = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t]
            delta = processed_batch_rewards[t] + self.args.gamma * (batch_values_tp1[t] if nextnonterminal < 0.1 else
                                                                    batch_values[t + 1]) - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def model_rollout_for_q_estimation(self, start_obses, start_actions):
        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        actions_tile = self.tf.tile(start_actions, [self.M, 1])

        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]
        not_dones_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile, self.args.training_task)
        max_num_rollout = max(self.num_rollout_list_for_q_estimation)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards, dones = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))
                not_dones_list.append(1. - self.tf.cast(dones, self.tf.float32))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q_target(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)
            all_not_dones = self.tf.concat(not_dones_list, 0)
            all_targets = all_rewards_sums + all_gammas * all_Qs * all_not_dones

            final = self.tf.reshape(all_targets, (max_num_rollout + 1, self.M, -1))
            all_model_returns = self.tf.reduce_mean(final, axis=1)
        selected_model_returns = []
        for num_rollout in self.num_rollout_list_for_q_estimation:
            selected_model_returns.append(all_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        return self.tf.stop_gradient(selected_model_returns_flatten)

    def model_rollout_for_policy_update(self, start_obses):
        processed_start_obses = self.preprocessor.tf_process_obses(start_obses)
        start_actions, _ = self.policy_with_value.compute_action(processed_start_obses)
        # judge_is_nan(start_actions)

        max_num_rollout = max(self.num_rollout_list_for_policy_update)

        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        actions_tile = self.tf.tile(start_actions, [self.M, 1])
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]
        not_dones_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile, self.args.training_task)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards, dones = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_for_rollout.compute_action(processed_obses_tile) if not \
                    self.args.deriv_interval_policy else self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))
                not_dones_list.append(1. - self.tf.cast(dones, self.tf.float32))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)
            all_not_dones = self.tf.concat(not_dones_list, 0)

            final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs * all_not_dones,
                                    (max_num_rollout + 1, self.M, -1))
            # final [[[time0+traj0], [time0+traj1], ..., [time0+trajn]],
            #        [[time1+traj0], [time1+traj1], ..., [time1+trajn]],
            #        ...
            #        [[timen+traj0], [timen+traj1], ..., [timen+trajn]],
            #        ]
            all_model_returns = self.tf.reduce_mean(final, axis=1)
        interval = int(self.args.mini_batch_size / self.reduced_num_minibatch)
        all_reduced_model_returns = self.tf.stack(
            [self.tf.reduce_mean(all_model_returns[:, i * interval:(i + 1) * interval], axis=-1) for i in
             range(self.reduced_num_minibatch)], axis=1)

        selected_model_returns, minus_selected_reduced_model_returns = [], []
        for num_rollout in self.num_rollout_list_for_policy_update:
            selected_model_returns.append(all_model_returns[num_rollout])
            minus_selected_reduced_model_returns.append(-all_reduced_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        minus_selected_reduced_model_returns_flatten = self.tf.concat(minus_selected_reduced_model_returns, 0)
        value_mean = self.tf.reduce_mean(all_model_returns[0])
        return selected_model_returns_flatten, minus_selected_reduced_model_returns_flatten, value_mean

    # @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_targets):
        if self.args.model_based:
            processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
            model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
            with self.tf.GradientTape() as tape:
                with self.tf.name_scope('q_loss') as scope:
                    q_pred = self.policy_with_value.compute_Q(processed_mb_obs, mb_actions)[:, 0]
                    q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - model_targets))
            with self.tf.name_scope('q_gradient') as scope:
                q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)

            return model_targets, q_gradient, q_loss, [self.tf.constant(1.)]

        else:
            processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
            with self.tf.GradientTape() as tape:
                with self.tf.name_scope('q_loss') as scope:
                    q_pred = self.policy_with_value.compute_Q(processed_mb_obs, mb_actions)[:, 0]
                    q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - mb_targets))

            with self.tf.name_scope('q_gradient') as scope:
                q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)
            model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
            model_bias_list = []
            for i, num_rollout in enumerate(self.num_rollout_list_for_q_estimation):
                model_target_i = model_targets[i * self.args.mini_batch_size:
                                               (i + 1) * self.args.mini_batch_size]
                model_bias_list.append(self.tf.reduce_mean(self.tf.abs(model_target_i-mb_targets)))
            return model_targets, q_gradient, q_loss, model_bias_list

    # @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            model_returns, minus_reduced_model_returns, value_mean = self.model_rollout_for_policy_update(mb_obs)

        with self.tf.name_scope('policy_jacobian') as scope:
            jaco = tape.jacobian(minus_reduced_model_returns,
                                 self.policy_with_value.policy.trainable_weights,
                                 unconnected_gradients='zero',
                                 experimental_use_pfor=False)
                                 # )
            # shape is len(self.policy_with_value.models[1].trainable_weights) * len(model_returns)
            # [[dy1/dx1, dy2/dx1,...(rolloutnum1)|dy1/dx1, dy2/dx1,...(rolloutnum2)| ...],
            #  [dy1/dx2, dy2/dx2, ...(rolloutnum1)|dy1/dx2, dy2/dx2,...(rolloutnum2)| ...],
            #  ...]
            return model_returns, minus_reduced_model_returns, jaco, value_mean

    def export_graph(self, writer):
        start_idx, end_idx = 0, self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_mc_target = self.batch_data['batch_mc'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_mc_target)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        mb_mc_target = self.batch_data['batch_mc'][start_idx: end_idx]
        rewards_mean = np.abs(np.mean(self.preprocessor.np_process_rewards(self.batch_data['batch_rewards'])))

        with self.q_gradient_timer:
            model_targets, q_gradient, q_loss, model_bias_list = self.q_forward_and_backward(mb_obs, mb_actions,
                                                                                             mb_mc_target)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            model_returns, minus_reduced_model_returns, jaco, value_mean = self.policy_forward_and_backward(mb_obs)

        model_bias_list = [a.numpy() for a in model_bias_list]
        bias_min = min(model_bias_list)
        model_bias_list = [a-bias_min+rewards_mean for a in model_bias_list]
        policy_gradient_list = []
        heuristic_bias_list = model_bias_list
        var_list = []
        final_policy_gradient = []

        for rollout_index in range(len(self.num_rollout_list_for_policy_update)):
            jaco_for_this_rollout = list(map(lambda x: x[rollout_index * self.reduced_num_minibatch:
                                                         (rollout_index + 1) * self.reduced_num_minibatch], jaco))

            gradient_std = []
            gradient_mean = []
            var = 0.
            for x in jaco_for_this_rollout:
                gradient_std.append(self.tf.math.reduce_std(x, 0))
                gradient_mean.append(self.tf.reduce_mean(x, 0))
                var += self.tf.reduce_mean(self.tf.square(gradient_std[-1])).numpy()

            policy_gradient_list.append(gradient_mean)
            var_list.append(var)

        epsilon = 1e-8
        heuristic_bias_inverse_sum = self.tf.reduce_sum(
            list(map(lambda x: 1. / (x + epsilon), heuristic_bias_list))).numpy()
        var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1. / (x + epsilon), var_list))).numpy()

        w_heur_bias_list = list(
            map(lambda x: (1. / (x + epsilon)) / heuristic_bias_inverse_sum, heuristic_bias_list))
        w_var_list = list(map(lambda x: (1. / (x + epsilon)) / var_inverse_sum, var_list))

        w_list_new = list(map(lambda x, y: 0.5*x + 0.5*y, w_heur_bias_list, w_var_list))

        w_list = list(self.w_list_old + 0.01 * (np.array(w_list_new)-self.w_list_old))
        self.w_list_old = np.array(w_list)

        # w_list = w_heur_bias_list

        for i in range(len(policy_gradient_list[0])):
            tmp = 0
            for j in range(len(policy_gradient_list)):
                # judge_is_nan(policy_gradient_list[j])
                tmp += w_list[j] * policy_gradient_list[j][i]
            final_policy_gradient.append(tmp)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(final_policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(

            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            q_loss=q_loss.numpy(),
            value_mean=value_mean.numpy(),
            q_gradient_norm=q_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            num_traj_rollout=self.M,
            num_rollout_list=self.num_rollout_list_for_policy_update,
            var_list=var_list,
            heuristic_bias_list=heuristic_bias_list,
            w_var_list=w_var_list,
            w_heur_bias_list=w_heur_bias_list,
            w_list_new=w_list_new,
            w_list=w_list
        ))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
