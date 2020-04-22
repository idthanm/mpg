from abc import ABC
import gym
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_rad):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original rad
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: rad
    :return:
    transformed_x, transformed_y, transformed_d
    """

    transformed_x = orig_x * tf.cos(coordi_rotate_rad) + orig_y * tf.sin(coordi_rotate_rad)
    transformed_y = -orig_x * tf.sin(coordi_rotate_rad) + orig_y * tf.cos(coordi_rotate_rad)
    transformed_d = orig_d - coordi_rotate_rad
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


class VehicleDynamics(object):
    def __init__(self, if_model=False):
        if if_model:
            self.vehicle_params = OrderedDict(C_f=100000.,  # front wheel cornering stiffness [N/rad]
                                              C_r=100000.,  # rear wheel cornering stiffness [N/rad]
                                              a=1.5,  # distance from CG to front axle [m]
                                              b=1.5,  # distance from CG to rear axle [m]
                                              mass=3000.,  # mass [kg]
                                              I_z=4000.,  # Polar moment of inertia at CG [kg*m^2]
                                              miu=0.8,  # tire-road friction coefficient
                                              g=9.81,  # acceleration of gravity [m/s^2]
                                              )
        else:
            self.vehicle_params = OrderedDict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
                                              C_r=94000.,  # rear wheel cornering stiffness [N/rad]
                                              a=1.14,  # distance from CG to front axle [m]
                                              b=1.40,  # distance from CG to rear axle [m]
                                              mass=1500.,  # mass [kg]
                                              I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
                                              miu=1.0,  # tire-road friction coefficient
                                              g=9.81,  # acceleration of gravity [m/s^2]
                                              )
        self.if_model = if_model
        self.expected_vs = 20.
        self.path = ReferencePath()

    def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
        with tf.name_scope('f_xu') as scope:
            # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
            # 1, 2, 0.2, 2.4, 1, 2, 0.4

            # 0.2 * torch.tensor([1, 5, 10, 12, 5, 10, 2]
            # vx, vy, r, delta_phi, delta_y
            # veh_full_state: v_ys, rs, v_xs, phis, ys, xs
            v_x, v_y, r, delta_y, delta_phi = states[:, 0], states[:, 1], states[:, 2], \
                                                          states[:, 3], states[:, 4]
            steer, a_x = actions[:, 0], actions[:, 1]

            C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
            C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
            a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
            b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
            mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
            I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
            miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
            g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

            F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
            F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x, dtype=tf.float32))
            F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
            miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
            miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
            alpha_f = tf.atan((v_y + a * r) / v_x) - steer
            alpha_r = tf.atan((v_y - b * r) / v_x)

            Ff_w1 = tf.square(C_f) / (3 * F_zf * miu_f)
            Ff_w2 = tf.pow(C_f, 3) / (27 * tf.pow(F_zf * miu_f, 2))
            F_yf_max = F_zf * miu_f

            Fr_w1 = tf.square(C_r) / (3 * F_zr * miu_r)
            Fr_w2 = tf.pow(C_r, 3) / (27 * tf.pow(F_zr * miu_r, 2))
            F_yr_max = F_zr * miu_r

            F_yf = - C_f * tf.tan(alpha_f) + Ff_w1 * tf.tan(alpha_f) * tf.abs(
                tf.tan(alpha_f)) - Ff_w2 * tf.pow(tf.tan(alpha_f), 3)
            F_yr = - C_r * tf.tan(alpha_r) + Fr_w1 * tf.tan(alpha_r) * tf.abs(
                tf.tan(alpha_r)) - Fr_w2 * tf.pow(tf.tan(alpha_r), 3)

            F_yf = tf.minimum(F_yf, F_yf_max)
            F_yf = tf.maximum(F_yf, -F_yf_max)

            F_yr = tf.minimum(F_yr, F_yr_max)
            F_yr = tf.maximum(F_yr, -F_yr_max)

            # tmp_f = tf.square(C_f * tf.tan(alpha_f)) / (27 * tf.square(miu_f * F_zf)) - C_f * tf.abs(tf.tan(alpha_f)) / (
            #         3 * miu_f * F_zf) + 1
            # tmp_r = tf.square(C_r * tf.tan(alpha_r)) / (27 * tf.square(miu_r * F_zr)) - C_r * tf.abs(tf.tan(alpha_r)) / (
            #         3 * miu_r * F_zr) + 1
            #
            # F_yf = -tf.sign(alpha_f) * tf.minimum(tf.abs(C_f * tf.tan(alpha_f) * tmp_f), tf.abs(miu_f * F_zf))
            # F_yr = -tf.sign(alpha_r) * tf.minimum(tf.abs(C_r * tf.tan(alpha_r) * tmp_r), tf.abs(miu_r * F_zr))
            if self.if_model:
                state_deriv = [a_x + 1.5,
                               (F_yf + F_yr) / mass,
                               (a * F_yf - b * F_yr) / I_z,
                               v_x * tf.sin(delta_phi),
                               r+1,
                               ]
            else:
                state_deriv = [a_x + v_y * r,
                               (F_yf * tf.cos(steer) + F_yr) / mass - v_x * r,
                               (a * F_yf * tf.cos(steer) - b * F_yr) / I_z,
                               v_x * tf.sin(delta_phi) + v_y * tf.cos(delta_phi),
                               r,
                               ]

            state_deriv_stack = tf.stack(state_deriv, axis=1)
        return state_deriv_stack

    def prediction(self, x_1, u_1, frequency, RK):
        if RK == 1:
            f_xu_1 = self.f_xu(x_1, u_1)
            x_next = f_xu_1 / frequency + x_1

        elif RK == 2:
            f_xu_1 = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1
            f_xu_2 = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_next = x_1 + (K1 + K2) / 2
        else:
            assert RK == 4
            f_xu_1 = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1 / 2
            f_xu_2 = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_3 = x_1 + K2 / 2
            f_xu_3 = self.f_xu(x_3, u_1)
            K3 = (1 / frequency) * f_xu_3
            x_4 = x_1 + K3
            f_xu_4 = self.f_xu(x_4, u_1)
            K4 = (1 / frequency) * f_xu_4
            x_next = x_1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return x_next

    def simulation(self, states, full_states, actions, base_freq, simu_times):
        def cal_steer_bound(v):
            bounds = -1.2*np.pi/180. * (v-10.) + 1.2*np.pi/9.
            bounds[v < 10] = 1.2 * np.pi / 9
            bounds[v > 20] = 1.2*np.pi/18.
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
        # veh_full_state: v_xs, v_ys, rs, ys, phis, steers, a_xs, xs
        for i in range(simu_times):
            states = tf.convert_to_tensor(states.copy(), dtype=tf.float32)
            states = self.prediction(states, actions, base_freq, 1).numpy().copy()
            states[:, 0] = np.clip(states[:, 0], 1, 35)
            # states[:, 1] = np.clip(states[:, 1], -2, 2)

            v_xs, v_ys, rs, phis = full_states[:, 0], full_states[:, 1], full_states[:, 2], full_states[:, 4]

            full_states[:, 4] += rs / base_freq
            full_states[:, 3] += (v_xs * np.sin(phis) + v_ys * np.cos(phis)) / base_freq
            full_states[:, -1] += (v_xs * np.cos(phis) - v_ys * np.sin(phis)) / base_freq
            full_states[:, 0:3] = states[:, 0:3].copy()

            path_y, path_phi = self.path.compute_path_y(full_states[:, -1]), \
                               self.path.compute_path_phi(full_states[:, -1])
            states[:, 4] = full_states[:, 4] - path_phi
            states[:, 3] = full_states[:, 3] - path_y

            full_states[:, 4][full_states[:, 4] > np.pi] -= 2 * np.pi
            full_states[:, 4][full_states[:, 4] <= -np.pi] += 2 * np.pi

            states[:, 4][states[:, 4] > np.pi] -= 2 * np.pi
            states[:, 4][states[:, 4] <= -np.pi] += 2 * np.pi

        return states, full_states

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        with tf.name_scope('compute_reward') as scope:
            # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis
            # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
            v_xs, v_ys, rs, delta_ys, delta_phis = states[:, 0], states[:, 1], states[:, 2], \
                                                                 states[:, 3], states[:, 4],
            steers, a_xs = actions[:, 0], actions[:, 1]

            devi_v = -tf.square(v_xs - self.expected_vs)
            devi_y = -tf.square(delta_ys)
            devi_phi = -tf.square(delta_phis)
            punish_yaw_rate = -tf.square(rs)
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)

            rewards = 0.01 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.05 * punish_steer + 0.0005 * punish_a_x

        return rewards


class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(7.5, 200, 0.), (2.5, 300., 0.), (-5., 400., 0.)]

    def compute_path_y(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            y += magnitude * np.sin((x - shift) * 2 * np.pi / T)
        return y

    def compute_path_phi(self, x):
        deriv = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            deriv += magnitude * 2 * np.pi / T * np.cos(
                (x - shift) * 2 * np.pi / T)
        return np.arctan(deriv)

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        phi = delta_phi + phi_ref
        phi[phi > np.pi] -= 2 * np.pi
        phi[phi <= -np.pi] += 2 * np.pi
        return phi

    def compute_delta_phi(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        delta_phi = phi - phi_ref
        delta_phi[delta_phi > np.pi] -= 2 * np.pi
        delta_phi[delta_phi <= -np.pi] += 2 * np.pi
        return delta_phi


class EnvironmentModel(object):  # all tensors
    def __init__(self, num_future_data=5):
        self.vehicle_dynamics = VehicleDynamics(if_model=True)
        self.base_frequency = 10.
        self.obses = None
        self.veh_states = None
        self.num_future_data = num_future_data
        # veh_state: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
        # obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs, future_delta_ys1,..., future_delta_ysn,
        #      future_delta_phis1,..., future_delta_phisn
        # veh_full_state: v_xs, v_ys, rs, ys, phis, steers, a_xs, xs

    def reset(self, obses):
        self.obses = obses
        self.veh_states = self._get_state(self.obses)

    def _get_obs(self, veh_states):
        v_xs, v_ys, rs, delta_ys, delta_phis = veh_states[:, 0], veh_states[:, 1], veh_states[:, 2], \
                                                             veh_states[:, 3], veh_states[:, 4]
        lists_to_stack = [v_xs, v_ys, rs, delta_ys, delta_phis] + \
                         [delta_ys for _ in range(self.num_future_data)]  # + \
                         # [delta_phis for _ in range(self.num_future_data)]
        return tf.stack(lists_to_stack, axis=1)

    def _get_state(self, obses):
        return obses[:, :5]

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]

            # v_xs = self.obses[:, 0]
            # max_decels = np.min(np.stack([v_xs / 3, 3 * np.ones_like(v_xs)], 1), 1)
            # actions = np.stack([steer_norm * 1.2 * np.pi / 9, (a_xs_norm + 1.) / 2. * (3. + max_decels) - max_decels], 1)

            actions = tf.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
            rewards = self.vehicle_dynamics.compute_rewards(self.veh_states, actions)
            self.veh_states = self.vehicle_dynamics.prediction(self.veh_states, actions,
                                                               self.base_frequency, 1)
            v_xs, v_ys, rs, delta_ys, delta_phis = self.veh_states[:, 0], self.veh_states[:, 1], self.veh_states[:, 2], \
                                                                 self.veh_states[:, 3], self.veh_states[:, 4]
            v_xs = tf.clip_by_value(v_xs, 1, 35)
            delta_phis = tf.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
            delta_phis = tf.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
            self.veh_states = tf.stack([v_xs, v_ys, rs, delta_ys, delta_phis], axis=1)
            self.obses = self._get_obs(self.veh_states)

        return self.obses, rewards


class PathTrackingEnv(gym.Env, ABC):
    def __init__(self, num_future_data=5, **kwargs):
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
        # obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs, future_delta_ys1,..., future_delta_ysn,
        #         #      future_delta_phis1,..., future_delta_phisn
        # veh_full_state: v_xs, v_ys, rs, ys, phis, steers, a_xs, xs
        self.vehicle_dynamics = VehicleDynamics()
        self.num_future_data = num_future_data
        self.obs = None
        self.veh_state = None
        self.veh_full_state = None
        self.simulation_time = 0
        self.action = None
        self.num_agent = kwargs['num_agent']
        self.expected_vs = 20.
        self.done = np.zeros((self.num_agent,), dtype=np.int)
        self.base_frequency = 200
        self.interval_times = 20
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (5 + self.num_future_data)),
            high=np.array([np.inf] * (5 + self.num_future_data)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2*np.pi / 9, -3]),
                                           high=np.array([1.2*np.pi / 9, 3]),
                                           dtype=np.float32)

        self.history_positions = deque(maxlen=100)
        plt.ion()

    def _get_obs(self, veh_state, veh_full_state):
        future_delta_ys_list = []
        # future_delta_phi_list = []
        v_xs, v_ys, rs, delta_ys, delta_phis = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                             veh_state[:, 3], veh_state[:, 4]

        v_xs, v_ys, rs, ys, phis, xs = veh_full_state[:, 0], veh_full_state[:, 1], veh_full_state[:, 2], \
                                                     veh_full_state[:, 3], veh_full_state[:, 4], veh_full_state[:, 5]
        x_ = xs.copy()
        for _ in range(self.num_future_data):
            x_ += v_xs * 1./self.base_frequency * self.interval_times*2
            future_delta_ys_list.append(self.vehicle_dynamics.path.compute_delta_y(x_, ys))
            # future_delta_phi_list.append(self.vehicle_dynamics.path.compute_delta_phi(x_, phis))

        lists_to_stack = [v_xs, v_ys, rs, delta_ys, delta_phis] + \
                         future_delta_ys_list  # + \
                         # future_delta_phi_list
        return np.stack(lists_to_stack, axis=1)

    def _get_state(self, obses):
        return obses[:, :5]

    def reset(self, **kwargs):
        if 'init_obs' in kwargs.keys():
            init_obs = kwargs.get('init_obs')
            self.veh_state = self._get_state(init_obs)
            init_x = np.random.uniform(0, 600, (self.num_agent,)).astype(np.float32)
            path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(init_x), \
                               self.vehicle_dynamics.path.compute_path_phi(init_x)
            self.veh_full_state = self.veh_state.copy()
            self.veh_full_state[:, 4] = self.veh_state[:, 4] + path_phi
            self.veh_full_state[:, 3] = self.veh_state[:, 3] + path_y
            self.veh_full_state = np.concatenate([self.veh_full_state, init_x[:, np.newaxis]], 1)
            self.obs = self._get_obs(self.veh_state, self.veh_full_state)
            return self.obs

        if self.done[0] == 1:
            self.history_positions.clear()
        self.simulation_time = 0
        init_x = np.random.uniform(0, 600, (self.num_agent,)).astype(np.float32)

        init_delta_y = np.random.normal(0, 1, (self.num_agent,)).astype(np.float32)
        init_y = self.vehicle_dynamics.path.compute_y(init_x, init_delta_y)

        init_delta_phi = np.random.normal(0, np.pi / 9, (self.num_agent,)).astype(np.float32)
        init_phi = self.vehicle_dynamics.path.compute_phi(init_x, init_delta_phi)

        init_v_x = np.random.uniform(15, 25, (self.num_agent,)).astype(np.float32)
        beta = np.random.normal(0, 0.15, (self.num_agent,)).astype(np.float32)
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (self.num_agent,)).astype(np.float32)

        init_veh_full_state = np.stack([init_v_x, init_v_y, init_r, init_y, init_phi, init_x], 1)
        if self.veh_full_state is None:
            self.veh_full_state = init_veh_full_state
        else:
            for i, done in enumerate(self.done):
                self.veh_full_state[i, :] = np.where(done == 1, init_veh_full_state[i, :], self.veh_full_state[i, :])
        self.veh_state = self.veh_full_state[:, 0:-1].copy()
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(self.veh_full_state[:, -1]), \
                           self.vehicle_dynamics.path.compute_path_phi(self.veh_full_state[:, -1])
        self.veh_state[:, 4] = self.veh_full_state[:, 4] - path_phi
        self.veh_state[:, 3] = self.veh_full_state[:, 3] - path_y
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)

        self.history_positions.append((self.veh_full_state[0, -1], self.veh_full_state[0, 3]))
        return self.obs

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm, a_x_norm = action[:, 0], action[:, 1]
        action = np.stack([steer_norm * 1.2 * np.pi / 9, a_x_norm*3], 1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.simulation_time += self.interval_times * 1 / self.base_frequency
        self.action = action
        veh_state_tensor = tf.convert_to_tensor(self.veh_state, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(self.action, dtype=tf.float32)
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.veh_state, self.veh_full_state = \
            self.vehicle_dynamics.simulation(self.veh_state, self.veh_full_state, self.action,
                                             base_freq=self.base_frequency, simu_times=self.interval_times)
        self.history_positions.append((self.veh_full_state[0, -1], self.veh_full_state[0, 3]))
        self.done = self.judge_done(self.veh_state)
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)
        info = {}
        return self.obs, reward, self.done, info

    def judge_done(self, veh_state):
        v_xs, v_ys, rs, delta_ys, delta_phis = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                             veh_state[:, 3], veh_state[:, 4]
        done = (np.abs(delta_ys) > 3) | (np.abs(delta_phis) > np.pi / 4.) | (v_xs < 2) | \
               (np.abs(rs) > 0.8)
        return done

    def render(self, mode='human'):
        plt.cla()
        v_x, v_y, r, delta_y, delta_phi = self.veh_state[0, 0], self.veh_state[0, 1], self.veh_state[0, 2], \
                                                      self.veh_state[0, 3], self.veh_state[0, 4]
        v_x, v_y, r, y, phi, x = self.veh_full_state[0, 0], self.veh_full_state[0, 1], \
                                             self.veh_full_state[0, 2], self.veh_full_state[0, 3], \
                                             self.veh_full_state[0, 4], self.veh_full_state[0, 5]
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(x), self.vehicle_dynamics.path.compute_path_phi(x)

        future_ys = self.obs[0, 5:]
        xs = np.array([x + i*v_x/self.base_frequency*self.interval_times*2 for i in range(1, self.num_future_data+1)])

        plt.plot(xs, -future_ys+y, 'r*')

        plt.title("Demo")
        range_x, range_y = 100, 100
        plt.axis('equal')
        path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
        path_ys = self.vehicle_dynamics.path.compute_path_y(path_xs)
        plt.plot(path_xs, path_ys)

        history_positions = list(self.history_positions)
        history_xs = np.array(list(map(lambda x: x[0], history_positions)))
        history_ys = np.array(list(map(lambda x: x[1], history_positions)))
        plt.plot(history_xs, history_ys, 'g')

        def draw_rotate_rec(x, y, a, l, w, color='black'):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            plt.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
            plt.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
            plt.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
            plt.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

        draw_rotate_rec(x, y, phi, 4.8, 2.2)
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3, 'time: {:.2f}s'.format(self.simulation_time))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 2,
                 'x: {:.2f}, y: {:.2f}, path_y: {:.2f}, delta_y: {:.2f}m'.format(x, y, path_y, delta_y))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 3,
                 r'phi: {:.2f}rad (${:.2f}\degree$), path_phi: {:.2f}rad (${:.2f}\degree$), delta_phi: {:.2f}rad (${:.2f}\degree$)'.format(
                     phi,
                     phi * 180 / np.pi,
                     path_phi,
                     path_phi * 180 / np.pi,
                     delta_phi, delta_phi * 180 / np.pi,
                 ))

        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 4,
                 'v_x: {:.2f}m/s (expected: {:.2f}m/s), v_y: {:.2f}m/s'.format(v_x, self.expected_vs, v_y))
        plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 5, 'yaw_rate: {:.2f}rad/s'.format(r))

        if self.action is not None:
            steer, a_x = self.action[0, 0], self.action[0, 1]
            plt.text(x - range_x / 2 - 20, range_y / 2 - 3 * 6,
                     r'steer: {:.2f}rad (${:.2f}\degree$), a_x: {:.2f}m/s^2'.format(steer, steer * 180 / np.pi, a_x))

        plt.axis([x - range_x / 2, x + range_x / 2, -range_y / 2, range_y / 2])

        plt.pause(0.001)
        plt.show()


def test_path():
    path = ReferencePath()
    path_xs = np.linspace(500, 700, 1000)
    path_ys = path.compute_path_y(path_xs).numpy()
    plt.plot(path_xs, path_ys)
    plt.show()


def test_path_tracking_env():
    from mixed_pg_learner import judge_is_nan
    env = PathTrackingEnv(num_agent=1)
    obs = env.reset()
    print(obs)
    judge_is_nan([obs])
    action = np.array([[0, 1]])
    for _ in range(1000):
        obs, reward, done, info = env.step(action)
        env.render()
        # env.reset()


if __name__ == '__main__':
    test_path_tracking_env()
