#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: test_for_debug.py
# =====================================

import numpy as np
import ray
import tensorflow as tf

from utils.task_pool import TaskPool


class Sample_numpy(object):
    def __init__(self):
        pass

    def sample(self):
        samples = np.random.random([1000, 1000])
        return samples
#
#
def test_remote():
    ray.init(redis_max_memory=100 * 1024 * 1024, object_store_memory=100 * 1024 * 1024)
    sps = [ray.remote(Sample_numpy).remote() for _ in range(1)]
    sample_tasks = TaskPool()
    for sp in sps:
        sample_tasks.add(sp, sp.sample.remote())
    samples = None
    for _ in range(1000000000):
        for sp, objID in list(sample_tasks.completed(blocking_wait=True)):
            samples = ray.get(objID)
            sample_tasks.add(sp, sp.sample.remote())


def test_remote2():
    ray.init(redis_max_memory=100 * 1024 * 1024, object_store_memory=1000 * 1024 * 1024)
    sampler = ray.remote(Sample_numpy).remote()
    for _ in range(1000000000):
        objID = sampler.sample.remote()
        samples = ray.get(objID)
        # ray.internal.free([objID])

        # hxx = hpy()
        # heap = hxx.heap()
        # print(heap.byrcs)


# def test_local():
#     a = Sample_numpy()
#     for _ in range(1000000000):
#         sample = a.sample()
#
#
# def test_get_size():
#     import sys
#     samples = np.random.random([1000, 1000])
#     print(sys.getsizeof(samples)/(1024*1024))
#
#
# def test_memory_profiler():
#     c = []
#     a = [1, 2, 3] * (2 ** 20)
#     b = [1] * (2 ** 20)
#     c.extend(a)
#     c.extend(b)
#     del b
#     del c


def test_jacobian():
    variable = tf.Variable(1.0)
    inputs = (
        tf.constant(tf.random.uniform((1, 4))),
        tf.constant(tf.random.uniform((1, 3))),
    )
    print(inputs)

    with tf.GradientTape(persistent=True) as tape:
        outputs = variable * tf.pow(tf.concat(inputs, axis=-1), 2.0)

    print(outputs)

    jacobians_1 = tape.jacobian(
        outputs,
        variable,
        experimental_use_pfor=True,
    )
    print(jacobians_1)
    print("tape.jacobians(..., experimental_use_pfor=True) works!")

    try:
        jacobians_2 = tape.jacobian(
            outputs,
            variable,
            experimental_use_pfor=False,
        )
        print(jacobians_2)
        print("tape.jacobians(..., experimental_use_pfor=False) works!")
    except TypeError:
        print("tape.jacobians(..., experimental_use_pfor=False) doesn't work!")
        raise

def test_jacobian2():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(2,)),
    ])
    print(model.trainable_variables)
    inputs = tf.Variable([[1, 2]], dtype=tf.float32)

    with tf.GradientTape() as gtape:
        outputs = model(inputs)
    print(outputs)
    jaco = gtape.jacobian(outputs, model.trainable_variables)
    print(jaco)

def test_logger():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.error('Watch out!')

    logger.warning('2222')
    logger.info('sdsdf')

def test_tf22():
    from tensorflow.keras.optimizers import Adam
    a = Adam()
    print(a)

def tape_gra():
    from train_script import built_mixedpg_parser
    import gym
    from policy import PolicyWithQs
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
    inp = tf.Variable(3.)

    with tf.GradientTape(persistent=True) as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        c = np.array([1.,2.,3.])*inp
        out = []
        for ci in c:
            out.append(ci)
        a = Qs[0][0]
        c0=c[0]
        c1=c[1]

    gradient = tape.jacobian(c, inp)
    print(gradient)

def test_gym():
    import gym
    env = gym.make('PathTracking-v0')
    obs = env.reset()
    print(obs)
    action = np.array([0, 10000])
    obs = env.step(action)
    print(obs)
    a = 1

def test_ray():
    ray.init(redis_max_memory=100 * 1024 * 1024, object_store_memory=100 * 1024 * 1024)
    import time

    class evaluator(object):
        def __init__(self):
            self.a = (i for i in range(100))

        def outa(self):
            print(next(self.a))
            time.sleep(5)

    ev = ray.remote(num_cpus=1)(evaluator).remote()
    for i in range(100000):
        ev.outa.remote()
    time.sleep(100)


import threading
class UpdateThread(threading.Thread):
    """Background thread that updates the local model from gradient list.
    """

    def __init__(self, stats):
        threading.Thread.__init__(self)
        self.stats = stats
        self.stopped = False

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        print(self.stats)


def test_threading():
    import time
    stat = {'a':1}
    thr = UpdateThread(stat)
    thr.start()
    for i in range(100000):
        if i > 100:
            stat = {'a': i}
            print(i)
            time.sleep(0.1)
        if i > 40000:
            thr.stopped = True
            break

def test_stack():
    from offpolicy_mb_learner import TimerStat
    import time
    a = TimerStat()

    def out():
        with a:
            time.sleep(4)
        print(a.mean)
    for i in range(100):
        with a:
            out()



if __name__ == '__main__':
    test_stack()

