"""
Train an agent using a genetic algorithm.
"""

import os
import copy
import time
import numpy as np
import random
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers
from coinrun.config import Config

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    #utils.mpi_print("nenvs", nenvs)
    total_timesteps = int(2e3)

    datapoints = []

    def save_model(sess, base_name=None):
        base_dict = {'datapoints': datapoints}
        utils.save_params_in_scopes(sess, ['model'], Config.get_save_file(base_name=base_name), base_dict)

    env = utils.make_general_env(nenvs, seed=rank)

    env = wrappers.add_final_wrappers(env)
        
    policy = policies.get_policy()

    # setup graph
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    model = policy(sess, env.observation_space, env.action_space, nenvs, 1)
    model_init_op = tf.global_variables_initializer()

    params = tf.trainable_variables()
    params = [v for v in params if '/bias' not in v.name] # filter biases


    model_noise_ops = []
    total_num_params = 0
    for p in params:
        shape = p.get_shape()
        shape_list = shape.as_list()
        num_params = np.prod(shape_list)
        #utils.mpi_print('param', p, num_params)
        total_num_params += num_params

        noise = tf.random_normal(shape, mean=0, stddev=0.01, dtype=tf.float32)
        model_noise_ops.append(tf.assign_add(p, noise))

    utils.mpi_print('total num params:', total_num_params)
    
    def load_agent(agent):
        set_global_seeds(agent["seed"])
        sess.run(model_init_op)

        for mut in agent["mut"]:
            set_global_seeds(mut)
            for model_noise_op in model_noise_ops:
                sess.run(model_noise_op)

    def run(timesteps):
        obs = env.reset()
        state = model.initial_state
        done = np.zeros(nenvs)

        rew_accum = 0
        for i in range(timesteps):
            action, values, state, _ = model.step( obs, state, done)
            obs, rew, done, info = env.step(action)
            rew_accum += rew
        return sum(rew_accum)

    load_data = Config.get_load_data('default')
    if load_data is not None:
        utils.load_all_params(sess)
        for i in range(10):
            utils.mpi_print(run(500))

    # initialise population
    population_count = 24
    timesteps_per_agent = 500

    population = [{"seed":random.randint(0,10000), "mut":[], "fit": -1} for _ in range(population_count)]

    utils.mpi_print("== population size ", population_count, ", nenvs ", nenvs, ", t_agent ", timesteps_per_agent, " ==")


    tfirststart = time.time()

    # main loop
    generation = 0
    timesteps_done = 0
    while timesteps_done < total_timesteps:
        utils.mpi_print("__ gen ", generation, ", t_done ", timesteps_done, " __")

        # initialise and evaluate all new agents
        for agent in population:
            if agent["fit"] < 0:
                
                load_agent(agent)

                agent["fit"] = run(timesteps_per_agent)

                timesteps_done += timesteps_per_agent

        utils.mpi_print([agent["fit"] for agent in population]) 

        # sort by fitness
        population.sort(key=lambda k: k['fit'], reverse=True) 

        #
        datapoints.append([generation, population[0]["fit"]])

        if not timesteps_done < total_timesteps:
            break

        # replace weak agents
        for i in range(population_count//4):
            source_agent = population[i]
            for j in range(3):
                target_agent = population[-i-1-(j * population_count//4)]
                target_agent["seed"] = source_agent["seed"]
                target_agent["mut"] = copy.deepcopy(source_agent["mut"])
                target_agent["mut"].append(random.randint(0,10000))
                target_agent["fit"] = -1


        generation += 1
    
    utils.mpi_print(time.time() - tfirststart)

    # save best performing agent
    load_agent(population[0])
    for i in range(10):
        utils.mpi_print(run(500))

    save_model(sess)

    # cleanup
    sess.close()
    return 0

if __name__ == '__main__':
    main()

