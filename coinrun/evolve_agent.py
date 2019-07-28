"""
Train an agent using a genetic algorithm.
"""

import time
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ga
from coinrun.config import Config

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    utils.mpi_print("nenvs", nenvs)
    total_timesteps = int(1e4)

    env = utils.make_general_env(nenvs, seed=rank)

    env = wrappers.add_final_wrappers(env)
        
    policy = policies.get_policy()

    # setup graph
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        model = policy(sess, env.observation_space, env.action_space, nenvs, 1)
        model_initialiser = tf.global_variables_initializer()
    
    # initialise population
    population_count = 5
    timesteps_per_agent = 500

    population = [{"sess":tf.Session(graph=graph), "fit": -1} for _ in range(population_count)]

    # main loop
    timesteps_done = 0
    while timesteps_done < total_timesteps:
        utils.mpi_print("__________________")
        utils.mpi_print("n pop", len(population))

        # initialise and evaluate all new agents
        for agent in population:
            if agent["fit"] < 0:
                agent["sess"].run(model_initialiser)
                utils.load_all_params(agent["sess"])

                obs = env.reset()
                state = model.initial_state
                done = np.zeros(nenvs)

                rew_accum = 0
                for i in range(timesteps_per_agent):
                    action, values, state, _ = model.step_with_sess(agent["sess"], obs, state, done)
                    obs, rew, done, info = env.step(action)
                    rew_accum += rew

                #utils.mpi_print('rew_accum', rew_accum)

                timesteps_done += timesteps_per_agent
                agent["fit"] = sum(rew_accum)

        # sort by fitness
        population.sort(key=lambda k: k['fit'], reverse=True) 
        # kill weak agents
        for agent in population[population_count//2:]:
            agent["sess"].close()
        population = population[:population_count//2]
        # create new agents
        population += [{"sess":tf.Session(graph=graph), "fit": -1} for _ in range((population_count+1)//2)]
    
    # cleanup
    for agent in population:
        agent["sess"].close()
    return 0

if __name__ == '__main__':
    main()

