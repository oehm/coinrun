"""
Train an agent using a genetic algorithm.
"""

import os
import shutil
import copy
import time
import math
import numpy as np
import random
import uuid
import re
from threading import Thread
from mpi4py import MPI
import joblib
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers
from coinrun.config import Config
from coinrun.worker import Worker

def main():
    # general setup

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    # perpare directory
    sub_dir = utils.file_to_path(Config.get_save_file(base_name="tmp"))
    if os.path.isdir(sub_dir):
        shutil.rmtree(path=sub_dir)
    os.mkdir(sub_dir)

    # hyperparams
    nenvs = Config.NUM_ENVS
    total_timesteps = Config.TIMESTEPS
    population_size = Config.POPULATION_SIZE
    timesteps_per_agent = Config.TIMESTEPS_AGENT
    worker_count = Config.WORKER_COUNT
    duplicate_factor =  Config.DUP_F
    survive_factor = Config.SURV_F

    # create environment
    def make_env():
        env = utils.make_general_env(nenvs, seed=rank)
        env = wrappers.add_final_wrappers(env)
        return env
        
    # setup session and workers, and therefore tensorflow ops
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    policy = policies.get_policy()

    workers = [Worker(sess, i, nenvs, make_env, policy, sub_dir) for i in range(worker_count)]


    def clean_exit():

        for worker in workers:
            Thread.join(worker.thread)

        utils.mpi_print("")
        utils.mpi_print("== total duration", "{:.1f}".format(time.time() - t_first_start), " s ==")
        utils.mpi_print(" exit...")

        # save best performing agent
        population.sort(key=lambda k: k['fit'], reverse=True) 
        workers[0].restore_model(name=population[0]["name"])
        workers[0].dump_model()

        # cleanup
        sess.close()
        shutil.rmtree(path=sub_dir)

    # load data from restore point and seed the whole population
    loaded_name = None
    if workers[0].try_load_model():
        loaded_name = str(uuid.uuid1())
        workers[0].save_model(name=loaded_name)
        
    # initialise population
    # either all random and no mutations pending
    # or all from restore point with all but one to be mutated
    population = [{"name": loaded_name or str(uuid.uuid1()), 
                   "fit": -1, 
                   "need_mut": loaded_name != None and i != 0,
                   "age": -1} 
                   for i in range(population_size)]

    utils.mpi_print("== population size", population_size, ", t_agent ", timesteps_per_agent, " ==")

    t_first_start = time.time()
    try:
        # main loop
        generation = 0
        timesteps_done = 0
        while timesteps_done < total_timesteps:
            t_generation_start = time.time()

            utils.mpi_print("")
            utils.mpi_print("__ Generation", generation, " __")

            # initialise and evaluate all new agents
            for agent in population:
                #if agent["fit"] < 0: # test/
                if True: # test constant reevaluation, to dismiss "lucky runs" -> seems good
                    
                    # pick worker from pool and let it work on the agent
                    not_in_work = True
                    while not_in_work:
                        for worker in workers:
                            if worker.can_take_work():
                                worker.work(agent, timesteps_per_agent)
                                not_in_work = False
                                break

                    timesteps_done += timesteps_per_agent

            for worker in workers:
                Thread.join(worker.thread)

            # sort by fitness
            population.sort(key=lambda k: k['fit'], reverse=True) 

            # print stuff
            utils.mpi_print(*["{:5.3f}".format(agent["fit"]) for agent in population])
            utils.mpi_print(*["{:5}".format(agent["age"]) for agent in population])
            utils.mpi_print("__ average fit", "{:.1f}".format(np.mean([agent["fit"] for agent in population])),
                            ", t_done", timesteps_done,
                            ", took", "{:.1f}".format(time.time() - t_generation_start), "s",
                            ", total", "{:.1f}".format(time.time() - t_first_start), "s __")

            # cleanup to prevent disk clutter
            to_be_removed = set(re.sub(r'\..*$', '', f) for f in os.listdir(sub_dir)) - set([agent["name"] for agent in population])
            for filename in to_be_removed: 
                os.remove(sub_dir+ "/" + filename + ".index")
                os.remove(sub_dir+ "/" + filename + ".data-00000-of-00001")

            # break when times up
            if not timesteps_done < total_timesteps:
                break
        
            # mark weak agents for replacement
            cutoff_duplicate = math.floor(population_size*duplicate_factor)
            cutoff_survive = math.floor(population_size*survive_factor)
            source_agents = population[:cutoff_survive]
            k = 0
            for i in range(cutoff_survive, population_size):
                population[i]["name"] = source_agents[k]["name"]
                population[i]["fit"] = -1
                if k < cutoff_duplicate: # test degeneration protection
                #if True: # test/
                    population[i]["need_mut"] = True
                    
                k = (k + 1) % len(source_agents)

            generation += 1
        
        clean_exit()
    except KeyboardInterrupt:
        clean_exit()

    return 0

if __name__ == '__main__':
    main()
