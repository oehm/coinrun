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
from mpi4py import MPI
import joblib
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

    sub_dir = Config.get_save_file(base_name="tmp")
    if os.path.isdir(utils.file_to_path(sub_dir)):
        shutil.rmtree(path=utils.file_to_path(sub_dir))
    os.mkdir(utils.file_to_path(sub_dir))

    nenvs = Config.NUM_ENVS
    #utils.mpi_print("nenvs", nenvs)
    total_timesteps = int(1e7)

    datapoints = []

    env = utils.make_general_env(nenvs, seed=rank)
    env = wrappers.add_final_wrappers(env)
        
    policy = policies.get_policy()

    # setup graph
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    model = policy(sess, env.observation_space, env.action_space, nenvs, 1)
    model_init_op = tf.global_variables_initializer()

    params = tf.trainable_variables()
    params = [v for v in params if '/b' not in v.name] # filter biases

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
        # test normalisatiom

    utils.mpi_print('total num params:', total_num_params)
    
    # initialise population
    population_count = 64
    timesteps_per_agent = 500

    # load data dummy
    load_data = Config.get_load_data('default')
    if load_data is not None:
        utils.load_all_params(sess)
        for i in range(10):
            utils.mpi_print(run(timesteps_per_agent))
        return

    population = [{"name":str(uuid.uuid1()), "fit": -1, "need_mut":False} for _ in range(population_count)]

    utils.mpi_print("== population size ", population_count, ", nenvs ", nenvs, ", t_agent ", timesteps_per_agent, " ==")

    def save_model(sess, sub_dir=None, base_name=None):
        base_dict = {'datapoints': datapoints}
        file_name = Config.get_save_file(base_name=base_name)
        if sub_dir is not None:
            file_name = sub_dir + "/" + file_name
        utils.save_params_in_scopes(sess, ['model'], file_name, base_dict)

    def load_model(sess, sub_dir=None, base_name=None):
        file_name = Config.get_save_file(base_name=base_name)
        if sub_dir is not None:
            file_name = sub_dir + "/" + file_name
        try:
            load_data = joblib.load(utils.file_to_path(file_name))
        except IOError:
            load_data = None
        if load_data is not None:
            params_dict = load_data['params']
        else:
            return False

        if 'model' in params_dict:
            loaded_params = params_dict['model']

            loaded_params, params = utils.get_savable_params(loaded_params, 'model', keep_heads=True)

            utils.restore_params(sess, loaded_params, params)
            return True
        else:
            return False

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

    def clean_exit():
        utils.mpi_print("exit...")

        # save best performing agent
        population.sort(key=lambda k: k['fit'], reverse=True) 
        load_model(sess=sess, sub_dir=sub_dir, base_name=population[0]["name"])
        
        # final evaluation
        for _ in range(10):
            utils.mpi_print(run(timesteps_per_agent))
        save_model(sess)

        # cleanup
        sess.close()
        shutil.rmtree(path=utils.file_to_path(sub_dir))

    t_first_start = time.time()
    try:
        # main loop
        generation = 0
        timesteps_done = 0
        while timesteps_done < total_timesteps:
            t_generation_start = time.time()

            utils.mpi_print("__ gen ", generation, ", t_done ", timesteps_done, " __")

            # initialise and evaluate all new agents
            for agent in population:
                #if agent["fit"] < 0: # test/
                if True: # test constant reevaluation, to dismiss "lucky runs" -> seems good
                    
                    need_save = False
                    if not load_model(sess=sess, sub_dir=sub_dir, base_name=agent["name"]):
                        sess.run(model_init_op)
                        need_save = True

                    if agent["need_mut"]:
                        population[i]["name"] = str(uuid.uuid1())
                        sess.run(model_noise_ops)
                        need_save = True

                    if need_save:
                        save_model(sess=sess, sub_dir=sub_dir, base_name=agent["name"])
                        
                    agent["fit"] = run(timesteps_per_agent)

                    timesteps_done += timesteps_per_agent


            # sort by fitness
            population.sort(key=lambda k: k['fit'], reverse=True) 

            utils.mpi_print(*["{:5}".format(int(agent["fit"])) for agent in population])

            #
            datapoints.append([generation, population[0]["fit"]])

            utils.mpi_print("__ average", "{:.1f}".format(np.mean([agent["fit"] for agent in population])), " __")
            utils.mpi_print("__ took", "{:.1f}".format(time.time() - t_generation_start), " s __")
            utils.mpi_print("")

            if not timesteps_done < total_timesteps:
                break
        
            # replace weak agents
            survive_factor = 0.25
            cutoff_index = math.floor(population_count*survive_factor)
            source_agents = population[:cutoff_index]
            k = 0
            for i in range(cutoff_index, population_count):
                population[i] = copy.deepcopy(source_agents[k % len(source_agents)])
                population[i]["fit"] = -1
                if k != 0: # test degeneration protection
                #if True: # test/
                    population[i]["need_mut"] = True
                    
                    
                k += 1

            generation += 1
    except KeyboardInterrupt:
        clean_exit()
    
    utils.mpi_print(time.time() - t_first_start)

    clean_exit()

    return 0

if __name__ == '__main__':
    main()
