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

    # perpare directory
    sub_dir = Config.get_save_file(base_name="tmp")
    if os.path.isdir(utils.file_to_path(sub_dir)):
        shutil.rmtree(path=utils.file_to_path(sub_dir))
    os.mkdir(utils.file_to_path(sub_dir))

    # params
    nenvs = Config.NUM_ENVS
    #utils.mpi_print("nenvs", nenvs)
    total_timesteps = int(2e5)
    population_count = 32
    timesteps_per_agent = 500

    env = utils.make_general_env(nenvs, seed=rank)
    env = wrappers.add_final_wrappers(env)
        
    policy = policies.get_policy()

    # setup graph
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    model = policy(sess, env.observation_space, env.action_space, nenvs, 1)
    model_init_op = tf.global_variables_initializer()

    params = tf.trainable_variables("model")
    params_filtered = [v for v in params if '/b' not in v.name] # filter biases

    model_noise_ops = []
    total_num_params = 0
    for p in params_filtered:
        shape = p.get_shape()
        shape_list = shape.as_list()
        num_params = np.prod(shape_list)
        #utils.mpi_print('param', p, num_params)
        total_num_params += num_params

        noise = tf.random_normal(shape, mean=0, stddev=0.01, dtype=tf.float32)
        model_noise_ops.append(tf.assign_add(p, noise))
        # test normalisatiom

    utils.mpi_print('total num params:', total_num_params)

    # function definitions
    def save_model(name=None):
        file_name = sub_dir + "/" + name if name else Config.get_save_file()
        utils.save_params_in_scopes(sess, ['model'], file_name)

    def load_model(name=None):
        file_name = sub_dir + "/" + name
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
        load_model(name=population[0]["name"])
        
        # final evaluation
        #for _ in range(10):
        #    utils.mpi_print(run(timesteps_per_agent))

        save_model()

        # cleanup
        sess.close()
        shutil.rmtree(path=utils.file_to_path(sub_dir))

    # load data from restore point and seed the whole population
    load_data = Config.get_load_data('default')
    loaded_name = None
    if load_data is not None:
        utils.load_all_params(sess)
        loaded_name = str(uuid.uuid1())
        save_model(name=loaded_name)
        

    # initialise population
    # either all random and no mutations pending
    # or all from the restore point with all but one to be mutated
    population = [{"name": loaded_name or str(uuid.uuid1()), "fit": -1, "need_mut": loaded_name != None and i != 0} for i in range(population_count)]

    utils.mpi_print("== population size ", population_count, ", nenvs ", nenvs, ", t_agent ", timesteps_per_agent, " ==")

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
                    if not load_model(name=agent["name"]):
                        sess.run(model_init_op)
                        need_save = True

                    if agent["need_mut"]:
                        sess.run(model_noise_ops)
                        agent["need_mut"] = False
                        agent["name"] = str(uuid.uuid1())
                        need_save = True

                    if need_save:
                        save_model(name=agent["name"])
                        
                    agent["fit"] = run(timesteps_per_agent)

                    timesteps_done += timesteps_per_agent


            # sort by fitness
            population.sort(key=lambda k: k['fit'], reverse=True) 

            utils.mpi_print(*["{:5}".format(int(agent["fit"])) for agent in population])

            utils.mpi_print("__ average", "{:.1f}".format(np.mean([agent["fit"] for agent in population])), " __")
            utils.mpi_print("__ took", "{:.1f}".format(time.time() - t_generation_start), " s __")
            utils.mpi_print("")

            # cleanup to prevent hdd clutter
            to_be_removed = set(os.listdir(utils.file_to_path(sub_dir))) - set([agent["name"] for agent in population])
            for filename in to_be_removed: 
                os.remove(utils.file_to_path(sub_dir+ "/" + filename))

            if not timesteps_done < total_timesteps:
                break
        
            # replace weak agents
            duplicate_factor =  1.0 / 16
            survive_factor = 1.0 / 4
            cutoff_duplicate = math.floor(population_count*duplicate_factor)
            cutoff_survive = math.floor(population_count*survive_factor)
            source_agents = population[:cutoff_survive]
            k = 0
            for i in range(cutoff_survive, population_count):
                population[i] = copy.deepcopy(source_agents[k])
                population[i]["fit"] = -1
                if k < cutoff_duplicate: # test degeneration protection
                #if True: # test/
                    population[i]["need_mut"] = True
                    
                k = (k + 1) % len(source_agents)

            generation += 1
        
        utils.mpi_print(time.time() - t_first_start)

        clean_exit()
    except KeyboardInterrupt:
        clean_exit()

    return 0

if __name__ == '__main__':
    main()
