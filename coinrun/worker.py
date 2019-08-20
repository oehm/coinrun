import numpy as np
import uuid
import re
from threading import Thread
import joblib
import tensorflow as tf
import coinrun.main_utils as utils
from coinrun.config import Config


class Worker:
    def __init__(self, sess, thread_id, nenvs, make_env, policy, sub_dir):
        self.sess = sess
        self.thread_id = thread_id
        self.thread = None
        self.nenvs = nenvs
        self.env = make_env()
        self.working_dir = sub_dir + "/"
        self.cached = None
        scope_name = "thread_" + str(thread_id)
        self.scope_dir = scope_name + "/"
        with tf.variable_scope(scope_name):
            self.model = policy(sess, self.env.observation_space, self.env.action_space, nenvs, 1, create_additional=False)

            self.params = tf.trainable_variables(self.scope_dir + "model")
            # params_train = self.params
            params_train = [v for v in self.params if '/b' not in v.name] # filter biases
            # params_train_head = [v for v in params_train if '/pi' in v.name]
            # params_train = [v for v in params_train if '/pi' not in v.name]

            params_names = [p.name for p in self.params]
            params_unscoped = [re.sub(self.scope_dir, "", n) for n in params_names] 

            self.model_init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_dir + "model"))

            with tf.variable_scope("mutation"):
                self.model_noise_ops = []
                #total_num_params = 0
                for p in params_train:
                    shape = p.get_shape()
                    #shape_list = shape.as_list()
                    #num_params = np.prod(shape_list)
                    #utils.mpi_print('param', p, num_params)
                    #total_num_params += num_params

                    noise = tf.random_normal(shape, mean=0, stddev=Config.MUTATION_RATE, dtype=tf.float32)
                    self.model_noise_ops.append(tf.assign_add(p, noise))
                    # for p in params_train_head:
                    #     shape = p.get_shape()
                    #     noise = tf.random_normal(shape, mean=0, stddev=0.015, dtype=tf.float32)
                    #     self.model_noise_ops.append(tf.assign_add(p, noise))

            #utils.mpi_print('total num params:', total_num_params)

            params_dict = dict(zip(params_unscoped, self.params))

            self.model_saver = tf.train.Saver(params_dict)

    # load model from coinruns file format, if resore-id was given
    def try_load_model(self):
        load_data = Config.get_load_data('default')
        if load_data is None:
            return False
        
        params_dict = load_data['params']

        if "model" in params_dict:
            print('Loading saved file for scope', "model")

            loaded_params = params_dict["model"]

            if len(loaded_params) != len(self.params):
                print('param mismatch', len(loaded_params), len(self.params))
                assert(False)

            restore_ops = []
            for p, loaded_p in zip(self.params, loaded_params):
                restore_ops.append(tf.assign(p, loaded_p))
            self.sess.run(restore_ops)
            return True
        return False

    # dump compatible with coinruns file format
    def dump_model(self):
        #utils.save_params_in_scopes(self.sess, [self.scope_dir + "model"], Config.get_save_file())
        data_dict = {}

        save_path = utils.file_to_path(Config.get_save_file())

        data_dict['args'] = Config.get_args_dict()
        data_dict['args']['use_minimum_model'] = True
        param_dict = {}

        if len(self.params) > 0:
            #print('saving scope', scope, filename)
            ps = self.sess.run(self.params)

            param_dict["model"] = ps
            
        data_dict['params'] = param_dict
        joblib.dump(data_dict, save_path)

    # save tensorflow checkpoint in subdir
    def save_model(self, name=None): 
        file_path = self.working_dir + name
        self.model_saver.save(sess=self.sess, save_path=file_path, write_meta_graph=False, write_state=False)
        self.cached = name

    # restore tensorflow checkpoint from subdir
    # returns true if checkpoint was found otherwise false
    def restore_model(self, name=None):
        if name != self.cached:
            file_path = self.working_dir + name
            try:
                self.model_saver.restore(sess=self.sess, save_path=file_path)
            except ValueError:
                self.cached = None # we dont know
                return False
            self.cached = name
        return True

    def work_thread(self, agent, seed, timesteps):
        need_save = False
        # init new individual
        if not self.restore_model(name=agent["name"]):
            self.sess.run(self.model_init_op)
            need_save = True
        # mutate if necessary
        if agent["need_mut"]:
            self.sess.run(self.model_noise_ops)
            agent["need_mut"] = False
            agent["name"] = str(uuid.uuid1())
            need_save = True
        # save if necessary
        if need_save:
            self.save_model(name=agent["name"])
            agent["age"] = 0
        else:
            agent["age"] += 1
        # run agent
        obs = self.env.reset(seed=seed)
        state = self.model.initial_state
        done = np.zeros(self.nenvs)

        scores = np.zeros(self.nenvs)
        episode_counts = np.zeros(self.nenvs)
        episode_lengths = []
        for _ in range(timesteps):
            action, _, state, _ = self.model.step( obs, state, done)
            obs, rew, done, info = self.env.step(action)
            for i, d in enumerate(done):
                if d:
                    episode_counts[i] += 1
                    assert('episode' in info[i])
                    scores[i] += info[i].get('episode')['r']
                    episode_lengths.append(info[i].get('episode')['l'])

        percentage_solved = np.zeros(self.nenvs)
        for i, (score, episode_count) in enumerate(zip(scores, episode_counts)):
            if episode_count:
                percentage_solved[i] = score/episode_count

        if(len(episode_lengths) > 0):
            agent["mean_ep_len"] = np.mean(episode_lengths)
        else:
            agent["mean_ep_len"] = float('nan')

        agent["fit"] = np.mean( percentage_solved )

    def work(self, agent, seed, timesteps):
        self.thread = Thread(target=self.work_thread, args=[agent, seed, timesteps])
        self.thread.start()
        return self.thread

    def can_take_work(self):
        return self.thread == None or not self.thread.is_alive()