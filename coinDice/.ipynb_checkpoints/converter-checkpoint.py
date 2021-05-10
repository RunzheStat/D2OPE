from _util import *

class dataSpec():
    def __init__(self
                , observation = None, action = None):
        self.observation = observation
        self.action = action

class convertDataset():
    def __init__(self, data, min_A, max_A, min_S, max_S):
        ##############################################################################
        from tf_agents.specs import tensor_spec
        dim_S = len(data[0][0][0])
        observation = tensor_spec.BoundedTensorSpec(
            (dim_S,), tf.float32, name='observation', minimum=min_S, maximum=max_S)
        action = tensor_spec.BoundedTensorSpec(
            (), tf.int64, name='action', minimum=min_A, maximum=max_A)
        self.spec = dataSpec(observation = observation, action = action)
        self.seed = 42
        self.N = len(data)
        self.T = len(data[0])
        self.NT = self.N * self.T
        self.raw_data = data
        self.S, self.A, self.R, self.SS = [np.array([item[i] for traj in data for item in traj]) for i in range(4)]
        self.R = np.abs(self.R)
        ##############################################################################
    def sample_step_and_next(self, batch_size):
        np.random.seed(self.seed)
        self.seed += 1

        idx = choice(self.N * (self.T - 1), batch_size)
        step_num = idx % (self.T - 1)
        N_idx = idx // (self.T - 1)
        idx = N_idx * self.T + step_num

        env_steps = {"observation" : tf.cast(tf.stack([self.S[i] for i in idx], axis = 0), tf.float32)
                     , "action" : tf.cast(tf.concat([self.A[i] for i in idx], axis = 0), tf.int64)
                      , "reward" : tf.cast(tf.concat([self.R[i] for i in idx], axis = 0), tf.float32)
                     , "step_num" : step_num }

        next_steps = {"observation" : tf.cast(tf.stack([self.S[i + 1] for i in idx], axis = 0), tf.float32)
                     , "action" : tf.cast(tf.concat([self.A[i + 1] for i in idx], axis = 0), tf.int64)
                      , "reward" : tf.cast(tf.concat([self.R[i + 1] for i in idx], axis = 0), tf.float32)
                                           }
        return env_steps, next_steps

    def sample_init_steps(self, batch_size):
        np.random.seed(self.seed)
        self.seed += 1
        init_steps = [self.raw_data[i][0] for i in choice(self.N, batch_size)]
        init_steps = {"observation" : tf.cast(tf.stack([step[0] for step in init_steps], axis = 0), tf.float32)
                     , "action" : tf.cast(tf.concat([step[1] for step in init_steps], axis = 0), tf.int64)
                      , "reward" : tf.cast(tf.concat([step[1] for step in init_steps], axis = 0), tf.float32)
                     }
        return init_steps
