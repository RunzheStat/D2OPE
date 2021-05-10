from _util import * 

"""
!pip install --upgrade pip grpcio termcolor logger  torch torchvision
# !pip install "PyQt5<5.13"
# !pip install "pyqtwebengine<5.13"
# !pip install --upgrade tensorflow_probability
# !conda remove tensorflow -y
# !pip install "cloudpickle <= 1.3"
# restart
# !pip install --upgrade tensorflow tensorflow-serving-api 
# GPU!!!
# gast

# !autopep8 --in-place --aggressive --aggressive Competing_TRPO/BRAC/brac/train_eval_offline.py
# nvidia-smi -l

"""

""" conda_amazonei_tensorflow2_p36
!pip install --upgrade pip grpcio gast  termcolor logger tensorflow_probability
! pip install torch torchvision
!conda remove tensorflow -y
restart
!pip install --upgrade tensorflow tensorflow-serving-api 
GPU!!!
"""
################################################################################################################################################################################################################################################################################################


class InitialStateSampler():
    """ G(ds): sample the initial state distribution
    options:
    1. empirical initial states 
        1. resample ["resample_empirical"]
        2. fit a normal ["fit_empirical"]
    2. the specifi parameter option
    """
    def __init__(self, data = None, mode = "resample_empirical"):
        # data: [N, dim]
        self.seed = 42
        self.data = arr(data)
        self.N = len(data)
        if mode is "resample_empirical":
            self.resample_empirical = True
        else:
            self.resample_empirical = False
            self.u = np.mean(data, 0)
            self.cov = np.cov(arr(data).T)
        
    def sample(self, batch_size = 1):
        np.random.seed(self.seed)
        self.seed += 1
        if self.resample_empirical: # resample initial?
            return self.data[choice(self.N, batch_size)]
        else:
            return np.random.multivariate_normal(self.u, self.cov, batch_size)

class SimpleReplayBuffer():
    # trajs: a list of trajs, each traj is a list of 4-tuple (S, A, R, S') [[[S, A, R, S']]]
    # aim: [arr of S, arr of A, arr of SS]
    # his original code? much easier?
    # not complex enough. online, etc.
    def __init__(self, trajs):
        self.seed = 42
        self.states = np.array([item[0] for traj in trajs for item in traj])
        self.actions = np.array([item[1] for traj in trajs for item in traj])
        self.rewards = np.array([item[2] for traj in trajs for item in traj])
        self.next_states = np.array([item[3] for traj in trajs for item in traj])
        self.N, self.S_dims = self.states.shape
        
    def add(self, SARS):
        self.states = np.append(self.states, SARS[0][np.newaxis, :], axis = 0)
        self.next_states = np.append(self.next_states, SARS[3][np.newaxis, :], axis = 0)
        self.actions = np.append(self.actions, SARS[1])
        self.rewards = np.append(self.rewards, SARS[2])

        
    def sample(self, batch_size):
        
        np.random.seed(self.seed)
        self.seed += 1
        idx = np.random.choice(self.N, batch_size, replace = False)
        return [self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]]