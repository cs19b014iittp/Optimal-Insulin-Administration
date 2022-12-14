from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController

import gym
import numpy as np
import copy

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from simglucose.envs.simglucose_gym_env import T1DSimEnv


class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state

        self.model = DDPG.load("ddpg_insulin")
        

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method

        return cp

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. For
                      now, it only has one entry: blood glucose level measured
                      by CGM sensor.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        action, _states = self.model.predict(observation)
        self.state = observation
        return action

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state


# learn the model
'''
def learn_ddpg():
    env = T1DSimEnv()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("ddpg_insulin")

learn_ddpg()
'''