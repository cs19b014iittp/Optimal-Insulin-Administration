from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController

import gym
import numpy as np
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


from simglucose.envs.simglucose_gym_env import T1DSimEnv


class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state

        self.model = PPO.load("ppo_insulin")
        

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
    # env = make_vec_env(env, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_insulin")

learn_ddpg()
'''