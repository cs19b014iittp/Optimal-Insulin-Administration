from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController

import gym
import numpy as np
import copy

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC

from simglucose.envs.simglucose_gym_env import T1DSimEnv


class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state

        self.model = SAC.load("sac_insulin_intBG")

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
def learn_sac():
    env = T1DSimEnv()
    model = SAC(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("sac_insulin_intBG")
    return

learn_sac()
'''