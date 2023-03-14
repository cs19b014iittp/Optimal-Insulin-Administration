# from numba import jit

from simglucose.controller.base import Controller, Action
# from GuidanceRewards.simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.envs.simglucose_gym_env import T1DSimEnv

from main import main

import numpy as np
import pickle

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class MyController(Controller):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
        with open('models/dr40_adol1_2lac.pkl', 'rb') as inp:
            self.model = pickle.load(inp)
        # self.model = PPO.load("models/ppo61_200000")
        # self.model = SAC.load("models/sac61_200000")

    def policy(self, observation, reward, done, **info):
        action = self.model.act(observation)
        # action, _states = self.model.predict(np.array(observation, dtype=float))
        print('action:', action)
        self.state = observation
        return action

    def reset(self):
        self.state = self.init_state

'''

def delayed_rewards():
    with open('dr0.pkl', 'wb') as outp:
        actor_critic = main()
        pickle.dump(actor_critic, outp, pickle.HIGHEST_PROTOCOL)

def learn_sac(t):
    env = T1DSimEnv()
    model = SAC(MlpPolicy, env, gamma=0.1, verbose=1, device='cuda')
    model.learn(total_timesteps=t, log_interval=20)
    model.save("sac7adol2_magni_" + str(t))
    return

def learn_ppo(t):
    env = T1DSimEnv()
    model = PPO("MlpPolicy", env, gamma=0.5, verbose=1, device='cuda')
    model.learn(total_timesteps=t)
    model.save("ppo50_" + str(t))
    return

def learn_ddpg(t):
    env = T1DSimEnv()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, gamma=0.1, action_noise=action_noise, verbose=1, device='cuda')
    model.learn(total_timesteps=t, log_interval=40)
    model.save("ddpg6adol2_" + str(t))
    return

if __name__ == '__main__':
    # learn_sac(200000)
    learn_ppo(20000)
    # learn_ddpg(10000)
    # delayed_rewards()

# '''

'''
ppo1: statespace = {CGM}
ppo2: statespace = {CGM, insulin, cho}(last 30 min avg), done = 20 to340, reward=-risk_index
ppo3: similar to ppo2 with last 60min avg
ppo4: statespace = {last 10 measurements of cgm, insulin and cho}
ppo5: statespace = {last 1 hr measurements grouped by 5 mins, making 12 measurements of each}
ppo6: statespace = {last 2 hr measurements grouped by 3 mins, making 40 measurements of each}
ppo7:           ... last 3 hr ...
ppo8: df=0.999, done=5-1000, reward-magni extreme penalty, 
ppo9: done=140-160 CGM
ppo10: done=40-300, reward=60-250
ppo11: done=120-160, reward=130-150, custom_reward
ppo12: done=130-150, reward=135-145, h=20, custom_reward
ppo13: similar, statespace=3min history of cgm and insulin
ppo14: slope height=200
ppo15: statespace= 2 states

'''