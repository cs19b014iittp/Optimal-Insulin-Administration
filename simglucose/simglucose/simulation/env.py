from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import numpy as np
import torch
from encoder import AE
import pkg_resources
import math
import pickle

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


# Observation = namedtuple("Observation", ["CGM", "insulin", "CHO"])
Observation = namedtuple('Observation', ['CGM'])
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def risk_diff(BG_last_hour):
    # if len(BG_last_hour) < 2:
    #     return 0
    # else:
    # _, _, risk_current = risk_index(BG_last_hour, 30)
    _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
    _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)

    diff = risk_prev - risk_current
    return math.exp(diff)
    # return risk_prev - risk_current
    # return -100*risk_current

def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk

def custom_reward(bg_hist):
    bg = bg_hist[-1]
    lbg = 130
    ubg = 150
    
    if bg < lbg or bg > ubg:
        return -1000000

    h = 20
    m = h / (140-lbg)
    if bg > 140:
        return -1*(m*bg + (h - m*ubg))
    return -1*(-m*bg + (h + m*lbg))

def exp_reward(bg_hist):
    G = bg_hist[-1]
    Gt = 127
    eps = 0.1
    return 10*math.exp(-eps*(abs(Gt - G)))

class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario):
        # print('init')
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self._reset()
        self.timestamps = 0
        self.lastdone = 0
        self.obs_hist = []
        # for history
        self.avg_size=1
        self.hist_size=150

        self.autoencoder = torch.load('D:\\IITTP\\Academics\\sem7\BTP\\Optimal-Insulin-Administration\\simglucose\\ae_model3.pt')
        self.autoencoder.eval()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action

        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus

        vpatient_params = pd.read_csv(PATIENT_PARA_FILE)
        bw = vpatient_params.query('Name=="{}"'.format(self.patient.name))['BW'].item()
        u2ss = vpatient_params.query('Name=="{}"'.format(self.patient.name))['u2ss'].item()
        ideal_basal = bw * u2ss / 6000.
        insulin = insulin * 43.2 * ideal_basal

        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)
        # print('cho', patient_mdl_act)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=magni_reward):
        # print('step-function time-stamps: ', self.timestamps)
        self.timestamps = self.timestamps+1
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
        CHO = tmp_CHO 
        insulin = tmp_insulin 
        BG = tmp_BG 
        CGM = tmp_CGM 
        
        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        # window_size = int(60 / self.sample_time)
        window_size = int(60)
        BG_last_hour = self.CGM_hist[-window_size:]
        # BG_last_hour = self.BG_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        
        done = BG < 40 or BG > 300 or (self.timestamps > (self.lastdone + 5000))

        if done == True:
            self.lastdone = self.timestamps
            reward = reward - 100000
            
            # obs_data = []
            # with open('D:\\IITTP\\Academics\\sem7\BTP\\Optimal-Insulin-Administration\\simglucose\\obs_hist.pt', 'rb') as inp:
            #     obs_data = pickle.load(inp)
            #     obs_data.extend(self.obs_hist)

            # with open('D:\\IITTP\\Academics\\sem7\BTP\\Optimal-Insulin-Administration\\simglucose\\obs_hist.pt', 'wb') as inp:
            #     pickle.dump(obs_data, inp, pickle.HIGHEST_PROTOCOL)

        #     print('done', self.timestamps, BG, CGM, reward)

        CGM_obs = np.full(self.hist_size, CGM, dtype=np.float32)
        CGM_obs[-min(self.hist_size,len(self.CGM_hist)):] = self.CGM_hist[-self.hist_size:]
        CGM_obs = [sum(CGM_obs[n:n+self.avg_size]) / self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        insulin_obs = np.full(self.hist_size, 0, dtype=np.float32)
        insulin_obs[-min(self.hist_size,len(self.insulin_hist)):] = self.insulin_hist[-self.hist_size:]
        insulin_obs = [sum(insulin_obs[n:n+self.avg_size]) / self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        cho_obs = np.full(self.hist_size, 0)
        cho_obs[-min(self.hist_size,len(self.CHO_hist)):] = self.CHO_hist[-self.hist_size:]
        cho_obs = [sum(cho_obs[n:n+self.avg_size]) / self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        # obs = Observation(CGM=CGM)
        obsh = []
        obsh.extend(CGM_obs)
        obsh.extend(insulin_obs)

        if self.timestamps % 5 == 0:
            self.obs_hist.append(obsh)

        obs, dec_obs = self.autoencoder(torch.FloatTensor(obsh).to(device))
        obs = obs.to(torch.device('cpu')).tolist()

        # obs = obsh
        # print(CGM, BG, insulin, reward)

        return Step(observation=obs,
                    reward=reward,
                    done=done,
                    sample_time=self.sample_time,
                    patient_name=self.patient.name,
                    meal=CHO,
                    patient_state=self.patient.state,
                    time=self.time,
                    bg=BG,
                    lbgi=LBGI,
                    hbgi=HBGI,
                    risk=risk)

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []
        self.obs_hist = []

    def reset(self):
        print('timesteps= ', self.timestamps)
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        # obs = Observation(CGM=CGM)
        obsh = []
        hlen = int(self.hist_size / self.avg_size)
        obsh.extend(np.full(hlen, CGM))
        obsh.extend(np.full(hlen,0))

        obs, dec_obs = self.autoencoder(torch.FloatTensor(obsh).to(device))
        obs = obs.to(torch.device('cpu')).tolist()
        # obs = obsh
        
        return Step(observation=obs,
                    reward=0,
                    done=False,
                    sample_time=self.sample_time,
                    patient_name=self.patient.name,
                    meal=0,
                    patient_state=self.patient.state,
                    time=self.time,
                    bg=self.BG_hist[0],
                    lbgi=self.LBGI_hist[0],
                    hbgi=self.HBGI_hist[0],
                    risk=self.risk_hist[0])

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df