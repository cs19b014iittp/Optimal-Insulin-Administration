from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import numpy as np

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


Observation = namedtuple("Observation", ["CGM", "insulin", "CHO"])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    # if len(BG_last_hour) < 2:
    #     return 0
    # else:
    _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
    # _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
    # return risk_prev - risk_current
    return -1*risk_current

def magni_reward(bg_hist, **kwargs):
    # if bg_hist[-1] < 130 or bg_hist[-1] > 250:
    #     return -1000000
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1000*risk

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
        
        # for history
        self.avg_size=5
        self.hist_size=50

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action
        # print(self.time)
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

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

        # print(self.time)
        # for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
        tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
        # CHO += tmp_CHO / self.sample_time
        # insulin += tmp_insulin / self.sample_time
        # BG += tmp_BG / self.sample_time
        # CGM += tmp_CGM / self.sample_time
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
        # done = BG < 70 or BG > 350
        # done = BG < 20 or BG > 340
        # if self.timestamps < 100000:
        # done = BG < 140 or BG > 16000
        # print('Insulin: ', insulin)
        done = BG < 50 or BG > 190

        # if done == True:
        #     self.lastdone = self.timestamps
        #     print('CGM:', CGM)
        #     print('done', self.timestamps, CGM, reward)

        CGM_last_60min = np.full(self.hist_size, CGM, dtype=np.float32)
        CGM_last_60min[-min(self.hist_size,len(self.CGM_hist)):] = self.CGM_hist[-self.hist_size:]
        CGM_last_60min = [sum(CGM_last_60min[n:n+self.avg_size])/self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        insulin_last_60min = np.full(self.hist_size, 0, dtype=np.float32)
        insulin_last_60min[-min(self.hist_size,len(self.insulin_hist)):] = self.insulin_hist[-self.hist_size:]
        insulin_last_60min = [sum(insulin_last_60min[n:n+self.avg_size])/self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        cho_last_60min = np.full(self.hist_size, 0)
        cho_last_60min[-min(self.hist_size,len(self.CHO_hist)):] = self.CHO_hist[-self.hist_size:]
        cho_last_60min = [sum(cho_last_60min[n:n+self.avg_size])/self.avg_size for n in range(0,self.hist_size,self.avg_size)]

        # obs = [CGM]
        # obs = [0]
        # if CGM < 150 and CGM > 130:
        #     obs = [1]
        # obs = Observation(CGM=CGM)
        obs = []
        obs.extend(CGM_last_60min)
        obs.extend(insulin_last_60min)
        # obs.append(cho_last_60min)

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

    def reset(self):
        print('timesteps= ', self.timestamps)
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        # obs = Observation(CGM=np.full(5,CGM), insulin=np.full(5,0), CHO=np.full(5,0))  # assmued initial insulin and cho as 0
        # obs = [CGM]
        obs = []
        hlen = int(self.hist_size / self.avg_size)
        obs.extend(np.full(hlen, CGM))
        obs.extend(np.full(hlen,0))
        # obs.append(np.full(40,0))
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