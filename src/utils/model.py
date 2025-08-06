import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from IHSetYates09 import yates09

class Model(object):
    """
    cal_Yates09
    
    Configuration to calibrate and run the Yates et al. (2009) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, data, cfg):

        self.calibr_cfg = fo.config_cal(cfg)
        self.switch_Yini = cfg['switch_Yini']

        self.lb = cfg['lb']
        self.ub = cfg['ub']

       
        self.hs = data.hs.values
        self.time = pd.to_datetime(data.time.values)
        self.E = self.hs ** 2
        self.Obs = data.obs.values
        self.time_obs = pd.to_datetime(data.time_obs.values)
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))


        if self.switch_Yini == 0:
            # @jit
            def model_simulation(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Ymd, _ = yates09(self.E_splited,
                                 self.dt_splited,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Ymd, _ = yates09(self.E,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 self.Yini)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
                population = np.zeros((population_size, 4))
                for i in range(4):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            # @jit
            def model_simulation(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Yini = par[4]

                Ymd, _ = yates09(self.E_splited,
                                 self.dt_splited,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                a = -np.exp(par[0])
                b = par[1]
                cacr = -np.exp(par[2])
                cero = -np.exp(par[3])
                Yini = par[4]

                Ymd, _ = yates09(self.E,
                                 self.dt,
                                 a,
                                 b,
                                 cacr,
                                 cero,
                                 Yini)
                return Ymd

            self.run_model = run_model


            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3]), 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3]), 1.25*np.max(self.Obs_splited)])
                population = np.zeros((population_size, 5))
                for i in range(5):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.E = self.E[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.idx_calibration = idx
        self.E_splited = self.E[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)

        self.full_run = self.run_model(self.solution)

        if self.switch_Yini == 1:
            self.par_names = [r'a', r'b', r'C+', r'C-', r'Y_i']
            self.par_values = self.solution.copy()
            self.par_values[0] = -np.exp(self.par_values[0])
            self.par_values[2] = -np.exp(self.par_values[2])
            self.par_values[3] = -np.exp(self.par_values[3])
        elif self.switch_Yini == 0:
            self.par_names = [r'a', r'b', r'C+', r'C-']
            self.par_values = self.solution.copy()
            self.par_values[0] = -np.exp(self.par_values[0])
            self.par_values[2] = -np.exp(self.par_values[2])
            self.par_values[3] = -np.exp(self.par_values[3])
