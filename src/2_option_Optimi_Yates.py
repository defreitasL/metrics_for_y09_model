import xarray as xr
import time as t_
import matplotlib.pyplot as plt
from utils.model import Model

metrics_name_list = [
    'mss',                      #Max Mielke Skill Score (MSS) ok
    'nashsutcliffe',            #Max Nash-Sutcliffe Efficiency (NSE) ok
    'lognashsutcliffe',         #Max log(NSE) ok
    'pearson',                  #Max Pearson Correlation ($\rho$) ok
    'spearman',                 #Max Spearman Correlation ($S_{rho}$) ok
    'agreementindex',           #Max Agreement Index (AI) ok
    'kge',                      #Max Kling-Gupta Efficiency (KGE) ok
    'npkge',                    #Max Non-parametric KGE (npKGE) ok
    'log_p',                    #Max Logarithmic Probability Distribution (LPD) ok
    'bias',                     #Min Bias (BIAS) ok
    'pbias',                    #Min Percent Bias (PBIAS) ok
    'mse',                      #Min Mean Squared Error (MSE) ok
    'rmse',                     #Min Root Mean Squared Error (RMSE) ok
    'mae',                      #Min Mean Absolute Error (MAE) ok
    'rrmse',                    #Min Relative RMSE (RRMSE) ok
    'rsr',                      #Min RMSE-observations standard deviation ratio (RSR) ok
    'covariance',               #Min Covariance ok
    'decomposed_mse',           #Min Decomposed MSE (DMSE) ok
]

# Avaliable methods: nsgaii, sa, sceua, spea2

# ## NSGAII
# config = {
#          'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
#          'start_date': r'1999-1-1',
#          'end_date': r'2007-1-1',
#          'cal_alg': 'NSGAII',    # Avaliable methods: NSGAII
#          'metrics': ['kge', 'mss', 'rsr'],# metrics_name_list,  # Metrics to be minimized 
#          'population_size': 100,            # Number of individuals in the population
#          'num_generations': 30,    # Number of generations for the calibration algorithm,
#          'cross_prob': 0.9,          # Crossover probability [0.6 - 1.0]
#          'mutation_rate': 0.5,       # Mutation rate [0.01 - 0.8]
#          'regeneration_rate': 0.1,   # Regeneration rate [0.1 - 0.5]
#          'pressure': 2,              # Pressure [2 - 5]
#          'kstop': 5,                 # Maximum number of stagnation before restarting [5 - 100]
#          'pcento': 0.02,               # Percentage improvement allowed in the past kstop loops. [0.01 - 0.2]
#          'peps': 0.0001,    # Convergence threshold [1e-6 - 1e-3]
#          'lb': [1e-3, 1e-3, 1e-3, 1e-3], # Lower bounds for the parameters
#          'ub': [1e+1, 1e+3, 1e-1, 1e-1], # Upper bounds for the parameters,
#          'trs': 2}

# SPEA2
# config = {
#          'switch_Yini': 0,              # Calibrate the initial position? (0: No, 1: Yes)
#          'start_date': r'1999-1-2',
#          'end_date': r'2007-1-1',
#          'cal_alg': 'SPEA2',            # Avaliable methods: NSGAII
#          'metrics': metrics_name_list,  # metrics_name_list,  # Metrics to be minimized 
#          'population_size': 100,        # Number of individuals in the population [50-1000]
#          'num_generations': 25,         # Number of generations for the calibration algorithm, [5-1000]
#          'cross_prob': 0.9,             # Crossover probability [0.6 - 1.0]
#          'mutation_rate': 0.5,          # Mutation rate [0.01 - 0.8]
#          'regeneration_rate': 0.3,      # Regeneration rate [0.1 - 0.5]
#          'pressure': 2,                 # Pressure [2 - 5]
#          'm':3,                         # Environmental selection parameter [1-5]
#          'eta_mut': 5,                  # Mutation parameter for polynomial mutation [5-20]
#          'kstop': 5,                    # Maximum number of stagnation before restarting [5 - 100]
#          'pcento': 0.01,                # Percentage improvement allowed in the past kstop loops.
#          'peps': 0.0001,                    # Threshold for the normalized geometric range of parameters to determine convergence. [1e-6 - 1e-3]
#          'lb': [1e-3, 1e-1, 1e-4, 1e-3],    # Lower bounds for the parameters
#          'ub': [1e+1, 1e+3, 5e-1, 5e-1],    # Upper bounds for the parameters,
#          'trs': 5}

# # SCE-UA
# config = {
#          'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
#          'start_date': r'1999-1-2',
#          'end_date': r'2007-1-1',
#          'cal_alg': 'SCE-UA',    # Avaliable methods: NSGAII
#          'metrics': ['mss'],# metrics_name_list,  # Metrics to be minimized 
#          'population_size': 200,            # Number of individuals in the population  [50-1000]
#          'num_generations': 30,    # Number of generations for the calibration algorithm, [5-1000]
#         #  'magnitude': 0.1,          # Perturbation magnitude for offspring generation
#          'cross_prob': 0.95,          # Crossover probability
#          'mutation_rate': 0.5,       # Mutation rate
#          'regeneration_rate': 0.1,   # Regeneration rate
#          'eta_mut': 5,               # Mutation parameter for polynomial mutation
#          'num_complexes': 10,          # Number of complexes for partitioning the population.
#          'kstop': 100,                 # Maximum number of stagnation before restarting
#          'pcento': 0.1,               # Percentage improvement allowed in the past kstop loops.
#          'peps': 0.000001,               # Minimum value for the standard deviation
#          'lb': [1e-3, 1e-2, 1e-4, 1e-3], # Lower bounds for the parameters
#          'ub': [1e+1, 1e+3, 5e-1, 5e-1], # Upper bounds for the parameters,
#          'trs': 5}

# # Simulated Annealing
config = {
         'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
         'start_date': r'1986-11-9',
         'end_date': r'2025-1-1',
         'cal_alg': 'Simulated Annealing',    # Avaliable methods: NSGAII
         'metrics': ['mss'],# metrics_name_list,  # Metrics to be minimized 
         'max_iterations': 5000,    # Number of generations for the calibration algorithm, [100 - 100000]
         'initial_temperature': 100, # Starting temperature for annealing. [100-1000]
         'cooling_rate': 0.99, # Rate at which the temperature decreases. [0.9-0.99]
         'lb': [1e-2, 1e-2, 1e-3, 1e-3], # Lower bounds for the parameters
         'ub': [1e+1, 1e+2, 1e-1, 1e-1], # Upper bounds for the parameters,
         'trs': 'Average',
         'n_restarts': 5}


path = r"/mnt/c/Users/freitasl/Documents/ReposGIT/metrics_for_y09_model/data/Angourie_daily.nc"

dataset = xr.open_dataset(path)

dataset.load()
dataset.close()

# Verifica se o arquivo existe


starting_time = t_.time()
print('Starting JIT compilation...')

my_model = Model(dataset, config)

my_model.calibrate()

par_values = my_model.par_values
par_names = my_model.par_names

for value, name in zip(par_values, par_names):
    print(f'{name}: {value}')

full_run = my_model.full_run

print('Time elapsed: %.2f seconds' % (t_.time() - starting_time))

fig, ax = plt.subplots(1, 1, figsize=(12, 2))

ax.plot(my_model.time_obs, my_model.Obs, 'k-s', label='Observations', markersize=1, linewidth=0.3)
ax.plot(my_model.time, full_run, 'r-', label='Model', linewidth=1.5)
ax.set_title('Calibration results')
ax.set_ylabel('Shoreline position (m)')

plt.tight_layout()
plt.show()