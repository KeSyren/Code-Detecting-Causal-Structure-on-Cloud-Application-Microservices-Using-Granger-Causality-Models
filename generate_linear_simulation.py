import numpy as np
import os
from synthetic import simulate_var
from utils import get_project_root


# Simulate data
# num_nonzero = int(p * sparsity) - 1
# sparsity=0.2
p = 20 # num_of_time series
T = 1000 # samples
var_lag = 1
X_np, beta, GC = simulate_var(p, T, var_lag)


root = str(get_project_root()) + os.path.sep
data_dir = root + 'data/linear-simulation' + os.path.sep

if not os.path.exists(data_dir + 'linear_p=' + str(p) + '_T=' + str(T)):
    os.makedirs(data_dir + 'linear_p=' + str(p) + '_T=' + str(T))
linear_simulation_data = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'temporal.data.npy'
# generate the linear simulation data

linear_simulation_data_dir = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep

if not os.path.exists(linear_simulation_data_dir):
    os.makedirs(linear_simulation_data_dir)

linear_simulation = linear_simulation_data_dir + 'temporal.data'
ground_truth = linear_simulation_data_dir + 'meta.data'

# ground truth
meta_filename = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'temporal.meta'

np.savetxt(linear_simulation, X_np, delimiter=',', fmt="%s")
np.save(linear_simulation, X_np)
np.savetxt(ground_truth, GC.flatten().reshape(1, -1), delimiter=',', fmt="%s")
np.save(ground_truth, GC)