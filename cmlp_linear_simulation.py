import numpy as np
import torch
from models.cmlp import cMLP, cMLPSparse, train_model_adam, train_model_gista
import time
from utils import get_project_root
import os

# For CPU or GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulate data
p = 20  # num_of_time series
T = 1000 # samples
sub_model = 'cMLP'
max_iter = 30

root = str(get_project_root()) + os.path.sep
data_dir = root + 'data' + os.path.sep + 'linear-simulation' + os.path.sep
if not os.path.exists(data_dir + 'linear_p=' + str(p) + '_T=' + str(T)+ os.path.sep):
    os.makedirs(data_dir + 'linear_p=' + str(p) + '_T=' + str(T))
if not os.path.exists(data_dir + 'linear_p=' + str(p) + '_T=' + str(T)+ os.path.sep + sub_model+ os.path.sep):
    os.makedirs(data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + sub_model)


# ts data filename
linear_simulation_data = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'temporal.data.npy'

# ground truth
meta_filename = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'meta.data.npy'

# output filename
output_filename = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + sub_model + os.path.sep + 'weights_iters=' + str(max_iter) + '.data'

X_np = np.load(linear_simulation_data)
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
GC = np.load(meta_filename)

# save simulation data and ground truth
# np.savetxt(linear_simulation_data, X_np, delimiter=',', fmt="%s")
# np.savetxt(meta_filename, GC.flatten().reshape(1, -1), delimiter=',', fmt="%s")
# np.save(linear_simulation_data, X_np)
# np.save(meta_filename, GC)

# Set up model
print("Start : %s" % time.ctime())
lag = 1
hidden = [10]  # hidden units layer
if device.type == 'cpu':
    cmlp = cMLP(p, lag, hidden)
else:
    cmlp = cMLP(p, lag, hidden).cuda(device=device)  # GPU

# Pretrain (no regularization)
check_every = 1
train_loss_list = train_model_adam(cmlp, X, lr=1e-5, niter=10000, check_every=check_every)

# Train with GISTA
check_every = 5
train_loss_list, train_mse_list = train_model_gista(
    cmlp, X, lam=0.012, lam_ridge=1e-4, lr=1e-5, penalty='H', max_iter=max_iter, check_every=check_every)

# Verify learned Granger causality
GC_est = cmlp.GC(threshold=False).cpu().data.numpy()
print(GC)
print(GC_est)

# save to file
np.savetxt(output_filename, np.array(GC_est).flatten().reshape(1, -1), delimiter=',', fmt="%s")
np.save(output_filename, GC_est)
print("End : %s" % time.ctime())
# print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
# print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
# print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))
