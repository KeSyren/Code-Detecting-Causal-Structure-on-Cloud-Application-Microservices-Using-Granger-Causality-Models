import numpy as np
import torch
from models.clstm import cLSTM, train_model_gista
from utils import get_project_root
import time
import os

# For CPU or GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = 10   # num_of_time series
T = 10000 # samples
max_iter = 30000
sub_model = 'cLSTM'

root = str(get_project_root()) + '/'
data_dir = root + 'data' + os.path.sep + 'linear-simulation' + os.path.sep

root = str(get_project_root()) + os.path.sep
data_dir = root + 'data' + os.path.sep + 'linear-simulation' + os.path.sep
if not os.path.exists(data_dir + 'linear_p=' + str(p) + '_T=' + str(T)+ os.path.sep):
    os.makedirs(data_dir + 'linear_p=' + str(p) + '_T=' + str(T))
if not os.path.exists(data_dir + 'linear_p=' + str(p) + '_T=' + str(T)+ os.path.sep + sub_model+ os.path.sep):
    os.makedirs(data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + sub_model)

# ground truth
meta_filename = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'meta.data.npy'
# ts data
linear_simulation_data = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + 'temporal.data.npy'

# output estimated matrix
output_filename = data_dir + 'linear_p=' + str(p) + '_T=' + str(T) + os.path.sep + sub_model + os.path.sep + 'weights_iters=' + str(max_iter) + '.data'

X_np = np.load(linear_simulation_data)
GC = np.load(meta_filename)

X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

print(X.shape)
print(GC)

# Set up model
if device.type == 'cpu':
    clstm = cLSTM(p, hidden=10)
else:
    clstm = cLSTM(p, hidden=10).cuda(device=device) # GPU

# Train with GISTA
print("Start : %s" % time.ctime())
check_every = 1
train_loss_list, train_mse_list = train_model_gista(
    clstm, X, lam=0.012, lam_ridge=1e-4, lr=0.00001, max_iter=max_iter, check_every=check_every, truncation=5)


# Verify learned Granger causality
GC_est = clstm.GC(threshold=False).cpu().data.numpy()
print(GC)
print(GC_est)
# save to file
np.savetxt(output_filename, np.array(GC_est).flatten().reshape(1, -1), delimiter=',', fmt="%s")
np.save(output_filename, GC_est)
print("End : %s" % time.ctime())
