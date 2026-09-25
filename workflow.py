import torch
import numpy as np
from datetime import datetime
from train import call_train
from test import call_test
from plot_losses import call_losses

print('Workflow start', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))

torch.manual_seed(42)

arr = np.loadtxt('/home/src/experiments_new_data.csv', delimiter=',', dtype=str)
loss_paths = []

for line in arr:
    is_supervised =  True if line[0] == 'True' else False
    train_y_path = line[1]
    n_folds = int(line[2])
    epochs = int(line[3])
    lr = float(line[4])
    base_channels = int(line[5])
    test_y_path = line[6]

    result_dir = call_train(is_supervised, train_y_path, n_folds, epochs, lr, base_channels)
    loss_paths.append(f'{result_dir}/loss_history.npy')

    if train_y_path != test_y_path:
        call_test(is_supervised, train_y_path, epochs, lr, base_channels, test_y_path)

call_losses(loss_paths)

print('Workflow end', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))
