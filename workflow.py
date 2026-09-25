import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from train import call_train
from test import call_test
from plot_losses import call_losses

print('Workflow start', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))

torch.manual_seed(42)

arr = np.loadtxt('/home/src/experiments.csv', delimiter=',', dtype=str, ndmin=2)
loss_paths = []

for line in arr:
    if line[0][0] == '#':
        continue
    is_supervised =  True if line[0] == 'True' else False
    epochs = int(line[1])
    lr = float(line[2])
    base_channels = int(line[3])
    train_y_path = line[4]
    test_y_path = line[5]
    train_x_path = line[6]
    test_x_path = line[7]

    dir_name = str(Path(train_y_path).parent.name)

    result_dir = call_train(is_supervised, epochs, lr, base_channels, dir_name, train_y_path, train_x_path)
    loss_paths.append(f'{result_dir}/loss_history.npy')

    if train_y_path != test_y_path:
        call_test(base_channels, result_dir, test_y_path, test_x_path)

call_losses(loss_paths)

print('Workflow end', datetime.now().strftime("%d/%m/%Y - %H:%M:%S"))
