import numpy as np
from skimage import metrics
from sklearn.metrics import mean_squared_error

def SNR2(input, output, target):
       
    diff_clean = output - target
    mse_pred_clean = np.mean(np.power(diff_clean, 2))

    diff_noisy = input - target
    mse_noisy_clean = np.mean(np.power(diff_noisy, 2))

    snr2 = 1 - (np.sqrt(mse_pred_clean) / np.sqrt(mse_noisy_clean))

    return snr2

def PSNR(input, output, target):          
    psnr = metrics.peak_signal_noise_ratio(target, output, data_range=2)
    return psnr

def MSE(input, output, target):
    mse = mean_squared_error(target, output)
    return mse

def export_metrics_csv(input, output, results_dir, target=None):

    if target is None:
        return None

    snr2 = SNR2(input, output, target)
    psnr = PSNR(input, output, target)
    mse = MSE(input, output, target)

    metrics = np.array([
        snr2,
        psnr,
        mse
    ])

    headers = [
        "snr2",
        "psnr",
        "mse"
    ]

    np.savetxt(
        f'{results_dir}/metrics.csv',
        metrics.reshape(1, -1),
        delimiter=",",
        fmt="%.2f",
        header=",".join(headers),
        comments=""
    )

    return snr2