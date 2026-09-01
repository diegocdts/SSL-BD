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

def export_metrics_csv(filename, y, x, reflectivity, reflectivity_sparse):

    if x is None:
        return None, None

    reflectivity_snr2 = SNR2(y, reflectivity, x)
    reflectivity_psnr = PSNR(y, reflectivity, x)
    reflectivity_mse = MSE(y, reflectivity, x)

    reflectivity_sparse_snr2 = SNR2(y, reflectivity_sparse, x)
    reflectivity_sparse_psnr = PSNR(y, reflectivity_sparse, x)
    reflectivity_sparse_mse = MSE(y, reflectivity_sparse, x)

    metrics = np.array([
        reflectivity_snr2,
        reflectivity_psnr,
        reflectivity_mse,
        reflectivity_sparse_snr2,
        reflectivity_sparse_psnr,
        reflectivity_sparse_mse
    ])

    headers = [
        "reflectivity_snr2",
        "reflectivity_psnr",
        "reflectivity_mse",
        "reflectivity_sparse_snr2",
        "reflectivity_sparse_psnr",
        "reflectivity_sparse_mse"
    ]

    np.savetxt(
        filename,
        metrics.reshape(1, -1),
        delimiter=",",
        fmt="%.2f",
        header=",".join(headers),
        comments=""
    )

    return reflectivity_snr2, reflectivity_sparse_snr2