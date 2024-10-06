import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

def CalculateMetrics(original, altered):
    original = original.astype(np.float32)
    altered = altered.astype(np.float32)
    
    data_range = max(original.max(), altered.max()) - min(original.min(), altered.min())
    
    psnrValue = peak_signal_noise_ratio(original, altered, data_range=data_range)
    mseValue = mean_squared_error(original, altered)
    rmseValue = np.sqrt(mseValue)
    return psnrValue, mseValue, rmseValue