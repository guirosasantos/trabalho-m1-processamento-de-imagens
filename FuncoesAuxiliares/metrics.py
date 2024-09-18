from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

def CalculateMetrics(original, altered):
    psnrValue = peak_signal_noise_ratio(original, altered)
    mseValue = mean_squared_error(original, altered)
    return psnrValue, mseValue