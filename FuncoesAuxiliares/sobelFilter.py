from matplotlib import pyplot as plt
import numpy as np
import cv2

def ApplySobelFilter(image):
    """
    Aplica o filtro de Sobel em uma imagem.

    Args:
        image: A imagem de entrada.

    Returns:
        A imagem resultante da aplicação do filtro de Sobel.
    """
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelX**2 + sobelY**2)
    sobel = np.array(sobel, dtype=np.float32)
    sobel -= sobel.min()
    sobel = sobel*255/sobel.max()
    return np.array(sobel, dtype=np.uint8)

def createPA(shape, center, radius=35, lpType=0, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # Ideal high pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 0
        lpFilter[lpFilter >= pow(radius, 2.0)] = 1
    elif lpType == 1: #Butterworth Highpass Filters 
        lpFilter = 1.0 - 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == 2: # Gaussian Highpass Filter 
        lpFilter = 1.0 - np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix