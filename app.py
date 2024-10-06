# Trabalho de processamento de imagens 2 

import cv2
import numpy as np
from FuncoesAuxiliares import metrics, gaussianFilter, sobelFilter, cannyEdgeDetection

originalImage = cv2.imread('Images/imagem-da-lib.png', cv2.IMREAD_GRAYSCALE)
height, width = originalImage.shape

# Mostrando a imagem original

cv2.imshow('Imagem original', originalImage)
cv2.waitKey(0)
print("Clicado")

#* Primeira parte: Comparação filtros gaussiano e passa baixa

# Obtendo resultados dos filtros

convertedImage = gaussianFilter.ConvertImage(originalImage)
convertedImgRows, convertedImgCols = convertedImage.shape[:2]
real = np.power(convertedImage[:, :, 0], 2.0)
imaginary = np.power(convertedImage[:, :, 1], 2.0)
amplitude = np.sqrt(real+imaginary)
minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)

gaussianFilteredImage = gaussianFilter.ApplyGaussianFilter(originalImage)
idealPassLowFilteredImage = gaussianFilter.ApplyIdealPassLowFilter(convertedImage.shape, center=maxLoc)
idealPassLowFilteredImage = convertedImage * idealPassLowFilteredImage
idealPassLowFilteredImage = np.fft.ifftshift(idealPassLowFilteredImage)
idealPassLowFilteredImage = cv2.idft(idealPassLowFilteredImage)
idealPassLowFilteredImage = cv2.magnitude(idealPassLowFilteredImage[:, :, 0], idealPassLowFilteredImage[:, :, 1])
idealPassLowFilteredImage = np.array(idealPassLowFilteredImage, dtype=np.float32)
idealPassLowFilteredImage = np.abs(idealPassLowFilteredImage)
idealPassLowFilteredImage -= idealPassLowFilteredImage.min()
idealPassLowFilteredImage = idealPassLowFilteredImage*255/idealPassLowFilteredImage.max()
idealPassLowFilteredImage = idealPassLowFilteredImage.astype(np.uint8)

gaussianPassLowFilteredImage = gaussianFilter.ApplyIdealPassLowFilter(convertedImage.shape, center=maxLoc, lpType=2)
gaussianPassLowFilteredImage = convertedImage * gaussianPassLowFilteredImage
gaussianPassLowFilteredImage = np.fft.ifftshift(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage = cv2.idft(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage = cv2.magnitude(gaussianPassLowFilteredImage[:, :, 0], gaussianPassLowFilteredImage[:, :, 1])
gaussianPassLowFilteredImage = np.array(gaussianPassLowFilteredImage, dtype=np.float32)
gaussianPassLowFilteredImage = np.abs(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage -= gaussianPassLowFilteredImage.min()
gaussianPassLowFilteredImage = gaussianPassLowFilteredImage*255/gaussianPassLowFilteredImage.max()
gaussianPassLowFilteredImage = gaussianPassLowFilteredImage.astype(np.uint8)

# Mostrando os resultados

gaussianFilter.showGaussianResults(gaussianFilteredImage, idealPassLowFilteredImage, "Imagem com Filtro Passa Baixa Ideal")
gaussianFilter.showGaussianResults(gaussianFilteredImage, gaussianPassLowFilteredImage, "Imagem com Filtro Passa Baixa Gaussiano")

# Métricas

psnrIdealPassLow, mseIdealPassLow, rmseIdealPassLow = metrics.CalculateMetrics(gaussianFilteredImage, idealPassLowFilteredImage)
psnrGaussianPassLow, mseGaussianPassLow, rmseGaussianPassLow = metrics.CalculateMetrics(gaussianFilteredImage, gaussianPassLowFilteredImage)

print(f"Métricas da imagem com filtro passa baixa ideal: PSNR: {psnrIdealPassLow}, MSE: {mseIdealPassLow}, RMSE: {rmseIdealPassLow}")
print(f"Métricas da imagem com filtro passa baixa gaussiano: PSNR: {psnrGaussianPassLow}, MSE: {mseGaussianPassLow}, RMSE: {rmseGaussianPassLow}") 

#* Segunda parte: Comparação filtros sobel e passa alta

# Obtendo resultados dos filtros

sobelFilteredImage = sobelFilter.ApplySobelFilter(originalImage)
idealPassHighFilteredImage = sobelFilter.ApplyPassHighFilter(originalImage)
gaussianPassHighFilteredImage = sobelFilter.ApplyGaussianPassHighFilter(originalImage)

# Mostrando os resultados

sobelFilter.showSobelResults(sobelFilteredImage, idealPassHighFilteredImage)
sobelFilter.showSobelResults(sobelFilteredImage, gaussianPassHighFilteredImage)

# Métricas

psnrSobel, mseSobel, rmseSobel = metrics.CalculateMetrics(originalImage, sobelFilteredImage)
psnrIdealPassHigh, mseIdealPassHigh, rmseIdealPassHigh = metrics.CalculateMetrics(originalImage, idealPassHighFilteredImage)
psnrGaussianPassHigh, mseGaussianPassHigh, rmseGaussianPassHigh = metrics.CalculateMetrics(originalImage, gaussianPassHighFilteredImage)

print(f"Métricas da imagem com filtro de Sobel: PSNR: {psnrSobel}, MSE: {mseSobel}, RMSE: {rmseSobel}")
print(f"Métricas da imagem com filtro passa alta ideal: PSNR: {psnrIdealPassHigh}, MSE: {mseIdealPassHigh}, RMSE: {rmseIdealPassHigh}")
print(f"Métricas da imagem com filtro passa alta gaussiano: PSNR: {psnrGaussianPassHigh}, MSE: {mseGaussianPassHigh}, RMSE: {rmseGaussianPassHigh}")