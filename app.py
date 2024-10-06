# Trabalho de processamento de imagens 2 

import cv2
import numpy as np
from FuncoesAuxiliares import metrics, gaussianFilter, sobelFilter, cannyEdgeDetection

originalImage = cv2.imread('Images/imagem-da-lib.png', cv2.IMREAD_GRAYSCALE)

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

idealPassHighFilteredImage = sobelFilter.createPA(convertedImage.shape, center=maxLoc)
idealPassHighFilteredImage = convertedImage * idealPassHighFilteredImage
idealPassHighFilteredImage = np.fft.ifftshift(idealPassHighFilteredImage)
idealPassHighFilteredImage = cv2.idft(idealPassHighFilteredImage)
idealPassHighFilteredImage = cv2.magnitude(idealPassHighFilteredImage[:, :, 0], idealPassHighFilteredImage[:, :, 1])
idealPassHighFilteredImage = np.array(idealPassHighFilteredImage, dtype=np.float32)
idealPassHighFilteredImage = np.abs(idealPassHighFilteredImage)
idealPassHighFilteredImage -= idealPassHighFilteredImage.min()
idealPassHighFilteredImage = idealPassHighFilteredImage*255/idealPassHighFilteredImage.max()
idealPassHighFilteredImage = idealPassHighFilteredImage.astype(np.uint8)

gaussianPassHighFilteredImage = sobelFilter.createPA(convertedImage.shape, center=maxLoc, lpType=2)
gaussianPassHighFilteredImage = convertedImage * gaussianPassHighFilteredImage
gaussianPassHighFilteredImage = np.fft.ifftshift(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage = cv2.idft(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage = cv2.magnitude(gaussianPassHighFilteredImage[:, :, 0], gaussianPassHighFilteredImage[:, :, 1])
gaussianPassHighFilteredImage = np.array(gaussianPassHighFilteredImage, dtype=np.float32)
gaussianPassHighFilteredImage = np.abs(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage -= gaussianPassHighFilteredImage.min()
gaussianPassHighFilteredImage = gaussianPassHighFilteredImage*255/gaussianPassHighFilteredImage.max()
gaussianPassHighFilteredImage = gaussianPassHighFilteredImage.astype(np.uint8)

# Mostrando os resultados

gaussianFilter.showGaussianResults(sobelFilteredImage, idealPassHighFilteredImage, "Imagem com Filtro Passa Alta Ideal", "Imagem com Filtro de Sobel")
gaussianFilter.showGaussianResults(sobelFilteredImage, gaussianPassHighFilteredImage, "Imagem com Filtro Passa Alta Gaussiano" , "Imagem com Filtro de Sobel")

# Métricas

psnrIdealPassHigh, mseIdealPassHigh, rmseIdealPassHigh = metrics.CalculateMetrics(sobelFilteredImage, idealPassHighFilteredImage)
psnrGaussianPassHigh, mseGaussianPassHigh, rmseGaussianPassHigh = metrics.CalculateMetrics(sobelFilteredImage, gaussianPassHighFilteredImage)

print("\n")
print(f"Métricas da imagem com filtro passa alta ideal: PSNR: {psnrIdealPassHigh}, MSE: {mseIdealPassHigh}, RMSE: {rmseIdealPassHigh}")
print(f"Métricas da imagem com filtro passa alta gaussiano: PSNR: {psnrGaussianPassHigh}, MSE: {mseGaussianPassHigh}, RMSE: {rmseGaussianPassHigh}")

#* Terceira parte: Comparação detecção de bordas

# Obtendo resultados dos filtros

cannyFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(originalImage)
cannyGaussianFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianFilteredImage)
cannyIdealPassLowFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(idealPassLowFilteredImage)
cannyGaussianPassLowFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianPassLowFilteredImage)
cannySobelFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(sobelFilteredImage)
cannyIdealPassHighFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(idealPassHighFilteredImage)
cannyGaussianPassHighFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianPassHighFilteredImage)

# Mostrando os resultados

gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianFilteredImage, "Gauss + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyIdealPassLowFilteredImage, "Ideal Low Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianPassLowFilteredImage, "Gaussian Low Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannySobelFilteredImage, "Sobel + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyIdealPassHighFilteredImage, "Ideal High Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianPassHighFilteredImage, "Gaussian High Pass + Canny", "Original Canny")

