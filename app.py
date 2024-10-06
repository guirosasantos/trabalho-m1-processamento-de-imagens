# Importações necessárias
import cv2
import numpy as np
import time  # Importa o módulo de tempo
from FuncoesAuxiliares import metrics, gaussianFilter, sobelFilter, cannyEdgeDetection

# Carrega a imagem original
originalImage = cv2.imread('Images/imagem-da-lib.png', cv2.IMREAD_GRAYSCALE)

# Exibe a imagem original
cv2.imshow('Imagem original', originalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dicionário para armazenar os tempos de execução
execution_times = {}

#* Primeira parte: Comparação filtros gaussiano e passa baixa

# Medição de tempo para a conversão da imagem
start_time = time.perf_counter()
convertedImage = gaussianFilter.ConvertImage(originalImage)
end_time = time.perf_counter()
execution_times['ConvertImage'] = end_time - start_time

convertedImgRows, convertedImgCols = convertedImage.shape[:2]
real = np.power(convertedImage[:, :, 0], 2.0)
imaginary = np.power(convertedImage[:, :, 1], 2.0)
amplitude = np.sqrt(real + imaginary)
minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)

# Medição de tempo para o filtro Gaussiano
start_time = time.perf_counter()
gaussianFilteredImage = gaussianFilter.ApplyGaussianFilter(originalImage)
end_time = time.perf_counter()
execution_times['GaussianFilter'] = end_time - start_time

# Medição de tempo para o filtro Ideal Passa-Baixa
start_time = time.perf_counter()
idealPassLowFilteredImage = gaussianFilter.ApplyIdealPassLowFilter(convertedImage.shape, center=maxLoc)
idealPassLowFilteredImage = convertedImage * idealPassLowFilteredImage
idealPassLowFilteredImage = np.fft.ifftshift(idealPassLowFilteredImage)
idealPassLowFilteredImage = cv2.idft(idealPassLowFilteredImage)
idealPassLowFilteredImage = cv2.magnitude(idealPassLowFilteredImage[:, :, 0], idealPassLowFilteredImage[:, :, 1])
idealPassLowFilteredImage = np.array(idealPassLowFilteredImage, dtype=np.float32)
idealPassLowFilteredImage = np.abs(idealPassLowFilteredImage)
idealPassLowFilteredImage -= idealPassLowFilteredImage.min()
idealPassLowFilteredImage = idealPassLowFilteredImage * 255 / idealPassLowFilteredImage.max()
idealPassLowFilteredImage = idealPassLowFilteredImage.astype(np.uint8)
end_time = time.perf_counter()
execution_times['IdealLowPassFilter'] = end_time - start_time

# Medição de tempo para o filtro Gaussiano Passa-Baixa
start_time = time.perf_counter()
gaussianPassLowFilteredImage = gaussianFilter.ApplyIdealPassLowFilter(convertedImage.shape, center=maxLoc, lpType=2)
gaussianPassLowFilteredImage = convertedImage * gaussianPassLowFilteredImage
gaussianPassLowFilteredImage = np.fft.ifftshift(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage = cv2.idft(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage = cv2.magnitude(gaussianPassLowFilteredImage[:, :, 0], gaussianPassLowFilteredImage[:, :, 1])
gaussianPassLowFilteredImage = np.array(gaussianPassLowFilteredImage, dtype=np.float32)
gaussianPassLowFilteredImage = np.abs(gaussianPassLowFilteredImage)
gaussianPassLowFilteredImage -= gaussianPassLowFilteredImage.min()
gaussianPassLowFilteredImage = gaussianPassLowFilteredImage * 255 / gaussianPassLowFilteredImage.max()
gaussianPassLowFilteredImage = gaussianPassLowFilteredImage.astype(np.uint8)
end_time = time.perf_counter()
execution_times['GaussianLowPassFilter'] = end_time - start_time

# Exibe os resultados
gaussianFilter.showGaussianResults(gaussianFilteredImage, idealPassLowFilteredImage, "Ideal Low Pass", "Gaussian Filter")
gaussianFilter.showGaussianResults(gaussianFilteredImage, gaussianPassLowFilteredImage, "Gaussian Low Pass", "Gaussian Filter")

# Métricas
psnrIdealPassLow, mseIdealPassLow, rmseIdealPassLow = metrics.CalculateMetrics(gaussianFilteredImage, idealPassLowFilteredImage)
psnrGaussianPassLow, mseGaussianPassLow, rmseGaussianPassLow = metrics.CalculateMetrics(gaussianFilteredImage, gaussianPassLowFilteredImage)

print(f"Ideal Low Pass      Metrics: PSNR: {psnrIdealPassLow}, MSE: {mseIdealPassLow}, RMSE: {rmseIdealPassLow}")
print(f"Gaussian Low Pass   Metrics: PSNR: {psnrGaussianPassLow}, MSE: {mseGaussianPassLow}, RMSE: {rmseGaussianPassLow}")

#* Segunda parte: Comparação filtros Sobel e passa alta

# Medição de tempo para o filtro Sobel
start_time = time.perf_counter()
sobelFilteredImage = sobelFilter.ApplySobelFilter(originalImage)
end_time = time.perf_counter()
execution_times['SobelFilter'] = end_time - start_time

# Medição de tempo para o filtro Ideal Passa-Alta
start_time = time.perf_counter()
idealPassHighFilteredImage = sobelFilter.createPA(convertedImage.shape, center=maxLoc)
idealPassHighFilteredImage = convertedImage * idealPassHighFilteredImage
idealPassHighFilteredImage = np.fft.ifftshift(idealPassHighFilteredImage)
idealPassHighFilteredImage = cv2.idft(idealPassHighFilteredImage)
idealPassHighFilteredImage = cv2.magnitude(idealPassHighFilteredImage[:, :, 0], idealPassHighFilteredImage[:, :, 1])
idealPassHighFilteredImage = np.array(idealPassHighFilteredImage, dtype=np.float32)
idealPassHighFilteredImage = np.abs(idealPassHighFilteredImage)
idealPassHighFilteredImage -= idealPassHighFilteredImage.min()
idealPassHighFilteredImage = idealPassHighFilteredImage * 255 / idealPassHighFilteredImage.max()
idealPassHighFilteredImage = idealPassHighFilteredImage.astype(np.uint8)
end_time = time.perf_counter()
execution_times['IdealHighPassFilter'] = end_time - start_time

# Medição de tempo para o filtro Gaussiano Passa-Alta
start_time = time.perf_counter()
gaussianPassHighFilteredImage = sobelFilter.createPA(convertedImage.shape, center=maxLoc, lpType=2)
gaussianPassHighFilteredImage = convertedImage * gaussianPassHighFilteredImage
gaussianPassHighFilteredImage = np.fft.ifftshift(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage = cv2.idft(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage = cv2.magnitude(gaussianPassHighFilteredImage[:, :, 0], gaussianPassHighFilteredImage[:, :, 1])
gaussianPassHighFilteredImage = np.array(gaussianPassHighFilteredImage, dtype=np.float32)
gaussianPassHighFilteredImage = np.abs(gaussianPassHighFilteredImage)
gaussianPassHighFilteredImage -= gaussianPassHighFilteredImage.min()
gaussianPassHighFilteredImage = gaussianPassHighFilteredImage * 255 / gaussianPassHighFilteredImage.max()
gaussianPassHighFilteredImage = gaussianPassHighFilteredImage.astype(np.uint8)
end_time = time.perf_counter()
execution_times['GaussianHighPassFilter'] = end_time - start_time

# Exibe os resultados
gaussianFilter.showGaussianResults(sobelFilteredImage, idealPassHighFilteredImage, "Ideal High Pass", "Sobel Filter")
gaussianFilter.showGaussianResults(sobelFilteredImage, gaussianPassHighFilteredImage, "Gaussian High Pass", "Sobel Filter")

# Métricas
psnrIdealPassHigh, mseIdealPassHigh, rmseIdealPassHigh = metrics.CalculateMetrics(sobelFilteredImage, idealPassHighFilteredImage)
psnrGaussianPassHigh, mseGaussianPassHigh, rmseGaussianPassHigh = metrics.CalculateMetrics(sobelFilteredImage, gaussianPassHighFilteredImage)

print("\n")
print(f"Ideal High Pass     Metrics: PSNR: {psnrIdealPassHigh}, MSE: {mseIdealPassHigh}, RMSE: {rmseIdealPassHigh}")
print(f"Gaussian High Pass  Metrics: PSNR: {psnrGaussianPassHigh}, MSE: {mseGaussianPassHigh}, RMSE: {rmseGaussianPassHigh}")

#* Terceira parte: Comparação detecção de bordas

# Medição de tempo para a detecção de bordas com Canny na imagem original
start_time = time.perf_counter()
cannyFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(originalImage)
end_time = time.perf_counter()
execution_times['CannyOriginal'] = end_time - start_time

# Medição de tempo para a detecção de bordas nas outras imagens
start_time = time.perf_counter()
cannyGaussianFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianFilteredImage)
end_time = time.perf_counter()
execution_times['CannyGaussianFiltered'] = end_time - start_time

start_time = time.perf_counter()
cannyIdealPassLowFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(idealPassLowFilteredImage)
end_time = time.perf_counter()
execution_times['CannyIdealLowPassFiltered'] = end_time - start_time

start_time = time.perf_counter()
cannyGaussianPassLowFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianPassLowFilteredImage)
end_time = time.perf_counter()
execution_times['CannyGaussianLowPassFiltered'] = end_time - start_time

start_time = time.perf_counter()
cannySobelFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(sobelFilteredImage)
end_time = time.perf_counter()
execution_times['CannySobelFiltered'] = end_time - start_time

start_time = time.perf_counter()
cannyIdealPassHighFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(idealPassHighFilteredImage)
end_time = time.perf_counter()
execution_times['CannyIdealHighPassFiltered'] = end_time - start_time

start_time = time.perf_counter()
cannyGaussianPassHighFilteredImage = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianPassHighFilteredImage)
end_time = time.perf_counter()
execution_times['CannyGaussianHighPassFiltered'] = end_time - start_time

# Exibe os resultados
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianFilteredImage, "Gauss + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyIdealPassLowFilteredImage, "Ideal Low Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianPassLowFilteredImage, "Gaussian Low Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannySobelFilteredImage, "Sobel + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyIdealPassHighFilteredImage, "Ideal High Pass + Canny", "Original Canny")
gaussianFilter.showGaussianResults(cannyFilteredImage, cannyGaussianPassHighFilteredImage, "Gaussian High Pass + Canny", "Original Canny")

# Exibe os tempos de execução
print("\nTempos de execução:")
for operation, exec_time in execution_times.items():
    print(f"{operation}: {exec_time:.6f} segundos")
