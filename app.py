# Trabalho de processamento de imagens

import cv2
from FuncoesAuxiliares import lfa, metrics, gaussianNoise, gaussianFilter, cannyEdgeDetection, unsharpMasking, highboostFilter

originalImage = cv2.imread('Images/imagem-da-lib.png', cv2.IMREAD_GRAYSCALE)
height, width = originalImage.shape

# Mostrando a imagem original

cv2.imshow('Imagem original', originalImage)
cv2.waitKey(0)
print("Clicado")

#* Primeira parte: Implementação do Filtro de Difusão Anisotrópica (FDA) 

# Obtendo resultados dos filtros

fdaImage = lfa.ApplyLfaFilter(originalImage)

noisedImage, noise = gaussianNoise.GenerateImageWithGaussianNoise(originalImage, height, width)

fdaNoisedImage = lfa.ApplyLfaFilter(noisedImage)

gaussianFilteredImage = gaussianFilter.ApplyGaussianFilter(originalImage)

gaussianFilteredNoisedImage = gaussianFilter.ApplyGaussianFilter(noisedImage)

# Mostrando os resultados

lfa.ShowLfaResults(originalImage, fdaImage)

gaussianNoise.ShowImageAndNoiseResults(originalImage, noise, noisedImage)

lfa.ShowLfaResults(noisedImage, fdaNoisedImage)

gaussianFilter.showGaussianResults(originalImage, gaussianFilteredImage)
gaussianFilter.showGaussianResults(noisedImage, gaussianFilteredNoisedImage)

# Métricas

psnrFda, mseFda = metrics.CalculateMetrics(originalImage, fdaImage)

psnrGaussian, mseGaussian = metrics.CalculateMetrics(originalImage, gaussianFilteredImage)

print(f"PSNR da imagem filtrada: {psnrFda}, MSE da imagem filtrada: {mseFda}")
print(f"PSNR da imagem com filtro gaussiano: {psnrGaussian}, MSE da imagem com filtro gaussiano: {mseGaussian}")

#* Segunda parte: Aplicação de técnicas de aguçamento.

# Obtendo resultados dos filtros

fdaEdges = cannyEdgeDetection.ApplyCannyEdgeDetection(fdaImage)
gaussianEdges = cannyEdgeDetection.ApplyCannyEdgeDetection(gaussianFilteredImage)

sharpenedFda = unsharpMasking.ApplyUnsharpMask(originalImage, fdaImage)
sharpenedGaussian = unsharpMasking.ApplyUnsharpMask(originalImage, gaussianFilteredImage)

highboostFda = highboostFilter.ApplyHighboostFilter(originalImage, fdaImage)
highboostGaussian = highboostFilter.ApplyHighboostFilter(originalImage, gaussianFilteredImage)

edgesSharpenedFda = cannyEdgeDetection.ApplyCannyEdgeDetection(sharpenedFda)
edgesSharpenedGaussian = cannyEdgeDetection.ApplyCannyEdgeDetection(sharpenedGaussian)

edgesHighboostFda = cannyEdgeDetection.ApplyCannyEdgeDetection(highboostFda)
edgesHighboostGaussian = cannyEdgeDetection.ApplyCannyEdgeDetection(highboostGaussian)

# Mostrando os resultados

cannyEdgeDetection.ShowEdgeDetectionResults(originalImage, fdaEdges, gaussianEdges)

unsharpMasking.ShowUnsharpMaskingResults(originalImage, sharpenedFda, sharpenedGaussian)

highboostFilter.ShowHighboostResults(originalImage, highboostFda, highboostGaussian)

cannyEdgeDetection.ShowEdgeDetectionResults(originalImage, edgesSharpenedFda, edgesSharpenedGaussian)

cannyEdgeDetection.ShowEdgeDetectionResults(originalImage, edgesHighboostFda, edgesHighboostGaussian)