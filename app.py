# Importações necessárias
import cv2
from FuncoesAuxiliares import print, skinDetection, kmeans, seeding

# Carregando as imagens

fingerPrintImage = cv2.imread('Images/dedo.jpg', cv2.IMREAD_GRAYSCALE)
faceImage = cv2.imread('Images/tiozao.jpg')

print.ShowImage(fingerPrintImage, 'Finger Print')

print.ShowImage(faceImage, 'Face')

#* Primeira parte: – Limiarização Básica x Otsu:

#* Segunda parte: Detecção de rostos em imagens

skinDetection.SkinSegmentation(faceImage, "Images/tiozao_skin.png")
kmeans.KMeans3D(faceImage, imgNameOut="Images/tiozao_kmeans.png")
seeding.Seeding(faceImage, imgNameOut="Images/tiozao_seeding.png")