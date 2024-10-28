# Importações necessárias
import cv2
from FuncoesAuxiliares import print, skinDetection, kmeans, seeding

# Carregando as imagens

fingerPrintImage = cv2.imread('Images/dedo.jpg', cv2.IMREAD_GRAYSCALE)
faceImage = cv2.imread('Images/tiozao.jpg')
secondFaceImage = cv2.imread('Images/tiozao2.png')

print.ShowImage(fingerPrintImage, 'Finger Print')

print.ShowImage(faceImage, 'Face')

#* Primeira parte: – Limiarização Básica x Otsu:

#* Segunda parte: Detecção de rostos em imagens

skinDetection.SkinSegmentation(faceImage, "Images/tiozao_skin.png")
skinDetection.SkinSegmentation(secondFaceImage, "Images/tiozao2_skin.png")

kmeans.KMeansImplementation(faceImage, clusters_to_show=[0,4])
kmeans.KMeansImplementation(secondFaceImage, clusters_to_show=[0,3,6])

seeding.Seeding(faceImage, imgNameOut="Images/tiozao_seeding.png")
seeding.Seeding(secondFaceImage, imgNameOut="Images/tiozao2_seeding.png")