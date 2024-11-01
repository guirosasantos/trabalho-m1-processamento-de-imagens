# Importações necessárias
import cv2
from FuncoesAuxiliares import print, skinDetection, kmeans, seeding

# Carregando as imagens

faceImage = cv2.imread('Images/tiozao.jpg')
secondFaceImage = cv2.imread('Images/tiozao2.png')

print.ShowImage(faceImage, 'Face')
print.ShowImage(secondFaceImage, 'Second Face')

#* Segunda parte: Detecção de rostos em imagens

skinDetection.ApplySkinMask(faceImage, "Images/tiozao_skin.png")
skinDetection.ApplySkinMask(secondFaceImage, "Images/tiozao2_skin.png")

kmeans.KMeansImplementation(faceImage, clusters_to_show=[0,4])
kmeans.KMeansImplementation(secondFaceImage, clusters_to_show=[0,2,4,5])

seeding.Seeding(
    "Images/tiozao.jpg",
    "Images/tiozao_SEEDS.png",
    num_superpixels = 500,
    num_levels = 20,
    prior = 2,
    num_histogram_bins = 5,
    num_iterations = 100)
seeding.Seeding(
    "Images/tiozao2.png",
    "Images/tiozao2_SEEDS.png",
    num_superpixels = 500,
    num_levels = 20,
    prior = 2,
    num_histogram_bins = 5,
    num_iterations = 100)
