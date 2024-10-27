import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('Images/tiozao.jpg')
if image is None:
    print("Erro ao carregar a imagem.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


num_superpixels = 400    # Número de superpixels desejados
num_levels = 4           # Número de níveis na pirâmide de imagem
prior = 2                # Peso do termo de prior
num_histogram_bins = 5   # Número de bins no histograma de cores

(height, width) = image_rgb.shape[:2]

seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, image_rgb.shape[2],
                                           num_superpixels, num_levels, prior, num_histogram_bins)

seeds.iterate(image_rgb, 10)  # Número de iterações

labels = seeds.getLabels()

mask = seeds.getLabelContourMask(False)

image_superpixels = image_rgb.copy()
image_superpixels[mask == 255] = (0, 0, 0)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image_rgb)
plt.title('Imagem Original'), plt.axis('off')
plt.subplot(122), plt.imshow(image_superpixels)
plt.title('Superpixels com SEEDS'), plt.axis('off')
plt.show()