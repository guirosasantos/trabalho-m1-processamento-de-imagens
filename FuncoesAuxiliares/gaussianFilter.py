import cv2
from matplotlib import pyplot as plt

def ApplyGaussianFilter(image, kernelSize=(3, 3), sigma=0):
    return cv2.GaussianBlur(image, kernelSize, sigma)

def showGaussianResults(image, imageWithGaussianFilter):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 2, 2)
    plt.imshow(imageWithGaussianFilter, cmap='gray')
    plt.axis("off")
    plt.title("Imagem com Filtro Gaussiano")

    plt.show()