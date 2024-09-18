import cv2
import numpy as np
from matplotlib import pyplot as plt

def GenerateImageWithGaussianNoise(image, height, width):
    gaussNoise = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(gaussNoise, 128, 50)
    gaussNoise = (gaussNoise * 0.5).astype(np.uint8)
    imageWithNoise = cv2.add(image, gaussNoise)
    return imageWithNoise, gaussNoise

def ShowImageAndNoiseResults(image, noise, imageWithNoise):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(noise, cmap='gray')
    plt.axis("off")
    plt.title("Ruído Gaussiano")

    fig.add_subplot(1, 3, 3)
    plt.imshow(imageWithNoise, cmap='gray')
    plt.axis("off")
    plt.title("Imagem com Ruído")

    plt.show()