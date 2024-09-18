import cv2
from matplotlib import pyplot as plt

def ApplyUnsharpMask(image, blurredImage, blurStrength=1.5):
    image8Units = cv2.convertScaleAbs(image)
    blurredimage8Units = cv2.convertScaleAbs(blurredImage)

    sharpened = cv2.addWeighted(image8Units, 1 + blurStrength, blurredimage8Units, -blurStrength, 0)
    return sharpened

def ShowUnsharpMaskingResults(original, sharpenedFda, sharpenedGaussian):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(sharpenedFda, cmap='gray')
    plt.axis("off")
    plt.title("Unsharp Mask - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(sharpenedGaussian, cmap='gray')
    plt.axis("off")
    plt.title("Unsharp Mask - Gaussiano")

    plt.show()