import cv2
from matplotlib import pyplot as plt

def ApplyHighboostFilter(image, blurredImage, A=1.5):
    image8Units = cv2.convertScaleAbs(image)
    blurredImage8Units = cv2.convertScaleAbs(blurredImage)

    highboost = cv2.addWeighted(image8Units, A, blurredImage8Units, -(A - 1), 0)
    return highboost

def ShowHighboostResults(original, highboostFda, highboostGaussian):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(highboostFda, cmap='gray')
    plt.axis("off")
    plt.title("Highboost - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(highboostGaussian, cmap='gray')
    plt.axis("off")
    plt.title("Highboost - Gaussiano")

    plt.show()
