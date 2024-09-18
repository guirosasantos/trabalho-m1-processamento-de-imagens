import cv2
from matplotlib import pyplot as plt

def ApplyCannyEdgeDetection(image):
    image8units = cv2.convertScaleAbs(image)
    edges = cv2.Canny(image8units, 100, 200)
    return edges

def ShowEdgeDetectionResults(original, fdaEdges, gaussianEdges):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(fdaEdges, cmap='gray')
    plt.axis("off")
    plt.title("Sobel Edge Detection - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(gaussianEdges, cmap='gray')
    plt.axis("off")
    plt.title("Sobel Edge Detection - Gaussiano")

    plt.show()