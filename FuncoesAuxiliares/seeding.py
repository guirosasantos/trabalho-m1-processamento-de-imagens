import cv2
import numpy as np
from matplotlib import pyplot as plt

def Seeding(imagem, num_superpixels=5, num_levels=2, 
                         prior=2, num_histogram_bins=5, num_iterations=2000):
    """
    Função que realiza segmentação por superpixels e destaca regiões de pele.

    Parâmetros:
    imagem (str): Caminho para a imagem a ser carregada.
    num_superpixels (int): Número aproximado de superpixels.
    num_levels (int): Níveis da pirâmide para SEEDS.
    prior (int): Prior para o algoritmo SEEDS.
    num_histogram_bins (int): Número de bins no histograma.
    num_iterations (int): Número de iterações do algoritmo.

    Retorna:
    Exibe a imagem original, a imagem com contornos e a segmentada.
    """
    # Carrega a imagem
    image = cv2.imread(imagem)
    if image is None:
        print("Erro ao carregar a imagem.")
        return

    # Converte a imagem para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (height, width) = image_rgb.shape[:2]

    # Cria o objeto SEEDS
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        width, height, image_rgb.shape[2], num_superpixels, 
        num_levels, prior, num_histogram_bins
    )

    # Executa o algoritmo de superpixels
    seeds.iterate(image_rgb, num_iterations)

    # Obtém as labels e a máscara de contorno
    labels = seeds.getLabels()
    mask = seeds.getLabelContourMask(False)

    # Desenha os contornos dos superpixels
    image_superpixels = image_rgb.copy()
    image_superpixels[mask == 255] = [0, 0, 0]

    # Calcula a cor média de cada superpixel
    num_superpixels_actual = seeds.getNumberOfSuperpixels()
    superpixel_means = np.zeros((num_superpixels_actual, 3), dtype=np.uint64)
    counts = np.zeros(num_superpixels_actual, dtype=np.uint64)

    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            superpixel_means[label] += image_rgb[y, x]
            counts[label] += 1

    superpixel_means = (superpixel_means / counts[:, np.newaxis]).astype(np.uint8)

    # Converte para o espaço HSV
    superpixel_means_hsv = cv2.cvtColor(
        superpixel_means.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)

    # Define intervalo para detectar pele
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = np.all((superpixel_means_hsv >= lower_hsv) & 
                       (superpixel_means_hsv <= upper_hsv), axis=1)

    # Cria imagem segmentada
    segmented_image = image_rgb.copy()
    for i in range(num_superpixels_actual):
        mask_i = labels == i
        if skin_mask[i]:
            segmented_image[mask_i] = [255, 0, 0]  # Destaque em vermelho
        else:
            segmented_image[mask_i] = image_rgb[mask_i] // 2  # Escurece

    # Exibe as imagens
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_superpixels)
    plt.title('Contornos dos Superpixels')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image)
    plt.title('Superpixels Segmentados')
    plt.axis('off')

    plt.show()
