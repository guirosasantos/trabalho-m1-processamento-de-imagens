import numpy as np
from matplotlib import pyplot as plt

def CalculateW(lambda_val, kappa, alfa, beta):
    # Cálculo correto do fator da exponencial
    fator = ((abs(alfa-beta))**(1/5)/lambda_val)/5

    # Cálculo do valor de w com o expoente correto
    w = (1 - np.exp(-8 * kappa * np.exp(-fator))) / 8

    return w

def CalculateWc(w1, w2, w3, w4, w6, w7, w8, w9):
    return 1 - (w1 + w2 + w3 + w4 + w6 + w7 + w8 + w9)


def ApplyLfaFilter(image, lambda_val=0.25, kappa=10, num_iterations=10):
    image = image.astype(np.float32)
    for t in range(num_iterations):
        print(f"Iteração {t+1}")
        updated_image = image.copy()
        kernel = np.zeros((3, 3))
        for x in range(1, image.shape[0]-1):
            for y in range(1, image.shape[1]-1):

                pixelCentral = image[x, y]
                w1 = CalculateW(lambda_val, kappa, pixelCentral, image[x-1, y+1])
                w2 = CalculateW(lambda_val, kappa, pixelCentral, image[x, y+1])
                w3 = CalculateW(lambda_val, kappa, pixelCentral, image[x+1, y+1])
                w4 = CalculateW(lambda_val, kappa, pixelCentral, image[x-1, y])
                w6 = CalculateW(lambda_val, kappa, pixelCentral, image[x+1, y])
                w7 = CalculateW(lambda_val, kappa, pixelCentral, image[x-1, y-1])
                w8 = CalculateW(lambda_val, kappa, pixelCentral, image[x, y-1])
                w9 = CalculateW(lambda_val, kappa, pixelCentral, image[x+1, y-1])

                pixelCentral = CalculateWc(w1, w2, w3, w4, w6, w7, w8, w9)

                kernel[0, 0] = w1
                kernel[0, 1] = w2
                kernel[0, 2] = w3
                kernel[1, 0] = w4
                kernel[1, 1] = pixelCentral
                kernel[1, 2] = w6
                kernel[2, 0] = w7
                kernel[2, 1] = w8
                kernel[2, 2] = w9

                updated_image[x, y] = np.sum(kernel * image[x-1:x+2, y-1:y+2])

        image = updated_image.copy()
    return image

def ShowLfaResults(image, blur, useCmap=True, cmap='gray'):
    fig, axes = plt.subplots(nrows = 2,ncols=1, figsize=(20,30), sharex=True, sharey=True)
    ax = axes.ravel()
    if useCmap:
        ax[0].imshow(image, cmap=cmap)
    else:
        ax[0].imshow(image)
    ax[0].set_title('Imagem Original')
    if useCmap:
        ax[1].imshow(blur, cmap=cmap)
    else:
        ax[1].imshow(blur)
    ax[1].set_title('Resultado Com Filtro de Difusão Anisotrópica')
    for a in ax:
        a.axis('off')
    plt.show()