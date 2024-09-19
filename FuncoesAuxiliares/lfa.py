import numpy as np
from matplotlib import pyplot as plt

def ApplyLfaFilter(image, lambda_val=0.25, kappa=10, num_iterations=20):
    image = image.astype(np.float32)
    for t in range(num_iterations):
        # Cria uma cópia da imagem para atualização
        updated_image = image.copy()
        # Percorre cada pixel excluindo as bordas
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                # Calcula as diferenças com os vizinhos
                deltaN = image[i-1, j] - image[i, j]
                deltaS = image[i+1, j] - image[i, j]
                deltaE = image[i, j+1] - image[i, j]
                deltaW = image[i, j-1] - image[i, j]
                deltaNE = image[i-1, j+1] - image[i, j]
                deltaNW = image[i-1, j-1] - image[i, j]
                deltaSE = image[i+1, j+1] - image[i, j]
                deltaSW = image[i+1, j-1] - image[i, j]

                # Calcula as funções de condução
                cN = np.exp(-(deltaN / kappa)**2)
                cS = np.exp(-(deltaS / kappa)**2)
                cE = np.exp(-(deltaE / kappa)**2)
                cW = np.exp(-(deltaW / kappa)**2)
                cNE = np.exp(-(deltaNE / kappa)**2)
                cNW = np.exp(-(deltaNW / kappa)**2)
                cSE = np.exp(-(deltaSE / kappa)**2)
                cSW = np.exp(-(deltaSW / kappa)**2)

                # Atualiza o pixel
                updated_image[i, j] += lambda_val * (
                    cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW +
                    cNE * deltaNE + cNW * deltaNW + cSE * deltaSE + cSW * deltaSW
                )
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