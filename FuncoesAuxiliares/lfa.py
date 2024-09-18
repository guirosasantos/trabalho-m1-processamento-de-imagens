import numpy as np
from matplotlib import pyplot as plt

def ApplyLfaFilter(image,niter=150,kappa=10,gamma=0.1,step=(1.,1.),option=1):
    image = image.astype('float32')
    imageOut = image.copy()

    deltaS = np.zeros_like(imageOut)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imageOut)
    gE = gS.copy()

    for ii in range(niter):
        deltaS[:-1,: ] = np.diff(imageOut,axis=0)
        deltaE[: ,:-1] = np.diff(imageOut,axis=1)

        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        E = gE*deltaE
        S = gS*deltaS

        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        imageOut += gamma*(NS+EW)

    return imageOut

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