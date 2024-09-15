# Trabalho de processamento de imagens

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error

img = cv2.imread('Images/imagem-da-lib.png', cv2.IMREAD_GRAYSCALE)
height, width = img.shape

cv2.imshow('Imagem original', img)
cv2.waitKey(0)

# Primeira parte: Implementação do Filtro de Difusão Anisotrópica (FDA) 

# Funções auxiliares

def showResults(img, blur, use_cmap=False, cmap='gray'):
    fig, axes = plt.subplots(nrows = 2,ncols=1, figsize=(20,30), sharex=True, sharey=True)
    ax = axes.ravel()
    if use_cmap:
        ax[0].imshow(img, cmap=cmap)
    else:
        ax[0].imshow(img)
    ax[0].set_title('Imagem Original')
    if use_cmap:
        ax[1].imshow(blur, cmap=cmap)
    else:
        ax[1].imshow(blur)
    ax[1].set_title('Resultado Com Filtro de Difusão Anisotrópica')
    for a in ax:
        a.axis('off')
    # Às vezes a linha abaixo faz com que o NB não reserve espaço para a imagem
    # Nesse caso comente-a
    #fig.tight_layout()
    plt.show()

def anisodiff(img,niter=150,kappa=10,gamma=0.1,step=(1.,1.),option=1):

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout

def calculate_metrics(original, noisy_or_filtered):
    psnr_value = peak_signal_noise_ratio(original, noisy_or_filtered)
    mse_value = mean_squared_error(original, noisy_or_filtered)
    return psnr_value, mse_value

def generate_image_with_gaussian_noise(img, height, width):
    gauss_noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randn(gauss_noise, 128, 50)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    gn_img = cv2.add(img, gauss_noise)
    return gn_img, gauss_noise

def show_image_and_noise_results(img, noise, img_with_noise):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(noise, cmap='gray')
    plt.axis("off")
    plt.title("Ruído Gaussiano")

    fig.add_subplot(1, 3, 3)
    plt.imshow(img_with_noise, cmap='gray')
    plt.axis("off")
    plt.title("Imagem com Ruído")

    plt.show()

def showGaussianResults(img, img_with_gaussian_filter):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 2, 2)
    plt.imshow(img_with_gaussian_filter, cmap='gray')
    plt.axis("off")
    plt.title("Imagem com Filtro Gaussiano")

    plt.show()

def apply_canny_edge_detection(img):
    img_8u = cv2.convertScaleAbs(img)
    edges = cv2.Canny(img_8u, 100, 200)
    return edges

def show_edge_detection_results(original, fda_edges, gaussian_edges):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(fda_edges, cmap='gray')
    plt.axis("off")
    plt.title("Sobel Edge Detection - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(gaussian_edges, cmap='gray')
    plt.axis("off")
    plt.title("Sobel Edge Detection - Gaussiano")

    plt.show()

def apply_unsharp_mask(img, blurred_img, blur_strength=1.5):
    img_8u = cv2.convertScaleAbs(img)
    blurred_img_8u = cv2.convertScaleAbs(blurred_img)

    sharpened = cv2.addWeighted(img_8u, 1 + blur_strength, blurred_img_8u, -blur_strength, 0)
    return sharpened

def show_unsharp_masking_results(original, sharpened_fda, sharpened_gaussian):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(sharpened_fda, cmap='gray')
    plt.axis("off")
    plt.title("Unsharp Mask - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(sharpened_gaussian, cmap='gray')
    plt.axis("off")
    plt.title("Unsharp Mask - Gaussiano")

    plt.show()

def apply_highboost_filter(img, blurred_img, A=1.5):
    # Converter imagens para uint8 para garantir compatibilidade com OpenCV
    img_8u = cv2.convertScaleAbs(img)
    blurred_img_8u = cv2.convertScaleAbs(blurred_img)
    
    # Aplicar o filtro Highboost
    highboost = cv2.addWeighted(img_8u, A, blurred_img_8u, -(A - 1), 0)
    return highboost

def show_highboost_results(original, highboost_fda, highboost_gaussian):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 3, 2)
    plt.imshow(highboost_fda, cmap='gray')
    plt.axis("off")
    plt.title("Highboost - FDA")

    fig.add_subplot(1, 3, 3)
    plt.imshow(highboost_gaussian, cmap='gray')
    plt.axis("off")
    plt.title("Highboost - Gaussiano")

    plt.show()


fda_diff = anisodiff(img)

showResults(img, fda_diff, use_cmap=True)

img_with_noise, gauss_noise = generate_image_with_gaussian_noise(img, height, width)

show_image_and_noise_results(img, gauss_noise, img_with_noise)

fda_diff_gn = anisodiff(img_with_noise)

showResults(img_with_noise, fda_diff_gn, use_cmap=True)

img_original_with_gaussian_filter = cv2.GaussianBlur(img, (3, 3), 0)
img_with_noise_with_gaussian_filter = cv2.GaussianBlur(img_with_noise, (3, 3), 0)

showGaussianResults(img, img_original_with_gaussian_filter)
showGaussianResults(img_with_noise, img_with_noise_with_gaussian_filter)

# Métricas

psnr_fda, mse_fda = calculate_metrics(img, fda_diff)
psnr_gaussian, mse_gaussian = calculate_metrics(img, img_original_with_gaussian_filter)

print(f"PSNR da imagem filtrada: {psnr_fda}, MSE da imagem filtrada: {psnr_fda}")
print(f"PSNR da imagem com filtro gaussiano: {psnr_gaussian}, MSE da imagem com filtro gaussiano: {mse_gaussian}")

# Segunda parte: Aplicação de técnicas de aguçamento.

# Aplicar a detecção de bordas nas imagens filtradas
fda_edges = apply_canny_edge_detection(fda_diff)
gaussian_edges = apply_canny_edge_detection(img_original_with_gaussian_filter)

show_edge_detection_results(img, fda_edges, gaussian_edges)

sharpened_fda = apply_unsharp_mask(img, fda_diff)
sharpened_gaussian = apply_unsharp_mask(img, img_original_with_gaussian_filter)

show_unsharp_masking_results(img, sharpened_fda, sharpened_gaussian)

highboost_fda = apply_highboost_filter(img, fda_diff)
highboost_gaussian = apply_highboost_filter(img, img_original_with_gaussian_filter)

show_highboost_results(img, highboost_fda, highboost_gaussian)

edges_sharpened_fda = apply_canny_edge_detection(sharpened_fda)
edges_sharpened_gaussian = apply_canny_edge_detection(sharpened_gaussian)

show_edge_detection_results(img, edges_sharpened_fda, edges_sharpened_gaussian)

edges_highboost_fda = apply_canny_edge_detection(highboost_fda)
edges_highboost_gaussian = apply_canny_edge_detection(highboost_gaussian)

show_edge_detection_results(img, edges_highboost_fda, edges_highboost_gaussian)