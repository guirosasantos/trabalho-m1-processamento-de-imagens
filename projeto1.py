import cv2
import numpy as np
import matplotlib.pyplot as plt
import math  # Importante para cálculos matemáticos
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error  # Importar métricas

# Funções de processamento de imagem
def create_element(size_x, size_y):
    element = np.ones((size_x, size_y), dtype=np.uint8) * 255
    return element

def threashold(img_in):
    img_out = np.zeros(img_in.shape)
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            if img_in[i, j] < 127:
                img_out[i, j] = 0
            else:
                img_out[i, j] = 255
    return np.array(img_out, dtype=np.uint8)

def add_padding(img, padding_height, padding_width):
    n, m = img.shape
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height:n + padding_height, padding_width:m + padding_width] = img
    return padded_img

def morf_dilate_erode(img, kernel, padding=True, dilate=True):
    k_height, k_width = kernel.shape
    img_height, img_width = img.shape
    pad_height = k_height // 2
    pad_width = k_width // 2
    padded_img = add_padding(img, pad_height, pad_width) if padding else img
    output = np.zeros((img_height, img_width), dtype=np.uint8)

    if dilate:
        for i_img in range(img_height):
            for j_img in range(img_width):
                match_element = 0
                for i_kernel in range(k_height):
                    for j_kernel in range(k_width):
                        if (kernel[i_kernel, j_kernel] == 255 and 
                            padded_img[i_img + i_kernel, j_img + j_kernel] == 255):
                            match_element = 1
                            break
                    if match_element:
                        break
                output[i_img, j_img] = 255 if match_element == 1 else 0
    else:
        for i_img in range(img_height):
            for j_img in range(img_width):
                match_element = 1
                for i_kernel in range(k_height):
                    for j_kernel in range(k_width):
                        if (kernel[i_kernel, j_kernel] == 255 and 
                            padded_img[i_img + i_kernel, j_img + j_kernel] != 255):
                            match_element = 0
                            break
                    if match_element == 0:
                        break
                output[i_img, j_img] = 255 if match_element == 1 else 0

    return output

# Funções de ruído e limiarização
def aplicar_ruido_gaussiano(imagem, mean=2, std_dev=20):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    ruido = np.random.normal(mean, std_dev, imagem_cinza.shape).astype(np.uint8)
    return cv2.add(imagem_cinza, ruido)

def limiarizacao_otsu(imagem):
    T, imagem_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Valor do limiar de Otsu:", T)
    return imagem_otsu

# Funções de abertura e fechamento
# def abertura(img, kernel):
#     eroded = morf_dilate_erode(img, kernel, dilate=False)
#     opened = morf_dilate_erode(eroded, kernel, dilate=True)
#     return opened

# def fechamento(img, kernel):
#     dilated = morf_dilate_erode(img, kernel, dilate=True)
#     closed = morf_dilate_erode(dilated, kernel, dilate=False)
#     return closed

#com biblioteca
def abertura(img, kernel):
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opened
def fechamento(img, kernel):
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

# Função para calcular métricas
def CalculateMetrics(original, altered):
    original = original.astype(np.float32)
    altered = altered.astype(np.float32)
    
    data_range = max(original.max(), altered.max()) - min(original.min(), altered.min())
    
    psnrValue = peak_signal_noise_ratio(original, altered, data_range=data_range)
    mseValue = mean_squared_error(original, altered)
    rmseValue = np.sqrt(mseValue)
    return psnrValue, mseValue, rmseValue

# Carregar a imagem
imagem = cv2.imread("images/dedo.jpg")
if imagem is None:
    raise ValueError("Erro: imagem não encontrada. Verifique o caminho.")

# Converter a imagem original para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Adicionar ruído e aplicar limiarizações
imagem_ruidosa = aplicar_ruido_gaussiano(imagem)
_, thresh_binaria = cv2.threshold(imagem_ruidosa, 127, 255, cv2.THRESH_BINARY)
imagem_otsu = limiarizacao_otsu(imagem_ruidosa)

# Criar um kernel e aplicar abertura e fechamento em ambas as limiarizações
kernel = create_element(3, 3)
opened_binaria = abertura(thresh_binaria, kernel)
closed_binaria = fechamento(thresh_binaria, kernel)
opened_otsu = abertura(imagem_otsu, kernel)
closed_otsu = fechamento(imagem_otsu, kernel)
open_closed_otsu = abertura(closed_otsu, kernel)
open_closed_binaria = abertura(closed_binaria, kernel)
close_original = fechamento(imagem_cinza, kernel)

# Função para exibir imagens individualmente
def show_image_individual(img, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Exibir individualmente as imagens especificadas
# Imagem Original
show_image_individual(imagem_cinza, "Imagem Original")

# Imagem com Ruído
show_image_individual(imagem_ruidosa, "Imagem com Ruído")

# Imagem com Abertura - Limiarização Binária
show_image_individual(opened_binaria, "Binária + Abertura")

# Imagem com Abertura - Limiarização Otsu
show_image_individual(opened_otsu, "Otsu + Abertura")

#imagem original com fechamento
show_image_individual(close_original, "original + fechamento")

# Calcular e exibir as métricas entre a imagem original e as imagens processadas
psnr_binaria, mse_binaria, rmse_binaria = CalculateMetrics(imagem_cinza, opened_binaria)
psnr_otsu, mse_otsu, rmse_otsu = CalculateMetrics(imagem_cinza, opened_otsu)

print("Métricas para Binária + Abertura:")
print(f"MSE: {mse_binaria:.2f}")
print(f"PSNR: {psnr_binaria:.2f} dB")
print(f"RMSE: {rmse_binaria:.2f}")

print("\nMétricas para Otsu + Abertura:")
print(f"MSE: {mse_otsu:.2f}")
print(f"PSNR: {psnr_otsu:.2f} dB")
print(f"RMSE: {rmse_otsu:.2f}")

# Exibir as imagens usando matplotlib (grid)
def show_images(images, titles):
    plt.figure(figsize=(15, 10))
    num_images = len(images)
    cols = 5  # Número de colunas desejado
    rows = math.ceil(num_images / cols)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Exibir os resultados finais (grid)
show_images(
    [imagem_cinza, imagem_ruidosa,
     thresh_binaria, opened_binaria, closed_binaria,
     imagem_otsu, opened_otsu, closed_otsu, open_closed_otsu, open_closed_binaria],
    ["Imagem Original", "Imagem com Ruído",
     "Limiarização Binária", "Binária + Abertura", "Binária + Fechamento",
     "Limiarização Otsu", "Otsu + Abertura", "Otsu + Fechamento",
     "Otsu + Fechamento e Abertura", "Binária + Fechamento e Abertura"]
)