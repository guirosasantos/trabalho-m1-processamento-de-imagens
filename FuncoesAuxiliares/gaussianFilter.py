from matplotlib import pyplot as plt
import numpy as np
import cv2

def convolucao(imagem, kernel):
  """
  Função para realizar a convolução de uma imagem com um kernel.

  Args:
    imagem: A imagem de entrada.
    kernel: O kernel para a convolução.

  Returns:
    A imagem resultante da convolução.
  """
  linhas, colunas = imagem.shape[:2]
  linhas_kernel, colunas_kernel = kernel.shape

  # Calcula as dimensões da imagem de saída
  linhas_saida = linhas - linhas_kernel + 1
  colunas_saida = colunas - colunas_kernel + 1
  imagem_saida = np.zeros((linhas_saida, colunas_saida), dtype=np.uint8)

  # Realiza a convolução
  for i in range(linhas_saida):
    for j in range(colunas_saida):
      soma = 0
      for k in range(linhas_kernel):
        for l in range(colunas_kernel):
          soma += imagem[i + k, j + l] * kernel[k, l]
      imagem_saida[i, j] = int(soma)

  return imagem_saida


def filtro_gaussiano(tamanho, sigma):
  """
  Cria um kernel para o filtro gaussiano.

  Args:
    tamanho: O tamanho do kernel (ex: 3 para um kernel 3x3).
    sigma: O desvio padrão da distribuição gaussiana.

  Returns:
    O kernel para o filtro gaussiano.
  """
  x, y = np.meshgrid(np.linspace(-1, 1, tamanho), np.linspace(-1, 1, tamanho))
  calc = 1 / ((2 * np.pi * (sigma ** 2)))
  exp = np.exp(-(((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))))
  kernel = exp * calc
  kernel = kernel / np.sum(kernel)
  return kernel


def gauss_create(sigma=1, size_x=3, size_y=3):
    '''
    Create normal (gaussian) distribuiton
    '''
    x, y = np.meshgrid(np.linspace(-1,1,size_x), np.linspace(-1,1,size_y))
    calc = 1/((2*np.pi*(sigma**2)))
    exp = np.exp(-(((x**2) + (y**2))/(2*(sigma**2))))

    return exp*calc

def add_padding(img, padding_height, padding_width):
    n, m = img.shape

    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img

    return padded_img

def conv2d(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img

    #print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output

    # Perform convolution
    for i_img in range(img_height):  # Loop usando i
        for j_img in range(img_width):  # Loop usando j
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] = output[i_img, j_img] + (padded_img[i_img+i_kernel, j_img+j_kernel] * kernel[i_kernel, j_kernel])  # Atribui valor à variável output[i, j]
            output[i_img, j_img] = int(output[i_img, j_img])

    return np.array(output, dtype=np.uint8)


def ApplyGaussianFilter(image, sigma=5):
    kenel = gauss_create(sigma)
    gaussImage = conv2d(image, kenel)
    return cv2.filter2D(src=image, ddepth=-1, kernel=kenel, borderType=cv2.BORDER_CONSTANT)

def showGaussianResults(image, imageWithGaussianFilter, tittleSecondImage, tittleFirstImage="Imagem com Filtro Gaussiano"):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.title(tittleFirstImage)

    fig.add_subplot(1, 2, 2)
    plt.imshow(imageWithGaussianFilter, cmap='gray')
    plt.axis("off")
    plt.title(tittleSecondImage)

    plt.show()

def ApplyIdealPassLowFilter(shape, center, radius=35, lpType=0, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # ideal low-pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    elif lpType == 1: #Butterworth low-pass filter 
        lpFilter = 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == 2: # Gaussian low pass filter
        lpFilter = np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix

def ApplyGaussianPassLowFilter(image):
    kernel = filtro_gaussiano(3, 1)
    return convolucao(image, kernel)

def ConvertImage(image):
    image_f32 = np.float32(image)
    dft = cv2.dft(image_f32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift