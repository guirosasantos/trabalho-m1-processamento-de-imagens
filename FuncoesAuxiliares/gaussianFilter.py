from matplotlib import pyplot as plt
import numpy as np

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



def ApplyGaussianFilter(image, sigma=1):
    kenel = filtro_gaussiano(3, sigma)
    return convolucao(image, kenel)

def showGaussianResults(image, imageWithGaussianFilter):
    fig = plt.figure(dpi=300)

    fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.title("Imagem Original")

    fig.add_subplot(1, 2, 2)
    plt.imshow(imageWithGaussianFilter, cmap='gray')
    plt.axis("off")
    plt.title("Imagem com Filtro Gaussiano")

    plt.show()