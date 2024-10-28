import cv2
import numpy as np
import morfologia as mf

# Carregar a imagem
imagem = cv2.imread("dedao.jpg")

def aplicar_ruido_gaussiano(imagem, mean = 0, std_dev = 25):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    ruido = np.random.normal(mean, std_dev, imagem_cinza.shape).astype(np.uint8)
    return cv2.add(imagem_cinza, ruido)

def aplicar_ruido_sal_pimenta(imagem, prob = 0.05):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ruidosa = np.copy(imagem_cinza)
    num_salt = np.ceil(prob * imagem_cinza.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in imagem_cinza.shape]
    imagem_ruidosa[coords[0], coords[1]] = 255
    num_pepper = np.ceil(prob * imagem_cinza.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in imagem_cinza.shape]
    imagem_ruidosa[coords[0], coords[1]] = 0
    return imagem_ruidosa

def limiarizacao_otsu(imagem):
    # Aplica a limiarização de Otsu
    T, imagem_otsu = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Valor do limiar de Otsu:", T)
    return imagem_otsu

# Adicionar o ruído à imagem original
imagem_ruidosa = aplicar_ruido_gaussiano(imagem)
(T, thresh) = cv2.threshold(imagem_ruidosa, 156, 255, cv2.THRESH_BINARY)
imagem_otsu = limiarizacao_otsu(imagem_ruidosa)

thresh = mf.plot_eroded_dilated(thresh, False)
imagem_otsu = mf.plot_eroded_dilated(imagem_otsu, False)

# Exibir a imagem com ruído
cv2.imshow("Imagem com Ruído Gaussiano", imagem_ruidosa)
cv2.imshow("Limiarização Binária", thresh)
cv2.imshow("Limiarização com Otsu", imagem_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
