import numpy as np
import cv2
from tqdm import tqdm

def Threshold(bgra, hsv, yCrCb):
    # RGB components
    b = float(bgra[0])
    g = float(bgra[1])
    r = float(bgra[2])
    a = float(bgra[3])

    # YCrCb components
    y = float(yCrCb[0])
    Cr = float(yCrCb[1])
    Cb = float(yCrCb[2])

    # RGB Thresholds
    E1 = r > 95
    E2 = g > 40
    E3 = b > 20
    E4 = r > g
    E5 = r > b
    E6 = abs(r - g) > 15
    E7 = a > 15

    RGB_cond = E1 and E2 and E3 and E4 and E5 and E6 and E7

    # HSV Thresholds
    H = hsv[0] * 2.0  # Convert H to degrees (0-360)
    S = hsv[1] / 255.0  # Normalize S to 0.0 - 1.0

    E8 = 0.0 <= H <= 50.0
    E9 = 0.23 <= S <= 0.68

    HSV_cond = E8 and E9

    # YCrCb Thresholds
    E10 = Cr > 135
    E11 = Cb > 85
    E12 = y > 80
    E13 = Cr <= (1.5862 * Cb) + 20
    E14 = Cr >= (0.3448 * Cb) + 76.2069
    E15 = Cr >= (-4.5652 * Cb) + 234.5652
    E16 = Cr <= (-1.15 * Cb) + 301.75
    E17 = Cr <= (-2.2857 * Cb) + 432.85

    YCrCb_cond = E10 and E11 and E12 and E13 and E14 and E15 and E16 and E17

    # Final Decision
    Condition1 = RGB_cond and HSV_cond
    Condition2 = RGB_cond and YCrCb_cond

    return Condition1 or Condition2

# https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
def SkinSegmentation(img, imgNameOut="out.png"):
    result = np.copy(img)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            if not Threshold(bgra[i, j], hsv[i, j], yCrCb[i, j]):
                result[i, j] = [0, 0, 0]  # Set pixel to black

    cv2.imwrite(imgNameOut, result)
