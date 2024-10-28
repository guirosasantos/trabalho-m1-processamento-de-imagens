import numpy as np
import cv2

def dilate(img, kernel):
    kernel_height, kernel_length = kernel.shape
    margin_height = kernel_height // 2
    margin_length = kernel_length // 2
    
    height, length = img.shape
    dilated_img = np.zeros((height, length), dtype=np.uint8)
    
    for i in range(margin_height, height - margin_height):
        for j in range(margin_length, length - margin_length):
            neighbor = img[i - margin_height:i + margin_height + 1, j - margin_length:j + margin_length + 1]
            if np.any(neighbor & kernel):
                dilated_img[i, j] = 255
    
    return dilated_img

def erode(img, kernel):
    kernel_height, kernel_length = kernel.shape
    margin_height = kernel_height // 2
    margin_length = kernel_length // 2
    
    height, length = img.shape
    eroded_img = np.zeros((height, length), dtype=np.uint8)
    
    for i in range(margin_height, height - margin_height):
        for j in range(margin_length, length - margin_length):
            neighbor = img[i - margin_height:i + margin_height + 1, j - margin_length:j + margin_length + 1]
            if np.all(neighbor & kernel == kernel):
                eroded_img[i, j] = 255
    
    return eroded_img


def plot_eroded_dilated(img, dilate1, kernel_size=3):
    custom_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if dilate1 == True:
        return dilate(img, custom_kernel)
    else:
        return erode(img, custom_kernel)