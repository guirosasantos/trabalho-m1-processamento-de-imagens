import cv2
import numpy as np
from FuncoesAuxiliares import skinDetection
from matplotlib import pyplot as plt

def Seeding(image_path, image_out_path, num_superpixels=5, num_levels=2, 
            prior=2, num_histogram_bins=5, num_iterations=2000, ):
    """
    Performs superpixel segmentation using the SEEDS algorithm and highlights skin regions
    using the SkinSegmentation algorithm for consistent skin detection.

    Parameters:
    - image_path (str): Path to the input image.
    - image_out_path (str): Path to the output image.
    - num_superpixels (int): Approximate number of superpixels.
    - num_levels (int): Number of pyramid levels for SEEDS.
    - prior (int): Prior for the SEEDS algorithm.
    - num_histogram_bins (int): Number of histogram bins.
    - num_iterations (int): Number of iterations for the algorithm.

    Displays:
    - Original image.
    - Image with superpixel contours.
    - Segmented image highlighting skin regions.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading the image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (height, width) = image_rgb.shape[:2]

    skin_mask = skinDetection.SkinSegmentation(image)

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        width, height, image_rgb.shape[2], num_superpixels, 
        num_levels, prior, num_histogram_bins
    )

    seeds.iterate(image_rgb, num_iterations)

    labels = seeds.getLabels()
    mask = seeds.getLabelContourMask(False)

    image_superpixels = image_rgb.copy()
    image_superpixels[mask == 255] = [0, 0, 0]

    num_superpixels_actual = seeds.getNumberOfSuperpixels()
    skin_superpixels = np.zeros(num_superpixels_actual, dtype=bool)

    for label in range(num_superpixels_actual):
        superpixel_mask = labels == label
        num_skin_pixels = np.sum(skin_mask[superpixel_mask])
        num_pixels = np.sum(superpixel_mask)
        if num_pixels > 0 and (num_skin_pixels / num_pixels) > 0.5:
            skin_superpixels[label] = True

    segmented_image = image_rgb.copy()
    for label in range(num_superpixels_actual):
        mask_i = labels == label
        if skin_superpixels[label]:
            segmented_image[mask_i] = [255, 0, 0]
        else:
            segmented_image[mask_i] = image_rgb[mask_i] // 2

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_superpixels)
    plt.title('Superpixel Contours')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image)
    plt.title('Segmented Superpixels')
    plt.axis('off')

    plt.savefig(image_out_path)
    plt.show()
