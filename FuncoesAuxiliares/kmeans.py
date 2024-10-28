import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def visualize_cluster_distribution(pixel_values, labels, centers):
    # Converte os centros dos clusters para valores RGB inteiros
    centers = np.uint8(centers)

    # Cria uma figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define os limites dos eixos
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    # Define os rótulos dos eixos
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    # Cria um array de cores para cada ponto baseado no cluster ao qual pertence
    colors = centers[labels]

    # Plotar os pixels no espaço RGB, coloridos de acordo com seu cluster
    ax.scatter(pixel_values[:, 0], pixel_values[:, 1], pixel_values[:, 2],
               c=colors / 255.0, s=1)

    plt.title('Distribuição dos Clusters no Espaço RGB')
    plt.show()

def visualize_all_clusters(image_rgb, labels, centers):
    num_clusters = len(centers)
    plt.figure(figsize=(15, 5))
    for i in range(num_clusters):
        mask = (labels == i)
        masked_image = np.copy(image_rgb)
        masked_image = masked_image.reshape((-1, 3))
        masked_image[~mask] = [0, 0, 0]  # Define outros clusters como preto
        masked_image = masked_image.reshape(image_rgb.shape)
        plt.subplot(1, num_clusters, i + 1)
        plt.imshow(masked_image)
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.show()

def KMeansImplementation(img, n_clusters=8, clusters_to_show=None):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values_float = np.float32(pixel_values)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values_float)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Print cluster centers
    print("Cluster Centers (RGB):")
    for idx, center in enumerate(centers):
        print(f"Cluster {idx}: {center}")

    # Visualize all clusters
    visualize_all_clusters(image_rgb, labels, centers)

    # Visualize cluster distribution in RGB space
    visualize_cluster_distribution(pixel_values, labels, centers)

    # Create segmented image
    centers_uint8 = np.uint8(centers)
    segmented_data = centers_uint8[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)

    # Display images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f'Segmented Image with {n_clusters} Clusters')
    plt.axis('off')

    # Mask for specific clusters (optional)
    if clusters_to_show is None:
        clusters_to_show = [0]  # Default cluster to show if none provided

    mask = np.isin(labels, clusters_to_show)
    masked_image = np.copy(image_rgb)
    masked_image = masked_image.reshape((-1, 3))
    masked_image[~mask] = [0, 0, 0]  # Set other clusters to black
    masked_image = masked_image.reshape(image_rgb.shape)

    plt.subplot(1, 3, 3)
    plt.imshow(masked_image)
    plt.title(f'Clusters {clusters_to_show} Isolated')
    plt.axis('off')

    plt.show()

    return segmented_image, masked_image
