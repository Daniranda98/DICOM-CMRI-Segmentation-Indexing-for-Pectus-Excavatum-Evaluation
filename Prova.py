import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from pathlib import Path
from sklearn.cluster import KMeans
from skimage import measure
import matplotlib.pyplot as plt
from scipy import ndimage

def K_means(image,centroids,iterations=10):
    n_pixels = image.shape[0] * image.shape[1]
    pixels = np.reshape(image, (n_pixels, 1))

    # Applica k-means con 2 cluster
    n_clusters = centroids
    n_init = iterations  # Puoi scegliere il numero di inizializzazioni che preferisci
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_init)
    kmeans.fit(pixels)

    # Prendi i centroidi dei cluster
    centroids = kmeans.cluster_centers_

    # Ordina i centroidi in ordine crescente
    sorted_centroids = np.sort(centroids, axis=0)
    return sorted_centroids

def Bwareafilt_N_Range(Image,min_area,max_area):
    labels = measure.label(Image, connectivity=1)

    # Calcola l'area di ciascun oggetto
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]

    # Filtra gli oggetti in base all'area desiderata
    filtered_labels = np.zeros_like(labels)
    for label in np.unique(labels):
        if min_area <= areas[label - 1] <= max_area:
            filtered_labels[labels == label] = label

    # Ottiene l'immagine finale mantenendo solo gli oggetti filtrati
    Image_new = (filtered_labels > 0).astype(np.uint8)
    return Image_new



immagine = MatToUintcv2.imread("Figure_1.png")

im_heart_mod = cv2.medianBlur(immagine, 5)
im_heart_mod = cv2.GaussianBlur(im_heart_mod,(3,3),0)

centroids = K_means(im_heart_mod, 3)
th1 = np.mean([centroids[0], centroids[1]])
th2 = np.mean([centroids[1], centroids[2]])

edges = cv2.Canny(im_heart_mod, threshold1=th1, threshold2=th2)
edges = Bwareafilt_N_Range(edges,4,float('inf'))

kernel = np.ones((2,2)) # Puoi regolare le dimensioni del kernel
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
aorta_mask = np.zeros_like(edges)
lowest_y = -float('inf')
for contour in contours:
    image = np.zeros_like(edges)
    cv2.drawContours(image, contour, -1, (255, 255, 255), thickness=cv2.FILLED)
    image = ndimage.binary_fill_holes(image)
    coordinate_bianche = np.argwhere(image != 0)
    centroide_y = np.mean(coordinate_bianche[:, 0])
    if centroide_y > lowest_y:
        lowest_y = centroide_y
        aorta_mask = image.copy()


plt.figure()
plt.imshow(aorta_mask,'gray')
plt.show()
