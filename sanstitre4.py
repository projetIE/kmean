import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
img = cv2.imread('./1.jpg')

# Convertir l'image en une matrice de pixels
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Définir les paramètres de l'algorithme k-means
k = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

# Exécuter l'algorithme k-means
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, flags)

# Convertir les centres en entiers
centers = np.uint8(centers)

# Associer chaque pixel à un centre de couleur
segmented_data = centers[labels.flatten()]

# Reshaper les données segmentées pour les convertir en image
segmented_image = segmented_data.reshape(img.shape)

# Afficher la palette de couleurs
colors, counts = np.unique(segmented_data, axis=0, return_counts=True)
proportions = counts / len(labels.flatten())
palette = np.hstack([colors, np.expand_dims(proportions, axis=1)])
palette = palette[palette[:, 1].argsort()[::-1]]
palette = palette[:, :3]
palette = np.uint8(palette)

plt.imshow(palette.reshape((50, -1, 3)))
plt.axis('off')
plt.show()
