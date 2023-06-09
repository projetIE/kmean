import cv2
import numpy as np
import os

# Fonction pour créer les bandes de couleurs
def create_bande(image, centroids, liste):
    # Créer une image de bandes de couleurs horizontales
    palette_bands = np.zeros((len(liste)*50 , 100, 3), dtype=np.uint8)
    for i, k in enumerate(liste):
        palette_bands[i*50:(i+1)*50, :] = np.uint8(centroids[k] * 255)
    # Redimensionner l'image pour qu'elle ait la même taille que l'image originale
        palette_bands = cv2.resize(palette_bands, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return palette_bands
        
        
# Charger l'image
image = cv2.imread("./3.jpg")

# Convertir l'image en matrice numpy
image_data = np.array(image, dtype=np.float64) / 255

# Spécifier le nombre de clusters (K) pour l'algorithme K-means
K = 5

# Initialiser les centroides aléatoirement
centroids = np.random.rand(K, 3)

# Répéter jusqu'à convergence
for i in range(10):
    # Calculer la distance euclidienne entre chaque pixel et les centroides
    distances = np.sqrt(np.sum((image_data[:, :, np.newaxis] - centroids) ** 2, axis=3))

    # Assigner chaque pixel au cluster le plus proche
    labels = np.argmin(distances, axis=2)

    # Mettre à jour les centroides en calculant la moyenne de chaque cluster
    for k in range(K):
        centroids[k] = np.mean(image_data[labels == k], axis=0)

# Convertir les labels en une image segmentée
segmented_image = np.zeros_like(image_data)
for k in range(K):
    segmented_image[labels == k] = centroids[k]

# Afficher les palettes d'origine
palette_bands = create_bande(image, centroids, range(K))

cv2.imshow("Palettes", palette_bands)

# Afficher l'image originale et l'image segmentée
cv2.imshow("Image originale", image)
cv2.imshow("Image segmentée", segmented_image)

# Enregistrer les images
if not os.path.exists("Saved"):
    # Créer le dossier s'il n'existe pas
    os.mkdir("Saved")

cv2.imwrite('./Saved/palette_bands1.jpg', palette_bands)
cv2.imwrite('./Saved/segmented_image.jpg', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()