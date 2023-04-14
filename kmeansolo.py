import cv2
import numpy as np
import os

meanpp = True
K = 5
nocluser = 0
centroids = 0

def createbande(image,centroids,K,angle):
    if angle == "horizontal":
        # Afficher des bandes de couleurs horizontales
        palette_bands = np.zeros((K * 50, 100, 3), dtype=np.uint8)
        for k in range(K):
            palette_bands[k*50:(k+1)*50, :] = np.uint8(centroids[k] * 255)
            
            # Redimensionner l'image palette_bands pour qu'elle ait la même taille que l'image originale

        palette_bands = cv2.resize(palette_bands, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


    elif angle == "vertical":
        
        # Afficher des bandes de couleurs verticales
        palette_bands = np.zeros(( 100 ,K * 50, 3), dtype=np.uint8)
        for k in range(K):
            palette_bands[:, k*50:(k+1)*50] = np.uint8(centroids[k] * 255)

        # Redimensionner l'image palette_bands pour qu'elle ait la même taille que l'image originale

        #palette_bands = cv2.resize(palette_bands, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Afficher des bandes de couleurs verticales
        palette_bands = np.zeros((100,K * 50, 3), dtype=np.uint8)
        for k in range(K):
            palette_bands[k*50:(k+1)*50, :] = np.uint8(centroids[k] * 255)
        
    # Redimensionner l'image palette_bands pour qu'elle ait la même taille que l'image originale
    cv2.imshow("Bandes de couleurssss", palette_bands)
    cv2.waitKey(0)
    return palette_bands

#get liste of pictures in the folder
picList = os.listdir('./')

# pics =[]
# #sort them to get all the png
# for pic in picList:
#     if pic == "Saved" : continue
#     if pic.split(".")[1] == "jpg":
#         pics.append(pic)

# Charger l'image
image = cv2.imread("./1.jpg")

# Convertir l'image en matrice numpy
image_data = np.array(image, dtype=np.float64) / 255

# Spécifier le nombre de clusters (K) pour l'algorithme K-means


while nocluser == 0:
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
            
        
        
    if np.isnan(centroids).any():
        print("There are NaN values in the array.")
        
        nocluser = 0
    else:
        print("There are no NaN values in the array")
        
        # Initialiser les centroides aléatoirement   meanpp =false
        nocluser = 1





# Convertir les labels en une image segmentée
segmented_image = np.zeros_like(image_data)
for k in range(K):
    segmented_image[labels == k] = centroids[k]





    
    

palette_bands = createbande(image, centroids, K, "horizontal")

# Afficher l'image originale
cv2.imshow("Image originale", image)

# Afficher l'image segmentée
cv2.imshow("Image segmentée", segmented_image)


# Afficher l'image palette
cv2.imshow("Bandes de couleurs", palette_bands)



if not os.path.exists("Saved"):
    # Créer le dossier s'il n'existe pas
    os.mkdir("Saved")

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./Saved/Test_gray.jpg', palette_bands)
cv2.imwrite('./Saved/Test_gray.jpg', segmented_image)



