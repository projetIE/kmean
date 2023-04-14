# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:02:29 2023

@author: yrolland
"""
import cv2
import os
from PIL import Image
import numpy as np
import colorsys
import matplotlib.pyplot as plt
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img,center




def get_closest_color(color, colors):
    """
    Retourne la couleur la plus proche dans la liste de couleurs
    """
    hsv = colorsys.rgb_to_hsv(*color[:3])
    distances = []
    for c in colors:
        c_hsv = colorsys.rgb_to_hsv(*c[:3])
        distance = sum((a - b) ** 2 for a, b in zip(hsv, c_hsv))
        distances.append(distance)
    closest_index = np.argmin(distances)
    return colors[closest_index]

def colorize_image2(image, colors):     #can used the same color twice
    """
    Divise l'image en bandes horizontales et assigne à chaque bande la couleur la plus proche dans la liste de couleurs
    """

    # Conversion de l'image en tableau numpy
    pixels = np.array(image)

    # Division de l'image en bandes horizontales
    num_bands = len(colors)
    height, width, _ = pixels.shape
    band_height = height // num_bands
    bands = [pixels[i*band_height:(i+1)*band_height, :, :] for i in range(num_bands)]

    # Attribution des couleurs
    for i, band in enumerate(bands):
        band_color = get_closest_color(np.mean(band, axis=(0, 1)), colors)
        band_color = tuple([int(c) for c in band_color])
        bands[i][:,:,:] = band_color

    # Reconstruction de l'image
    pixels = np.concatenate(bands, axis=0)
    result_image = Image.fromarray(pixels)

    return result_image

    
    


def colorize_image(image, colors, orientation='horizontal'):
    """
    Divise l'image en bandes horizontales ou verticales et attribue à chaque bande une couleur unique de la liste de couleurs en fonction de la ressemblance.
    Chaque couleur de la liste colors ne sera utilisée qu'une seule fois.
    """

    # Conversion de l'image en tableau numpy
    pixels = np.array(image)

    # Division de l'image en bandes
    if orientation == 'horizontal':
        num_bands = len(colors)
        height, width, _ = pixels.shape
        band_height = height // num_bands
        bands = [pixels[i*band_height:(i+1)*band_height, :, :] for i in range(num_bands)]
    elif orientation == 'vertical':
        num_bands = len(colors)
        height, width, _ = pixels.shape
        band_width = width // num_bands
        bands = [pixels[:, i*band_width:(i+1)*band_width, :] for i in range(num_bands)]
    else:
        raise ValueError("L'orientation doit être 'horizontal' ou 'vertical'")

    # Attribution des couleurs
    for i, band in enumerate(bands):
        # Calculer la couleur moyenne de la bande
        band_color = np.mean(band, axis=(0,1)).astype(int)
        # Trouver la couleur la plus proche dans la liste de couleurs
        closest_color = get_closest_color(band_color, colors)
        # Assigner la couleur à tous les pixels de la bande
        bands[i][:,:,:] = closest_color
        # Supprimer la couleur utilisée de la liste de couleurs
        colors.remove(closest_color)

    # Reconstruction de l'image
    if orientation == 'horizontal':
        pixels = np.concatenate(bands, axis=0)
    else:
        pixels = np.concatenate(bands, axis=1)
    result_image = Image.fromarray(pixels)

    return result_image




# Charger la liste des images dans le dossier courant
picList = os.listdir('./')

pics =[]
# Filtrer les images pour ne garder que les fichiers JPG
# for pic in picList:
#     try:
#         if pic.split(".")[1] == "png":
#             pics.append(pic)
#     except:
#         print(pic,'is not a pic')
        
        
pics = ["3.jpg"]        
for png in pics:
        
    image = cv2.imread(png)
    image_data = np.array(image, dtype=np.float64) / 255

    plt.imshow(image_data)
    plt.axis('off')
    plt.show()
    
    
    
    
    for k in range(2,15):
        quantized_image,center = quantimage(image,k)
        plt.imshow(quantized_image)
        plt.axis('off')
        plt.show()
        cv2.imwrite("./Saved/"+png +str(k*3)+".jpg", quantized_image)
        # récupération des centroids des clusters
        centroids = center.tolist()
        a =colorize_image(quantized_image, centroids,orientation='vertical')
        a.save(f"./Saved/Weird{k*3}_colorized.jpg")
    
        plt.imshow(a)
        plt.axis('off')
    
        plt.show()
