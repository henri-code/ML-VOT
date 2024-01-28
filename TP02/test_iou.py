import iou
import os
import cv2

# Usage example chatGPT :
print("Start test")
file_path = 'ADL-Rundle-6/det/det.txt'
print(f"Load the file {file_path}")
loaded_detections = iou.load_detections(file_path)
print("load finish")
# print(f"{loaded_detections[0]}")

print("Start similarity creation matrix")
# Créer la matrice de similarité
# similarity_matrix = iou.create_similarity_matrix(loaded_detections)
# print("creation similarity matrix finish")
# print(similarity_matrix[0])

# Initialiser les pistes (tracks)
print("Start creat tracks")
tracks = [{'id': i, 'detections': [loaded_detections[i]]} for i in range(len(loaded_detections))]
print(tracks[0])
print("finish create tracks")

# Définir le chemin vers le dossier contenant les images
print("Start load all pictures")
images_folder_path = 'ADL-Rundle-6\img1'  
# Lire les images du dossier
image_files = sorted([os.path.join(images_folder_path, file) for file in os.listdir(images_folder_path)])
print("finish load all picture")

# Définir le seuil sigma_iou
print("Start associate detection to track")
sigma_iou = 0.5  # Vous pouvez ajuster cette valeur selon vos besoins
# Associer les détections aux pistes
# updated_tracks = iou.associate_detections_to_tracks(loaded_detections, tracks, sigma_iou)
# print(updated_tracks[0])
print("Finish associate detection to track")

print("Start track management")
updated_tracks = iou.track_management(tracks, loaded_detections, sigma_iou)
print(updated_tracks[0])
print("Finish track_management")

print("Start Drawing")
# Chemin du dossier contenant les images
image_folder = 'ADL-Rundle-6/img1'

# Afficher les résultats du suivi
iou.draw_tracking_results(image_folder, updated_tracks)
print("Finish Drawing")

