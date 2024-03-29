### Copier coller du TP02 ###
import cv2
import os 

def load_detections(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            frame = int(fields[0])
            obj_id = int(fields[1])
            bb_left = float(fields[2])
            bb_top = float(fields[3])
            bb_width = float(fields[4])
            bb_height = float(fields[5])
            conf = float(fields[6])
            x = float(fields[7])
            y = float(fields[8])
            z = float(fields[9])
            detections.append({
                'frame': frame,
                'id': obj_id,
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': conf,
                'x': x,
                'y': y,
                'z': z
            })
    return detections


def compute_iou(box1, box2):
    """
    Compute intersection over union (IoU) between two bounding boxes.
    :param box1: [left, top, width, height] of box 1
    :param box2: [left, top, width, height] of box 2
    :return: IoU score
    """
    # Convert bounding boxes to format [x1, y1, x2, y2]
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def create_similarity_matrix(detections):
    """
    Create a similarity matrix that stores the IoU for all bounding boxes.
    :param detections: List of dictionaries containing detection information
    :return: Similarity matrix
    """
    num_detections = len(detections)
    similarity_matrix = [[0.0] * num_detections for _ in range(num_detections)]

    for i in range(num_detections):
        for j in range(num_detections):
            if i != j:
                box1 = [detections[i]['bb_left'], detections[i]['bb_top'], detections[i]['bb_width'], detections[i]['bb_height']]
                box2 = [detections[j]['bb_left'], detections[j]['bb_top'], detections[j]['bb_width'], detections[j]['bb_height']]
                iou = compute_iou(box1, box2)
                similarity_matrix[i][j] = iou

    return similarity_matrix

def associate_detections_to_tracks(detections, tracks, sigma_iou):
    """
    Associate detections to tracks in a greedy manner using IoU and a threshold sigma_iou.
    :param detections: List of dictionaries containing detection information
    :param tracks: List of dictionaries containing track information
    :param sigma_iou: Threshold for IoU
    :return: Updated tracks with associated detections
    """
    for track in tracks:
        last_detection = track['detections'][-1]
        best_iou = -1
        best_detection = None

        for detection in detections:
            iou = compute_iou([last_detection['bb_left'], last_detection['bb_top'], last_detection['bb_width'], last_detection['bb_height']],
                              [detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']])
            if iou > best_iou and iou > sigma_iou:
                best_iou = iou
                best_detection = detection

        if best_detection is not None:
            track['detections'].append(best_detection)
            detections.remove(best_detection)

    return tracks

def track_management(tracks, detections, sigma_iou):
    """
    Perform track management based on IoU threshold sigma_iou.
    :param tracks: List of dictionaries containing track information
    :param detections: List of dictionaries containing detection information
    :param sigma_iou: Threshold for IoU
    :return: Updated list of tracks
    """
    updated_tracks = []

    # First, associate detections to existing tracks
    for track in tracks:
        last_detection = track['detections'][-1]
        best_iou = -1
        best_detection = None

        for detection in detections:
            iou = compute_iou([last_detection['bb_left'], last_detection['bb_top'], last_detection['bb_width'], last_detection['bb_height']],
                              [detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']])
            if iou > best_iou and iou > sigma_iou:
                best_iou = iou
                best_detection = detection

        if best_detection is not None:
            track['detections'].append(best_detection)
            updated_tracks.append(track)
            detections.remove(best_detection)

    # Then, create new tracks for unmatched detections
    for detection in detections:
        new_track = {'id': len(tracks) + len(updated_tracks), 'detections': [detection]}
        updated_tracks.append(new_track)

    return updated_tracks


def draw_tracking_results(image_dir, tracks, iou_threshold=0.5):
    """
    Draw bounding boxes, IDs, and trajectories on images to visualize tracking results.
    :param image_dir: Directory containing images
    :param tracks: List of dictionaries containing track information
    :param iou_threshold: Threshold for IoU to consider detections as the same object
    """
    image_files = sorted(os.listdir(image_dir))
    
    # Dictionnaire pour stocker les identifiants uniques des objets détectés
    unique_ids = {}
    unique_id_counter = 0
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        # Dictionnaire pour stocker les correspondances entre les identifiants de pistes et les identifiants uniques des objets détectés
        id_mapping = {}
        
        for track in tracks:
            for detection in track['detections']:
                if detection['frame'] == idx:
                    # Calculer l'IoU avec les détections précédentes
                    detection_bbox = [detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']]
                    
                    same_object_found = False
                    for obj_id, bbox in unique_ids.items():
                        iou = compute_iou(bbox, detection_bbox)
                        if iou >= iou_threshold:
                            id_mapping[detection['id']] = obj_id
                            same_object_found = True
                            break
                    
                    # Si une correspondance est trouvée, utiliser l'identifiant unique existant
                    if same_object_found:
                        obj_id = id_mapping[detection['id']]
                    else:
                        # Créer un nouvel identifiant unique
                        obj_id = unique_id_counter
                        unique_ids[obj_id] = detection_bbox
                        unique_id_counter += 1
                    
                    # Draw bounding box
                    bb_left = int(detection['bb_left'])
                    bb_top = int(detection['bb_top'])
                    bb_width = int(detection['bb_width'])
                    bb_height = int(detection['bb_height'])
                    cv2.rectangle(image, (bb_left, bb_top), (bb_left + bb_width, bb_top + bb_height), (0, 255, 0), 2)
                    
                    # Draw ID
                    cv2.putText(image, str(obj_id), (bb_left, bb_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow('Tracking Results', image)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Attendre 100 ms entre chaque image et vérifier si 'q' est pressé pour quitter
            break
    
    cv2.destroyAllWindows()

### Fin du coppier coller du TP02 ###
    
### Section TP03 ###
from scipy.optimize import linear_sum_assignment
import numpy as np

def track_management_with_hungarian(tracks, detections, similarity_matrix):
    """
    Perform track management using Hungarian algorithm to find the optimal assignment.
    :param tracks: List of dictionaries containing track information
    :param detections: List of dictionaries containing detection information
    :param similarity_matrix: Similarity matrix containing IoU values for all bounding boxes
    :return: Updated list of tracks
    """
    num_tracks = len(tracks)
    num_detections = len(detections)

    if num_tracks == 0 or num_detections == 0:
        return tracks

    # Convertir la matrice de similarité en une matrice de coût en la multipliant par -1
    cost_matrix = -np.array(similarity_matrix)

    # Utiliser l'algorithme hongrois pour trouver l'assignation optimale entre les pistes et les détections
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Créer un dictionnaire pour stocker les associations entre les identifiants de piste et de détection
    assignment_dict = {}
    for row_idx, col_idx in zip(row_indices, col_indices):
        assignment_dict[col_idx] = row_idx

    updated_tracks = []

    # Mettre à jour les pistes avec les détections associées
    for track in tracks:
        if track['id'] in assignment_dict:
            detection_idx = assignment_dict[track['id']]
            track['detections'].append(detections[detection_idx])
            updated_tracks.append(track)

    # Créer de nouvelles pistes pour les détections non associées
    for col_idx in range(num_detections):
        if col_idx not in assignment_dict.values():
            new_track = {'id': num_tracks + col_idx, 'detections': [detections[col_idx]]}
            updated_tracks.append(new_track)

    return updated_tracks

import csv

def save_tracking_results(tracks, sequence_name):
    """
    Save tracking results in a txt file.
    :param tracks: List of dictionaries containing track information
    :param sequence_name: Name of the sequence
    """
    output_file = f"{sequence_name}.txt"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        
        for track in tracks:
            for detection in track['detections']:
                frame = detection['frame']
                obj_id = track['id']
                bb_left = detection['bb_left']
                bb_top = detection['bb_top']
                bb_width = detection['bb_width']
                bb_height = detection['bb_height']
                conf = 1  # Flag as 1
                x = detection['x']
                y = detection['y']
                z = detection['z']
                
                writer.writerow([frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z])

### Fin de section TP03 ### 