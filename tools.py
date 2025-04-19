import numpy as np
from openpyxl import Workbook
from statistics import mode
import cv2

def detecter_lignes_hough(image:np.array,
                          rho=1, theta=np.pi/180, seuil=100,
                          longueur_min=175, ecart_max=20, epaisseur=2):
    """
    Détecte et filtre les lignes d'une grille (horizontales ou verticales) avec HoughLinesP.
    
    Args:
        masque_lignes (np.array): Masque binaire où les lignes seront dessinées
        orientation (str): 'horizontal' ou 'vertical'
        rho (float): Résolution de distance pour l'accumulateur
        theta (float): Résolution angulaire (radians)
        seuil (int): Seuil de l'accumulateur
        longueur_min (int): Longueur minimale des lignes (pixels)
        ecart_max (int): Écart maximal entre segments de ligne
        tolerance_angle (int): Déviation angulaire autorisée (degrés)
        epaisseur (int): Épaisseur des lignes dessinées
        
    Returns:
        np.array: Masque avec les lignes détectées
    """
    
    # Détection des lignes avec Hough
    masque_lignes = np.zeros_like(image, dtype=image.dtype)
    lignes = cv2.HoughLinesP(
        image,
        rho=rho,
        theta=theta,
        threshold=seuil,
        minLineLength=longueur_min,
        maxLineGap=ecart_max
    )
    
    if lignes is not None:
        for ligne in lignes:
            x1, y1, x2, y2 = ligne[0]
            cv2.line(masque_lignes, (x1, y1), (x2, y2), 255, epaisseur)
    
    return masque_lignes

def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle


def decode_qr(raw_text:str):
    if not raw_text:
        return None
    dic={}
    t = raw_text.split(";")
    for text in t:
        temp = text.split(":")
        try:
            dic[temp[0].strip()]=int(temp[1].strip())
        except (ValueError):
            dic[temp[0].strip()]=temp[1].strip()
    return dic


def add_offset_bbox(row,offset:tuple[int,int]):
    new_row = []
    offset_x,offset_y=offset
    for cell in row:
        new_row.append((cell[0]+offset_x,cell[1]+offset_y,cell[2],cell[3]))
    return new_row

def sort_cells(bounding_boxes):
    rows = []
    heights = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(h)
        
    avg_height = np.average(heights)
    half_avg = avg_height / 2
    if not bounding_boxes:
        return []
    
    curr_row = [bounding_boxes[0]]
        
    for cell in bounding_boxes[1:]:
        curr_cell_y = cell[1]
        prev_cell_y = curr_row[-1][1]
        same_row = abs(curr_cell_y-prev_cell_y)<=half_avg
        if same_row:
            curr_row.append(cell)
        else:
            rows.append(curr_row)
            curr_row = [cell]
        
    rows.append(curr_row)
    
    for row in rows:
        row.sort(key=lambda x: x[0])
    
    rows.reverse()
    return rows
