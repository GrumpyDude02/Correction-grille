import numpy as np
from openpyxl import Workbook
from statistics import mode
import cv2


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
