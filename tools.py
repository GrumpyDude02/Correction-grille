import numpy as np
from openpyxl import Workbook
from statistics import mode
import cv2


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

def find_cells(contours,epsilon,areas, median):
    cells = []

    for i in range(len(contours)):
        approximate_points = cv2.approxPolyDP(contours[i], 0.1* cv2.arcLength(contours[i], True), True)
        if len(approximate_points)==4 and median*(1-epsilon)< areas[i] <median*(1+epsilon):
            cells.append(cv2.boundingRect(contours[i]))

    return sort_cells(cells)


def add_offset_bbox(row,offset:tuple[int,int]):
    new_row = []
    offset_x,offset_y=offset
    for cell in row:
        new_row.append((cell[0]+offset_x,cell[1]+offset_y,cell[2],cell[3]))
    return new_row

def compute_dynamic_threshold(checkmark_area, median_cell_area, base_threshold=0.3):
    """
    Computes a dynamic overlap threshold based on the checkmark's area.
    - If the checkmark is much larger than a typical cell, increase threshold.
    - If it's smaller, decrease threshold.
    """
    size_ratio = checkmark_area / median_cell_area
    return min(0.5, max(0.2, base_threshold * size_ratio))  # Clamp between 0.2 and 0.5


def assign_checkmarks_with_voting(bbox, rows, checked_cells):
    """Assigns checkmarks to cells using a voting system based on overlap."""
    
    x, y, w, h = bbox  # Bounding box of the detected checkmark
    checkmark_area = w * h  

    for row_index, row in enumerate(rows):
        for col_index, (cell_x, cell_y, cell_w, cell_h) in enumerate(row):
            # Compute intersection area
            x_overlap = max(0, min(x + w, cell_x + cell_w) - max(x, cell_x))
            y_overlap = max(0, min(y + h, cell_y + cell_h) - max(y, cell_y))
            intersection_area = x_overlap * y_overlap

            # Compute overlap percentage
            overlap_ratio = intersection_area / checkmark_area
            checked_cells[row_index][col_index] += overlap_ratio  # Accumulate votes
    return finalize_vote(checked_cells)

def finalize_vote(checked_cells,threshold = 0.2):
    
    for i,row in enumerate(checked_cells):
        for j,cell in enumerate(row):
            if cell < threshold:
                checked_cells[i][j]=0
    
    # for i,row in enumerate(checked_cells):
    #     m = max(row)
    #     for j,cell in enumerate(row):
    #         if m - cell > threshold:
    #             checked_cells[i][j] = 0
   

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


# def get_checked_cells(inv_binary_img, rows):
#     """Detects checkmarks and checks for multiple selections in a row."""
#     checked_cells = {}

#     # Find contours (potential checkmarks)
#     contours, _ = cv2.findContours(inv_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Ignore small noise
#         if cv2.contourArea(contour) < 50:  
#             continue

#         # Compute centroid of the checkmark
#         M = cv2.moments(contour)
#         if M["m00"] == 0:  # Avoid division by zero
#             continue
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])

#         # Assign checkmark to the correct row & column
#         for row_index, row in enumerate(rows, start=1):
#             for col_index, (x, y, w, h) in enumerate(row, start=1):
#                 if x <= cx <= x + w and y <= cy <= y + h:
#                     if row_index not in checked_cells:
#                         checked_cells[row_index] = []
#                     checked_cells[row_index].append(col_index)  # Store all checked columns in a list

#     return checked_cells

def merge_overlapping_bboxes(bboxes, threshold=30):
    """Merge bounding boxes that overlap or are close to each other."""
    merged_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        merged = False
        for idx, (mx, my, mw, mh) in enumerate(merged_bboxes):
            # Check if the bounding boxes overlap or are close enough to merge
            if abs(x - mx) < threshold and abs(y - my) < threshold:
                # Merge the boxes
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged_bboxes[idx] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_bboxes.append((x, y, w, h))
    return merged_bboxes


def get_checked_cells_with_spanning(inv_binary_img, rows, min_contour_area=50, threshold=10):
    """Detects checkmarks, merges overlapping checkmarks, and handles spanning checkmarks."""
    checked_cells = {}

    # Find contours (potential checkmarks)
    contours, _ = cv2.findContours(inv_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        # Ignore small noise
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))

    # Merge overlapping bounding boxes (if the checkmark spans multiple cells)
    merged_bboxes = merge_overlapping_bboxes(bboxes, threshold)

    for bbox in merged_bboxes:
        x, y, w, h = bbox
        # Check which rows and columns this merged box intersects
        for row_index, row in enumerate(rows, start=1):
            for col_index, (cell_x, cell_y, cell_w, cell_h) in enumerate(row, start=1):
                if cell_x <= x + w / 2 <= cell_x + cell_w and cell_y <= y + h / 2 <= cell_y + cell_h:
                    if row_index not in checked_cells:
                        checked_cells[row_index] = []
                    checked_cells[row_index].append(col_index)  # Store all checked columns in a list

    return checked_cells

def save_to_excel(rows, inv_binary_img, file_name="output.xlsx"):
    """Extracts checkmark data, warns about multiple selections, and saves to an Excel file."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Extracted Data"

    checked_cells = get_checked_cells_with_spanning(inv_binary_img, rows)

    for row_index in range(1, len(rows) + 1):
        checked_cols = checked_cells.get(row_index, [])

        if not checked_cols:
            sheet.cell(row=row_index, column=1).value = "Non cochée"  # No checkmark
        elif len(checked_cols) == 1:
            sheet.cell(row=row_index, column=checked_cols[0]).value = "cochée"  # Normal case
        else:
            # Multiple checkmarks detected: Warn user
            for col_index in checked_cols:
                sheet.cell(row=row_index, column=col_index).value = "cochée"
            sheet.cell(row=row_index, column=len(rows[0]) + 1).value = "⚠ Multiple cochées"

            print(f"Warning: Multiple checkmarks detected in row {row_index}")

    workbook.save(file_name)




def save_rows_coord_excel(rows, file_name="coord.xlsx"):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Extracted Data"
    for row_index, row in enumerate(rows, start=1):
        for col_index, cell in enumerate(row, start=1):
            sheet.cell(row=row_index, column=col_index).value = f"({cell[0]},{cell[1]})"
            #sheet.cell(row=row_index, column=col_index).value = f'{cell[0]},{cell[1]}'
    workbook.save(file_name)
    
    
def save_cell_imgs(img, rows,folder_path:str):
    for row_index,row in enumerate(rows,start=1):
        for col_index, cell in enumerate(row,start=1):
            x,y,w,h = cell
            roi = img[y:y+h,x:x+w]
            cv2.imwrite(f"{folder_path}/cell-{row_index}-{col_index}.png",roi)
            
def draw_bounding_boxes(img, rows):
    for row in rows:
        for cell in row:
            x, y, w, h = cell
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

