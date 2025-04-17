import cv2, numpy as np
import imutils
import tools
import constants as cst
from enum import Enum


class GridType(Enum):
    Unknown = "Unknown"
    PFE_Finale = "Grille Projet Fin d'Etude - Finale"
    PFE_Inter = "Grille Projet Fin d'Etude - Intermediaire"
    PFA = "Grille Projet Fin d'Année"


class Grid:

    k_size = 7
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
    square_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    square_kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tolerance = 0.45
    save = True
    threshold = 0.1

    def __init__(self, cv_image: np.ndarray):

        # to change, absolutely
        self.type = GridType.Unknown
        self.collisions_per_checkmark_per_row={}
        self.checkmark_bboxes = []
        self.expected_row_cols = (0, 0)
        self.detect_type_n_rotation(cv_image)
        self.drawn_og_img = self.original_matrix.copy()
        self.middle_x = self.original_matrix.shape[1] // 2
        self.gray_img = None
        self.binary_img = None
        self.inverted_img = None
        self.combined_lines = None
        self.no_lines = None
        self.sorted_cells = None
        self.cells_state = None
        self.warnings = None
        
    def detect_type_n_rotation(self,original_img):
        qcd = cv2.QRCodeDetector()
        decoded_info, points, _ = qcd.detectAndDecode(original_img)
        qr_code_data = tools.decode_qr(decoded_info)
        
        if qr_code_data:
            rotation = tools.calculate_angle(points[0][0],points[0][1])
            self.original_matrix = imutils.rotate_bound(original_img, -rotation)
            match qr_code_data["Type"]:
                case "PFE-F":
                    self.type = GridType.PFE_Finale
                case "PFE-Inter":
                    self.type = GridType.PFE_Inter
                case "PFA":
                    self.type = GridType.PFA
            self.expected_row_cols = (qr_code_data["Lines"], qr_code_data["Cols"])
        elif original_img.shape[1] > original_img.shape[0]:
            self.original_matrix = imutils.rotate_bound(original_img, 90)
        else:
            self.original_matrix = original_img

    def _preprocess(self):
        self.gray_img = cv2.cvtColor(self.original_matrix, cv2.COLOR_BGR2GRAY)
        self.binary_img = cv2.adaptiveThreshold(
            self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
        
        # cv2.ADAPTIVE_GAUSSIAN_C : Méthode utilisée pour calculer le seuil localement (Moyenne pondérée selon une distribution gaussienne)
        # 51: Taille du voisinage (fenêtre) 51
        # 5: Valeur constante a soustraire au seuil calculé
        
        self.inverted_img = cv2.bitwise_not(self.binary_img.copy())

    def _isolate_lines(self):
        
        # element strurcuturant horizontal === horizontal_kernel = [1,1,1,1,1,1,1] 
        
        # 8 iterations d'erosion
        self.horizontal_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.horizontal_kernel, iterations=8
        )
        
        # 14 iteration de dilatation
        self.horizontal_lines = cv2.morphologyEx(
            self.horizontal_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=14,
        )

        
        # element strurcuturant vertical === vertical_kernel = [[1],
            #                                                   [1],
            #                                                   [1],
            #                                                   [1],
            #                                                   [1],
            #                                                   [1],
            #                                                   [1]] 
            
        # 8 iterations d'erosion
        self.vertical_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.vertical_kernel, iterations=8
        )
        
        # 8 iterations de dilatation
        self.vertical_lines = cv2.morphologyEx(
            self.vertical_lines, cv2.MORPH_DILATE, Grid.vertical_kernel, iterations=8
        )

        self.combined_lines = cv2.bitwise_or(self.horizontal_lines, self.vertical_lines)

    def _extract_cells(self):
        rhs_combined_lines = self.combined_lines[:, self.middle_x :]
        outer_contours, _ = cv2.findContours(
            rhs_combined_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = []
        for i in range(len(outer_contours)):
            approximate_points = cv2.approxPolyDP(
                outer_contours[i], 0.02 * cv2.arcLength(outer_contours[i], True), True
            )
            if len(approximate_points) == 4:
                rects.append(outer_contours[i])

        self.bbox_biggest_rect = cv2.boundingRect(
            max(rects, key=lambda x: cv2.contourArea(x))
        )

        x, y, w, h = self.bbox_biggest_rect
        self.cropped_combined_lines = rhs_combined_lines[y : y + h, x : x + w]

        self.cropped_combined_lines = cv2.ximgproc.thinning(self.cropped_combined_lines)

        inner_contours, _ = cv2.findContours(
            self.cropped_combined_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )



        bboxes = []

        areas = [cv2.contourArea(contour) for contour in inner_contours]
        median = np.median(areas)

        for i in range(len(inner_contours)):
            approximate_points = cv2.approxPolyDP(
                inner_contours[i], 0.1 * cv2.arcLength(inner_contours[i], True), True
            )
            if len(approximate_points) == 4 and median * (1 - Grid.tolerance) < areas[
                i
            ] < median * (1 + Grid.tolerance):
                bboxes.append(cv2.boundingRect(inner_contours[i]))

        self.sorted_cells = tools.sort_cells(bboxes)
        self.cells_state = [
            [[0, 1] for j in range(len(self.sorted_cells[i]))]
            for i in range(len(self.sorted_cells))
        ]

    def _isolate_checkmarks(self):
        x, y, w, h = self.sorted_cells[0][0]
        x1, y1, w1, h1 = self.bbox_biggest_rect

        # Suppression des lignes verticales
        self.no_lines = cv2.absdiff(self.inverted_img, self.vertical_lines)
        
        

        # Extraction de la partie droite de l'image
        right_no_lines = self.no_lines[:, self.middle_x :]
        self.cropped_no_lines = right_no_lines[y1 : y1 + h1, x1 : x1 + w1][y:, x:]

        # Suppression du bruit avec une ouverture morphologique
        self.cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )

        # Dilatation pour reconnecter les éléments disjoints
        
        
        dilated_cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=3,
        )
        
        cv2.imwrite("temp/1.png",dilated_cropped_no_lines)
        
        
        # Suppression des lignes résiduelles
        mask_horizontal = self.horizontal_lines[:, self.middle_x :][
            y1 : y1 + h1, x1 : x1 + w1
        ][y:, x:]

        dilated_cropped_no_lines = cv2.absdiff(
            dilated_cropped_no_lines, mask_horizontal
        )

        # Fermeture puis ouverture morphologique pour éliminer le bruit
        dilated_cropped_no_lines = cv2.morphologyEx(
            dilated_cropped_no_lines, cv2.MORPH_CLOSE, Grid.square_kernel5, iterations=1
        )
        dilated_cropped_no_lines = cv2.morphologyEx(
            dilated_cropped_no_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        
        
         # Filtrage supplémentaire pour améliorer l'extraction des checkmarks
        dilated_cropped_no_lines = cv2.bitwise_and(
            dilated_cropped_no_lines,
            self.inverted_img[:, self.middle_x :][y1 : y1 + h1, x1 : x1 + w1][y:, x:],
        )

        
        

        # Mise à jour de l'attribut final
        self.cropped_no_lines = dilated_cropped_no_lines

        checkmark_contours, _ = cv2.findContours(
            dilated_cropped_no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in checkmark_contours:
            bbox = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 175:
                self.checkmark_bboxes.append(bbox)

        
        offset_cells = [tools.add_offset_bbox(row,(-x,-y)) for row in self.sorted_cells]
        self.collisions_per_checkmark_per_row = self._get_occupied_cells_per_row(offset_cells)


    def change_cell_color(self,row,col,color_code):
        self.cells_state[row][col][1] = color_code
        self._draw_cells_bboxs()
    
    def _draw_cells_bboxs(self):
        xcell, ycell, _, _ = self.sorted_cells[0][0]
        x, y, w, h = self.bbox_biggest_rect
        self.drawn_og_img = self.original_matrix.copy()
        for bbox in self.checkmark_bboxes:
            xb, yb, wb, hb = bbox
            cv2.rectangle(self.cropped_no_lines, (xb, yb), (xb + wb, yb + hb), 255, 2)
            cv2.rectangle(
                self.drawn_og_img,
                (xb + x + self.middle_x + xcell, ycell + y + yb),
                (xb + x + self.middle_x + xcell + wb, ycell + y + yb + hb),
                (0, 0, 255),
                2,
            )

        for i, row in enumerate(self.sorted_cells):
            for j, cell in enumerate(row):
                xcell, ycell, wcell, hcell = cell
                cv2.rectangle(
                    self.drawn_og_img,
                    (xcell + self.middle_x + 10, y + ycell + 10),
                    (xcell + self.middle_x + wcell - 10, y + ycell + hcell - 10),
                    cst.COLORS[self.cells_state[i][j][1]],
                    2,
                )

    def save_imgs(self, folder):
        if not Grid.save:
            return
        try:
            cv2.imwrite(f"{folder}/roi.png", self.drawn_og_img)
            cv2.imwrite(f"{folder}/og.png", self.original_matrix)
            cv2.imwrite(f"{folder}/inverted.png", self.inverted_img)
            cv2.imwrite(f"{folder}/binary.png", self.binary_img)
            cv2.imwrite(f"{folder}/checkmarks.png", self.cropped_no_lines)
        except (ValueError, FileExistsError, FileNotFoundError):
            print("Sauvegarde Impossible")

    def run_analysis(self):
        if not self.cells_state:
            self._preprocess()
            self._isolate_lines()
            self._extract_cells()
            self._isolate_checkmarks()
            self._draw_cells_bboxs()
            self.save_imgs("temp")
        return {"image": self.drawn_og_img, "type": self.type}
    
    
    def calculate_score(self):
        score = 0
        ponderation = 0
        for i, row in enumerate(self.cells_state):
            if i > 19:
                multiplier = 2 
            else :
                multiplier = 1
            ponderation +=1 * multiplier
            for j in range(len(row)):
                    score += row[j][0] * j * 0.2 * multiplier
        return ((score/ponderation)*20)

        
        
    def _get_occupied_cells_per_row(self,offset_cells):
        collisions:dict[int:list] = {}
        collisions_per_checkmark_per_row = {}
        for index, bbox in enumerate(self.checkmark_bboxes):
            xb, yb, wb, hb = bbox
            checkmark_area = hb * wb
            for row_index, row in enumerate(offset_cells):
                for cell_index, cell in enumerate(row):
                    cx, cy, cw, ch = cell
                    x_overlap = max(0, min(xb + wb, cx + cw) - max(xb, cx))
                    y_overlap = max(0, min(yb + hb, cy + ch) - max(yb, cy))
                    intersection_area = x_overlap * y_overlap
                    if intersection_area == 0:
                        continue
                    ratio = intersection_area / checkmark_area
                    if collisions.get(index) is None:
                        collisions[index] = [(row_index, cell_index, ratio)]
                    else:
                        collisions[index].append((row_index, cell_index, ratio))

        for index, bbox in enumerate(self.checkmark_bboxes):
            collision = collisions.get(index)
            if not collision:
                continue

            max_collision = max(collision, key=lambda x: x[2])

            if max_collision[2] >= 0.6:
                # Strong enough match: take only the first (main) collision
                row, col, _ = collision[0]
                self.cells_state[row][col][0] = 1                
                # Append a new individual match (as its own list)
                if row not in collisions_per_checkmark_per_row:
                    collisions_per_checkmark_per_row[row] = [[col]]
                else:
                    collisions_per_checkmark_per_row[row].append([col])

            else:
                # Multiple weak overlaps: collect all of them together
                cols = []
                for row, col, _ in collision:
                    self.cells_state[row][col][0] = 0.5
                    self.cells_state[row][col][1] = 3
                    cols.append(col)

                if row not in collisions_per_checkmark_per_row:
                    collisions_per_checkmark_per_row[row] = [cols]
                else:
                    collisions_per_checkmark_per_row[row].append(cols)
        return collisions_per_checkmark_per_row 
        
    def update_cell_state_color(self):
        for i in range(len(self.cells_state)):
            empty_row = True
            for j in range(len(self.cells_state[i])):
                if self.cells_state[i][j][0] > 0:
                    empty_row = False
            if empty_row:
                for j in range(len(self.cells_state[i])):
                    self.cells_state[i][j][1] = 0
 
    
    
    def set_selected_cell(self, row, cols):
            """Keeps only the selected cell checked and unchecks others."""
            for j in range(len(self.cells_state[row])):
                self.cells_state[row][j] = [1,0]
            if len(cols)>1:
                self.cells_state[row][cols[0]] = [0.5,0]
                self.cells_state[row][cols[1]] = [0.5,0]
            else:
                self.cells_state[row][cols[0]] = [1,0]
                
    def get_warnings_errors(self):
        """a ce stade cette fonction ne retourne que les avertissements"""
        warnings=[]
        errors = []
        n_cells = 0
        for i in range(len(self.sorted_cells)):
            for j in range(len(self.sorted_cells[i])):
                n_cells += 1
        min_cells_row = len(self.sorted_cells[0])

        for i in range(len(self.sorted_cells)):
            t = len(self.sorted_cells[0])
            min_cells_row = t if t < min_cells_row else min_cells_row

        if self.type is GridType.Unknown:
            warnings.append("Type de grille inconnue")
        if (
            self.expected_row_cols[0] != min_cells_row
            or self.expected_row_cols[0] * self.expected_row_cols[1] != n_cells
        ):
            warnings.append("Le nombre de cellules détectées ne correspond pas au nombre attendu.")
        
        return warnings
    
    
    def get_problematic_cells_per_row(self):
        problematic_rows = {}
        for key,value in self.collisions_per_checkmark_per_row.items():
            if len(value)>1:
                problematic_rows[key] = value
        return problematic_rows