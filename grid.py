import cv2, numpy as np
import imutils
import tools
import constants as cst
from enum import Enum

"""
Module `grid.py`
Ce module contient des classes et des fonctions pour analyser et traiter des grilles contenant des cases et des marques de vérification (checkmarks). 
Il utilise OpenCV pour le traitement d'image et des outils personnalisés pour extraire des informations spécifiques.
Classes:
---------
1. `GridType`:
    - Enumération pour définir les différents types de grilles.
    - Types possibles : `Unknown`, `PFE_Finale`, `PFE_Inter`, `PFA`.
2. `Grid`:
    - Classe principale pour analyser une grille.
    - Contient des méthodes pour détecter le type de grille, isoler les lignes, extraire les cellules, détecter les marques de vérification, et calculer des scores.
Attributs de la classe `Grid`:
------------------------------
- `k_size`: Taille des kernels pour les opérations morphologiques.
- `horizontal_kernel`, `vertical_kernel`: Kernels pour isoler les lignes horizontales et verticales.
- `square_kernel3`, `square_kernel5`: Kernels carrés pour des opérations morphologiques spécifiques.
- `tolerance`: Tolérance pour la détection des cellules.
- `save`: Indique si les images doivent être sauvegardées.
- `threshold`: Seuil pour certaines opérations.
Méthodes de la classe `Grid`:
-----------------------------
1. `__init__(cv_image: np.ndarray)`:
    - Initialise une instance de la classe `Grid`.
    - Détecte le type et la rotation de la grille.
2. `detect_type_n_rotation(original_img)`:
    - Détecte le type de grille à partir d'un QR code et ajuste la rotation de l'image.
3. `_preprocess()`:
    - Prétraite l'image pour la convertir en niveaux de gris, binariser et inverser les couleurs.
4. `_isolate_lines()`:
    - Isole les lignes horizontales et verticales de la grille à l'aide d'opérations morphologiques.
5. `_extract_cells()`:
    - Extrait les cellules de la grille en détectant les contours et en triant les boîtes englobantes.
6. `_isolate_checkmarks()`:
    - Isole les marques de vérification (checkmarks) en supprimant les lignes et en appliquant des opérations morphologiques.
7. `change_cell_color(row, col, color_code)`:
    - Change la couleur d'une cellule spécifique.
8. `_draw_cells_bboxs()`:
    - Dessine les boîtes englobantes des cellules et des marques de vérification sur l'image originale.
9. `save_imgs(folder)`:
    - Sauvegarde les images intermédiaires dans un dossier spécifié.
10. `run_analysis()`:
    - Exécute l'analyse complète de la grille et retourne l'image annotée et le type de grille.
11. `calculate_score()`:
    - Calcule un score basé sur l'état des cellules.
12. `_get_occupied_cells_per_row(offset_cells)`:
    - Identifie les cellules occupées par les marques de vérification pour chaque ligne.
13. `update_cell_state_color()`:
    - Met à jour les couleurs des cellules en fonction de leur état.
14. `set_selected_cell(row, cols)`:
    - Définit une cellule sélectionnée et désélectionne les autres.
15. `get_warnings_errors()`:
    - Retourne les avertissements et erreurs détectés lors de l'analyse de la grille.
16. `get_problematic_cells_per_row()`:
    - Retourne les lignes contenant des cellules problématiques (par exemple, plusieurs marques de vérification).
"""


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
    tolerance = 0.65
    seuil_filtrage_surface_croix = 175

    def __init__(self, cv_image: np.ndarray):

        # to change, absolutely
        self.type = GridType.Unknown
        self.collisions_per_checkmark_per_row = {}
        self.checkmark_bboxes = []
        self.expected_row_cols = (0, 0)
        self.detect_type_n_rotation(cv_image)
        self.image_annotee = self.original_matrix.copy()
        self.middle_x = self.original_matrix.shape[1] // 2
        self.gray_img = None
        self.binary_img = None
        self.inverted_img = None
        self.combined_lines = None
        self.no_lines = None
        self.sorted_cells = None
        self.cells_state = None
        self.warnings = None
        self.imgs_dict = {}

    def detect_type_n_rotation(self, original_img):
        qcd = cv2.QRCodeDetector()
        decoded_info, points, _ = qcd.detectAndDecode(original_img)
        qr_code_data = tools.decode_qr(decoded_info)

        if qr_code_data:
            rotation = tools.calculate_angle(points[0][0], points[0][1])
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
        self.imgs_dict["0original"] = self.original_matrix.copy()
        self.gray_img = cv2.cvtColor(self.original_matrix, cv2.COLOR_BGR2GRAY)
        self.imgs_dict["1niveau_gris"] = self.gray_img.copy()
        self.binary_img = cv2.adaptiveThreshold(
            self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
        self.imgs_dict["2binaire"] = self.binary_img.copy()

        # cv2.ADAPTIVE_GAUSSIAN_C : Méthode utilisée pour calculer le seuil localement (Moyenne pondérée selon une distribution gaussienne)
        # 51: Taille du voisinage (fenêtre) 51
        # 5: Valeur constante a soustraire au seuil calculé

        self.inverted_img = cv2.bitwise_not(self.binary_img.copy())
        self.imgs_dict["3inverse"] = self.inverted_img.copy()

    def _isolate_lines(self):

        # element strurcuturant horizontal === horizontal_kernel = [1,1,1,1,1,1,1]

        # 8 iterations d'erosion
        self.horizontal_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.horizontal_kernel, iterations=8
        )
        self.imgs_dict["4lignes_horizontales_erosion"] = self.horizontal_lines.copy()
        # 14 iteration de dilatation
        self.horizontal_lines = cv2.morphologyEx(
            self.horizontal_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=14,
        )
        self.imgs_dict["5lignes_horizontales_finale"] = self.horizontal_lines.copy()

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
        self.imgs_dict["6lignes_verticales_erosion"] = self.vertical_lines.copy()

        # 8 iterations de dilatation
        self.vertical_lines = cv2.morphologyEx(
            self.vertical_lines, cv2.MORPH_DILATE, Grid.vertical_kernel, iterations=8
        )
        self.imgs_dict["7lignes_verticales_finale"] = self.vertical_lines.copy()

        self.combined_lines = cv2.bitwise_or(self.horizontal_lines, self.vertical_lines)
        self.imgs_dict["8lignes_combines"] = self.combined_lines.copy()

    def _extract_cells(self):
        rhs_combined_lines = self.combined_lines[:, self.middle_x :]
        self.imgs_dict["9moitie_droite_lignes_combines"] = rhs_combined_lines.copy()

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
        self.imgs_dict["10roi_lignes_combinees"] = self.cropped_combined_lines.copy()

        self.cropped_combined_lines = cv2.ximgproc.thinning(self.cropped_combined_lines)
        self.imgs_dict["11roi_lignes_combines_minces"] = (
            self.cropped_combined_lines.copy()
        )

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
            [[0, 2] for j in range(len(self.sorted_cells[i]))]
            for i in range(len(self.sorted_cells))
        ]

    def _isolate_checkmarks(self):
        x, y, w, h = self.sorted_cells[0][0]
        x1, y1, w1, h1 = self.bbox_biggest_rect

        # Suppression des lignes verticales
        self.no_lines = cv2.absdiff(self.inverted_img, self.vertical_lines)

        self.imgs_dict["12sans_lignes_verticales"] = self.no_lines.copy()

        # Extraction de la partie droite de l'image
        right_no_lines = self.no_lines[:, self.middle_x :]
        self.cropped_no_lines = right_no_lines[y1 : y1 + h1, x1 : x1 + w1][y:, x:]
        self.imgs_dict["13sans_lignes_verticales_roi"] = self.cropped_no_lines.copy()

        # Suppression du bruit avec une ouverture morphologique
        self.cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        self.imgs_dict["14sans_lignes_verticales_roi_ouverture"] = (
            self.cropped_no_lines.copy()
        )

        # Dilatation pour reconnecter les éléments disjoints

        dilated_cropped_no_lines = cv2.morphologyEx(
            self.cropped_no_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=3,
        )
        self.imgs_dict["15dilatee_roi_ouverture"] = dilated_cropped_no_lines.copy()

        # Suppression des lignes résiduelles
        mask_horizontal = self.horizontal_lines[:, self.middle_x :][
            y1 : y1 + h1, x1 : x1 + w1
        ][y:, x:]
        self.imgs_dict["16masque_lignes_horizontales"] = mask_horizontal.copy()

        dilated_cropped_no_lines = cv2.absdiff(
            dilated_cropped_no_lines, mask_horizontal
        )
        self.imgs_dict["17dilatee_roi_ouverture_sans_lignes_horizontales"] = (
            dilated_cropped_no_lines.copy()
        )

        # Fermeture puis ouverture morphologique pour éliminer le bruit
        dilated_cropped_no_lines = cv2.morphologyEx(
            dilated_cropped_no_lines, cv2.MORPH_CLOSE, Grid.square_kernel5, iterations=1
        )
        self.imgs_dict["18dilatee_roi_ouverture_sans_lignes_horizontales_fermeture"] = (
            dilated_cropped_no_lines.copy()
        )
        dilated_cropped_no_lines = cv2.morphologyEx(
            dilated_cropped_no_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        self.imgs_dict["19dilatee_roi_ouverture_sans_lignes_horizontales_ouverture"] = (
            dilated_cropped_no_lines.copy()
        )

        # Filtrage supplémentaire pour améliorer l'extraction des checkmarks
        dilated_cropped_no_lines = cv2.bitwise_and(
            dilated_cropped_no_lines,
            self.inverted_img[:, self.middle_x :][y1 : y1 + h1, x1 : x1 + w1][y:, x:],
        )
        self.imgs_dict["20et_logique_img-dilatee_img-inversee"] = (
            dilated_cropped_no_lines.copy()
        )

        # Mise à jour de l'attribut final
        self.cropped_no_lines = dilated_cropped_no_lines

        checkmark_contours, _ = cv2.findContours(
            dilated_cropped_no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.imgs_dict["21contours_checkmarks_img"] = self.cropped_no_lines.copy()
        for contour in checkmark_contours:
            bbox = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > Grid.seuil_filtrage_surface_croix:
                self.checkmark_bboxes.append(bbox)

        offset_cells = [
            tools.add_offset_bbox(row, (-x, -y)) for row in self.sorted_cells
        ]
        self.collisions_per_checkmark_per_row = self._get_occupied_cells_per_row(
            offset_cells
        )
        
    def _draw_checkmarks_bboxes(self):
        xcell, ycell, _, _ = self.sorted_cells[0][0]
        x, y, w, h = self.bbox_biggest_rect
        for bbox in self.checkmark_bboxes:
            xb, yb, wb, hb = bbox
            cv2.rectangle(self.cropped_no_lines, (xb, yb), (xb + wb, yb + hb), 255, 2)
            cv2.rectangle(
                self.image_annotee,
                (xb + x + self.middle_x + xcell, ycell + y + yb),
                (xb + x + self.middle_x + xcell + wb, ycell + y + yb + hb),
                (0, 0, 255),
                2,
            )

    def _draw_cells_bboxs(self):
        xcell, ycell, _, _ = self.sorted_cells[0][0]
        x, y, w, h = self.bbox_biggest_rect
        for i, row in enumerate(self.sorted_cells):
            for j, cell in enumerate(row):
                xcell, ycell, wcell, hcell = cell
                if self.cells_state[i][j][1]!=-1:
                    cv2.rectangle(
                        self.image_annotee,
                        (xcell + self.middle_x + 10, y + ycell + 10),
                        (xcell + self.middle_x + wcell - 10, y + ycell + hcell - 10),
                        cst.COLORS[self.cells_state[i][j][1]],
                        2,
                    )
    
    def draw_all_bboxs(self):
        self.image_annotee = self.original_matrix.copy()
        self._draw_cells_bboxs()
        self._draw_checkmarks_bboxes()
        

    def save_imgs(self, folder):
        if not cst.SAVE:
            return
        try:
            for key, img in self.imgs_dict.items():
                cv2.imwrite(f"{folder}/{key}.png", img)
        except (ValueError, FileExistsError, FileNotFoundError):
            print("Sauvegarde Impossible")

    def run_analysis(self):
        if not self.cells_state:
            self._preprocess()
            self._isolate_lines()
            self._extract_cells()
            self._isolate_checkmarks()
            self.save_imgs("temp")
        return {"image": self.image_annotee, "type": self.type}

    def calculate_score(self):
        score = 0
        ponderation = 0
        for i, row in enumerate(self.cells_state):
            multiplier = 2 if i > 19 else 1

            # Vérifie si la ligne contient au moins une case cochée
            has_checked = any(bool(cell[0]) for cell in row)
            # Ajoute une pondération supplémentaire si la ligne est partiellement remplie
            if has_checked:
                ponderation += multiplier
            # Calcule le score pondéré pour chaque case cochée
            score += sum(cell[0] * j * 0.2 * multiplier for j, cell in enumerate(row))
        return (score / ponderation) * 20 if ponderation else 0


    def _get_occupied_cells_per_row(self, offset_cells):
        collisions: dict[int:list] = {}
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
                row, col, _ = collision[0]
                self.cells_state[row][col][0] = 1
                if row not in collisions_per_checkmark_per_row:
                    collisions_per_checkmark_per_row[row] = [[col]]
                else:
                    collisions_per_checkmark_per_row[row].append([col])

            else:
                cols = []
                for row, col, _ in collision:
                    self.cells_state[row][col][0] = 0.5
                    self.cells_state[row][col][1] = 1
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
            self.cells_state[row][j] = [0, 2]
        if len(cols) == 2:
            self.cells_state[row][cols[0]] = [0.5, 1]
            self.cells_state[row][cols[1]] = [0.5, 1]
        else:
            self.cells_state[row][cols[0]] = [1, 1]

    def get_warnings_errors(self):
        """a ce stade cette fonction ne retourne que les avertissements"""
        warnings = []
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
            self.expected_row_cols[1] != min_cells_row or self.expected_row_cols[0] != len(self.sorted_cells)
            or self.expected_row_cols[0] * self.expected_row_cols[1] != n_cells
        ):
            warnings.append(
                f"Le nombre de cellules détectées ne correspond pas au nombre attendu de cellule de la grille.\n Nombre de cellules trouvées: {n_cells}\n Nombre de cellules attendues: {self.expected_row_cols[0] * self.expected_row_cols[1]}"
            )

        return warnings

    def get_problematic_cells_per_row(self):
        problematic_rows = {}
        for key, value in self.collisions_per_checkmark_per_row.items():
            if len(value) > 1:
                problematic_rows[key] = value
        return problematic_rows

    def highlight_row(self, row,cols, color=(2, 80, 246), thickness=10):
        if not self.sorted_cells:
            return
        x_start, y_start, w_start, h_start = self.sorted_cells[row][0]
        x_end, y_end, w_end, h_end = self.sorted_cells[row][-1]
        x_bbox, y_bbox, w_bbox, h_bbox = self.bbox_biggest_rect
        top_left = (x_start + x_bbox + self.middle_x, y_start + y_bbox)
        bottom_right = (x_end + w_end + x_bbox + self.middle_x, y_end + h_end + y_bbox)
        cv2.rectangle(self.image_annotee, top_left, bottom_right, color, thickness)
        if len(cols)>1:
            x_start, y_start, w_start, h_start = self.sorted_cells[row][cols[0]]
            x_end, y_end, w_end, h_end = self.sorted_cells[row][cols[-1]]
            top_left = (x_start + x_bbox + self.middle_x, y_start + y_bbox)
            bottom_right = (x_end + w_end + x_bbox + self.middle_x, y_end + h_end + y_bbox)
            cv2.rectangle(self.image_annotee, top_left, bottom_right, (0,255,0), thickness)
        else:
            x_start, y_start, w_start, h_start = self.sorted_cells[row][cols[0]]
            top_left = (x_start + x_bbox + self.middle_x, y_start + y_bbox)
            bottom_right = (x_start + w_start + x_bbox + self.middle_x, y_start + h_start + y_bbox)
            cv2.rectangle(self.image_annotee, top_left, bottom_right, (0,255,0), thickness)

    def clear_image(self,draw_bboxes=False):
        self.image_annotee = self.original_matrix.copy()
        if draw_bboxes:
            self.draw_all_bboxs()
            
