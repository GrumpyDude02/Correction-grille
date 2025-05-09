import cv2, numpy as np
import imutils
import tools
import constants as cst
from enum import Enum

# -------------------------------------------------------------------------------------------------------------
#                                   Structure du fichier et catégorisation des fonctions
# -------------------------------------------------------------------------------------------------------------

# Catégorisation des fonctions :
# 1. Fonctions d'aide:
#    - _add_img_to_dict : Ajoute une image au dictionnaire de sauvegarde avec un nom spécifique.
#    - _draw_checkmarks_bboxes : Dessine les contours des croix détectées sur l'image annotée.
#    - _draw_cells_bboxs : Dessine les contours des cellules avec leur couleur d'état respective.
#    - _get_occupied_cells_per_row : Détermine les cellules contenant des croix en calculant les zones de chevauchement.
#    - _save_imgs : Sauvegarde les images intermédiaires dans un dossier spécifié.

# 2. Fonctions principales :
#    - _detect_type_n_rotation : Détecte le type de grille et ajuste la rotation de l'image.
#    - _pretraitement : Effectue le prétraitement de l'image (conversion en niveaux de gris, binarisation, etc.).
#    - _extraction_lignes : Isole les lignes structurelles de la grille par morphologie mathématique.
#    - _extraction_cellules : Extrait les cellules de la grille par analyse des contours.
#    - _extraction_croix : Détecte et extrait les croix de validation dans les cellules.
#    - run_analysis : Exécute l'analyse complète de la grille.
#    - calculate_score : Calcule le score pondéré selon les cases cochées.

# 3. Fonctions d'interface avec l'application :
#    - draw_all_bboxs : Dessine toutes les boîtes englobantes (cellules et croix) sur l'image annotée.
#    - update_cell_state_color : Met à jour les couleurs des lignes vides.
#    - set_selected_cell : Force la sélection d'une cellule dans une ligne.
#    - get_warnings_errors : Détecte les anomalies structurelles.
#    - get_problematic_cells_per_row : Identifie les lignes avec détections ambiguës.
#    - highlight_row : Surligne une ligne et ses cellules sélectionnées.
#    - clear_image : Réinitialise l'image annotée.

# Les fonction precédées de "_" sont privées et ne doivent pas être appelées directement à l'extérieur de la classe
# Les fonctions publiques sont celles qui n'ont pas de "_" au début de leur nom


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
    seuil_croix = 0.7
    seuil_filtrage_surface_croix = 175

    def __init__(self, cv_image: np.ndarray):

        # to change, absolutely
        self.type = GridType.Unknown
        self.collisions_per_checkmark_per_row = {}
        self.checkmark_bboxes = []
        self.expected_row_cols = (0, 0)
        self._detect_type_n_rotation(cv_image)
        self.image_annotee = self.original_matrix.copy()
        self.middle_x = self.original_matrix.shape[1] // 2
        self.gray_img = None
        self.binary_img = None
        self.inverted_img = None
        self.combined_lines = None
        self.cropped_no_lines = None
        self.sorted_cells = None
        self.cells_state = None
        self.warnings = None
        self.save_counter = 0
        self.imgs_dict = {}

    
    def _add_img_to_dict(self, name,img):
        """Ajoute une image au dictionnaire de sauvegarde un nom spécifique."""
        self.imgs_dict[f"{self.save_counter}-{name}"] = img.copy()
        self.save_counter += 1
        
    # ----------------------------------------------------------------------------------------------------------------------
    #           Fonctions de Traitement de l'image (Fonctions Principales)
    # ----------------------------------------------------------------------------------------------------------------------

    def _detect_type_n_rotation(self, original_img):
        qcd = cv2.QRCodeDetector()
        decoded_info, points, _ = qcd.detectAndDecode(original_img)
        qr_code_data = tools.decode_qr(decoded_info)

        if qr_code_data:
            rotation = tools.calculate_angle(points[0][0], points[0][1])
            self.original_matrix = imutils.rotate_bound(original_img, -rotation)
            match qr_code_data.get("Type"):
                case "PFE-F":
                    self.type = GridType.PFE_Finale
                case "PFE-Inter":
                    self.type = GridType.PFE_Inter
                case "PFA":
                    self.type = GridType.PFA
            self.expected_row_cols = (
                qr_code_data.get("Lines"),
                qr_code_data.get("Cols"),
            )
        elif original_img.shape[1] > original_img.shape[0]:
            self.original_matrix = imutils.rotate_bound(original_img, 90)
        else:
            self.original_matrix = original_img

    def _pretraitement(self):
        """Prétraitement de l'image avant analyse"""
        # Sauvegarde de l'image originale
        self._add_img_to_dict("originale", self.original_matrix.copy())
        # Conversion en niveaux de gris (réduction à 1 canal)
        self.gray_img = cv2.cvtColor(self.original_matrix, cv2.COLOR_BGR2GRAY)
        self._add_img_to_dict("niveau_gris", self.gray_img.copy())

        # Binarisation adaptative avec seuillage gaussien
        # - Paramètres:
        #   * Taille de voisinage: 51px (zone analysée pour calculer le seuil)
        #   * Constante de soustraction: 5 (ajustement fin du seuil)
        self.binary_img = cv2.adaptiveThreshold(
            self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
        self._add_img_to_dict("binaire", self.binary_img.copy())

        # Inversion des couleurs (noir <-> blanc)
        # - Nécessaire pour les opérations morphologiques suivantes
        self.inverted_img = cv2.bitwise_not(self.binary_img.copy())
        self._add_img_to_dict("inverse", self.inverted_img.copy())

    def _extraction_lignes(self):
        """Isolation des lignes structurelles par morphologie mathématique"""

        # ---------------------------------------------------------------
        # Traitement des LIGNES HORIZONTALES
        # ---------------------------------------------------------------

        # Érosion horizontale agressive
        # - Noyau: [1×7] - Sensibilité horizontale
        # - 8 itérations pour une suppression complète
        self.horizontal_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.horizontal_kernel, iterations=8
        )
        self._add_img_to_dict("lignes_horizontales_erosion", self.horizontal_lines.copy())
        # Dilatation de reconstruction
        # - 14 itérations pour compenser l'érosion initiale
        self.horizontal_lines = cv2.morphologyEx(
            self.horizontal_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=14,
        )
        # ---------------------------------------------------------------
        # Traitement des LIGNES VERTICALES
        # ---------------------------------------------------------------

        # Érosion verticale ciblée
        # - Noyau: [7×1] - Sensibilité verticale
        # - 8 itérations pour éliminer l'horizontal
        self.vertical_lines = cv2.morphologyEx(
            self.inverted_img, cv2.MORPH_ERODE, Grid.vertical_kernel, iterations=8
        )
        self._add_img_to_dict("lignes_verticales_erosion", self.vertical_lines.copy())
        # Dilatation modérée
        # - Même nombre d'itérations que l'érosion pour équilibre
        self.vertical_lines = cv2.morphologyEx(
            self.vertical_lines, cv2.MORPH_DILATE, Grid.vertical_kernel, iterations=8
        )
        self._add_img_to_dict("lignes_verticales", self.vertical_lines.copy())
        self._add_img_to_dict("lignes_horizontales", self.horizontal_lines.copy())
        
        # - HoghLinesP pour affiner les lignes détectées
        hough_horizontal = tools.detecter_lignes_hough(
            self.horizontal_lines,
            longueur_min=int(self.original_matrix.shape[1] * 0.1),
            epaisseur=3,  # epaisseur de la ligne a dessiner sur l'image retournée par la fonction
        )
        hough_vertical = tools.detecter_lignes_hough(
            self.vertical_lines,
            longueur_min=int(self.original_matrix.shape[0] * 0.1),
            epaisseur=3,  # epaisseur de la ligne a dessiner sur l'image retournée par la fonction
        )
        
        
        self._add_img_to_dict("hough_horizontal", hough_horizontal.copy())
        self._add_img_to_dict("hough_vertical", hough_vertical.copy())
        
        self.horizontal_lines = cv2.bitwise_and(hough_horizontal,self.horizontal_lines)
        self.vertical_lines = cv2.bitwise_and(hough_vertical,self.vertical_lines)
        
        self._add_img_to_dict("lignes_verticales_finale", self.vertical_lines.copy())
        self._add_img_to_dict("lignes_horizontales_finale", self.horizontal_lines.copy())
        # Combinaison des résultats
        # - Fusion des lignes horizontales et verticales détectées

        self.combined_lines = cv2.bitwise_or(self.horizontal_lines, self.vertical_lines)
        self._add_img_to_dict("lignes_combines", self.combined_lines.copy())

    def _extraction_cellules(self):
        """Extraction des cellules de la grille par analyse des contours"""

        # ---------------------------------------------------------------
        # Étape 1: Découpage de la moitié droite de l'image
        # ---------------------------------------------------------------
        # On isole la partie droite qui contient les cellules à analyser
        rhs_combined_lines = self.combined_lines[:, self.middle_x :]
        self._add_img_to_dict("moitie_droite_lignes_combinees", rhs_combined_lines.copy())
        # ---------------------------------------------------------------
        # Étape 2: Détection du cadre principal
        # ---------------------------------------------------------------
        # Trouver tous les contours externes dans la partie droite
        outer_contours, _ = cv2.findContours(
            rhs_combined_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sélection du plus grand contour détecté
        self.bbox_biggest_rect = cv2.boundingRect(
            max(outer_contours, key=lambda x: cv2.contourArea(x))
        )

        # Découpage de la région d'intérêt (ROI)
        x, y, w, h = self.bbox_biggest_rect
        self.cropped_combined_lines = rhs_combined_lines[y : y + h, x : x + w]
        self._add_img_to_dict("ligne_combinees_cadree", self.cropped_combined_lines.copy())  
        # ---------------------------------------------------------------
        # Étape 3: Affinement des lignes
        # ---------------------------------------------------------------
        # Amincissement des lignes pour mieux séparer les cellules
        # self.combined_cropped_lines = cv2.morphologyEx(self.cropped_combined_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1)
        self.cropped_combined_lines = cv2.ximgproc.thinning(self.cropped_combined_lines)
        self._add_img_to_dict("lignes_minces_cadree", self.cropped_combined_lines.copy())

        # ---------------------------------------------------------------
        # Étape 4: Détection des cellules individuelles (peut être optimisé en utilisant les contours déjà detectés qui sont a l'intérieur du plus grand rectangle)
        # ---------------------------------------------------------------
        # Recherche des contours internes (les cellules)
        inner_contours, _ = cv2.findContours(
            self.cropped_combined_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes = []
        areas = [cv2.contourArea(c) for c in inner_contours]
        median = np.median(areas)  # Surface médiane de référence

        # Filtrage des cellules valides
        for contour in inner_contours:
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )
            area = cv2.contourArea(contour)

            # Critères de validation:
            # 1. Forme ayant 3,4 ou 5 côtés (tolerance de 1 coté)
            # 2. Surface dans la plage acceptable (médiane ± tolérance)
            if 3 <= len(approx) <= 5 and median * (
                1 - Grid.tolerance
            ) < area < median * (1 + Grid.tolerance):
                bboxes.append(cv2.boundingRect(contour))

        # ---------------------------------------------------------------
        # Étape 5: Organisation des cellules
        # ---------------------------------------------------------------
        # Tri des cellules par position (ligne/colonne)
        self.sorted_cells = tools.sort_cells(bboxes)

        # Initialisation de la matrice d'état:
        # - Premier indice: état (0=vide, 1=cochée)
        # - Second indice: couleur (1=par défaut)
        self.cells_state = [[[0, 1] for _ in ligne] for ligne in self.sorted_cells]

    def _extraction_croix(self):
        """Détection et extraction des croix de validation (croix/coches) dans les cellules"""

        # Récupération des coordonnées de référence
        x_cell, y_cell, w_cell, h_cell = self.sorted_cells[0][0]  # Première cellule
        x_roi, y_roi, w_roi, h_roi = self.bbox_biggest_rect  # Zone d'intérêt principale

        # ---------------------------------------------------------------
        # Étape 1: Suppression des lignes verticales
        # ---------------------------------------------------------------

        no_vertical_lines = cv2.absdiff(
            self.vertical_lines,
            self.inverted_img,
        )
        cropped_no_vertical_lines = no_vertical_lines[:, self.middle_x :][
            y_roi : y_roi + h_roi, x_roi : x_roi + w_roi
        ][
            y_cell:, x_cell:
        ]  # cadrage a partir des coordonnées de la premiere cellule detectée
        self._add_img_to_dict("sans_lignes_verticales_cadree", cropped_no_vertical_lines.copy())
        # ---------------------------------------------------------------
        # Étape 2: Nettoyage initial
        # ---------------------------------------------------------------
        # Ouverture morphologique pour supprimer le bruit (noyau 3x3)
        cropped_no_vertical_lines = cv2.morphologyEx(
            cropped_no_vertical_lines, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        self._add_img_to_dict("sans_lignes_verticales_ouverture_cadree",cropped_no_vertical_lines.copy())

        # ---------------------------------------------------------------
        # Étape 3: Reconstitution des croix fragmentées
        # ---------------------------------------------------------------
        # Dilatation horizontale pour reconnecter les segments de croix
        dilated_img = cv2.morphologyEx(
            cropped_no_vertical_lines,
            cv2.MORPH_DILATE,
            Grid.horizontal_kernel,
            iterations=3,
        )
        self.imgs_dict["sans_lignes_verticales_dilatee_cadree"] = dilated_img.copy()
        self._add_img_to_dict("sans_lignes_verticales_dilatee_cadree", dilated_img.copy())

        # ---------------------------------------------------------------
        # Étape 4: Suppression des artefacts horizontaux résiduels
        # ---------------------------------------------------------------
        # Masquage des lignes horizontales restantes
        mask_horizontal = self.horizontal_lines[:, self.middle_x :][
            y_roi : y_roi + h_roi, x_roi : x_roi + w_roi
        ][y_cell:, x_cell:]
        self._add_img_to_dict("masque_horizontal_cadree", mask_horizontal.copy())

        dilated_img = cv2.absdiff(dilated_img, mask_horizontal)
        self._add_img_to_dict("diff_masque_horizontal_sans_lignes_verticales_dilatee",dilated_img)

        # ---------------------------------------------------------------
        # Étape 5: Post-traitement morphologique
        # ---------------------------------------------------------------
        # 1. Fermeture (noyau 5x5) pour combler les petits trous
        # 2. Ouverture (noyau 3x3) pour lisser les contours
        dilated_img = cv2.morphologyEx(
            dilated_img, cv2.MORPH_CLOSE, Grid.square_kernel5, iterations=1
        )
        self._add_img_to_dict("apres_fermeture", dilated_img.copy())
        
        dilated_img = cv2.morphologyEx(
            dilated_img, cv2.MORPH_OPEN, Grid.square_kernel3, iterations=1
        )
        self._add_img_to_dict("apres_ouverture", dilated_img.copy())

        # ---------------------------------------------------------------
        # Étape 6: Filtrage final par cohérence lumineuse
        # ---------------------------------------------------------------
        # Intersection avec l'image originale inversée
        final_img = cv2.bitwise_and(
            dilated_img,
            self.inverted_img[:, self.middle_x :][
                y_roi : y_roi + h_roi, x_roi : x_roi + w_roi
            ][y_cell:, x_cell:],
        )
        self._add_img_to_dict("image_dilatee_ET_image_inversee", final_img.copy())
        self.cropped_no_lines = final_img.copy()

        # ---------------------------------------------------------------
        # Étape 7: Détection des contours des croix
        # ---------------------------------------------------------------
        contours, _ = cv2.findContours(
            final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtrage par taille minimale
        self.checkmark_bboxes = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > Grid.seuil_filtrage_surface_croix
        ]

        # ---------------------------------------------------------------
        # Étape 8: Cartographie des collisions
        # ---------------------------------------------------------------
        # Ajustement des coordonnées des cellules
        cellules_decalees = [
            tools.add_offset_bbox(ligne, (-x_cell, -y_cell))
            for ligne in self.sorted_cells
        ]
        # Calcul des intersections croix/cellules
        self.collisions_per_checkmark_per_row = self._get_occupied_cells_per_row(
            cellules_decalees
        )

    def _get_occupied_cells_per_row(self, offset_cells):
        """
        Détermine les cellules contenant des croix en calculant les zones de chevauchement.

        Args:
            offset_cells: Liste des cellules avec coordonnées ajustées

        Returns:
            Dictionnaire {num_ligne: [liste_colonnes_occupées]}
            avec distinction entre croix certaines (ratio >= 60%) et ambiguës
        """
        # Dictionnaire temporaire des collisions par croix
        croix_collisions = {}
        # Dictionnaire final des collisions par ligne
        collisions_par_ligne = {}

        # ---------------------------------------------------------------
        # Étape 1: Calcul des intersections croix-cellule
        # ---------------------------------------------------------------
        for index, (x_croix, y_croix, w_croix, h_croix) in enumerate(
            self.checkmark_bboxes
        ):
            surface_croix = w_croix * h_croix

            for row_idx, ligne in enumerate(offset_cells):
                for col_idx, (x_cell, y_cell, w_cell, h_cell) in enumerate(ligne):

                    # Calcul de la zone d'intersection
                    x_intersection = max(
                        0,
                        min(x_cell + w_cell, x_croix + w_croix) - max(x_cell, x_croix),
                    )
                    y_intersection = max(
                        0,
                        min(y_cell + h_cell, y_croix + h_croix) - max(y_cell, y_croix),
                    )

                    # Si l'intersection est valide (si la surface est positive)
                    surface_intersec = x_intersection * y_intersection
                    if surface_intersec > 0:
                        ratio = surface_intersec / surface_croix
                        # Enregistrement de la collision
                        if index not in croix_collisions:
                            croix_collisions[index] = [
                                (row_idx, col_idx, ratio)
                            ]  # Première collision pour cette croix
                        else:
                            croix_collisions[index].append((row_idx, col_idx, ratio))

        # ---------------------------------------------------------------
        # Étape 2: Analyse des collisions
        # ---------------------------------------------------------------
        for index, _ in enumerate(self.checkmark_bboxes):
            if index not in croix_collisions:
                continue

            collisions = croix_collisions[index]
            meilleure_collision = max(collisions, key=lambda x: x[2])

            # Seuil de certitude (70% de recouvrement)
            if meilleure_collision[2] >= Grid.seuil_croix:
                ligne, colonne, _ = (
                    meilleure_collision  # Utilisation de la meilleure collision
                )
                self.cells_state[ligne][colonne][0] = 1  # Croix certaine

                # Format: {ligne: [[col1], [col2], ...]}
                if ligne not in collisions_par_ligne:
                    collisions_par_ligne[ligne] = [[colonne]]
                else:
                    collisions_par_ligne[ligne].append([colonne])

            # Cas des croix ambiguës (ex: à cheval sur 2 colonnes)
            else:
                colonnes_concernees = (
                    set()
                )  # Utilisation d'un ensemble pour éviter les doublons
                for ligne, colonne, ratio in collisions:
                    self.cells_state[ligne][colonne][0] = 0.5  # Croix incertaine
                    self.cells_state[ligne][colonne][1] = 2  # Code couleur orange
                    colonnes_concernees.add(colonne)  # Ajouter la colonne sans doublons

                # Format: {ligne: [[col1, col2], ...]}
                if ligne not in collisions_par_ligne:
                    collisions_par_ligne[ligne] = [list(colonnes_concernees)]
                else:
                    collisions_par_ligne[ligne].append(list(colonnes_concernees))

        # Éliminer les doublons pour chaque ligne
        for ligne in collisions_par_ligne:
            unique_col_lists = list({tuple(col) for col in collisions_par_ligne[ligne]})
            collisions_par_ligne[ligne] = [list(col) for col in unique_col_lists]

        return collisions_par_ligne

    def _save_imgs(self, folder):
        if not cst.SAVE:
            return
        i = self.cropped_no_lines.copy()
        for bbox in self.checkmark_bboxes:
            x_croix, y_croix, w_croix, h_croix = bbox
            cv2.rectangle(
                i, (x_croix, y_croix), (x_croix + w_croix, y_croix + h_croix), 255, 2
            )
        self._add_img_to_dict("image_dilatee_ET_image_inversee", i)
        image = self.image_annotee.copy()
        self._draw_checkmarks_bboxes(image)
        self._draw_cells_bboxs(image)
        self._add_img_to_dict("image_annotée", image)
        try:
            for key, img in self.imgs_dict.items():
                cv2.imwrite(f"{folder}/{key}.png", img)

        except (ValueError, FileExistsError, FileNotFoundError):
            print("Sauvegarde Impossible")

    # -------------------------------------------------------------------------------------------------------------
    #                                   Fonctions permettant l'interaction avec l'application
    # -------------------------------------------------------------------------------------------------------------
    def _draw_checkmarks_bboxes(self, image):
        # Coordonnées de référence pour l'alignement
        x_cell_ref, y_cell_ref, _, _ = self.sorted_cells[0][0]  # Première cellule
        x_roi, y_roi, w_roi, h_roi = self.bbox_biggest_rect  # Zone d'intérêt principale

        for x_croix, y_croix, w_croix, h_croix in self.checkmark_bboxes:
            # Dessin sur l'image finale (en rouge)
            cv2.rectangle(
                image,
                # Calcul précis de la position absolue:
                (
                    x_croix + x_roi + self.middle_x + x_cell_ref,
                    y_cell_ref + y_roi + y_croix,
                ),
                (
                    x_croix + x_roi + self.middle_x + x_cell_ref + w_croix,
                    y_cell_ref + y_roi + y_croix + h_croix,
                ),
                (0, 0, 255),  # Rouge en BGR
                2,
            )

    def _draw_cells_bboxs(self, image):
        """
        Dessine les contours des cellules avec leur couleur d'état respective.
        - Ignore les cellules marquées comme masquées (état -1)
        - Marge intérieure de 10px pour mieux visualiser les cases
        """
        _, y_cell_ref, _, _ = self.sorted_cells[0][0]  # Référence verticale
        x_roi, y_roi, _, _ = self.bbox_biggest_rect  # Offset du ROI

        for i, row in enumerate(self.sorted_cells):
            for j, (x_cell, y_cell, w_cell, h_cell) in enumerate(row):
                # Ne pas dessiner les cellules masquées
                if self.cells_state[i][j][1] == -1:
                    continue

                # Dessin avec marge intérieure de 10px
                cv2.rectangle(
                    image,
                    (x_cell + self.middle_x + 10, y_roi + y_cell + 10),
                    (
                        x_cell + self.middle_x + w_cell - 10,
                        y_roi + y_cell + h_cell - 10,
                    ),
                    cst.COLORS[self.cells_state[i][j][1]],  # Couleur selon l'état
                    2,  # Épaisseur du trait
                )

    def draw_all_bboxs(self):
        """Dessine toutes les boîtes englobantes (cellules et croix) sur l'image annotée"""
        self.image_annotee = self.original_matrix.copy()
        self._draw_cells_bboxs(self.image_annotee)  # Dessine les contours des cellules
        self._draw_checkmarks_bboxes(
            self.image_annotee
        )  # Dessine les contours des croix

    def run_analysis(self):
        """Exécute l'analyse complète de la grille si pas déjà fait"""
        if not self.cells_state:
            self._pretraitement()  # Étape 1: Prétraitement
            self._extraction_lignes()  # Étape 2: Détection des lignes
            self._extraction_cellules()  # Étape 3: Extraction cellules
            self._extraction_croix()  # Étape 4: Détection croix
            self._save_imgs(cst.SAVE_PATH)  # Sauvegarde images intermédiaires
        return {
            "image": self.image_annotee,  # Image avec annotations
            "type": self.type,  # Type de grille détecté
        }

    def calculate_score(self):
        """Calcule le score pondéré selon les cases cochées"""
        score = 0
        ponderation = 0

        for i, row in enumerate(self.sorted_cells):
            # Pondération double pour les lignes > 19
            multiplier = 2 if (i > 19 and self.type == GridType.PFE_Finale) else 1
            # Vérification cases cochées
            has_checked = any(cell[0] > 0 for cell in self.cells_state[i])

            if has_checked:
                ponderation += multiplier
                # Calcul score avec pondération colonne (j*0.2)
                score += sum(
                    cell[0] * j * 0.2 * multiplier
                    for j, cell in enumerate(self.cells_state[i])
                )

        # Normalisation sur 20
        return (score / ponderation) * 20 if ponderation else 0

    def update_cell_state_color(self):
        """Met à jour les couleurs des lignes vides"""
        for i, row in enumerate(self.cells_state):
            if all(cell[0] == 0 for cell in row):  # Si ligne vide
                for cell in row:
                    cell[1] = 0  # Couleur spéciale pour lignes vides

    def set_selected_cell(self, row, cols):
        """Force la sélection d'une cellule dans une ligne"""
        # Réinitialisation de la ligne
        for cell in self.cells_state[row]:
            cell[0], cell[1] = 0, 2  # Déselection

        # Cas normal: une seule colonne sélectionnée
        if len(cols) == 1:
            self.cells_state[row][cols[0]] = [1, 1]  # [état, couleur]

        # Cas ambigu: deux colonnes possibles
        else:
            for col in cols[:2]:  # Prend au plus 2 colonnes
                self.cells_state[row][col] = [0.5, 1]  # État incertain

    def get_warnings_errors(self):
        """Détecte les anomalies structurelles"""
        data = {"warnings": [], "errors": []}
        total_cells = sum(len(row) for row in self.sorted_cells)

        # Vérification type de grille
        if self.type is GridType.Unknown:
            data["warnings"].append(
                "Type de grille inconnu. le calcule effectué ne sera pas fiable."
            )

        # Vérification nombre de cellules
        expected_total = self.expected_row_cols[0] * self.expected_row_cols[1]
        if (
            self.expected_row_cols[1] != len(self.sorted_cells[0])
            or self.expected_row_cols[0] != len(self.sorted_cells)
            or expected_total != total_cells
        ):

            data["warnings"].append(
                f"Incohérence détectée:\n"
                f"- Cellules trouvées: {total_cells}\n"
                f"- Cellules attendues: {expected_total}"
            )
        lignes_non_vides = set(self.collisions_per_checkmark_per_row.keys())
        lignes_vides = list(
            set(range(len(self.sorted_cells))) - lignes_non_vides
        )
        if lignes_vides:
            lignes_vides.sort()
            for ligne in lignes_vides:
                data["warnings"].append(f"Ligne vide détectée: {ligne + 1}")
        return data

    def get_problematic_cells_per_row(self):
        """Identifie les lignes avec détections ambiguës"""
        return {
            row: cases
            for row, cases in self.collisions_per_checkmark_per_row.items()
            if len(cases) > 1  # Plus d'une croix détectée par ligne
        }

    def highlight_row(self, row, cols, color=(2, 80, 246), thickness=10):
        """Surligne une ligne et ses cellules sélectionnées"""
        if not self.sorted_cells:
            return

        # Surlignage de toute la ligne
        x1, y1, _, _ = self.sorted_cells[row][0]
        x2, y2, w2, h2 = self.sorted_cells[row][-1]
        x_off, y_off, _, _ = self.bbox_biggest_rect

        cv2.rectangle(
            self.image_annotee,
            (x1 + x_off + self.middle_x, y1 + y_off),
            (x2 + w2 + x_off + self.middle_x, y2 + h2 + y_off),
            color,
            thickness,
        )

        # Surlignage des cellules spécifiques (en vert)
        for col in cols[:2]:  # Maximum 2 colonnes
            x, y, w, h = self.sorted_cells[row][col]
            cv2.rectangle(
                self.image_annotee,
                (x + x_off + self.middle_x, y + y_off),
                (x + w + x_off + self.middle_x, y + h + y_off),
                (0, 255, 0),
                thickness,
            )

    def clear_image(self, draw_bboxes=False):
        """Réinitialise l'image annotée"""
        self.image_annotee = self.original_matrix.copy()
        if draw_bboxes:
            self.draw_all_bboxs()  # Option: redessiner les contours
