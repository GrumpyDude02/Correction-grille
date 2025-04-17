import pymupdf,cv2,numpy as np
from grid import Grid
"""
Module: convert_pdf
Ce module contient une classe `PDFFile` qui permet de manipuler des fichiers PDF pour extraire des images et les convertir en grilles exploitables.
Classes:
--------
- PDFFile: Représente un fichier PDF et fournit des méthodes pour extraire des images et les manipuler sous forme de grilles.
Dependencies:
-------------
- pymupdf: Utilisé pour lire et extraire des données des fichiers PDF.
- cv2 (OpenCV): Utilisé pour manipuler les images extraites.
- numpy: Utilisé pour convertir les données d'image en tableaux.
- Grid: Une classe externe utilisée pour représenter les grilles générées à partir des images.
Classe PDFFile:
---------------
Méthodes:
---------
- __init__(self, path) -> None:
    Constructeur de la classe. Initialise le fichier PDF et les attributs nécessaires.
    - path: Chemin du fichier PDF.
- extract_grids(self):
    Extrait les images du fichier PDF et les convertit en objets `Grid`.
- get_current_grid(self):
    Retourne la grille actuellement sélectionnée. Si aucune grille n'est sélectionnée, retourne la première.
- get_next_grid(self):
    Passe à la grille suivante et la retourne. Si la dernière grille est atteinte, revient à la première.
- get_previous_grid(self):
    Passe à la grille précédente et la retourne. Si la première grille est atteinte, revient à la dernière.
- extract_images(self):
    Extrait les données brutes des images contenues dans le fichier PDF.
    Retourne une liste de dictionnaires contenant les données des images.
- save_images(self):
    Sauvegarde les images extraites dans un répertoire local sous forme de fichiers.
- get_cv_images(self):
    Convertit les données brutes des images en objets d'image OpenCV.
- __del__(self):
    Ferme le fichier PDF lors de la destruction de l'objet.
Attributs:
----------
- file: Objet représentant le fichier PDF ouvert.
- path: Chemin du fichier PDF.
- _raw_images_dict: Liste contenant les données brutes des images extraites.
- grids: Liste d'objets `Grid` générés à partir des images.
- current_grid: Index de la grille actuellement sélectionnée.
Notes:
------
- La méthode `extract_images` sélectionne la plus grande image de chaque page du PDF.
- Les images sont converties en tableaux OpenCV pour faciliter leur manipulation.
"""

# Classe représentant un fichier PDF et fournissant des méthodes pour extraire et manipuler des images
class PDFFile:
    def __init__(self, path) -> None:
        """Constructeur de la classe PDFFile"""
        try:
            # Tente d'ouvrir le fichier PDF à partir du chemin donné
            self.file = pymupdf.open(path)
        except (FileExistsError, FileNotFoundError):
            # Lève une exception si le fichier n'existe pas ou n'est pas trouvé
            raise FileNotFoundError
        self.path = path  # Chemin du fichier PDF
        self._raw_images_dict = []  # Liste pour stocker les données brutes des images extraites
        self.grids: list[Grid] = []  # Liste des grilles générées à partir des images
        self.current_grid = None  # Index de la grille actuellement sélectionnée

    def extract_grids(self):
        """Extrait les images du PDF et les convertit en objets Grid"""
        self.extract_images()  # Extrait les données brutes des images
        images = self.get_cv_images()  # Convertit les données en images OpenCV
        if not self.grids:
            # Crée des objets Grid pour chaque image et les ajoute à la liste
            for image in images:
                self.grids.append(Grid(image))

    def get_current_grid(self):
        """Retourne la grille actuellement sélectionnée"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = 0  # Initialise à la première grille si aucune n'est sélectionnée
        return self.grids[self.current_grid]

    def get_next_grid(self):
        """Passe à la grille suivante et la retourne"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = 0  # Initialise à la première grille si aucune n'est sélectionnée
        else:
            self.current_grid += 1  # Passe à l'index suivant
            self.current_grid %= len(self.grids)  # Revient au début si la fin est atteinte
        return self.grids[self.current_grid]

    def get_previous_grid(self):
        """Passe à la grille précédente et la retourne"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = 0  # Initialise à la première grille si aucune n'est sélectionnée
        else:
            self.current_grid -= 1  # Passe à l'index précédent
            self.current_grid %= len(self.grids)  # Revient à la fin si le début est atteint
        return self.grids[self.current_grid]

    def extract_images(self):
        """
        Extrait les données brutes des images contenues dans le fichier PDF.
        Retourne une liste de dictionnaires contenant les données des images.
        """
        if self._raw_images_dict:
            return self._raw_images_dict  # Retourne les données si elles ont déjà été extraites
        biggest_image_dict = {'height': 0}  # Initialise un dictionnaire pour la plus grande image
        for i in range(len(self.file)):
            # Parcourt chaque page du PDF pour extraire les images
            images = self.file.load_page(i).get_images(True)
            for image in images:
                base_image_dict = self.file.extract_image(xref=image[0])
                # Sélectionne l'image avec la plus grande hauteur
                if base_image_dict["height"] > biggest_image_dict["height"]:
                    biggest_image_dict = base_image_dict
            self._raw_images_dict.append(biggest_image_dict)  # Ajoute l'image à la liste
        return self._raw_images_dict.copy()

    def save_images(self):
        """Sauvegarde les images extraites dans un répertoire local"""
        if not self._raw_images_dict:
            self.extract_images()  # Extrait les images si ce n'est pas déjà fait
        for i, image in enumerate(self._raw_images_dict):
            # Sauvegarde chaque image dans un fichier local
            f = open(f"images/img-{i}.{image['ext']}", "wb")
            f.write(image["image"])
            f.close()

    def get_cv_images(self):
        """Convertit les données brutes des images en objets d'image OpenCV"""
        if not self._raw_images_dict:
            self.extract_images()  # Extrait les images si ce n'est pas déjà fait
        images = []
        for image in self._raw_images_dict:
            # Convertit les données brutes en tableau numpy
            image_array = np.frombuffer(image["image"], dtype=np.uint8)
            # Décode le tableau en image OpenCV
            images.append(cv2.imdecode(image_array, cv2.IMREAD_COLOR))
        return images

    def __del__(self):
        """Ferme le fichier PDF lors de la destruction de l'objet"""
        self.file.close()