import pymupdf, cv2, numpy as np,fitz
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
        self._raw_images_dict = (
            []
        )  # Liste pour stocker les données brutes des images extraites
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

    def get_count(self):
        return len(self.grids)  # Retourne le nombre de grilles disponibles

    def get_current_grid(self):
        """Retourne la grille actuellement sélectionnée"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = (
                0  # Initialise à la première grille si aucune n'est sélectionnée
            )
        return (
            self.grids[self.current_grid],
            self.current_grid,
        )  # Retourne la grille actuelle et son index

    def get_next_grid(self):
        """Passe à la grille suivante et la retourne"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = (
                0  # Initialise à la première grille si aucune n'est sélectionnée
            )
        else:
            self.current_grid += 1  # Passe à l'index suivant
            self.current_grid %= len(
                self.grids
            )  # Revient au début si la fin est atteinte
        return (
            self.grids[self.current_grid],
            self.current_grid,
        )  # Retourne la grille actuelle et son index

    def get_previous_grid(self):
        """Passe à la grille précédente et la retourne"""
        if not self.grids:
            return None  # Retourne None si aucune grille n'est disponible
        if self.current_grid is None:
            self.current_grid = (
                0  # Initialise à la première grille si aucune n'est sélectionnée
            )
        else:
            self.current_grid -= 1  # Passe à l'index précédent
            self.current_grid %= len(
                self.grids
            )  # Revient à la fin si le début est atteint
        return (
            self.grids[self.current_grid],
            self.current_grid,
        )  # Retourne la grille actuelle et son index

    def extract_images(self):
        """
        Extrait l'image la plus grande par page si elle couvre une grande partie de la page.
        Sinon, effectue un rendu de la page à 300 PPI.
        """
        if self._raw_images_dict:
            return (
                self._raw_images_dict
            )  # Si les images sont déjà extraites, on les retourne directement
        for i in range(len(self.file)):
            page : pymupdf.Page = self.file.load_page(i)
            minimum_page_area = 0.7 * (page.rect.width * page.rect.height)
            images = page.get_images(full=True)
            max_area = 0
            xref_max = None
            # Parcourt toutes les images de la page
            for img in images:
                xref = img[0]
                for info in page.get_image_info(xref):
                    bbox = fitz.Rect(info["bbox"])
                    area = bbox.width * bbox.height
        
                    # Vérifie si l'image couvre au moins ~70% de la surface de la page
                    if area > max_area and area >= minimum_page_area:
                        max_area = area
                        xref_max = xref

            if xref_max is not None:
                # Une image suffisamment grande a été trouvée
                base_image_dict = self.file.extract_image(xref_max)
                self._raw_images_dict.append(base_image_dict)
            else:
                # Aucune image adéquate – on effectue un rendu de la page en 300 PPI
                pix = page.get_pixmap(dpi=300)
                image_bytes = pix.tobytes("png")
                self._raw_images_dict.append(
                    {
                        "image": image_bytes,
                        "width": pix.width,
                        "height": pix.height,
                        "ext": "png",
                    }
                )
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
