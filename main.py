import customtkinter as ctk, cv2, openpyxl
from constants import *
from GUI_warning import ConflictFrame, WarningFrame
from GUI_cyclebuttons import NavigationButtons
from PIL import Image, ImageTk
from convert_pdf import PDFFile
from grid import Grid
from styling import *


# Structure de l'application :
# - La classe `App` est le point d'entrée principal de l'application. Elle gère la fenêtre principale, les widgets, et les interactions utilisateur.
# - La classe `ImageViewer` est un widget personnalisé pour afficher et interagir avec des images (zoom, déplacement, etc.).
# - Les classes `ConflictFrame` et `WarningFrame` sont des cadres personnalisés pour afficher les conflits et les avertissements détectés dans les grilles.
# - `NavigationButtons` gère les boutons de navigation pour parcourir les fichiers PDF et les grilles.
# - `PDFFile` est une classe qui gère l'extraction d'images à partir de fichiers PDF et la conversion de ces images en grilles exploitables.
# - `Grid` est une classe qui représente une grille extraite d'une image PDF et fournit des méthodes pour analyser et manipuler cette grille.
# - `openpyxl` est utilisé pour manipuler des fichiers Excel, permettant de sauvegarder les grilles modifiées.
# - `constants` contient des constantes utilisées dans l'application (comme les couleurs, les polices, etc.).
# - `styling` contient des styles et des thèmes pour l'interface utilisateur.
# - `convert_pdf` contient la classe `PDFFile` qui gère l'extraction d'images à partir de fichiers PDF.

# Fonctionnalités principales :
# - Chargement et navigation dans des fichiers PDF.
# - Extraction et affichage de grilles détectées dans les PDF.
# - Interaction avec les grilles (sélection de cellules, affichage des conflits, etc.).
# - Sauvegarde des grilles modifiées dans un fichier Excel.
# - Affichage d'images avec des fonctionnalités de zoom et de déplacement.

# Points clés :
# - Les widgets CustomTkinter sont utilisés pour créer une interface moderne et personnalisée.
# - OpenCV est utilisé pour manipuler les images (redimensionnement, conversion de couleurs, etc.).
# - La navigation entre les fichiers PDF et les grilles est gérée par des fonctions utilitaires centralisées.
# - Les événements utilisateur (clics, survols, etc.) sont liés à des callbacks pour des interactions dynamiques.

# Explication des fonctions :
# - `App.__init__`: Initialise l'application, configure la fenêtre principale et les widgets.
# - `App.run`: Lance la boucle principale de l'application.
# - `App.open_file`: Permet de sélectionner et charger des fichiers PDF.
# - `App.save_file`: Sauvegarde les grilles modifiées dans un fichier Excel.
# - `App.update`: Met à jour l'interface utilisateur avec les données actuelles.
# - `App.show_next_pdf` et `App.show_previous_pdf`: Naviguent entre les fichiers PDF.
# - `App.show_next_grid` et `App.show_previous_grid`: Naviguent entre les grilles d'un PDF.
# - `ImageViewer.__init__`: Configure le widget pour afficher des images avec des fonctionnalités de zoom et de déplacement.
# - `ImageViewer.zoom`: Gère le zoom avant/arrière sur l'image.
# - `ImageViewer.pan`: Permet de déplacer l'image dans le widget.
# - `ImageViewer.redraw_image`: Redessine l'image après un zoom ou un déplacement.
# - `ImageViewer.resize`: Redimensionne l'image pour s'adapter au widget.


class ImageViewer(ctk.CTkCanvas):
    def __init__(self, master, row, column, rowspan, columnspan, cv_image):
        super().__init__(master=master, bg=offwhite)

        if App.cv_image is None:
            App.cv_image = cv_image

        self.grid(
            column=column,
            columnspan=columnspan,
            row=row,
            rowspan=rowspan,
            sticky="nsew",
        )

        # Zoom and Pan Variables
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.last_x = 0
        self.last_y = 0
        self.resize_job = None  # For debouncing

        # Bind Events
        self.bind("<Configure>", self.resize_callback)

        self.bind("<MouseWheel>", self.zoom)  # Windows + trackpad
        self.bind("<Control-MouseWheel>", self.zoom)  # macOS trackpad
        self.bind("<Button-4>", self.zoom)  # Linux scroll up
        self.bind("<Button-5>", self.zoom)  # Linux scroll down

        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<ButtonRelease-1>", self.end_pan)

        # Set initial image size based on canvas size
        self.initial_scale = 1.0

    def resize(self, width, height):
        if App.cv_image is None:
            return

        # Resize image using OpenCV (faster than PIL)
        new_size = (
            max(1, int(App.cv_image.shape[1] * self.scale)),
            max(1, int(App.cv_image.shape[0] * self.scale)),
        )
        resized_cv_img = cv2.resize(
            App.cv_image, new_size, interpolation=cv2.INTER_AREA
        )

        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(resized_rgb)
        App.resized_tk_img = ImageTk.PhotoImage(image)

    def resize_callback(self, event):
        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(100, self._do_resize)

    def _do_resize(self, keep_scale=False):
        if App.cv_image is None:
            return
        width, height = self.winfo_width(), self.winfo_height()

        # Adjust scale to fit image into canvas
        if not keep_scale:
            self.scale = min(
                width / App.cv_image.shape[1], height / App.cv_image.shape[0]
            )
        self.delete("all")
        self.resize(width, height)
        self.redraw_image()

    def redraw_image(self):
        if App.resized_tk_img:
            self.delete("all")
            self.create_image(
                self.winfo_width() // 2 + self.offset_x,
                self.winfo_height() // 2 + self.offset_y,
                image=App.resized_tk_img,
                anchor="center",
            )

    def zoom(self, event):
        # Determine zoom direction
        zoom_factor = 1.1 if event.delta > 0 else 0.9

        # New scale calculation
        new_scale = self.scale * zoom_factor

        # Apply zoom limits between 0.2 and 0.55
        if 0.1 <= new_scale <= 0.55:
            self.scale = new_scale
            self.resize(self.winfo_width(), self.winfo_height())

        # Clamp pan offsets to avoid the image going out of bounds
        self.clamp_offsets()
        self.redraw_image()

    def start_pan(self, event):
        """Records the starting point of a pan."""
        self.last_x = event.x
        self.last_y = event.y

    def pan(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y

        self.offset_x += dx
        self.offset_y += dy

        self.clamp_offsets()
        self.redraw_image()

        self.last_x = event.x
        self.last_y = event.y

    def clamp_offsets(self):
        """Clamp pan so image doesn't go out of bounds."""
        if App.resized_tk_img is None:
            return

        img_w = App.resized_tk_img.width()
        img_h = App.resized_tk_img.height()
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        max_x = max(0, (img_w - canvas_w) // 2)
        max_y = max(0, (img_h - canvas_h) // 2)

        self.offset_x = max(-max_x, min(max_x, self.offset_x))
        self.offset_y = max(-max_y, min(max_y, self.offset_y))

    def end_pan(self, event=None):
        pass


class App:
    resized_tk_img = None
    cv_image = None
    decoded_info = None
    hover = None

    def __init__(
        self,
        title: str,
        width: int,
        height: int,
        resizable: list[bool],
        icon_path: str = None,
    ):
        ctk.set_appearance_mode("light")
        self.width = width
        self.height = height
        self.pdfs: list[PDFFile] = []
        self.current_pdf_index: int | None = None
        self.current_grid: Grid | None = None

        self.window = ctk.CTk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.minsize(width, height)
        self.window.resizable(resizable[0], resizable[1])
        self.window.title(title)

        self.window.columnconfigure((0, 1), weight=2, uniform="b")
        self.window.columnconfigure((2, 3, 4), weight=1, uniform="b")
        self.window.rowconfigure(0, weight=2, uniform="b")
        self.window.rowconfigure(5, weight=2, uniform="b")
        self.window.rowconfigure(1, weight=0, uniform="b")
        self.window.rowconfigure((2, 3, 4), weight=3, uniform="b")

        logo_eilco = Image.open("assets/EILCO.png")
        aspect_ratio_eilco = logo_eilco.width / logo_eilco.height
        eilco_dimensions = (int(logo_eilco_height * aspect_ratio_eilco), logo_eilco_height)
        self.eilco_logo = ctk.CTkImage(light_image=logo_eilco, size=eilco_dimensions)
        logo_eilco_label = ctk.CTkLabel(self.window, image=self.eilco_logo, text="")

        logo_ulco = Image.open("assets/ULCO.png")
        aspect_ratio_ulco = logo_ulco.width / logo_ulco.height
        ulco_dimensions = (int(logo_ulco_height * aspect_ratio_ulco), logo_ulco_height)
        self.ulco_logo = ctk.CTkImage(light_image=logo_ulco, size=ulco_dimensions)
        logo_ulco_label = ctk.CTkLabel(self.window, image=self.ulco_logo, text="")

        self.rhs_frame = ctk.CTkFrame(self.window)
        self.rhs_frame.grid(
            row=0, rowspan=6, column=2, columnspan=3, padx=4, pady=10, sticky="nsew"
        )

        self.rhs_frame.rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1, uniform="a")
        self.rhs_frame.rowconfigure(6, weight=0)
        self.rhs_frame.columnconfigure((0, 1), weight=1, uniform="a")

        # Define StringVar variables
        self.grid_type = ctk.StringVar(value="Type de Grille: N/A")
        self.score = ctk.StringVar(value="Score : N/A")
        self.current_pdf_var = ctk.StringVar(value="Aucun fichier PDF ouvert")

        self.grid_type_label = ctk.CTkLabel(
            self.rhs_frame, textvariable=self.grid_type, font=font1
        )
        self.score_label = ctk.CTkLabel(
            self.rhs_frame, text="Score: ", textvariable=self.score, font=font1
        )
        self.current_pdf_label = ctk.CTkLabel(
            self.window,
            text="Fichier PDF: ",
            font=font1,
            textvariable=self.current_pdf_var,
        )
        self.show_detected_cells = ctk.CTkCheckBox(
            self.rhs_frame, text="Vue Détection", command=self.draw_detected_cells
        )
        self.add_file_button = ctk.CTkButton(
            self.rhs_frame, text="Ajouter", command=self.open_file
        )
        self.bottom_bottons = NavigationButtons(self.window, 5, 0, 2)
        self.image_viewer = ImageViewer(self.window, 2, 0, 3, 2, App.cv_image)
        self.confilct_frame = ConflictFrame(self.rhs_frame, row=1, row_span=2)
        self.warning_frame = WarningFrame(self.rhs_frame, row=3, row_span=2)

        logo_ulco_label.grid(row=0, column=1, padx=30, pady=6, sticky="ne")
        logo_eilco_label.grid(row=0, column=0, padx=30, pady=6, sticky="nw")
        self.grid_type_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew"
        )
        self.score_label.grid(
            row=5, column=0, padx=10, columnspan=2, pady=5, sticky="nsew"
        )
        self.show_detected_cells.grid(
            row=6, column=0, columnspan=1, padx=10, pady=10, sticky="ew"
        )
        self.add_file_button.grid(
            row=6, column=1, columnspan=1, padx=10, pady=10, sticky="ew"
        )
        self.current_pdf_label.grid(
            row=1, column=0, columnspan=2, padx=4, pady=0, sticky="ew"
        )

        self.rhs_frame.grid_columnconfigure(0, weight=1)

        self.bottom_bottons.set_command_functions(
            [
                self.show_previous_pdf,
                self.show_previous_grid,
                self.show_next_grid,
                self.show_next_pdf,
            ]
        )

        self.cell_buttons_callback = {
            "on_click": self._button_on_click,
            "on_hover": self._button_on_hover,
            "on_leave": self._button_on_leave,
        }

        if icon_path:
            self.window.iconbitmap(icon_path)

    def update(self, data: tuple = None):
        if self.current_grid is not None:
            self.current_pdf_var.set(
                f"Fichier PDF: {self.pdfs[self.current_pdf_index].path.split('/')[-1]}"
            )
        self.confilct_frame.clear()
        self.warning_frame.clear()
        self.update_displayed_score()
        if data is None:
            return
        App.cv_image = data["image"]
        self.grid_type.set(value=data["type"].value)
        self.image_viewer.after(
            10, lambda: self.image_viewer.event_generate("<Configure>")
        )
        self.draw_detected_cells()
        problematic_rows: dict = self.current_grid.get_problematic_cells_per_row()
        self.confilct_frame.add_button_frames(
            problematic_rows, self.cell_buttons_callback
        )

        self.warning_frame.add_warnings(self.current_grid.get_warnings_errors()["warnings"])

        if not self.confilct_frame.button_frames:
            self.update_displayed_score(self.current_grid.calculate_score())

    def update_displayed_score(self, value: str | float = "N/A"):
        if isinstance(value, str):
            self.score.set(value=f"Score: {value}")
        else:
            self.score.set(value=f"Score: {value:.2f}")

    def _set_current_grid(self, grid_index, grid_count, event=None):
        self.bottom_bottons.current_pdf_grid_count_var.set(
            f"{grid_index + 1} / {grid_count} images"
        )
        self.update(self.current_grid.run_analysis())
        
    def _load_pdf_and_grid(self, index_func=None, grid_func=None):
        """
        Fonction utilitaire centrale permettant de charger un fichier PDF et une grille.
        Elle est utilisée pour éviter la répétition de code dans les fonctions de navigation.
        - `index_func` : fonction pour déterminer l'index du PDF à charger (ex: suivant, précédent…)
        - `grid_func` : fonction pour déterminer quelle grille afficher (ex: grille suivante…)
        """
        if not self.pdfs:
            return  # Aucun fichier PDF chargé

        if self.current_pdf_index is None:
            self.current_pdf_index = 0  # On démarre avec le premier fichier
        elif index_func:
            self.current_pdf_index = index_func(self.current_pdf_index)  # Applique la logique d’index

        pdf: PDFFile = self.pdfs[self.current_pdf_index]
        pdf.extract_grids()  # Extraction des grilles du fichier

        if grid_func:
            self.current_grid, grid_index = grid_func(pdf)  # Grille définie via une fonction
        else:
            self.current_grid, grid_index = pdf.get_next_grid()  # Par défaut, la grille suivante

        self._set_current_grid(grid_index, pdf.get_count())  # Mise à jour de l’interface

    def _show_latest_file(self, event=None):
        """
        Affiche le dernier fichier PDF de la liste.
        Utilisé pour charger le fichier le plus récent ou le dernier dans l’ordre.
        """
        self._load_pdf_and_grid(index_func=lambda _: len(self.pdfs) - 1)


    def show_next_pdf(self, event=None):
        """
        Passe au fichier PDF suivant dans la liste.
        Fait une rotation circulaire : après le dernier, revient au premier.
        """
        self._load_pdf_and_grid(index_func=lambda i: (i + 1) % len(self.pdfs))
    
    def show_previous_pdf(self, event=None):
        """
        Passe au fichier PDF précédent dans la liste.
        Fait une rotation circulaire : avant le premier, passe au dernier.
        """
        self._load_pdf_and_grid(index_func=lambda i: (i - 1) % len(self.pdfs))

    def show_next_grid(self, event=None):
        """
        Affiche la grille suivante du PDF courant.
        Utile pour naviguer à travers les grilles d’un même fichier.
        """
        self._load_pdf_and_grid(grid_func=lambda pdf: pdf.get_next_grid())



    def show_previous_grid(self, event=None):
        """
        Affiche la grille précédente du PDF courant.
        Permet de revenir en arrière dans la navigation des grilles.
        """
        self._load_pdf_and_grid(grid_func=lambda pdf: pdf.get_previous_grid())


    def open_file(self, event=None):
        file_paths = ctk.filedialog.askopenfilenames(
            title="Sélectionner un fichier", filetypes=[("Fichiers PDF", "*.pdf")]
        )
        if file_paths:
            for file_path in file_paths:
                self.pdfs.append(PDFFile(file_path))
            self._show_latest_file()

    def save_file(self, event=None):
        if self.current_grid is None:
            return

        file_path = ctk.filedialog.asksaveasfilename(
            title="Sauvegarder fichier excel",
            filetypes=[("Fichiers Excel", "*.xlsx")],
            defaultextension=".xlsx",
        )
        if not file_path:
            return

        xl_original_grid_obj = openpyxl.load_workbook("documents/Grille.xlsx")
        xl_original_grid_obj.save(file_path)
        xl_original_grid_obj.close()

        checked_cells = self.current_grid.cells_state
        xl_copy = openpyxl.load_workbook(file_path)
        active_sheet = xl_copy.active
        for i, row in enumerate(checked_cells, start=3):
            for j, cell in enumerate(row, start=4):
                active_sheet.cell(row=i, column=j).value = "X" if cell[0] > 0 else ""

        xl_copy.save(file_path)
        xl_copy.close()

    def run(self):
        self.update()
        self.window.mainloop()

    def _button_on_click(self, button_frame, row, cols, event=None):
        self.current_grid.set_selected_cell(row, cols)
        self.confilct_frame.destroy_frame(button_frame)
        if not self.confilct_frame.button_frames:
            self.update_displayed_score(self.current_grid.calculate_score())
        self._button_on_leave(row, cols)

    def _button_on_hover(self, row, cols, event=None):
        self.current_grid.highlight_row(row, cols)
        App.cv_image = self.current_grid.image_annotee
        self.image_viewer._do_resize(keep_scale=True)

    def _button_on_leave(self, row, cols, event=None):
        self.current_grid.clear_image(self.show_detected_cells.get())
        App.cv_image = self.current_grid.image_annotee
        self.image_viewer._do_resize(keep_scale=True)

    def draw_detected_cells(self, event=None):
        if self.current_grid is None:
            return
        self.current_grid.clear_image(self.show_detected_cells.get())
        App.cv_image = self.current_grid.image_annotee
        self.image_viewer._do_resize(keep_scale=True)


app = App(APP_NAME, WIDTH, HEIGHT, [True, True],icon_path="assets/logo_app.ico")
app.run()
