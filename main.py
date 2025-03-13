import customtkinter as ctk
from constants import *
from PIL import Image, ImageTk
from convert_pdf import PDFFile
from grid import Grid
import cv2
from processing import process_pdf
import openpyxl



class ImageViewer(ctk.CTkCanvas):
    def __init__(self, master, cv_image):
        super().__init__(master=master, bg="#1F1F1F")  # Set background in constructor

        if App.cv_image is None:
            App.cv_image = cv_image

        self.grid(column=0, columnspan=2, row=0, sticky="nsew")
        self.bind("<Configure>", self.resize_callback)

    @staticmethod
    def resize(width, height):
        if App.cv_image is None:
            return

        # Convert BGR (OpenCV) to RGB (PIL)
        b, g, r = cv2.split(App.cv_image)
        re_image = cv2.merge((r, g, b))
        image = Image.fromarray(re_image)

        # Maintain aspect ratio
        canvas_ar = width / height
        image_ar = App.cv_image.shape[1] / App.cv_image.shape[0]

        if image_ar > canvas_ar:
            new_width = width
            new_height = int(width / image_ar)
        else:
            new_width = int(height * image_ar)
            new_height = height

        resized_img = image.resize((new_width, new_height))

        # Store as Tkinter-compatible image
        App.resized_tk_img = ImageTk.PhotoImage(resized_img)

    def resize_callback(self, event):
        if App.cv_image is None:
            return

        width, height = self.winfo_width(), self.winfo_height()
        self.delete("all")
        ImageViewer.resize(width, height)
        self.create_image(
            width // 2, height // 2, image=App.resized_tk_img, anchor="center"
        )


class NavigationButtons(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master=master)
        self.grid(row=1, columnspan=2, column=0, sticky="ew", padx=7, pady=4)

        self.columnconfigure(0, weight=1, uniform="b")
        self.columnconfigure(1, weight=1, uniform="b")
        self.columnconfigure(2, weight=1, uniform="b")
        self.columnconfigure(3, weight=1, uniform="b")
        self.columnconfigure(4, weight=1, uniform="b")

        self.previous_file = ctk.CTkButton(self, text="<<")
        self.previous_file.grid(column=0, row=0, padx=10, pady=15)

        self.next_file = ctk.CTkButton(self, text=">>")
        self.next_file.grid(column=4, row=0, padx=20, pady=15)

        self.previous_image = ctk.CTkButton(self, text="<")
        self.previous_image.grid(column=1, row=0, padx=10, pady=15)

        self.next_image = ctk.CTkButton(self, text=">")
        self.next_image.grid(column=3, row=0, padx=10, pady=15)
        
    def set_command_functions(self,functions):
        self.previous_file.configure(command = functions[0])
        self.previous_image.configure(command = functions[1])
        self.next_image.configure(command = functions[2])
        self.next_file.configure(command = functions[3])


class App:
    resized_tk_img = None
    cv_image = None
    decoded_info = None

    def __init__(
        self,
        title: str,
        width: int,
        height: int,
        resizable: list[bool],
        icon_path: str = None,
    ):
        self.width = width
        self.height = height
        self.pdfs: list[PDFFile] = []
        self.current_pdf: int | None = None
        self.current_grid: Grid | None = None

        self.window = ctk.CTk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.resizable(resizable[0], resizable[1])
        self.window.title(title)

        self.window.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="a")
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=0)

        self.rhs_frame = ctk.CTkFrame(self.window)
        self.rhs_frame.grid(
            row=0, rowspan=2, column=2, columnspan=4, padx=4, pady=10, sticky="nsew"
        )

        self.rhs_frame.rowconfigure((0, 1, 2, 3, 4), weight=1, uniform="a")
        self.rhs_frame.columnconfigure((0, 1), weight=1, uniform="a")

        # Define StringVar variables
        self.grid_type = ctk.StringVar(value="N/A")
        self.detected_cols = ctk.StringVar(value="N/A")
        self.detected_rows = ctk.StringVar(value="N/A")
        self.score = ctk.StringVar(value="N/A")

        self.grid_type_label = ctk.CTkLabel(
            self.rhs_frame, text="Grid Type: ", textvariable=self.grid_type
        )
        self.detected_rows_label = ctk.CTkLabel(
            self.rhs_frame, text="Rows: ", textvariable=self.detected_rows
        )
        self.detected_cols_label = ctk.CTkLabel(
            self.rhs_frame, text="Columns: ", textvariable=self.detected_cols
        )
        self.score_label = ctk.CTkLabel(
            self.rhs_frame, text="Score: ", textvariable=self.score
        )
        self.export_button = ctk.CTkButton(self.rhs_frame, text="Exporter", command=self.save_file)
        self.add_file_button = ctk.CTkButton(
            self.rhs_frame, text="Ajouter", command=self.open_file
        )

        self.grid_type_label.grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w"
        )
        self.detected_rows_label.grid(
            row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w"
        )
        self.detected_cols_label.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w"
        )
        self.score_label.grid(
            row=3, column=0, padx=10, columnspan=2, pady=5, sticky="w"
        )
        self.export_button.grid(
            row=4, column=0, columnspan=1, padx=10, pady=10, sticky="ew"
        )
        self.add_file_button.grid(
            row=4, column=1, columnspan=1, padx=10, pady=10, sticky="ew"
        )

        self.rhs_frame.grid_columnconfigure(0, weight=1)

        self.bottom_bottons = NavigationButtons(self.window)
        self.bottom_bottons.set_command_functions([self.show_previous_pdf,self.show_previous_grid,self.show_next_grid,self.show_next_pdf])

        if icon_path:
            self.window.iconbitmap(icon_path)

        self.image_viewer = ImageViewer(self.window, App.cv_image)

    def update(self, data: tuple):
        App.decoded_info = data[1]
        App.cv_image = data[0]
        if App.decoded_info is not None:
            self.grid_type.set(value=f"Grille : {App.decoded_info["Type"]}")
            self.detected_rows.set(
                value=f"Lignes Détéctées: {App.decoded_info["Lines"]}"
            )
            self.detected_cols.set(
                value=f"Colonnes Détéctées: {App.decoded_info["Cols"]}"
            )
        self.score = ctk.StringVar(value="N/A")

        self.image_viewer.after(
            10, lambda: self.image_viewer.event_generate("<Configure>")
        )

    def _set_current_grid(self, event=None):
        data = self.current_grid.run_analysis()
        temp = [data["image"], data["qr_code_info"]]
        self.update(temp)

    def _show_latest_file(self, event=None):
        if self.current_pdf is None:
            self.current_pdf = 0
        else:
            self.current_pdf = len(self.pdfs) - 1
        self.pdfs[self.current_pdf].extract_grids()
        self.current_grid = self.pdfs[self.current_pdf].get_next_grid()
        self._set_current_grid()

    def show_next_pdf(self, event=None):
        if not self.pdfs or self.current_pdf is None:
            return 
        self.current_pdf = (self.current_pdf + 1) % len(self.pdfs)
        self.pdfs[self.current_pdf].extract_grids()
        self.current_grid = self.pdfs[self.current_pdf].get_next_grid()
        self._set_current_grid()
    
    def show_previous_pdf(self, event=None):
        if not self.pdfs or self.current_pdf is None:
            return 
        self.current_pdf = (self.current_pdf + 1) % len(self.pdfs)
        self.pdfs[self.current_pdf].extract_grids()
        self.current_grid = self.pdfs[self.current_pdf].get_previous_grid()
        self._set_current_grid()

    def show_next_grid(self, event=None):
        if not self.pdfs or self.current_pdf is None:
            return
        self.current_grid = self.pdfs[self.current_pdf].get_next_grid()
        self._set_current_grid()

    def show_previous_grid(self, event=None):
        if not self.pdfs or self.current_pdf is None:
            return
        self.current_grid = self.pdfs[self.current_pdf].get_previous_grid()
        self._set_current_grid()

    def open_file(self,event=None):
        file_paths = ctk.filedialog.askopenfilenames(
            title="Sélectionner un fichier", filetypes=[("Fichiers PDF", "*.pdf")]
        )
        for file_path in file_paths:
            self.pdfs.append(PDFFile(file_path))
        self._show_latest_file()
        
    def save_file(self,event=None):
        if self.current_grid is None:
            return
        
        file_path = ctk.filedialog.asksaveasfilename(title="Sauvegarder fichier excel",filetypes=[("Fichiers Excel", "*.xlsx")],defaultextension=".xlsx")
        if not file_path:
            return
        
        xl_original_grid_obj = openpyxl.load_workbook("documents/Grille.xlsx")
        xl_original_grid_obj.save(file_path)
        xl_original_grid_obj.close()
        
        checked_cells = self.current_grid.checked_cells
        xl_copy = openpyxl.load_workbook(file_path)
        active_sheet = xl_copy.active
        for i,row in enumerate(checked_cells,start=3):
            for j,cell in enumerate(row,start=4):
                active_sheet.cell(row=i, column=j).value="X" if cell > 0 else ""
        
        xl_copy.save(file_path)
        xl_copy.close()

    def run(self):
        self.update(process_pdf())
        self.window.mainloop()


app = App("Demo", WIDTH, HEIGHT, [True, True])
app.run()
