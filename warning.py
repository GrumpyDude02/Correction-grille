import customtkinter as ctk

class GridButtonFrame(ctk.CTkFrame):
    def __init__(self, master, row, cols ,call_back_functions, buttons_per_row=1, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="orange")
        self.row = row

        self.buttons_per_row = buttons_per_row
        self.buttons: list[Button] = []

        self.label = ctk.CTkLabel(self, text=f'Ligne {self.row+1}', font=("Arial", 12, "bold"))
        self.label.grid(
            row=0, column=0, columnspan=self.buttons_per_row, sticky="nsew", pady=(0, 5)
        )
        
        self.add_buttons(cols,call_back_functions)

    def add_buttons(self, cols_indecies: list[list], call_back_functions:dict[callable]):

        for i, indecies in enumerate(cols_indecies, start=1):
            self.buttons.append(Button(self, indecies, call_back_functions))
            row = (i - 1) // self.buttons_per_row + 1
            col = (i - 1) % self.buttons_per_row
            self.buttons[-1].grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.buttons[-1].configure()
        for col in range(self.buttons_per_row):
            self.columnconfigure(col, weight=1)


class Button(ctk.CTkButton):
    def __init__(
        self, master: GridButtonFrame, cell_cols: list, call_back_functions, **kwargs
    ):

        self.cols = cell_cols
        self.on_click_call_back = call_back_functions["on_click"]
        self.on_hover_call_back = call_back_functions["on_hover"]
        self.on_leave_call_back = call_back_functions["on_leave"]

        if len(self.cols) > 1:
            txt= f"{int((sum(self.cols)*0.5)*20)}%"
        else:
            txt = f"{self.cols[0]*20}%"

        super().__init__(master, **kwargs, text=txt, command=self.on_click)
        self.bind("<Enter>", self.on_hover)

    def on_click(self, event=None):
        self.on_click_call_back(self.master,self.master.row, self.cols)

    def on_hover(self, event=None):
        self.on_hover_call_back(self.master.row, self.cols)

    def on_leave(self, event=None):
        self.on_leave_call_back()


class WarningFrame(ctk.CTkFrame):
    def __init__(self, master, label="Avertissements", row=2):
        super().__init__(master=master)
        self.configure(fg_color="black")
        self.grid(
            row=row, rowspan=2, columnspan=2, column=0, sticky="nsew", padx=7, pady=4
        )
        self.button_frames = []

        self.rowconfigure(0, weight=0)
        self.rowconfigure((1, 2), weight=1)
        self.columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text=label)
        self.label.grid(column=0, row=0, rowspan=1, columnspan=1, sticky="nsew")

        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.grid(column=0, row=1, rowspan = 2, columnspan=1, sticky="nsew", padx=4,pady =(0,4))
        
        self.button_frames:list[GridButtonFrame] =[]

    def add_button_frame(self, problematic_rows:dict,functions_callback_dict:dict):
        for row, cols in problematic_rows.items():
            self.button_frames.append(GridButtonFrame(self.scrollable_frame,row,cols,functions_callback_dict))
        self.button_frames.reverse()
    
    def show_button_frames(self):
        for button_frame in self.button_frames:
            button_frame.pack(fill="x", padx=5, pady=(5, 5))
            
    def destroy_frame(self,button_frame:GridButtonFrame):
        self.button_frames.remove(button_frame)
        button_frame.destroy()
        
    def clear(self):
        for button_frame in self.button_frames:
            button_frame.destroy()
        self.button_frames.clear()
