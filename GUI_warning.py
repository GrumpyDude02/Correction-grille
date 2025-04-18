import customtkinter as ctk
from styling import *

class GridButtonFrame(ctk.CTkFrame):
    def __init__(self, master, row, cols, call_back_functions, buttons_per_row=2, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=blue_notification)
        self.row = row
        self.buttons_per_row = buttons_per_row
        self.buttons: list[Button] = []

        # Add top border (row 0)
        self.top_border = ctk.CTkFrame(self, height=2, fg_color=light_blue)
        self.top_border.grid(row=0, column=0, columnspan=self.buttons_per_row, sticky="ew",pady=(1,0),padx=3)

        # Label (row 1)
        self.label = ctk.CTkLabel(self, text=f'Ligne {self.row+1}', font=font1)
        self.label.grid(
            row=1, column=0, columnspan=self.buttons_per_row, sticky="nsew", pady=(5, 5), padx=4
        )

        # Add buttons below (starting at row 2)
        self.add_buttons(cols, call_back_functions)

    def add_buttons(self, cols_indecies: list[list], call_back_functions: dict[callable]):
        for i, indecies in enumerate(cols_indecies, start=1):
            self.buttons.append(Button(self, indecies, call_back_functions, font=font1))
            row = (i - 1) // self.buttons_per_row + 2  # row offset +2 (border + label)
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
        self.bind("<Leave>", self.on_leave)

    def on_click(self, event=None):
        self.on_click_call_back(self.master,self.master.row, self.cols)

    def on_hover(self, event=None):
        self.on_hover_call_back(self.master.row, self.cols)

    def on_leave(self, event=None):
        self.on_leave_call_back(None, None)

class InteractiveFrame(ctk.CTkFrame):
    def __init__(self, master, label, color, row_span=2, row=1,**kwargs):
        super().__init__(master=master,**kwargs,fg_color=color)
        self.grid(
            row=row, rowspan=row_span, columnspan=2, column=0, sticky="nsew", padx=7, pady=4
        )
        self.rowconfigure(0, weight=0)
        self.rowconfigure((1, 2), weight=1)
        self.columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(self, text=label, font=("Arial", 16, "bold"))
        self.label.grid(column=0, row=0, rowspan=1, columnspan=1, sticky="nsew",pady=4)

        self.scrollable_frame = ctk.CTkScrollableFrame(self,fg_color="#efefef")
        self.scrollable_frame.grid(column=0, row=1, rowspan = 2, columnspan=1, sticky="nsew", padx=4,pady =(0,4))

class ConflictFrame(InteractiveFrame):
    def __init__(self, master,row_span=2, row=1,**kwargs):
        super().__init__(master=master,label="Conflits Détectés",color=light_blue,**kwargs)
        self.grid(
            row=row, rowspan=row_span, columnspan=2, column=0, sticky="nsew", padx=7, pady=4
        )
        self.button_frames = []
        self.button_frames:list[GridButtonFrame] =[]

    def add_button_frames(self, problematic_rows:dict,functions_callback_dict:dict):
        for row, cols in problematic_rows.items():
            self.button_frames.append(GridButtonFrame(self.scrollable_frame,row,cols,functions_callback_dict))
        self.button_frames.reverse()
        for button_frame in self.button_frames:
            button_frame.pack(fill="x", padx=5, pady=(5, 5))
          
    def destroy_frame(self,button_frame:GridButtonFrame):
        self.button_frames.remove(button_frame)
        button_frame.destroy()
        
    def clear(self):
        for button_frame in self.button_frames:
            button_frame.destroy()
        self.button_frames.clear()
        
class WarningFrame(InteractiveFrame):
    class WarningNotification(ctk.CTkFrame):
        def __init__(self, master, warning:str, **kwargs):
            super().__init__(master,fg_color=orange_notification, **kwargs)
            self.top_border = ctk.CTkFrame(self, height=2, fg_color=light_orange)
            self.top_border.grid(row=0, column=0, sticky="ew",pady=(1,0),padx=3)
            self.label = ctk.CTkLabel(self, text=warning, font=font1,wraplength=self.master.winfo_width()+10)
            self.label.grid(
                row=1, column=0, sticky="nsew", pady=(8, 8), padx=4)
            
            self.master.bind("<Configure>", self.update_wrap)
            
        def update_wrap(self,event = None):
                self.label.configure(wraplength=event.width - 20)
            
    
    def __init__(self, master,row_span=2, row=1,**kwargs):
        super().__init__(master=master,label="Avertissements",color=light_orange,**kwargs)
        self.configure(fg_color=light_orange)
        self.grid(
            row=row, rowspan=row_span, columnspan=2, column=0, sticky="nsew", padx=7, pady=4
        )
        self.warning_notifications : list[WarningFrame.WarningNotification] = []
        
    def add_warnings(self, warnings:str):
        for warning in warnings:
            self.warning_notifications.append(self.WarningNotification(self.scrollable_frame,warning))
            self.warning_notifications[-1].pack(fill="x", padx=4, pady=(5, 5))
    
    def clear(self):
        for warning_notification in self.warning_notifications:
            warning_notification.destroy()
        self.warning_notifications.clear()
