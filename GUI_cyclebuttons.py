import customtkinter as ctk

class NavigationButtons(ctk.CTkFrame):
    def __init__(self, master, row, column, columnspan):
        super().__init__(master=master)
        # Keep outer padding of the frame
        self.grid(
            row=row, columnspan=columnspan, column=column,
            sticky="nsew", padx=7, pady=10
        )

        # Internal frame to tightly pack buttons and label
        inner_frame = ctk.CTkFrame(self, fg_color="transparent")
        inner_frame.pack(expand=True)  # center inside the outer frame

        # Previous button
        self.previous_image = ctk.CTkButton(inner_frame, text="<", width=40)
        self.previous_image.pack(side="left", padx=(0, 5))

        # Label
        self.current_pdf_grid_count_var = ctk.StringVar(value="0 / 0 images")
        self.current_pdf_grid_count = ctk.CTkLabel(
            inner_frame, textvariable=self.current_pdf_grid_count_var
        )
        self.current_pdf_grid_count.pack(side="left", padx=5)

        # Next button
        self.next_image = ctk.CTkButton(inner_frame, text=">", width=40)
        self.next_image.pack(side="left", padx=(5, 0))

    
    def reset_counter(self):
        self.current_pdf_grid_count_var.set("0 / 0 images")

    def set_command_functions(self, functions):
        self.previous_image.configure(command=functions[0])
        self.next_image.configure(command=functions[1])
