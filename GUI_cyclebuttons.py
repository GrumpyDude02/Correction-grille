import customtkinter as ctk

class NavigationButtons(ctk.CTkFrame):
    def __init__(self, master,row,column,columnspan):
        super().__init__(master=master)
        self.grid(row=row, columnspan=columnspan, column=column, sticky="ew", padx=7, pady=4)

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

    def set_command_functions(self, functions):
        self.previous_file.configure(command=functions[0])
        self.previous_image.configure(command=functions[1])
        self.next_image.configure(command=functions[2])
        self.next_file.configure(command=functions[3])

