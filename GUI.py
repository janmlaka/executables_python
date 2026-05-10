from transformation_json import transformation_json_D48_D96
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk 

class App():
    def __init__(self, root):
        self.root = root

        self.input_file_D48 = None
        self.input_file_D96 = None

        D48_entry = ttk.Label(root, text="D48 coordinates:").grid(row=0, column=0)
        D96_entry = ttk.Label(root, text="D96 coordinates:").grid(row=0, column=1)

        D48_button = ttk.Button(root, text="D48 coordinates data", width=25, command=self.file_finder_D48).grid(row=1, column=0)
        D96_button = ttk.Button(root, text="D96 coordinates data", width=25, command=self.file_finder_D96).grid(row=1, column=1)

        call_function_button = ttk.Button(root, text="Transform data", width=50, command=self.calculate).grid(columnspan=2)
        
    def file_finder_D48(self):
        self.input_file_D48 = filedialog.askopenfilename()

    def file_finder_D96(self):
        self.input_file_D96 = filedialog.askopenfilename()

    def calculate(self):

        transformation_json_D48_D96(
            self.input_file_D48,
            self.input_file_D96
        )

root = ttk.Window(themename="darkly")
root.geometry("500x400")

app = App(root)

root.mainloop()
