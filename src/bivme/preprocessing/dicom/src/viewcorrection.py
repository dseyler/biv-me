from doctest import master
import tkinter as tk
from tkinter import StringVar, filedialog, OptionMenu, ttk
import re
import os
from pathlib import Path
from csv import writer
from PIL import Image, ImageTk
import shutil
import threading
import pandas as pd
import numpy as np

LIST_OF_VIEWS = ['SAX', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RT', 'RVOT-T', 'SAX-atria', 'OTHER']

# Global variables


# Initialising Window and Child Frames
class VSGUIOLD:
    def __init__(self, patient, dst, viewSelector):
        self.patient = patient
        self.dst = dst
        self.view_predictions = pd.read_csv(viewSelector.csv_path)
        self.img_dict = {}
        self.viewSelector = viewSelector
        self.create_window()

    def create_window(self):
        self.window = tk.Tk()
        self.window.columnconfigure([0,1], weight = 1, minsize=150)
        self.window.rowconfigure([0,1], weight = 1, minsize=50)

        self.frame_header = tk.Frame(master=self.window, relief = tk.RAISED, borderwidth=1)
        self.frame_header.grid(column = 0, row = 0, columnspan = 2, sticky = tk.W + tk.E + tk.N)
        self.frame_header.anchor(tk.CENTER)

        self.frame_scrollable = tk.Frame(master=self.window, width = 350, height = 800)
        self.frame_scrollable.columnconfigure([0], weight=1)
        self.frame_scrollable.rowconfigure([0], weight=1)

        self.canvas_scrollable = tk.Canvas(self.frame_scrollable, width = 350, height = 800)
        self.scrollbar = ttk.Scrollbar(self.frame_scrollable, orient="vertical", command=self.canvas_scrollable.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_scrollable, width = 350, height = 800)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas_scrollable.configure(scrollregion=self.canvas_scrollable.bbox("all"))
        )

        self.canvas_scrollable.create_window((0,0), window=self.scrollable_frame, anchor="nw")

        self.canvas_scrollable.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_scrollable.bind_all("<MouseWheel>", self.on_mousewheel)

        self.frame_scrollable.grid(column = 0, row = 1)
        self.canvas_scrollable.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.frame_confirmation = tk.Frame(master=self.window, width = 500, height = 800)
        self.frame_confirmation.grid(column = 1, row = 1, sticky = tk.E)

        # Setup tkinter string variables which hold values for widgets
        current_patient_var = StringVar(master=self.frame_header)
        self.current_option_view = StringVar(master=self.frame_confirmation)
        grade_var = StringVar(master=self.frame_confirmation)

    def correct_views_gui(self):
        print(f'debug correct_views_gui')
        self.setup()
        self.load_patient_images()

    def setup(self):
        print('debug setup')
        # Initialise save button at the top
        btn_optn_confirm = tk.Button(master=self.frame_header, text="Save view predictions", command=lambda: self.save_corrections)
        btn_optn_confirm.grid(row=2, column=2)
        self.window.update()

    # Scroll left hand panel when mousewheel is used
    def on_mousewheel(self, event):
        scroll = -1 if event.delta > 0 else 1

        self.canvas_scrollable.yview_scroll(scroll, "units")

    # Load and display list of images for current patient
    def load_patient_images(self):
        print('debug load_patient_images')
        # Get directory for png images
        sorted_img_directory = Path(self.dst, 'view-classification', 'sorted')
        unsorted_img_directory = Path(self.dst, 'view-classification', 'unsorted')
            
        # Scroll back to the top when going to the next patient
        self.canvas_scrollable.yview_moveto('0.0')

        # Destroy all existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for widget in self.frame_confirmation.winfo_children():
            widget.destroy() 

        # Get list of all pngs
        all_imgs = os.listdir(unsorted_img_directory)
        # Format as full paths
        all_imgs = [os.path.join(unsorted_img_directory, i) for i in all_imgs if i.endswith('_0.png')]

        for i in all_imgs:
            series = list(os.path.basename(i).split('_'))
            series = int(series[0])  # Get the series number from the filename
            # Get view classification
            vp = self.view_predictions[self.view_predictions['Series Number'] == series]
            view = vp['Predicted View'].values[0]
            self.img_dict[i] = view

        list_of_images = []

        # For each file, convert it to a PIL image and display it in a Tkinter widget
        for i, img in enumerate(all_imgs):
            print(f"----- Loading image {img} ({i+1}/{len(all_imgs)})")

            # Load image
            image = Image.open(img)
            image = image.resize((100, int(image.size[1] * 100 / image.size[0])))
            
            image_tk = ImageTk.PhotoImage(image)
            list_of_images.append(image_tk)

            view_class = self.img_dict[img]
            series = int(os.path.basename(img).split('_')[0])


            btn_image = tk.Button(master=self.scrollable_frame, image = image_tk, relief=tk.RAISED, command=lambda: self.display_image_to_confirm(img, view_class, series))
            btn_image.anchor(tk.CENTER)
            btn_image.grid(column=0, row=i)

            self.window.update()

        # self.window.mainloop()

        
        print(f"----- All images loaded.\n")

    # This function displays a selected image on the right panel to confirm save
    def display_image_to_confirm(self, image_location, currentview, series):
        print(f"----- Displaying image {image_location} for confirmation.")
        # Load image into right hand panel
        image = Image.open(image_location)
        image = image.resize((500, int(image.size[1] * 500 / image.size[0])))
        image_tk = ImageTk.PhotoImage(image)
        lbl_image = tk.Label(master=self.frame_confirmation, image = image_tk)
        lbl_image.anchor(tk.CENTER)
        lbl_image.grid(column=0, row=0, columnspan=2)
        self.window.update()

        lbl_file_name = tk.Label(master=self.frame_confirmation, text = f"File name: {image_location}")
        lbl_file_name.grid(column=0, row=1, columnspan=2)

        # Display drop down with list of different views
        # Populate with current view
        self.current_option_view.set(currentview)
        optn_menu_views = ttk.Combobox(self.frame_confirmation, values=LIST_OF_VIEWS, textvariable=self.current_option_view, state="readonly")
        optn_menu_views.grid(row = 2, column=0, columnspan=2)

        # Give hint to user
        lbl_grade_hint = tk.Label(master=self.frame_confirmation, text="Select a view class for this series") 
        lbl_grade_hint.grid(row=3, column=0, columnspan=2)

        # Get selected option from the drop down menu
        selected_view = optn_menu_views.get()
        self.img_dict[image_location] = selected_view
        self.view_predictions.loc[self.view_predictions['Series Number'] == series, 'Predicted View'] = selected_view

        self.window.update()

    # This function saves the corrected predictions to the processing and states directories
    def save_corrections(self):
        # Save the corrected predictions to the CSV file
        self.view_predictions.to_csv(self.viewSelector.csv_path, index=False)

        # Add text confirmation to the header
        lbl_confirmation = tk.Label(master=self.frame_header, text="View predictions saved successfully!", fg="green")
        lbl_confirmation.grid(row=1, column=2, sticky=tk.W + tk.E)
        self.window.update()
        print("----- View predictions saved successfully!")


        self.window.mainloop()

class VSGUI:
    def __init__(self, patient, dst, viewSelector):
        self.patient = patient
        self.dst = dst
        self.view_predictions = pd.read_csv(viewSelector.csv_path)
        self.img_dict = {}
        self.viewSelector = viewSelector
        self.create_window()

    def create_window(self):
        self.window = tk.Tk()
        unique_series = self.view_predictions['Series Number'].unique()
        unique_series = sorted(unique_series, key=lambda x: int(x))  # Sort series numbers numerically
        num_series = len(unique_series)
        self.num_rows = 6
        self.num_cols = 8
        self.gridlayout = {}
        for i in range(0, self.num_rows):
            for j in range(self.num_cols):
                self.gridlayout[i * self.num_cols + j] = [i+1, j]
        # Create a grid layout for the window, with num_rows and num_cols
        self.series_mapping = {}

        for i,s in enumerate(unique_series):
            series = int(s)
            self.series_mapping[series] = i

        print(f"Grid layout: {self.gridlayout}")
        print(f"Series mapping: {self.series_mapping}")

        # Create a grid layout for the window, with num_rows and num_cols
        self.window.title("View Correction GUI")

    def correct_views_gui(self):
        print(f'debug correct_views_gui')

        # Initialise save button at the top
        btn_optn_confirm = tk.Button(master=self.window, text="Save view predictions", command=lambda: self.save_corrections)
        btn_optn_confirm.grid(row=0, column=0)

        # Get directory for png images
        sorted_img_directory = Path(self.dst, 'view-classification', 'sorted')
        unsorted_img_directory = Path(self.dst, 'view-classification', 'unsorted')

        # Get list of all pngs
        all_imgs = os.listdir(unsorted_img_directory)
        # Format as full paths
        all_imgs = [os.path.join(unsorted_img_directory, i) for i in all_imgs if i.endswith('_0.png')]

        for i in all_imgs:
            series = list(os.path.basename(i).split('_'))
            series = int(series[0])  # Get the series number from the filename
            # Get view classification
            vp = self.view_predictions[self.view_predictions['Series Number'] == series]
            view = vp['Predicted View'].values[0]
            self.img_dict[i] = view

        list_of_images = []

        # For each file, convert it to a PIL image and display it in a Tkinter widget
        for i, img in enumerate(all_imgs):
            print(f"----- Loading image {img} ({i+1}/{len(all_imgs)})")

            # Load image
            image = Image.open(img)
            image = image.resize((150, int(image.size[1] * 150 / image.size[0])))
            
            image_tk = ImageTk.PhotoImage(image)
            list_of_images.append(image_tk)

            view_class = self.img_dict[img]
            series = int(os.path.basename(img).split('_')[0])
            mapped_series = self.series_mapping[series]


            btn_image = tk.Button(master=self.window, image = image_tk, relief=tk.RAISED)
            btn_image.anchor(tk.CENTER)
            btn_image.grid(row=self.gridlayout[mapped_series][0], column=self.gridlayout[mapped_series][1])

            # Option menu for selecting view class

            # Display drop down with list of different views
            # Populate with current view
            currentview = StringVar(value=view_class)
            currentview.set(view_class)

            optn_menu_views = ttk.Combobox(self.window, values=LIST_OF_VIEWS, textvariable=currentview, state="readonly")
            optn_menu_views.grid(row=self.gridlayout[mapped_series][0], column=self.gridlayout[mapped_series][1])

        self.window.mainloop()
        print(f"----- All images loaded.\n")

        # Get selected option from the drop down menu
        selected_view = optn_menu_views.get()
        self.img_dict[img] = selected_view
        self.view_predictions.loc[self.view_predictions['Series Number'] == series, 'Predicted View'] = selected_view
        self.window.update()


    # This function saves the corrected predictions to the processing and states directories
    def save_corrections(self):
        # Save the corrected predictions to the CSV file
        self.view_predictions.to_csv(self.viewSelector.csv_path, index=False)

        # Add text confirmation to the header
        lbl_confirmation = tk.Label(master=self.window, text="View predictions saved successfully!", fg="green")
        lbl_confirmation.grid(row=1, column=2, sticky=tk.W + tk.E)
        self.window.update()
        print("----- View predictions saved successfully!")
