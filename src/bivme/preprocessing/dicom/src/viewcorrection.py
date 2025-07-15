from doctest import master
import tkinter as tk
from tkinter import StringVar, ttk
import os
from pathlib import Path
from csv import writer
from PIL import Image, ImageTk
import pandas as pd
from idlelib.tooltip import Hovertip
import numpy as np

LIST_OF_VIEWS = ['SAX', '2ch', '3ch', '4ch', 'RVOT', 'LVOT', '2ch-RT', 'RVOT-T', 'SAX-atria', 'OTHER', 'Excluded']

class VSGUI:
    def __init__(self, patient, dst, viewSelector, my_logger):
        self.patient = patient
        self.dst = dst
        self.view_predictions = pd.read_csv(viewSelector.csv_path)
        self.img_dict = {}
        self.viewSelector = viewSelector
        self.my_logger = my_logger
        self.create_window()

    def create_window(self):
        self.window = tk.Tk()
        self.window.geometry("1280x900")  # Set the window size based on screen size
        self.window.resizable(width=True, height=True)  # Allow resizing of the window

        unique_series = self.view_predictions['Series Number'].unique()
        unique_series = sorted(unique_series, key=lambda x: int(x))  # Sort series numbers numerically

        self.num_rows = 6*2 + 2 # Doubled to allow for buttons above each image
        self.num_cols = 8
        self.gridlayout = {}

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Only put images on odd rows
                self.gridlayout[i * self.num_cols + j] = (2*i+2, j)

        # Create a grid layout for the window, with num_rows and num_cols
        self.series_mapping = {}

        for i,s in enumerate(unique_series):
            series = int(s)
            self.series_mapping[series] = i

        # Create a grid layout for the window, with num_rows and num_cols
        self.window.title("View Correction GUI")

        self.window.after(1000, lambda: self.window.focus_force())

    # This function saves the corrected predictions to the processing and states directories
    def save_corrections(self):
        self.my_logger.info("----- Saving view predictions...")
        # Get selected option from the drop down menu
        for i, dropdown in enumerate(self.list_of_dropdowns):
            selected_view = dropdown.get()
            series = self.series[i]
            img = f'{series}_0.png'  # Construct image filename from series number
            self.img_dict[img] = selected_view
            self.view_predictions.loc[self.view_predictions['Series Number'] == series, 'Predicted View'] = selected_view

            self.my_logger.info(f"----- Series {series} corrected to {selected_view}.")

        # Save the corrected predictions to the CSV file
        self.view_predictions.to_csv(self.viewSelector.csv_path, index=False)

        # Add text confirmation to the header
        lbl_confirmation = tk.Label(master=self.window, text="View predictions saved successfully!", fg="green")
        lbl_confirmation.grid(row=0, column=5, sticky=tk.W + tk.E)
        self.my_logger.success("----- View predictions saved successfully. Close the window to continue.")

    def correct_views_gui(self):
        # Initialise save button at the top
        btn_optn_confirm = tk.Button(master=self.window, text="Save view predictions", command=self.save_corrections)
        btn_optn_confirm.grid(row=0, column=4)

        # Get directory for png images
        unsorted_img_directory = Path(self.dst, 'view-classification', 'unsorted')

        # Get list of all pngs
        all_imgs = os.listdir(unsorted_img_directory)
        # Format as full paths
        all_imgs = [os.path.join(unsorted_img_directory, i) for i in all_imgs if i.endswith('_0.png')]

        stringvars = []
        confidences = []
        descriptions = []
        locations = []
        self.series = []

        for i in all_imgs:
            series = list(os.path.basename(i).split('_'))
            series = int(series[0])  # Get the series number from the filename
            # Get view classification
            vp = self.view_predictions[self.view_predictions['Series Number'] == series]
            view = vp['Predicted View'].values[0]

            stringvars.append(StringVar(master=self.window, value=view))  # Create a StringVar for each image view class
            self.series.append(series)

            confidence = vp['Confidence'].values[0]
            confidences.append(confidence)

            description = vp['Series Description'].values[0]
            descriptions.append(description)

            location = vp['Slice Location'].values[0]
            locations.append(location)

        self.list_of_images = []
        self.list_of_dropdowns = []

        # For each file, convert it to a PIL image and display it in a Tkinter widget
        for i, img in enumerate(all_imgs):
            # Load image
            image = Image.open(img)
            image = image.resize((150, int(image.size[1] * 150 / image.size[0])))
            
            image_tk = ImageTk.PhotoImage(image)
            self.list_of_images.append(image_tk)

            series = int(os.path.basename(img).split('_')[0])
            mapped_series = self.series_mapping[series]
            
            confidence = confidences[i]
            if confidence < 0.66:
                color = 'red'
            else:
                color = 'green'

            # Create a label with the image
            lbl_image = tk.Label(master=self.window, image = image_tk, highlightcolor=color, highlightbackground=color,
                                  highlightthickness=1, relief=tk.RAISED)
            lbl_image.anchor(tk.CENTER)
            lbl_image.grid(row=self.gridlayout[mapped_series][0], column=self.gridlayout[mapped_series][1])

            Hovertip(lbl_image, f'Series {series}, Confidence: {confidences[i]:.2f}, Description: {descriptions[i]}, Location: {locations[i]:.2f}')

            # Display drop down with list of different views
            # Populate with current view
            self.list_of_dropdowns.append(ttk.Combobox(self.window, values=LIST_OF_VIEWS, textvariable=stringvars[i], state="readonly"))
            self.list_of_dropdowns[-1].grid(row=self.gridlayout[mapped_series][0]-1, column=self.gridlayout[mapped_series][1])

        # Configure grids
        for i in range(self.num_rows):
            self.window.grid_rowconfigure(i, weight=1)
        for j in range(self.num_cols):
            self.window.grid_columnconfigure(j, weight=1)

        self.window.mainloop()