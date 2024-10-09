from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import nibabel as nib
import numpy as np
import gradio as gr
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events

class NiftiData(GradioModel):
    nifti_file: FileData


class NiftiViewer:
    """
    Class to handle displaying 3D medical images in .nii.gz format.
    The class allows the user to navigate through different slices of the 3D volume using a slider.
    """

    def __init__(self):
        self.data = None
        self.slice_index = 0

    def load_nifti(self, file_obj, slope="1", intercept="-1024") -> dict:
        """
        Load the NIfTI file and set up the data for slicing.
        """
        if file_obj is None:
            return None

        # Load the NIfTI image
        img = nib.load(file_obj)
        self.data = img.get_fdata() # Data loaded ranging between -1000,1000

        slope, intercept = float(slope), float(intercept)
        self.data = slope * self.data + intercept
        # Norm. between 0, 1
        self.data = np.clip(self.data, -1000, 1000)
        self.data = self.data / 1000
        self.data = (self.data + 1) / 2
        
        # Initialize the slice index and return the first slice and max index for the slider
        self.data = np.transpose(self.data, (1,0,2))
        self.slice_index = 0
        return {
            "image": self.data[:, :, self.slice_index],
            "max_index": self.data.shape[2] - 1
        }
    
    def update_slice(self, index: int, window="full") -> np.ndarray:
        """
        Update the displayed slice based on the slider index.
        """
        if self.data is None:
            return None
        
        self.slice_index = index

        if window == "full":
            return self.data[:, :, self.slice_index]
        elif window == "lung": # Lung windowing values (W:1500 L:-600) from https://radiopaedia.org/articles/windowing-ct
            # Set range between -1000,150 for lung
            return np.clip(self.data[:, :, self.slice_index], 0, 0.575) / 0.575
        elif window == "mediastinum": # Mediastinum windowing values (W:350 L:50) from https://radiopaedia.org/articles/windowing-ct
            return np.clip(((np.clip(self.data[:, :, self.slice_index], 0.4375, 0.6125)-0.4375) / 0.175), 0, 1)
        else: 
            return "Incorrect mode entered. Please enter one of the listed: full, lung, mediastinum"
        
        
        