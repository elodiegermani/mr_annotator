import streamlit as st
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import io 
from skimage import measure
from skimage.segmentation import random_walker
from skimage.filters import sobel
import os
from skimage import morphology
from src.mask import draw_masks

def make_heatmap(slice2d: np.ndarray, title: str, colorscale: str, 
    view: str = 'axial', seg_slice: np.ndarray | None = None, labels: list[int] | None = None,
    line_width: int = 2, show_boundaries: bool = True, show_seg: bool = True, bbox_lesion:dict = None):

    z, bbox_dict, masks_refined = draw_masks(slice2d, seg_slice, show_seg)

    return z