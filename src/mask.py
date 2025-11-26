import streamlit as st
import numpy as np
import pandas as pd
import os
from skimage.transform import resize

def draw_masks(img, masks, show_seg=True): 
    masked_image = np.asarray(img).copy() 
    masked_image = resize(masked_image, (512, 512))
    img_size = masked_image.shape
    if np.max(masked_image) <= 10: 
        masked_image = masked_image * 255 
    masked_image = masked_image.astype(np.uint8) 

    colors = {'uterus': np.array([255,0,0]), 'rectum':np.array([0,0,255]), 'bladder':np.array([255,255,0])} 

    masked_image = np.repeat(masked_image[:, :, np.newaxis], 3, axis=2) 
    bbox_dict = None
    masks_refined = {}
    
    return masked_image, bbox_dict, masks_refined