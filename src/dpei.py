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


def make_heatmap(slice2d: np.ndarray, title: str, colorscale: str, 
    view: str = 'axial', seg_slice: np.ndarray | None = None, labels: list[int] | None = None,
    line_width: int = 2, show_boundaries: bool = True, show_seg: bool = True, bbox_lesion:dict = None):

    z, bbox_dict, masks_refined = draw_masks(slice2d, seg_slice, show_seg)

    # if view=='axial': 
    #     if j_ant and j_post:
    #         if show_boundaries:
    #             z = cv2.line(z, (0, j_ant), (z.shape[0], j_ant), (255, 250, 255), 2)
    #             z = cv2.line(z, (0, j_post), (z.shape[0], j_post), (255, 250, 255), 2)
    #             z = cv2.line(z, (i_right, 0), (i_right, z.shape[1]), (255, 250, 255), 2)
    #             z = cv2.line(z, (i_left, 0), (i_left, z.shape[1]), (255, 250, 255), 2)

            # curve_L, curve_R = lateral_curves_from_chain(
            #     masks_refined['rectum'], masks_refined['uterus'], masks_refined['bladder'],
            #     close_px=3, dilate_px=1, smooth_sigma=1.0)

            # y_L = curve_L[:, 0]
            # x_L = curve_L[:, 1]
            # y_R = curve_R[:, 0]
            # x_R = curve_R[:, 1]

            # curve_L = curve_L.astype(int)
            # curve_R = curve_R.astype(int)

            # if show_boundaries:
            #     z = cv2.polylines(z, [curve_L[:, [1,0]]], False, (255, 250, 255), 2)
            #     z = cv2.polylines(z, [curve_R[:, [1,0]]], False, (255, 250, 255), 2)

            # st.session_state.boundaries = {'Left':(x_L, y_L), 'Right':(x_R, y_R), 'Anterior':j_ant, 'Posterior':j_post}
            # st.session_state.boundaries = {'Left':i_left, 'Right': i_right, 'Anterior':j_ant, 'Posterior':j_post}

    return z