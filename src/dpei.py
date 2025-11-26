import streamlit as st
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import io 
from skimage import measure
from skimage.segmentation import random_walker
from skimage.filters import sobel
import cv2
import os
from skimage import morphology
from src.mask import draw_masks


def _safe_percentile_max_j(mask: np.ndarray):
    """Frontière 'antérieure' d’un organe : j ~ percentile haut des voxels."""
    jj = np.where(mask)[0]
    if jj.size == 0:
        return None
    return int(np.min(jj))

def _safe_min_max_i(mask: np.ndarray, j=0):
    """Frontières 'latérales' d’un organe : i ~ percentile haut des voxels."""
    if j!=0:
        ii = np.where(mask[j, :])#
    else:
        ii = np.where(mask)[0]
    if ii[0].size == 0:
        return None, None
    return np.max([int(np.min(ii)) - 20, 0]), np.min([int(np.max(ii)) + 20, mask.shape[1]])

# ---------- lignes A/P ----------
def compute_ap_cut_lines(masks):
    """
    Renvoie (j_ant, j_post)
    j_ant : ligne antérieure (en avant du col/vagin)
    j_post: ligne antérieure au rectum
    """
    j_ant = _safe_percentile_max_j(masks["uterus"])
    if j_ant is None:
        # fallback : bord postérieur de la vessie
        j_ant = _safe_percentile_max_j(masks["bladder"])  # très postérieur de la vessie
    j_post = _safe_percentile_max_j(masks["rectum"])  # paroi antérieure du rectum
    if j_ant is None or j_post is None:
        return None,None
    return j_ant, j_post

def compute_lat_cut_lines(masks, j_ant, j_post):
    """
    """
    i_right_ut, i_left_ut = _safe_min_max_i(masks["uterus"], int((j_ant+j_post)/2))

    j_ant_bl = _safe_percentile_max_j(masks["bladder"])
    j_post_bl = _safe_percentile_max_j(masks["uterus"])

    if j_ant_bl and j_post_bl:
        i_right_bl, i_left_bl = _safe_min_max_i(masks["bladder"])
    else:
        i_right_bl, i_left_bl = None, None

    if (i_right_ut is None or i_left_ut is None) and (i_right_bl is None or i_left_bl is None):
        return None, None

    elif i_right_ut is None or i_left_ut is None:
        return i_right_bl, i_left_bl

    elif i_right_bl is None or i_left_bl is None:
        return i_right_ut, i_left_ut

    else:
        i_right = min([i_right_ut, i_right_bl])
        i_left = max([i_left_ut, i_left_bl])

        return i_right, i_left

def _fill_gaps_1d(arr: np.ndarray) -> np.ndarray:
    """Interpole les NaN en 1D (linéaire)."""
    x = np.arange(arr.size)
    good = ~np.isnan(arr)
    if good.sum() == 0:
        return arr.copy()
    if good.sum() == 1:
        arr[~good] = arr[good][0]
        return arr
    return np.interp(x, x[good], arr[good])

def lateral_curves_from_chain(
    rect2d: np.ndarray,
    cerv2d: np.ndarray,
    vess2d: np.ndarray,
    *,
    close_px: int = 3,     # fermeture (arrondit les creux)
    dilate_px: int = 1,    # dilatation légère (branche les micro-coupures)
    smooth_sigma: float = 1.0,  # lissage Gaussien sur x(y)
    clip_to_band: bool = True,  # tronque aux y où U existe
):
    """
    Retourne (curve_left, curve_right) sous forme de polylignes (N×2) en (y,x),
    qui suivent les bords latéraux du masque union rectum∪cervix∪vessie.

    Les entrées doivent être des booléens 2D déjà alignés à la vue (même rot que l'image).
    """
    assert rect2d.shape == cerv2d.shape == vess2d.shape
    H, W = rect2d.shape

    # 1) Union + lissage morphologique léger
    U = (rect2d.astype(bool) | cerv2d.astype(bool) | vess2d.astype(bool))
    if close_px and close_px > 0:
        U = morphology.binary_closing(U, morphology.disk(close_px))
    if dilate_px and dilate_px > 0:
        U = morphology.binary_dilation(U, morphology.disk(dilate_px))

    # 2) Profils x_left / x_right par ligne y
    xL = np.full(H, np.nan, dtype=float)
    xR = np.full(H, np.nan, dtype=float)

    yy, xx = np.where(U)
    if yy.size == 0:
        return None, None

    y_min, y_max = int(yy.min()), int(yy.max())
    for y in range(y_min, y_max + 1):
        xs = np.where(U[y])[0]
        if xs.size:
            # -0.5/+0.5 : colle au bord des pixels quand on dessine sous Plotly
            xL[y] = xs.min() - 0.5
            xR[y] = xs.max() + 0.5

    # 3) Interpolation + lissage
    xL = _fill_gaps_1d(xL)
    xR = _fill_gaps_1d(xR)
    if smooth_sigma and smooth_sigma > 0:
        xL = ndi.gaussian_filter1d(xL, sigma=smooth_sigma)
        xR = ndi.gaussian_filter1d(xR, sigma=smooth_sigma)

    y_coords = np.arange(H, dtype=float)
    if clip_to_band:
        y_coords = y_coords[y_min:y_max + 1]
        xL = xL[y_min:y_max + 1]
        xR = xR[y_min:y_max + 1]

    curve_left  = np.vstack([y_coords, xL]).T  # (y,x)
    curve_right = np.vstack([y_coords, xR]).T
    return curve_left, curve_right

def compute_boundaries(masks_refined):
    j_ant, j_post = compute_ap_cut_lines(masks_refined)
    if j_ant and j_post:
        i_right, i_left = compute_lat_cut_lines(masks_refined, j_ant, j_post)
    return j_ant, j_post, i_right, i_left

def get_compartment(lesion_coords):
    if 'boundaries' not in st.session_state: 
        return 'N/A'

    else:
        x1_lesion = lesion_coords[0]
        y1_lesion = lesion_coords[1]
        x2_lesion = lesion_coords[2] + x1_lesion
        y2_lesion = lesion_coords[3] + y1_lesion

        y_ant = st.session_state.boundaries['Anterior']
        y_post = st.session_state.boundaries['Posterior']

        x_right = st.session_state.boundaries['Right']
        x_left = st.session_state.boundaries['Left']

        # cy_left = st.session_state.boundaries['Left'][1]
        # cy_right = st.session_state.boundaries['Right'][1]
        # cx_left = st.session_state.boundaries['Left'][0]
        # cx_right = st.session_state.boundaries['Right'][0]

        #if y1_lesion > max(cy_left) or y1_lesion > max(cy_right) or y2_lesion < min(cy_left) or y2_lesion < min(cy_right):
            # return 'N/A'

        comp = []

        if y1_lesion <= y_ant: # Anterior
            #x_left = cx_left[np.where((cy_left > int(y1_lesion)) & (cy_left < int(y2_lesion)) & (cy_left <= int(y_ant)))[0]]
            #x_right = cx_right[np.where((cy_right > int(y1_lesion)) & (cy_right < int(y2_lesion)) & (cy_right <= int(y_ant)))[0]]
            if np.any(x1_lesion < x_right):
                comp.append('ALL')

            if np.any(x2_lesion > x_right) and np.any(x1_lesion < x_left):
                comp.append('AC')

            if np.any(x2_lesion > x_left):
                comp.append('ALR')

        if y2_lesion > y_ant and y1_lesion < y_post: # Medial
            # x_left = cx_left[np.where((cy_left > int(y1_lesion)) & (cy_left < int(y2_lesion)) & (cy_left > int(y_ant)) & (cy_left < int(y_post)))[0]]
            # x_right = cx_right[np.where((cy_right > int(y1_lesion)) & (cy_right < int(y2_lesion)) & (cy_right > int(y_ant)) & (cy_right < int(y_post)))[0]]
            if np.any(x1_lesion < x_right):
                comp.append('MLL')

            if np.any(x2_lesion > x_right) and np.any(x1_lesion < x_left):
                comp.append('MC')

            if np.any(x2_lesion > x_left):
                comp.append('MLR')

        if y2_lesion >= y_post: # Posterior
            # x_left = cx_left[np.where((cy_left > int(y1_lesion)) & (cy_left < int(y2_lesion)) & (cy_left >= int(y_post)))[0]]
            # x_right = cx_right[np.where((cy_right > int(y1_lesion)) & (cy_right < int(y2_lesion)) & (cy_right >= int(y_post)))[0]]
            if np.any(x1_lesion < x_right):
                comp.append('PLL')

            if np.any(x2_lesion > x_right) and np.any(x1_lesion < x_left):
                comp.append('PC')

            if np.any(x2_lesion > x_left):
                comp.append('PLR')

        return comp

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

    # else:
    if bbox_lesion:
        for box in bbox_lesion:
            if box is not None:
                z = cv2.rectangle(z, (box[1], box[3]), (box[0], box[2]), (255, 0, 0), 2)
    return z