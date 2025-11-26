from skimage.segmentation import random_walker
from skimage.filters import sobel
from skimage import morphology
import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2

def edge_snap_guided(image_bgr: np.ndarray, mask: np.ndarray, radius=8, eps=1e-3, thr=0.5) -> np.ndarray: 
    """  """ 
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 
    m = (mask > 0).astype(np.float32) 
    # 0..1 
    # guidedFilter(guide, src, radius, eps, dDepth) 
    prob = cv2.ximgproc.guidedFilter(guide, m, radius, eps, -1) 
    return (prob >= thr).astype(np.uint8) * 255 

def clean_mask_basic(mask: np.ndarray, open_ks=3, close_ks=5, min_area=200, keep_largest=False): 
    """ mask: uint8 binary (0/255) """ 
    m = (mask > 0).astype(np.uint8) * 255 
    m = cv2.resize(m, (512,512))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks)) 
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks)) 
    # Remove salt noise; close small gaps 
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1) 
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1) 
    # Fill holes via flood fill from the border 
    h, w = m.shape 
    flood = np.zeros((h+2, w+2), np.uint8) 
    m_inv = (255 - m).copy() 
    cv2.floodFill(m_inv, flood, (0, 0), 0) # eats background 
    holes = (m_inv > 0).astype(np.uint8) * 255 
    m = cv2.bitwise_or(m, holes) # Drop tiny components 
    num, lab, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), connectivity=8) 

    out = np.zeros_like(m) 
    if keep_largest and num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA] 
        lid = 1 + np.argmax(areas) 
        out[lab == lid] = 255 
    else: 
        for i in range(1, num): 
            if stats[i, cv2.CC_STAT_AREA] >= min_area: 
                out[lab == i] = 255 

    return out 

def refine_random_walker(image_bgr: np.ndarray, mask: np.ndarray, erode_iters=2, dilate_iters=5, beta=150):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 
    # léger lissage pour réduire le bruit 
    gray = cv2.bilateralFilter(gray, 5, 20, 20) 
    m = (mask > 0).astype(np.uint8) 
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
    sure_fg = cv2.erode(m, k, iterations=erode_iters) # graines objet 
    sure_bg = cv2.erode(1 - m, k, iterations=erode_iters) # graines fond 

    sure_bg = cv2.dilate(sure_bg, k, iterations=dilate_iters) # étiquettes de graines pour random walker: 0=non étiqueté, 1=fond, 2=objet 
    seeds = np.zeros_like(m, np.uint8) 
    seeds[sure_bg > 0] = 1 
    seeds[sure_fg > 0] = 2 # image de coût basée sur le gradient (bords forts = barrières) 

    edges = sobel(gray.astype(np.float32) / 255.0) 
    labels = random_walker(edges, seeds, beta=beta, mode="bf") 
    out = (labels == 2).astype(np.uint8) * 255 

    return out 

def bbox2d_from_mask(mask2d: np.ndarray, margin: int = 0):
    assert mask2d.ndim == 2
    coords = np.array(np.nonzero(mask2d))
    if coords.size == 0:
        return (-1,-1),(-1,-1)
    lo = coords.min(axis=1)
    hi = coords.max(axis=1) + 1
    if margin > 0:
        lo = lo - margin; hi = hi + margin
        lo, hi = clamp(lo, hi, np.array(mask2d.shape))
    i0, j0 = map(int, lo)  # i = lignes (axe 0), j = colonnes (axe 1)
    i1, j1 = map(int, hi)
    return (i0, j0), (i1, j1)  # exclusif

def draw_masks(img, masks, show_seg=True): 
    masked_image = np.asarray(img).copy() 
    masked_image = cv2.resize(masked_image, (512, 512))
    img_size = masked_image.shape
    if np.max(masked_image) <= 10: 
        masked_image = masked_image * 255 
    masked_image = masked_image.astype(np.uint8) 

    colors = {'uterus': np.array([255,0,0]), 'rectum':np.array([0,0,255]), 'bladder':np.array([255,255,0])} 

    masked_image = np.repeat(masked_image[:, :, np.newaxis], 3, axis=2) 
    bbox_dict = None
    masks_refined = {}

    if masks: 
        bbox_dict = {}
        for n, (organ, mask) in enumerate(masks.items()): 
            refined = np.asarray(mask, dtype=np.uint8) 
            refined = clean_mask_basic(mask) 
            #refined = edge_snap_guided(masked_image, refined) 
            refined = refine_random_walker(masked_image, refined) 
            bbox_dict[organ] = bbox2d_from_mask(refined)
            masks_refined[organ] = refined
            if show_seg:
                masked_image[refined>1] = masked_image[refined>1] * 0.7 + colors[organ] * 0.3 

    return masked_image, bbox_dict, masks_refined