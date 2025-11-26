import streamlit as st
import numpy as np
import nibabel as nib
import pandas as pd
import tempfile
from pathlib import Path
from PIL import Image
import io 
from streamlit_plotly_events import plotly_events
import os
from glob import glob
from streamlit_image_annotation import detection
from src.vol_to_vol import project_bbox_from_axial
from monai.utils import first, set_determinism
from monai.networks.nets import vista3d132
from monai.transforms import Orientation
from src.utils import get_data, save_nifti_bytes
from src.dpei import make_heatmap
from pydicom import dcmread


st.set_page_config(page_title="MRAnnotator", layout="wide")

cmap = "Gray"

def img2buf(img):
    image = Image.fromarray(img)
    buf = io.BytesIO()
    image.save(buf, format='png')
    return buf

sc1, sc2 = st.columns([2,6])
with sc1:
    with st.expander('📥  **Upload**'):
        #uploaded = st.file_uploader(label="Axial", width=250)
        uploaded = st.file_uploader("Choose DICOM or NIfTI Files", accept_multiple_files=True, key="file_uploader")


if uploaded is None:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.stop()

@st.cache_data(show_spinner=False)
def _cached_load(uploaded):
    return get_data(uploaded)

try:
    data = _cached_load(uploaded)

except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

if data is not None:
    vol = data
    nx, ny, nz = data.shape

    with sc2:
        k_ax = st.slider("Axial", 0, nz - 1, nz // 2, key="axial", label_visibility="collapsed", width=750)
        axial = vol[:, :, k_ax]
        fig_ax = make_heatmap(axial, f"Slice", cmap, view="axial", seg_slice=None, line_width=2, show_boundaries=False, show_seg = False)

        if 'result_dict' not in st.session_state:
            st.session_state['result_dict'] = {}

        if f'img{k_ax}' not in st.session_state['result_dict'].keys(): 
            st.session_state['result_dict'][f'img{k_ax}'] = {'bboxes':[], 'labels':[]}

        new_labels = detection(image_path=img2buf(fig_ax), 
                bboxes=st.session_state['result_dict'][f'img{k_ax}']['bboxes'], 
                labels=st.session_state['result_dict'][f'img{k_ax}']['labels'], 
                label_list=['Lesion'], use_space=True, width=512, height=512, line_width=2.0, key = f'img{k_ax}')

        if new_labels is not None:
            st.session_state['result_dict'][f'img{k_ax}']['bboxes'] = [v['bbox'] for v in new_labels]
            st.session_state['result_dict'][f'img{k_ax}']['labels'] = [v['label_id'] for v in new_labels]

    with sc1:
        lesion_dict = {'x':[], 'y':[], 'z':[], 'h':[], 'w':[]}
        with st.expander("✏️ **Annotate**"):
            if 'result_dict' in st.session_state: 
                if st.session_state['result_dict'].keys() != []:
                    for im in st.session_state['result_dict'].keys():
                        for i in range(len(st.session_state['result_dict'][im]['bboxes'])):
                            lesion_dict['z'].append(im[3:]) # If not in Dataframe, add the lesion
                            lesion_dict['x'].append(int(st.session_state['result_dict'][im]['bboxes'][i][0] * nx / 512))
                            lesion_dict['y'].append(int(st.session_state['result_dict'][im]['bboxes'][i][1] * ny / 512))
                            lesion_dict['w'].append(int(st.session_state['result_dict'][im]['bboxes'][i][2] * nx / 512))
                            lesion_dict['h'].append(int(st.session_state['result_dict'][im]['bboxes'][i][3] * ny / 512))
            st.dataframe(pd.DataFrame(lesion_dict), hide_index=True)
