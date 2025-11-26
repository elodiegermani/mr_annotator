import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    ToTensord, 
)
import io 
from monai.data import DataLoader, Dataset, CacheDataset
import os
import gzip
import SimpleITK as sitk    

def load_nifti_from_bytes(name: str, file_bytes: bytes):
    """
    Load a NIfTI file from bytes, reorient in PLI and scale intensities to [0,1].

    Parameters
    ----------
        name, str: name of the file containing the NIfTI file
        file_bytes, bytes: content of the file in bytes, as output from fileloader

    Returns
    -------
        data, np.array
        affine, np.array
        zooms, np.array
    """
    suffix = ".nii.gz" if name.endswith(".nii.gz") else (Path(name).suffix or ".nii")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)

        test_data = [{"image": tmp_path}]

        test_org_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="PLI"),
                ScaleIntensityd(keys=["image"], minv=0,maxv=1),
            ]
        )

        ds = Dataset(data=test_data, transform=test_org_transforms)
        test_loader = DataLoader(ds, batch_size=1)

        for batch in test_loader:
            data = batch["image"][0][0]

        return data
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def load_and_store_dicom_series(directory, session_key):
    if session_key not in st.session_state:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image_sitk = reader.Execute()
        image_np = sitk.GetArrayFromImage(image_sitk)
        image_np = np.moveaxis(image_np, 0, -1)
        st.session_state[session_key] = image_np
    return st.session_state[session_key]

def load_dicom(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            #suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else (Path(uploaded_file.name).suffix or ".nii")
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(bytes_data)
        
        image_np = load_and_store_dicom_series(temp_dir, "dicom_image_data")
    try:
        test_data = [{"image": np.expand_dims(image_np, axis=0)}]
        test_org_transforms = Compose(
        [   
            ToTensord(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0,maxv=1),
        ]
        )
        ds = Dataset(data=test_data, transform=test_org_transforms)
        test_loader = DataLoader(ds, batch_size=1)

        for batch in test_loader:
            data = batch["image"][0][0]

        return data

    finally:
        Path(temp_dir).unlink(missing_ok=True)

def get_data(uploaded_files):
    if len(uploaded_files) > 1:
        return load_dicom(uploaded_files)

    elif len(uploaded_files) == 1:
        return load_nifti_from_bytes(uploaded_files[0].name, uploaded_files[0].getvalue())

    else:
        return None
        

def save_nifti_bytes(nii_file):
    bio = io.BytesIO()
    file_map = nii_file.make_file_map({'image': bio, 'header': bio})
    nii_file.to_file_map(file_map)
    data = gzip.compress(bio.getvalue())

    return bio