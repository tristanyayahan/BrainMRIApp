import torch
import os
import gdown
import requests
import base64
import pydicom
import uuid
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO


def go_to(page_name):
    st.session_state.page = page_name


def pthFile_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model_path = "cnn_model.pth"
    if not os.path.exists(cnn_model_path):
        file_id = "1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", cnn_model_path, quiet=False)
    return(cnn_model_path, device)


def get_base64_image(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# Orthanc server settings
ORTHANC_SERVER_URL = "http://localhost:8042"
ORTHANC_DICOM_URL = f"{ORTHANC_SERVER_URL}/instances"
UID_CSV = "data/csv/processed_uids.csv"

# Function to load processed UIDs
def load_processed_uids():
    if not os.path.exists(UID_CSV):
        return set()
    df = pd.read_csv(UID_CSV)
    return set(df['SOPInstanceUID'])

# Function to save processed UIDs
def save_processed_uids(new_uids):
    if not os.path.exists(UID_CSV):
        df = pd.DataFrame(columns=["SOPInstanceUID"])
    else:
        df = pd.read_csv(UID_CSV)
    updated_df = pd.concat([df, pd.DataFrame(new_uids, columns=["SOPInstanceUID"])], ignore_index=True)
    updated_df.drop_duplicates(inplace=True)
    updated_df.to_csv(UID_CSV, index=False)

# Function to check if a DICOM exists in Orthanc
def check_dicom_exists(sop_uid):
    """Check if the DICOM SOP Instance UID already exists in Orthanc."""
    response = requests.get(f"{ORTHANC_DICOM_URL}/{sop_uid}")
    if response.status_code == 200:
        return True  # DICOM exists in Orthanc
    elif response.status_code == 404:
        return False  # DICOM does not exist in Orthanc
    else:
        response.raise_for_status()

def get_orthanc_studies():
    response = requests.get(f"{ORTHANC_SERVER_URL}/studies")
    return response.json() if response.status_code == 200 else []

def get_study_details(study_id):
    return requests.get(f"{ORTHANC_SERVER_URL}/studies/{study_id}").json()

def get_series_from_study(study_id):
    return get_study_details(study_id).get("Series", [])

def download_series_dicom(series_id, download_dir="temp_dicom"):
    os.makedirs(download_dir, exist_ok=True)
    instances = requests.get(f"{ORTHANC_SERVER_URL}/series/{series_id}").json()["Instances"]

    image_paths = []
    for instance_id in instances:
        dicom_bytes = requests.get(f"{ORTHANC_SERVER_URL}/instances/{instance_id}/file").content
        dicom_path = os.path.join(download_dir, f"{instance_id}.dcm")
        with open(dicom_path, "wb") as f:
            f.write(dicom_bytes)
        image_paths.append(dicom_path)
    
    return image_paths

def delete_series_from_orthanc(series_id, orthanc_url="http://localhost:8042", username=None, password=None):
    url = f"{orthanc_url}/series/{series_id}"
    auth = (username, password) if username and password else None
    try:
        response = requests.delete(url, auth=auth)
        if response.status_code == 200:
            return True, "Series deleted successfully."
        else:
            return False, f"Failed to delete series. Status code: {response.status_code}, Message: {response.text}"
    except Exception as e:
        return False, f"Exception occurred: {str(e)}"

def convert_dicom_to_jpg(dicom_paths, output_dir="temp_jpg"):
    os.makedirs(output_dir, exist_ok=True)
    jpg_paths = []

    for path in dicom_paths:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array
        image = Image.fromarray(pixel_array)
        jpg_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.jpg")
        image.convert("L").save(jpg_path)
        jpg_paths.append(jpg_path)
    
    return jpg_paths