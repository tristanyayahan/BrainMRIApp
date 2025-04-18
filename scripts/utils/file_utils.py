import os
import requests
import pydicom
import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from datetime import datetime
from PIL import Image


def create_and_upload_compressed_dicom(original_dicom_path, compressed_image_path, orthanc_url="http://localhost:8042", username="orthanc", password="orthanc"):
    ds = pydicom.dcmread(original_dicom_path)
    
    # Read compressed image and convert to numpy array
    img = Image.open(compressed_image_path).convert("L")
    np_img = np.array(img)

    # Replace pixel data and update metadata
    ds.PixelData = np_img.tobytes()
    ds.Rows, ds.Columns = np_img.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Save to temporary path
    new_dicom_path = compressed_image_path.replace(".jpg", "_compressed.dcm")
    ds.save_as(new_dicom_path)

    # Upload to Orthanc
    with open(new_dicom_path, "rb") as f:
        r = requests.post(f"{orthanc_url}/instances", auth=(username, password), data=f)
        if r.status_code != 200:
            raise Exception(f"Failed to upload to Orthanc: {r.text}")

    return new_dicom_path


def get_FolderName(patient_name, patient_id):
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{patient_name}_{patient_id} {timestamp_str}"


def evaluate_compression(image, original_image_path, compressed_image_path):
    original_image = image
    compressed_image = io.imread(compressed_image_path, as_gray=True) / 255.0
    original_image = np.clip(original_image, 0, 1)
    compressed_image = np.clip(compressed_image, 0, 1)

    # PSNR and SSIM
    PSNR = round(psnr(original_image, compressed_image, data_range=1.0), 4)

    SSIM = round(ssim(original_image, compressed_image, data_range=1.0), 4)

    # file sizes
    original_size = os.path.getsize(original_image_path)
    compressed_size = os.path.getsize(compressed_image_path) 

    # compression ratio
    if compressed_size != 0:
        CR_ratio = round(compressed_size / original_size, 2)
    else:
        CR_ratio = float('inf')

    original_size = round(original_size / 1024, 2)  
    compressed_size = round(compressed_size / 1024, 2)  

    return original_size, compressed_size, CR_ratio, PSNR, SSIM


def saveCSV(batch_FolderName, patient_name, patient_id, orig_img_path, comp_img_path,
            orig_img_name, comp_img_name, original_size, compressed_size, cr_ratio,
            psnr, ssim, compression_time, filename,
            patient_birthdate="Unknown", patient_sex="Unknown"):

    csv_path = os.path.join("data/csv", filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = pd.DataFrame([{
        "batch_FolderName": batch_FolderName,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "patient_birthdate": patient_birthdate,
        "patient_sex": patient_sex,
        "orig_img_path": orig_img_path,
        "comp_img_path": comp_img_path,
        "orig_img_name": orig_img_name,
        "comp_img_name": comp_img_name,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "cr_ratio": cr_ratio,
        "psnr": psnr,
        "ssim": ssim,
        "compression_time": compression_time
    }])

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, new_data], ignore_index=True)
        updated.to_csv(csv_path, index=False)
    else:
        new_data.to_csv(csv_path, index=False)
