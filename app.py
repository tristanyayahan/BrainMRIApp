import streamlit as st
import pandas as pd
import os
import time
import torch
import gdown
import base64
import shutil
import pydicom
from io import BytesIO
from PIL import Image
from datetime import datetime

from file_utils import (
    evaluate_compression, saveCSV, get_FolderName, get_orthanc_studies, create_and_upload_compressed_dicom,
    get_study_details, get_series_from_study, download_series_dicom, convert_dicom_to_jpg
)
from main_script import load_cnn_model, load_image, encode_image_with_kdtree, decode_image

# ----------------------
# Session Initialization
# ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

# ----------------------
# Load CNN Model
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model_path = "cnn_model.pth"
if not os.path.exists(cnn_model_path):
    file_id = "1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", cnn_model_path, quiet=False)

cnn_model = load_cnn_model(cnn_model_path, device, input_size=8)

# ----------------------
# Compression Function
# ----------------------
def compress_and_show_images(uploaded_files, patient_name, patient_id, patient_birthdate="Unknown", patient_sex="Unknown", upload_to_pacs=False, source_dicom_paths=None, block_size=8):
    compressed_files = []
    batch_FolderName = get_FolderName(patient_name, patient_id)
    orig_data_path = os.path.join("data/original", batch_FolderName)
    comp_data_path = os.path.join("data/compressed", batch_FolderName)
    os.makedirs(orig_data_path, exist_ok=True)

    if isinstance(uploaded_files, str):
        uploaded_files = [uploaded_files]

    for idx, uploaded_file in enumerate(uploaded_files):
        if isinstance(uploaded_file, str):
            orig_img_path = uploaded_file
            orig_img_name = os.path.basename(uploaded_file)
        else:
            orig_img_name = uploaded_file.name
            base_name, ext = os.path.splitext(orig_img_name)
            orig_img_path = os.path.join(orig_data_path, orig_img_name)

            counter = 2
            while os.path.exists(orig_img_path):
                orig_img_name = f"{base_name}_{counter}{ext}"
                orig_img_path = os.path.join(orig_data_path, orig_img_name)
                counter += 1

            with open(orig_img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        image = load_image(orig_img_path)
        start_time = time.perf_counter()
        encoded_data, domain_blocks = encode_image_with_kdtree(image, block_size, cnn_model, device)
        comp_img_name = f"compressed_{os.path.basename(orig_img_path)}"
        comp_img_path = os.path.join(comp_data_path, comp_img_name)
        decode_image(encoded_data, domain_blocks, image.shape, block_size, comp_img_path, comp_data_path)
        end_time = time.perf_counter()
        compression_time = round(end_time - start_time, 4)

        original_size, compressed_size, cr_ratio, psnr, ssim = evaluate_compression(image, orig_img_path, comp_img_path)

        # Upload to Orthanc as new DICOM
        if upload_to_pacs and source_dicom_paths and idx < len(source_dicom_paths):
            try:
                create_and_upload_compressed_dicom(source_dicom_paths[idx], comp_img_path)
            except Exception as e:
                print(f"⚠️ PACS upload failed: {e}")

        compressed_files.append({
            "original_image": orig_img_path,
            "compressed_image": comp_img_path,
            "compression_time": compression_time,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "cr_ratio": cr_ratio
        })

        saveCSV(batch_FolderName, patient_name, patient_id, orig_img_path, comp_img_path, os.path.basename(orig_img_path),
                os.path.basename(comp_img_path), original_size, compressed_size, cr_ratio, psnr, ssim,
                compression_time, "DataCollection.csv", patient_birthdate, patient_sex)

    return compressed_files, batch_FolderName


def get_base64_image(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# ----------------------
# PAGES
# ----------------------

# HOME PAGE
if st.session_state.page == "home":
    st.title("🧠 Brain MRI Compression Tool")

    # First row (top)
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        if st.button("📥 Compress New Image"):
            go_to("compress"); st.rerun()
    with top_col2:
        if st.button("📂 View Previous Data"):
            go_to("view"); st.rerun()

    st.markdown("---")  # Optional separator line

    # Second row (bottom)
    bottom_col1, bottom_col2 = st.columns(2)
    with bottom_col1:
        if st.button("📡 Import from PACS"):
            go_to("pacs"); st.rerun()
    with bottom_col2:
        if st.button("🧾 Upload DICOM File"):
            go_to("dicom_upload"); st.rerun()


# COMPRESS IMAGE PAGE
elif st.session_state.page == "compress":
    st.title("📥 Compress New MRI Image")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    patient_name = st.text_input("👤 Patient Name")
    patient_id = st.text_input("🆔 Patient ID")

    if patient_name and patient_id:
        uploaded_files = st.file_uploader("📤 Upload Brain MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"✅ Compressing {len(uploaded_files)} images for {patient_name} (ID: {patient_id})...")
            compressed_images_data, batch_FolderName = compress_and_show_images(uploaded_files, patient_name, patient_id)
            st.session_state.selected_batch = batch_FolderName
            st.session_state.page = "view_folder"
            time.sleep(0.3)
            st.rerun()
    else:
        st.info("ℹ️ Please enter both the patient name and ID to proceed.")

# VIEW PREVIOUS DATA (FOLDERS)
elif st.session_state.page == "view":
    st.title("📂 View Previously Compressed Data")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    csv_path = "data/csv/DataCollection.csv"
    if not os.path.exists(csv_path):
        st.warning("No data found.")
    else:
        df = pd.read_csv(csv_path)
        unique_batches = df["batch_FolderName"].unique()
        st.subheader("📁 Select a Patient Folder")
        for batch in unique_batches:
            if st.button(f"📂 {batch}"):
                st.session_state.selected_batch = batch
                st.session_state.page = "view_folder"
                st.rerun()

# VIEW FOLDER PAGE
elif st.session_state.page == "view_folder":
    st.title(f"🗂️ Folder: {st.session_state.selected_batch}")
    if st.button("🔙 Back to Folders"):
        if "selected_image_details" in st.session_state:
            del st.session_state.selected_image_details
        st.session_state.page = "view"
        st.rerun()

    df = pd.read_csv("data/csv/DataCollection.csv")
    batch_df = df[df["batch_FolderName"] == st.session_state.selected_batch]

    columns = st.columns(5)
    col_index = 0

    for _, row in batch_df.iterrows():
        key = f"img_{row['comp_img_name']}"
        with columns[col_index].form(key=key):
            st.markdown(f"""
                <button class="img-button" type="submit" style="border:none;background:none;">
                <img src="data:image/jpeg;base64,{get_base64_image(row['comp_img_path'])}"
                     alt="{row['comp_img_name']}" style="width:100%;max-width:120px;border-radius:8px;">
                </button>""", unsafe_allow_html=True)
            submitted = st.form_submit_button("Details")
            if submitted:
                st.session_state.selected_image_details = row.to_dict()
        col_index += 1
        if col_index == 5: columns = st.columns(5); col_index = 0

    if "selected_image_details" in st.session_state:
        data = st.session_state.selected_image_details
        st.markdown("---")
        st.markdown(f"### 🖼️ {data['comp_img_name']} Details")
        col1, col2 = st.columns(2)
        with col1:
            st.image(data["orig_img_path"], caption="Original", width=250)
        with col2:
            st.image(data["comp_img_path"], caption="Compressed", width=250)

        st.markdown(f"""
        - **Patient Name**: `{data['patient_name']}`
        - **Patient ID**: `{data['patient_id']}`
        - **Birthdate**: `{data.get('patient_birthdate', 'N/A')}`
        - **Sex**: `{data.get('patient_sex', 'N/A')}`   
        - **Original Size**: `{data['original_size']} KB`
        - **Compressed Size**: `{data['compressed_size']} KB`
        - **Compression Ratio**: `{data['cr_ratio']}`
        - **PSNR**: `{data['psnr']} dB`
        - **SSIM**: `{data['ssim']}`
        - **Compression Time**: `{data['compression_time']} sec`
        """)

        if st.button("⬅ Hide Details"):
            del st.session_state.selected_image_details
            st.rerun()

    if st.button("📦 Download All Compressed Images (ZIP)"):
        folder_path = f"data/compressed/{st.session_state.selected_batch}"
        zip_path = shutil.make_archive(folder_path, 'zip', folder_path)
        with open(zip_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="{os.path.basename(zip_path)}">📥 Click here to download ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)

# PACS PAGE
elif st.session_state.page == "pacs":
    st.title("📡 PACS Import (Orthanc)")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    st.subheader("🩺 Available Studies from Orthanc")
    studies = get_orthanc_studies()
    if not studies:
        st.warning("No studies found.")
    else:
        for study_id in studies:
            if st.button(f"📁 Study ID: {study_id}"):
                st.session_state.selected_study = study_id
                st.rerun()

    if "selected_study" in st.session_state:
        st.subheader(f"📚 Series in Study: {st.session_state.selected_study}")
        series_list = get_series_from_study(st.session_state.selected_study)

        for series_id in series_list:
            if st.button(f"📥 Download & Compress Series: {series_id}"):
                dicom_paths = download_series_dicom(series_id)
                jpg_paths = convert_dicom_to_jpg(dicom_paths)

                # Extract patient info from first DICOM
                try:
                    ds = pydicom.dcmread(dicom_paths[0])
                    patient_name = str(ds.PatientName) if "PatientName" in ds else "Unknown"
                    patient_id = str(ds.PatientID) if "PatientID" in ds else "Unknown"
                    patient_sex = str(ds.PatientSex) if "PatientSex" in ds else "Unknown"
                    birth_raw = str(ds.PatientBirthDate) if "PatientBirthDate" in ds else "Unknown"
                    patient_birthdate = datetime.strptime(birth_raw, "%Y%m%d").strftime("%Y-%m-%d") if birth_raw != "Unknown" else "Unknown"
                except Exception as e:
                    st.error(f"Failed to read DICOM metadata: {e}")
                    patient_name = patient_id = patient_birthdate = patient_sex = "Unknown"

                st.success(f"✅ Name: {patient_name}, ID: {patient_id}, Sex: {patient_sex}, Birthdate: {patient_birthdate}")

                compressed_data, folder = compress_and_show_images(
                    jpg_paths,
                    patient_name,
                    patient_id,
                    patient_birthdate,
                    patient_sex,
                    upload_to_pacs=True,
                    source_dicom_paths=dicom_paths
                )
                st.session_state.selected_batch = folder
                st.session_state.page = "view_folder"
                st.rerun()



# DICOM UPLOAD PAGE
elif st.session_state.page == "dicom_upload":
    st.title("🧾 Upload DICOM File")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    dicom_files = st.file_uploader("Select DICOM Files", type=["dcm"], accept_multiple_files=True)

    if dicom_files:
        temp_dir = "temp_dicom_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        dicom_paths = []

        for f in dicom_files:
            file_path = os.path.join(temp_dir, f.name)
            with open(file_path, "wb") as out:
                out.write(f.read())
            dicom_paths.append(file_path)

        try:
            ds = pydicom.dcmread(dicom_paths[0])
            patient_name = str(ds.PatientName) if "PatientName" in ds else "Unknown"
            patient_id = str(ds.PatientID) if "PatientID" in ds else "Unknown"
            patient_sex = str(ds.PatientSex) if "PatientSex" in ds else "Unknown"
            birth_raw = str(ds.PatientBirthDate) if "PatientBirthDate" in ds else "Unknown"
            patient_birthdate = datetime.strptime(birth_raw, "%Y%m%d").strftime("%Y-%m-%d") if birth_raw != "Unknown" else "Unknown"
        except Exception as e:
            st.error(f"Failed to read DICOM metadata: {e}")
            patient_name = patient_id = patient_birthdate = patient_sex = "Unknown"

        st.success(f"✅ Patient Name: {patient_name}")
        st.success(f"✅ Patient ID: {patient_id}")
        st.success(f"✅ Birthdate: {patient_birthdate}")
        st.success(f"✅ Sex: {patient_sex}")

        jpg_paths = convert_dicom_to_jpg(dicom_paths)
        compressed_data, folder = compress_and_show_images(
            jpg_paths,
            patient_name,
            patient_id,
            patient_birthdate,
            patient_sex,
            upload_to_pacs=True,
            source_dicom_paths=dicom_paths
        )

        st.session_state.selected_batch = folder
        st.session_state.page = "view_folder"
        st.rerun()
