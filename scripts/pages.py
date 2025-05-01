import os
import time
import base64
import shutil
import pydicom
import zipfile
import pandas as pd
import streamlit as st
from datetime import datetime
from scripts.main_script import compress_and_show_images
from scripts.utils.helper import (
    go_to, load_processed_uids, save_processed_uids, check_dicom_exists, get_base64_image, 
    get_orthanc_studies, delete_series_from_orthanc, get_series_from_study, download_series_dicom, convert_dicom_to_jpg
)

def login_page():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.success("✅ Login successful!")
                st.session_state.logged_in = True
                go_to("home"); st.rerun()
            else:
                st.error("❌ Invalid credentials. Try again.")
        st.stop()

def home_page():
    # Centered title
    st.markdown("<h1 style='text-align: center;'>🧠 Brain MRI Compression Tool</h1>", unsafe_allow_html=True)

    st.markdown("###")  # Vertical spacing

    # Top buttons centered in one row
    top_col = st.columns([1, 2, 1])[1]
    with top_col:
        if st.button("-- SIMULATION --", use_container_width=True):
            go_to("dicom_upload"); st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)  # small spacing
        if st.button("📡 Import from PACS", use_container_width=True):
            go_to("pacs"); st.rerun()

    st.markdown("---")  # Divider

    # Bottom button centered
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        if st.button("📂 View Previous Data", use_container_width=True):
            go_to("view"); st.rerun()


def view_page():
    st.title("📂 View Previously Compressed Data")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    csv_path = "data/csv/DataCollection.csv"
    if not os.path.exists(csv_path):
        st.warning("No data found.")
    else:
        df = pd.read_csv(csv_path)
        patient_groups = df.groupby(["patient_name", "patient_id"])

        st.subheader("👥 Select a Patient")
        for (name, pid), _ in patient_groups:
            label = f"👤 {name} (ID: {pid})"
            if st.button(label, key=f"patient_{pid}_{name}"):
                st.session_state.selected_patient = {"name": name, "id": pid}
                st.session_state.page = "view_patient"
                st.rerun()


def view_patient_page():
    st.title(f"🧑‍⚕️ Patient: {st.session_state.selected_patient['name']} (ID: {st.session_state.selected_patient['id']})")

    if st.button("🔙 Back to Patient List"):
        del st.session_state.selected_patient
        st.session_state.page = "view"
        st.rerun()

    df = pd.read_csv("data/csv/DataCollection.csv")
    patient_df = df[(df["patient_name"] == st.session_state.selected_patient["name"]) &
                    (df["patient_id"] == st.session_state.selected_patient["id"])]

    unique_folders = patient_df["batch_FolderName"].unique()
    st.subheader("📁 Select a Folder (Timestamped Batch)")
    for folder in unique_folders:
        if st.button(f"📂 {folder}", key=f"folder_{folder}"):
            st.session_state.selected_batch = folder
            st.session_state.page = "view_folder"
            st.rerun()


def view_folder_page():
    if "selected_batch" not in st.session_state or "selected_patient" not in st.session_state:
        st.error("Missing context. Please select a patient and folder again.")
        st.stop()

    patient_name = st.session_state.selected_patient["name"]
    patient_id = st.session_state.selected_patient["id"]
    batch_name = st.session_state.selected_batch

    st.title(f"🗂️ Folder: {batch_name}")

    if st.button("🔙 Back to Patient Folders"):
        if "selected_image_details" in st.session_state:
            del st.session_state.selected_image_details
        st.session_state.page = "view_patient"
        st.rerun()

    df = pd.read_csv("data/csv/DataCollection.csv")
    batch_df = df[df["batch_FolderName"] == batch_name]

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
        if col_index == 5:
            columns = st.columns(5)
            col_index = 0

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
        folder_path = f"data/compressed/{batch_name}"
        zip_path = shutil.make_archive(folder_path, 'zip', folder_path)
        with open(zip_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="{os.path.basename(zip_path)}">📥 Click here to download ZIP</a>'
            st.markdown(href, unsafe_allow_html=True)


def pacs_page():
    st.title("📡 PACS Import (Orthanc)")

    # Back button: reset selected study and go back to home
    if st.button("🔙 Back"):
        st.session_state.selected_study = None
        go_to("home")
        st.rerun()

    # Initialize selected study state if not set
    if "selected_study" not in st.session_state:
        st.session_state.selected_study = None

    # List of studies from Orthanc
    st.subheader("🩺 Available Studies from Orthanc")
    studies = get_orthanc_studies()
    if not studies:
        st.warning("No studies found.")
    else:
        for study_id in studies:
            is_selected = st.session_state.selected_study == study_id
            label = f"{'🔵' if is_selected else '📁'} Study ID: {study_id}"
            if st.button(label, key=f"study_{study_id}"):
                st.session_state.selected_study = study_id
                st.rerun()

    # Display series only if a study is selected
    if st.session_state.selected_study:
        st.subheader(f"📚 Series in Study: {st.session_state.selected_study}")
        try:
            series_list = get_series_from_study(st.session_state.selected_study)
        except Exception as e:
            st.error(f"❌ Failed to fetch series: {e}")
            st.stop()

        for series_id in series_list:
            dicom_paths = download_series_dicom(series_id)
            jpg_paths = convert_dicom_to_jpg(dicom_paths)

            try:
                ds = pydicom.dcmread(dicom_paths[0])
                patient_name = str(ds.PatientName) if "PatientName" in ds else "Unknown"
                patient_id = str(ds.PatientID) if "PatientID" in ds else "Unknown"
                patient_sex = str(ds.PatientSex) if "PatientSex" in ds else "Unknown"
                birth_raw = str(ds.PatientBirthDate) if "PatientBirthDate" in ds else "Unknown"
                try:
                    patient_birthdate = datetime.strptime(birth_raw, "%Y%m%d").strftime("%Y-%m-%d") \
                        if birth_raw and birth_raw.strip() else "Unknown"
                except Exception:
                    patient_birthdate = "Unknown"
            except Exception as e:
                st.error(f"Failed to read DICOM metadata: {e}")
                patient_name = patient_id = patient_sex = patient_birthdate = "Unknown"

            with st.expander(f"📦 Series ID: {series_id}"):
                st.markdown(
                    f"**👤 Name:** {patient_name}  \n"
                    f"**🆔 ID:** {patient_id}  \n"
                    f"**🗓️ Birthdate:** {patient_birthdate}  \n"
                    f"**🚻 Sex:** {patient_sex}"
                )

                cols = st.columns(min(4, len(jpg_paths)))
                for i, path in enumerate(jpg_paths):
                    with cols[i % len(cols)]:
                        st.image(path, caption=f"Image {i+1}", use_container_width=True)

                download_key = f"download_{series_id}"
                delete_key = f"delete_button_{series_id}"
                confirm_key = f"confirm_delete_{series_id}"
                yes_key = f"yes_delete_{series_id}"
                cancel_key = f"cancel_delete_{series_id}"

                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button(f"📥 Download This Series", key=download_key):
                        # Download and zip the series DICOM files
                        zip_filename = f"{series_id}_dicoms.zip"
                        zip_path = os.path.join("temp_downloads", zip_filename)
                        os.makedirs("temp_downloads", exist_ok=True)

                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for dicom_file in dicom_paths:
                                arcname = os.path.basename(dicom_file)
                                zipf.write(dicom_file, arcname)

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Click here to download the DICOM series",
                                data=f,
                                file_name=zip_filename,
                                mime="application/zip"
                            )

                with col2:
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if not st.session_state[confirm_key]:
                        if st.button(f"🗑️ Delete This Series", key=delete_key):
                            st.session_state[confirm_key] = True
                            st.rerun()
                    else:
                        st.warning("⚠️ Are you sure you want to delete this series from the PACS?")
                        confirm_col1, confirm_col2 = st.columns([1, 1])
                        with confirm_col1:
                            if st.button("✅ Yes, Delete", key=yes_key):
                                success, msg = delete_series_from_orthanc(series_id)
                                st.session_state[confirm_key] = False
                                if success:
                                    st.success(msg)
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with confirm_col2:
                            if st.button("❌ Cancel", key=cancel_key):
                                st.session_state[confirm_key] = False
                                st.rerun()


def dicom_upload_page():
    st.title("🧾 Upload DICOM File")
    if st.button("🔙 Back"): go_to("home"); st.rerun()

    dicom_files = st.file_uploader("Select DICOM Files", type=["dcm"], accept_multiple_files=True)

    if dicom_files:
        temp_dir = "data/temp_dicom_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        dicom_paths = []

        processed_uids = load_processed_uids()
        new_uids = []

        for f in dicom_files:
            file_path = os.path.join(temp_dir, f.name)
            with open(file_path, "wb") as out:
                out.write(f.read())

            try:
                ds = pydicom.dcmread(file_path)
                sop_uid = str(ds.SOPInstanceUID)

                if sop_uid in processed_uids:
                    st.warning(f"⚠️ Duplicate file {f.name} (UID: {sop_uid}) skipped.")
                    continue
                else:
                    # Check if the DICOM already exists in Orthanc before uploading
                    if check_dicom_exists(sop_uid):
                        st.warning(f"⚠️ DICOM with SOPInstanceUID {sop_uid} already exists in Orthanc.")
                        continue
                    new_uids.append([sop_uid])
                    dicom_paths.append(file_path)
            except Exception as e:
                st.error(f"❌ Failed to read {f.name}: {e}")
                continue

        if not dicom_paths:
            st.info("No new files to process.")
        else:
            try:
                ds = pydicom.dcmread(dicom_paths[0])
                patient_name = str(ds.PatientName) if "PatientName" in ds else "Unknown"
                patient_id = str(ds.PatientID) if "PatientID" in ds else "Unknown"
                patient_sex = str(ds.PatientSex) if "PatientSex" in ds else "Unknown"
                birth_raw = str(ds.PatientBirthDate) if "PatientBirthDate" in ds else "Unknown"
                try:
                    patient_birthdate = datetime.strptime(birth_raw, "%Y%m%d").strftime("%Y-%m-%d") if birth_raw and birth_raw.strip() else "Unknown"
                except Exception:
                    patient_birthdate = "Unknown"

            except Exception as e:
                st.error(f"❌ Failed to read DICOM metadata: {e}")
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

            save_processed_uids(new_uids)

            st.session_state.selected_patient = {
                "name": patient_name,
                "id": patient_id
            }
            st.session_state.selected_batch = folder
            st.session_state.page = "view_folder"
            st.rerun()