import streamlit as st
import pandas as pd
import os
import time
import torch
import gdown
from file_utils import evaluate_compression, saveCSV, get_FolderName
from main_script import load_cnn_model, load_image, encode_image_with_kdtree, decode_image
import base64
from io import BytesIO
from PIL import Image

def get_base64_image(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

# Load CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model_path = "cnn_model.pth"
if not os.path.exists(cnn_model_path):
    file_id = "1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW"
    print(f"\n\nDownloading the CNN model with extracted features...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", cnn_model_path, quiet=False)

cnn_model = load_cnn_model(cnn_model_path, device, input_size=8)

# Compression function
def compress_and_show_images(uploaded_files, patient_name, patient_id, block_size=8):
    compressed_files = []
    batch_FolderName = get_FolderName(patient_name, patient_id)
    orig_data_path = os.path.join("data/original", batch_FolderName)
    comp_data_path = os.path.join("data/compressed", batch_FolderName)
    os.makedirs(orig_data_path, exist_ok=True)

    for uploaded_file in uploaded_files:
        # Save the original uploaded image with duplicate check
        orig_img_name = uploaded_file.name
        base_name, ext = os.path.splitext(orig_img_name)
        orig_img_path = os.path.join(orig_data_path, orig_img_name)

        # Check if a file with the same name already exists
        counter = 2
        while os.path.exists(orig_img_path):
            orig_img_name = f"{base_name}_{counter}{ext}"
            orig_img_path = os.path.join(orig_data_path, orig_img_name)
            counter += 1

        # Save the original image
        with open(orig_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = load_image(orig_img_path)
        start_time = time.perf_counter()
        encoded_data, domain_blocks = encode_image_with_kdtree(image, block_size, cnn_model, device)
        comp_img_name = f"compressed_{orig_img_name}"
        comp_img_path = os.path.join(comp_data_path, comp_img_name)
        decode_image(encoded_data, domain_blocks, image.shape, block_size, comp_img_path, comp_data_path)
        end_time = time.perf_counter()
        compression_time = round(end_time - start_time, 4)

        original_size, compressed_size, cr_ratio, psnr, ssim = evaluate_compression(image, orig_img_path, comp_img_path)

        compressed_files.append({
            "original_image": orig_img_path,
            "compressed_image": comp_img_path,
            "compression_time": compression_time,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "cr_ratio": cr_ratio
        })

        saveCSV(
                batch_FolderName, patient_name, patient_id, orig_img_path, comp_img_path, orig_img_name, comp_img_name,
                original_size, compressed_size, cr_ratio, psnr, ssim, compression_time, "DataCollection.csv"
            )

    return compressed_files, batch_FolderName

# ============================================================
# 🏠 HOME PAGE
# ============================================================
if st.session_state.page == "home":
    st.title("Brain MRI Compression Tool")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Compress New Image"):
            go_to("compress")
            st.rerun()

    with col2:
        if st.button("📂 View Previous Compressed Data"):
            go_to("view")
            st.rerun()

# ============================================================
# 📥 COMPRESSION PAGE
# ============================================================
elif st.session_state.page == "compress":
    st.title("📥 Compress New MRI Image")

    if st.button("🔙 Back"):
        go_to("home")
        st.rerun()

    # Input fields for patient details
    st.subheader("🔎 Enter Patient Details")
    patient_name = st.text_input("👤 Patient Name")
    patient_id = st.text_input("🆔 Patient ID")

    # Only show uploader if both fields are filled
    if patient_name and patient_id:
        uploaded_files = st.file_uploader(
            "📤 Upload Brain MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"✅ Compressing {len(uploaded_files)} images for **{patient_name}** (ID: {patient_id})...")
            
            compressed_images_data, batch_FolderName = compress_and_show_images(uploaded_files, patient_name, patient_id)

            # ✅ Save state first
            st.session_state.selected_batch = batch_FolderName
            st.session_state.page = "view_folder"

            # ✅ Add a short delay to ensure filesystem operations are complete
            time.sleep(0.3)

            # ✅ Only rerun after everything is ready
            st.rerun()


    else:
        st.info("ℹ️ Please enter both the patient name and ID to proceed.")


# ============================================================
# 📂 VIEW PREVIOUS COMPRESSED DATA PAGE | FOLDERS
# ============================================================
elif st.session_state.page == "view":
    st.title("📂 View Previously Compressed Data")

    if st.button("🔙 Back"):
        if "selected_image_details" in st.session_state:
            del st.session_state.selected_image_details
        go_to("home")
        st.rerun()

    csv_path = "data/csv/DataCollection.csv"

    if not os.path.exists(csv_path):
        st.warning("No data found.")
    else:
        df = pd.read_csv(csv_path)

        # Unique batch folders
        unique_batches = df["batch_FolderName"].unique()

        st.subheader("📁 Select a Patient Folder")
        for batch in unique_batches:
            if st.button(f"📂 {batch}"):
                st.session_state.selected_batch = batch
                st.session_state.page = "view_folder"
                st.rerun()


# ============================================================
# 📂 VIEW PREVIOUS COMPRESSED DATA PAGE | MRI IMAGES
# ============================================================
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
            st.markdown(
                f"""
                <style>
                    .img-button {{
                        border: none;
                        background: none;
                        padding: 0;
                    }}
                    .img-button img {{
                        width: 100%;
                        max-width: 120px;
                        border-radius: 8px;
                        transition: transform 0.2s ease;
                    }}
                    .img-button img:hover {{
                        transform: scale(1.05);
                        cursor: pointer;
                    }}
                </style>
                <button class="img-button" type="submit">
                    <img src="data:image/jpeg;base64,{get_base64_image(row['comp_img_path'])}" alt="{row['comp_img_name']}">
                </button>
                """,
                unsafe_allow_html=True
            )
            submitted = st.form_submit_button("Details")
            if submitted:
                st.session_state.selected_image_details = row.to_dict()

        col_index += 1
        if col_index == 5:
            columns = st.columns(5)
            col_index = 0

    # Show details if selected
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
