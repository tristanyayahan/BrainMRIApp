import streamlit as st
import os
import time
import torch
import gdown
from file_utils import evaluate_compression
from main_script import load_cnn_model, load_image, encode_image_with_kdtree, decode_image
import base64
from io import BytesIO
from PIL import Image

def get_base64_image(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Paths
orig_path = "data/original"
output_path = "data/compressed"

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
def compress_and_show_images(uploaded_files, block_size=8):
    compressed_files = []
    os.makedirs(orig_path, exist_ok=True)

    for uploaded_file in uploaded_files:
        # Save the original uploaded image with duplicate check
        image_name = uploaded_file.name
        base_name, ext = os.path.splitext(image_name)
        image_path = os.path.join(orig_path, image_name)

        # Check if a file with the same name already exists
        counter = 2
        while os.path.exists(image_path):
            image_name = f"{base_name}_{counter}{ext}"
            image_path = os.path.join(orig_path, image_name)
            counter += 1

        # Save the file
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


        image = load_image(image_path)
        start_time = time.perf_counter()
        encoded_data, domain_blocks, bps, buildingTree_time, nearestSearch_time, inference_time = encode_image_with_kdtree(
            image, block_size, cnn_model, device)
        output_file = os.path.join(output_path, f"compressed_{image_name}")
        decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file)
        end_time = time.perf_counter()

        original_size, compressed_size, cr_ratio, _, _ = evaluate_compression(image, image_path, output_file)

        compressed_files.append({
            "original_image": image_path,
            "compressed_image": output_file,
            "compression_time": round(end_time - start_time, 4),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "cr_ratio": cr_ratio
        })

    return compressed_files

# ============================================================
# 🏠 HOME PAGE
# ============================================================
if st.session_state.page == "home":
    st.title("🧠 Brain MRI Compression Tool")

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

    uploaded_files = st.file_uploader("Upload Brain MRI images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")

        compressed_images_data = compress_and_show_images(uploaded_files)

        for data in compressed_images_data:
            st.image([data['original_image'], data['compressed_image']], caption=["Original", "Compressed"], width=300)
            st.write("### Compression Details")
            st.write(f"**Compression Time**: {data['compression_time']} seconds")
            st.write(f"**Original Size**: {data['original_size']} KB")
            st.write(f"**Compressed Size**: {data['compressed_size']} KB")
            st.write(f"**Compression Ratio**: {data['cr_ratio']}")
            st.markdown("---")

# ============================================================
# 📂 VIEW PREVIOUS COMPRESSED DATA PAGE
# ============================================================
elif st.session_state.page == "view":
    st.title("📂 View Previously Compressed Images")
    if st.button("🔙 Back"):
        go_to("home")
        st.rerun()

    if not os.path.exists(orig_path) or not os.path.exists(output_path):
        st.warning("No previous data found.")
    else:
        original_images = sorted(os.listdir(orig_path))
        compressed_images = sorted(os.listdir(output_path))

        # Initialize selected image if not already
        if "selected_image" not in st.session_state:
            st.session_state.selected_image = None

        # Display compressed image thumbnails in rows of 5
        columns = st.columns(5)  # Create 5 columns for a row
        col_index = 0  # Track which column to use

        for orig_name, comp_name in zip(original_images, compressed_images):
            orig_full_path = os.path.join(orig_path, orig_name)
            comp_full_path = os.path.join(output_path, comp_name)

            key = f"click_{comp_name}"  # Unique key per image button

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
                        <img src="data:image/jpeg;base64,{get_base64_image(comp_full_path)}" alt="{comp_name}">
                    </button>
                    """,
                    unsafe_allow_html=True
                )
                submitted = st.form_submit_button("Details")
                if submitted:
                    st.session_state.selected_image = {
                        "original": orig_full_path,
                        "compressed": comp_full_path,
                        "name": comp_name
                    }

            col_index += 1
            # Start a new row after 5 images
            if col_index == 5:
                columns = st.columns(5)
                col_index = 0


        # Display the selected image details
        if st.session_state.selected_image:
            st.markdown("---")
            selected = st.session_state.selected_image
            st.markdown(f"### 🖼️ {selected['name']} Details")

            try:
                orig_img = load_image(selected["original"])
                original_size, compressed_size, CR_ratio, _, _ = evaluate_compression(
                    orig_img, selected["original"], selected["compressed"]
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.image(selected["original"], caption="Original", width=250)
                with col2:
                    st.image(selected["compressed"], caption="Compressed", width=250)

                st.markdown(f"""
                - **Original Size**: `{original_size} KB`
                - **Compressed Size**: `{compressed_size} KB`
                - **Compression Ratio**: `{CR_ratio}`
                """)

                if st.button("⬅️ Hide Details"):
                    st.session_state.selected_image = None

            except Exception as e:
                st.error(f"Error loading selected image: {e}")
