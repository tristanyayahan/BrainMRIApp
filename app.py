import streamlit as st
from scripts.main_script import load_cnn_model
from scripts.utils.helper import pthFile_check
from scripts.utils.helper import go_to
from scripts.pages import (
    login_page, home_page, view_page, view_patient_page, view_folder_page, pacs_page, dicom_upload_page
)


# Load CNN Model
cnn_model_path, device = pthFile_check()
cnn_model = load_cnn_model(cnn_model_path, device, input_size=8)


# Browser tab title and icon
st.set_page_config(
    page_title="Brain MRI Compression",
    page_icon="scripts/assets/icon.ico",        
    layout="centered",
    initial_sidebar_state="auto"
)

# Session Initialization
if "page" not in st.session_state:
    st.session_state.page = "login"


# -------------------------------------------
#                  PAGES
# -------------------------------------------

# LOGIN PAGE
if st.session_state.page == "login":
    login_page()

# HOME PAGE
elif st.session_state.page == "home":
    home_page()

# DICOM UPLOAD PAGE
elif st.session_state.page == "dicom_upload":
    dicom_upload_page()

# PACS PAGE
elif st.session_state.page == "pacs":
    pacs_page()

# VIEW PREVIOUS DATA (FOLDERS)
elif st.session_state.page == "view":
    view_page()

# VIEW FOLDERS (timestamps) FOR PATIENT
elif st.session_state.page == "view_patient":
    view_patient_page()

# VIEW IMAGES FOR PATIENTS (timestamps)
elif st.session_state.page == "view_folder":
    view_folder_page()


# If user is logged in, show logout option at the top of the page
if st.session_state.logged_in:
    st.markdown("---")  # Divider
    st.markdown("###")  # Vertical spacing
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        go_to("login"); st.rerun()
