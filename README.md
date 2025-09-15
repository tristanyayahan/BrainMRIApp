# 🧠 Brain MRI Compression Demo App

## 📌 Overview
This is a demo application developed as part of our undergraduate thesis:  
**“Enhanced Compression Time Efficiency of Brain MRI Fractal Image Compression Using CNN-Guided and KD-Tree Optimized Encoding.”**  

The app allows users to **upload, compress, and view brain MRI images** using our hybrid **CNN + KD-tree optimized fractal compression method**. It demonstrates how medical images can be compressed efficiently while maintaining diagnostic quality.  

---

## 🚀 Features
- Upload **MRI images** (DICOM or standard image formats).  
- Apply **CNN-guided fractal compression** with KD-tree optimization.  
- View compressed vs. original images side by side.  
- Display key metrics: **PSNR, SSIM, compression ratio, and encoding/decoding time**.  
- Simple **Streamlit web interface** for testing and demonstration.  

---

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (frontend demo)  
- **PyTorch** (CNN model)  
- **NumPy / OpenCV** (image processing)  
- **Scikit-learn** (KD-tree implementation)  

---

## ⚡ Installation & Setup
1. **Clone this repository**:
   ```bash
   git clone https://github.com/tristanyayahan/BrainMRIApp.git
   cd demo-app-repo

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

3. **Run the Streamlit app**:
    ```bash
    streamlit run app.py

4. **Open your browser and go to**:
    ```bash
    http://localhost:8501

---

## ⚠️ Disclaimer
This project is for **academic and research purposes only**.  
The dataset used is anonymized and does not contain any sensitive or identifiable patient information.  

---

## 👨‍💻 Authors
- **Cassius Wayne N. Reyes** – Back-End Developer, Researcher 
- **Princess May R. Tomongha** – Front-End Developer, Researcher 
- **Tristan D. Anyayahan** – Front-End/Back-End Developer, Researcher 