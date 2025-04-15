import os
import csv
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from datetime import datetime


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
    #print(f"\tPeak Signal-to-Noise Ratio (PSNR): {PSNR:.2f} dB")

    SSIM = round(ssim(original_image, compressed_image, data_range=1.0), 4)
    #print(f"\tStructural Similarity Index (SSIM): {SSIM:.4f}")

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


def saveCSV(batch_FolderName, patient_name, patient_id, orig_img_path, comp_img_path, orig_img_name, comp_img_name,
                original_size, compressed_size, cr_ratio, psnr, ssim, compression_time, csvFile_name):
    os.makedirs("data/csv", exist_ok=True)

    csv_filename = os.path.join("data/csv", csvFile_name)

    # prepare data for CSV
    data = [batch_FolderName, patient_name, patient_id, orig_img_path, comp_img_path, orig_img_name, comp_img_name,
                original_size, compressed_size, cr_ratio, psnr, ssim, compression_time]

    # Write to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["batch_FolderName", "patient_name", "patient_id", "orig_img_path", "comp_img_path", "orig_img_name", "comp_img_name",
                "original_size", "compressed_size", "cr_ratio", "psnr", "ssim", "compression_time"])

        # Write the data row
        writer.writerow(data)

    print(f"Metrics saved to {csv_filename}")