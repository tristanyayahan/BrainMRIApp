import os
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


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
        #CR = original_size / compressed_size
        CR_ratio = f"1:{original_size // compressed_size}"
    else:
        #CR = float('inf')
        CR_ratio = "Infinity:1"

    original_size = round(original_size / 1024, 2)  
    compressed_size = round(compressed_size / 1024, 2)  

    return original_size, compressed_size, CR_ratio, PSNR, SSIM