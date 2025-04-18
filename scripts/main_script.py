import os
import time
import torch
import shutil
import numpy as np
from skimage import io, img_as_ubyte, transform
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from cnn_model.cnn_model import CNNModel
from scripts.utils.helper import pthFile_check
from scripts.utils.file_utils import evaluate_compression
from scripts.utils.file_utils import (
    evaluate_compression, saveCSV, 
    get_FolderName, create_and_upload_compressed_dicom
)

# Load pre-trained MobileNetV2 model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_cnn_model(model_path, device, input_size=64):
    model = CNNModel(input_size=input_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Adjust the checkpoint to match the current model's structure
    checkpoint_state_dict = checkpoint
    model_state_dict = model.state_dict()

    for key in checkpoint_state_dict.keys():
        if key in model_state_dict and checkpoint_state_dict[key].shape != model_state_dict[key].shape:
            print(f"Resizing layer: {key} from {checkpoint_state_dict[key].shape} to {model_state_dict[key].shape}")
            if "weight" in key:
                checkpoint_state_dict[key] = torch.nn.functional.interpolate(
                    checkpoint_state_dict[key].unsqueeze(0).unsqueeze(0),
                    size=model_state_dict[key].shape,
                    mode="nearest"
                ).squeeze(0).squeeze(0)
            elif "bias" in key:
                checkpoint_state_dict[key] = torch.zeros_like(model_state_dict[key])

    # Load the adjusted checkpoint
    model.load_state_dict(checkpoint_state_dict, strict=False)
    model.eval()
    return model

def load_image(file_path, target_size=(256, 256)):
    try:
        image = io.imread(file_path, as_gray=True)
        image = transform.resize(image, target_size, anti_aliasing=True)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
        return image.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error loading image at {file_path}: {e}")

# Partition image into smaller blocks
def partition_image(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            
            if block.shape[:2] != (block_size, block_size):
                block = np.pad(block, ((0, block_size - block.shape[0]), (0, block_size - block.shape[1])), mode='edge')
            blocks.append(block)
    return blocks

# Apply affine transformation
def apply_affine_transformation(block, transformation):
    scale, rotation, tx, ty = transformation
    h, w = block.shape

    transformation_matrix = np.array([
        [scale * np.cos(rotation), -scale * np.sin(rotation), tx],
        [scale * np.sin(rotation), scale * np.cos(rotation), ty]
    ])

    y, x = np.indices((h, w))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())]) 

    transformed_coords = transformation_matrix @ coords
    x_transformed = transformed_coords[0, :].reshape(h, w)
    y_transformed = transformed_coords[1, :].reshape(h, w)

    x_transformed = np.clip(x_transformed, 0, w - 1).astype(np.int32)
    y_transformed = np.clip(y_transformed, 0, h - 1).astype(np.int32)

    transformed_block = block[y_transformed, x_transformed]
    return transformed_block

class KDNode:
    def __init__(self, point, index, left=None, right=None):
        self.point = point  # the feature vector
        self.index = index  # index of the original block
        self.left = left
        self.right = right

def build_kdtree(points, indices, depth=0):
    if len(points) == 0:
        return None

    k = points.shape[1]  # feature vector dimension
    axis = depth % k

    # Sort points and indices together based on the current axis
    sorted_indices = np.argsort(points[:, axis])
    points = points[sorted_indices]
    indices = indices[sorted_indices]
    
    median = len(points) // 2

    return KDNode(
        point=points[median],
        index=indices[median],
        left=build_kdtree(points[:median], indices[:median], depth + 1),
        right=build_kdtree(points[median + 1:], indices[median + 1:], depth + 1)
    )

def find_nearest_in_kdtree(node, target, best=None, best_dist=float('inf'), depth=0):
    if node is None:
        return best, best_dist

    k = len(target)
    axis = depth % k
    
    current_dist = np.sum((node.point - target) ** 2)
    
    if current_dist < best_dist:
        best = node
        best_dist = current_dist

    if target[axis] < node.point[axis]:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    best, best_dist = find_nearest_in_kdtree(first, target, best, best_dist, depth + 1)
    
    if abs(target[axis] - node.point[axis]) ** 2 < best_dist:
        best, best_dist = find_nearest_in_kdtree(second, target, best, best_dist, depth + 1)
    
    return best, best_dist

def encode_image_with_kdtree(image, block_size=8, cnn_model=None, device=None):
    range_blocks = partition_image(image, block_size)
    domain_blocks = range_blocks  # Use same blocks for both to reduce computation

    # Extract all features at once in a single batch
    batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in domain_blocks]).to(device)
    with torch.no_grad():
        all_features, _ = cnn_model(batch_tensor)
        all_features = all_features.view(all_features.size(0), -1).cpu().numpy()

    # Build KD-tree using domain features
    domain_indices = np.arange(len(domain_blocks))
    kd_tree = build_kdtree(all_features, domain_indices)

    # Search for nearest neighbors
    encoded_data = []
    transformation = (1.0, 0.0, 1, 1)  # Fixed transformation
    
    for feature in all_features:  # Use pre-computed features
        best_node, _ = find_nearest_in_kdtree(kd_tree, feature)
            
        encoded_data.append((best_node.index, transformation))

    return encoded_data, domain_blocks

# Decode the image
def decode_image(encoded_data, domain_blocks, image_shape, block_size=8, output_file=None, output_path='data/compressed'):
    os.makedirs(output_path, exist_ok=True)
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    h, w = image_shape
    idx = 0

    # Decode the data from the raw binary format and include transformation info
    decoded_data = [(int(entry[0]), entry[1]) for entry in encoded_data]

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx < len(decoded_data):
                best_index, transformation = decoded_data[idx]
                transformed_block = apply_affine_transformation(domain_blocks[best_index], transformation)
                reconstructed_image[i:i + block_size, j:j + block_size] = transformed_block
                idx += 1

    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    if is_low_contrast(reconstructed_image):
        reconstructed_image = rescale_intensity(reconstructed_image, in_range='image', out_range=(0, 1))

    reconstructed_image = img_as_ubyte(reconstructed_image)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imsave(output_file, reconstructed_image)


#=======================================================================================================================================

# ----------------------
# Compression Function
# ----------------------
def compress_and_show_images(
    uploaded_files,
    patient_name,
    patient_id,
    patient_birthdate="Unknown",
    patient_sex="Unknown",
    upload_to_pacs=False,
    source_dicom_paths=None,
    block_size=8
):
    # ----------------------
    # Load CNN Model
    # ----------------------
    cnn_model_path, device = pthFile_check()
    cnn_model = load_cnn_model(cnn_model_path, device, input_size=8)

    compressed_files = []
    batch_FolderName = get_FolderName(patient_name, patient_id)

    # Set up directory paths
    orig_data_path = os.path.join("data", "original", f"{patient_name}_{patient_id}", batch_FolderName)
    comp_data_path = os.path.join("data", "compressed", f"{patient_name}_{patient_id}", batch_FolderName)
    os.makedirs(orig_data_path, exist_ok=True)
    os.makedirs(comp_data_path, exist_ok=True)

    # Ensure uploaded_files is a list
    if isinstance(uploaded_files, str):
        uploaded_files = [uploaded_files]

    for idx, uploaded_file in enumerate(uploaded_files):
        # Save original image to proper folder
        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                raise FileNotFoundError(f"Image path does not exist: {uploaded_file}")
            orig_img_name = os.path.basename(uploaded_file)
            dest_img_path = os.path.join(orig_data_path, orig_img_name)
            shutil.copy(uploaded_file, dest_img_path)
            orig_img_path = dest_img_path
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

        # Load image for processing
        image = load_image(orig_img_path)

        # Compress image
        start_time = time.perf_counter()
        encoded_data, domain_blocks = encode_image_with_kdtree(image, block_size, cnn_model, device)
        comp_img_name = f"compressed_{orig_img_name}"
        comp_img_path = os.path.join(comp_data_path, comp_img_name)
        decode_image(encoded_data, domain_blocks, image.shape, block_size, comp_img_path, comp_data_path)
        end_time = time.perf_counter()

        # Evaluate compression metrics
        compression_time = round(end_time - start_time, 4)
        original_size, compressed_size, cr_ratio, psnr, ssim = evaluate_compression(image, orig_img_path, comp_img_path)

        # Optionally upload to PACS
        if upload_to_pacs and source_dicom_paths and idx < len(source_dicom_paths):
            try:
                create_and_upload_compressed_dicom(source_dicom_paths[idx], comp_img_path)
            except Exception as e:
                print(f"⚠️ PACS upload failed for {source_dicom_paths[idx]}: {e}")

        # Save to CSV
        saveCSV(
            batch_FolderName,
            patient_name,
            patient_id,
            orig_img_path,
            comp_img_path,
            os.path.basename(orig_img_path),
            os.path.basename(comp_img_path),
            original_size,
            compressed_size,
            cr_ratio,
            psnr,
            ssim,
            compression_time,
            "DataCollection.csv",
            patient_birthdate,
            patient_sex
        )

        # Collect for return
        compressed_files.append({
            "original_image": orig_img_path,
            "compressed_image": comp_img_path,
            "compression_time": compression_time,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "cr_ratio": cr_ratio,
            "psnr": psnr,
            "ssim": ssim
        })

    return compressed_files, batch_FolderName