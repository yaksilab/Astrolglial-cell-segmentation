import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
import os
from .combination import combine_masks
from scipy.signal import medfilt2d
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(base_dir, "../../data")

# Construct the absolute paths for the model directories
model_dir1 = os.path.join(base_dir, "models/CP2_s3_039234")  # processes
model_dir2 = os.path.join(base_dir, "models/CP2_s2_039234")  # body
model_dir3 = os.path.join(
    base_dir, "models/CP2_s1_039189"
)  # complete cell
model_dirs = [model_dir1, model_dir2, model_dir3]


def cellpose_segmentation(mean_image, model, flow_threshold, cellprob_threshold, file_name="mean_image"):
    """
    Perform cell segmentation using the Cellpose model.

    Args:
        mean_image (numpy.ndarray): The mean image to be segmented. Extracted from suite2p ops file.
        model (cellpose model): A custom trained cellpose model for a spesific cellpart.
        file_name (str, optional): The name of the file to save the segmentation results. Defaults to "mean_image".

    Returns:
        numpy.ndarray: The flows array containing flows[0] beings masks, flows[1] being gradient flow, flows[2] cellprobability.
    """
    masks, flows, _ = model.eval( mean_image, channels=[0, 0], flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold )

    #masks, flows, _ = model.eval(
    #    mean_image, flow_threshold, cellprob_threshold, channels=[0, 0],
    #)

    io.masks_flows_to_seg(
        mean_image, masks, flows, model.diam_labels, file_names=file_name
    )

    return flows


def compute_enhanced_mean_image(I, ops):
    """ computes enhanced mean image
    Median filters ops["meanImg"] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops["meanImgE"]
    """
    I = ops["meanImg"].astype(np.float32)
    if "spatscale_pix" not in ops:
        if isinstance(ops["diameter"], int):
            diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            diameter = np.array(ops["diameter"])
        if diameter[0] == 0:
            diameter[:] = 12
        ops["spatscale_pix"] = diameter[1]
        ops["aspect"] = diameter[0] / diameter[1]

    diameter = 4 * np.ceil(
        np.array([ops["spatscale_pix"] * ops["aspect"], ops["spatscale_pix"]])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0, np.minimum(1, mimg0))
    return mimg0


# Apply CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image.astype(np.uint8))

# Apply Gaussian High-Pass Filtering
def apply_gaussian_highpass(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    high_pass = image - blurred
    high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min())  # Normalize
    return high_pass

# Apply Unsharp Masking
def apply_unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    high_pass = image - blurred
    unsharp = image + 0.5 * high_pass  # Weight the high-pass component
    unsharp = (unsharp - unsharp.min()) / (unsharp.max() - unsharp.min())  # Normalize
    return unsharp



def segment_cells(data_path, flow_threshold, cellprob_threshold, model_dirs=model_dirs):
    io.logger_setup()


    try:
        ops = np.load(data_path + "/ops.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("Ops file not found.")
        raise SystemExit
    mean_image1 = ops["meanImg_chan2_corrected"]
    meanImgCorr_enhanced = compute_enhanced_mean_image(mean_image1, ops)

    # # Plot the original and enhanced images
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(mean_image1, cmap='gray')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.title("Enhanced Mean Corrected Image")
    # plt.imshow(mean_image, cmap='gray')
    # plt.colorbar()
    # plt.show()

    # ------------------ Enahnace Image ------------------
    
    # Load the original image
    original_image = mean_image1                                                                                                           
    normalized_image = ((mean_image1 - mean_image1.min()) / (mean_image1.max() - mean_image1.min()) * 255).astype(np.uint8)
    
    # Perform the methods
    clahe_image = apply_clahe(normalized_image)
    #gaussian_highpass_image = apply_gaussian_highpass(normalized_image)
    
    # Scale intensities to brighten the image with a higher scaling factor
    scaling_factor = 1.8  # Increased brightening factor
    brightened_image_1_5 = np.clip(normalized_image * scaling_factor, 0, 255)  # Clip to valid range

    # Convert back to uint8 for display
    brightened_image_1_5 = brightened_image_1_5.astype(np.uint8)

    # Plot the results
    plt.figure(figsize=(60, 40))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("CLAHE Enhanced Image")
    plt.imshow(clahe_image, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Corrcted and Enhanced Mean Image")
    plt.imshow(meanImgCorr_enhanced, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Brightened Image")
    plt.imshow(brightened_image_1_5, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(data_path + "/enhanced_mean_image.png")
    plt.show()
    
    mean_image = clahe_image

    # ------------------ Cellpose Segmentation ------------------
    for model_dir in model_dirs:
        print(f"Segmenting using {model_dir}")
        model = models.CellposeModel(pretrained_model=model_dir)
        # Save mean_image as PNG for compatibility with Cellpose
        model_name = os.path.basename(model_dir)
        plt.imsave(data_path + f"/{model_name}_mean_image.png", mean_image, cmap="gray")

        flows = cellpose_segmentation(
            mean_image,
            model, flow_threshold, cellprob_threshold,
            file_name=data_path + f"/{model_name}_mean_image",
        )

    masks = combine_masks(data_path)

    io.masks_flows_to_seg(
        mean_image,
        masks,
        flows=flows,
        diams=30.0,
        file_names=data_path + "/combined_mean_image",
    )

