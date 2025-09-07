import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
import os
from .combination import combine_masks


base_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(base_dir, "../../data")

# Construct the absolute paths for the model directories
model_dir1 = os.path.join(base_dir, "../models/CP2_s3_039234")  # processes
model_dir2 = os.path.join(base_dir, "../models/CP2_s2_039234")  # body
model_dir3 = os.path.join(base_dir, "../models/CP2_s1_039189")  # complete cell
model_dirs = [model_dir1, model_dir2, model_dir3]


def cellpose_segmentation(mean_image, model, file_name="mean_image"):
    """
    Perform cell segmentation using the Cellpose model.

    Args:
        mean_image (numpy.ndarray): The mean image to be segmented. Extracted from suite2p ops file.
        model (cellpose model): A custom trained cellpose model for a spesific cellpart.
        file_name (str, optional): The name of the file to save the segmentation results. Defaults to "mean_image".

    Returns:
        numpy.ndarray: The flows array containing flows[0] beings masks, flows[1] being gradient flow, flows[2] cellprobability.
    """

    masks, flows, _ = model.eval(
        mean_image, channels=[0, 0], flow_threshold=1.0, cellprob_threshold=0.0
    )

    io.masks_flows_to_seg(
        mean_image, masks, flows, diams=model.diam_labels, file_names=file_name
    )

    return flows


def segment_cells(data_path, model_dirs=model_dirs):
    """
    Segment cells using multiple Cellpose models and combine the results.
    Args:
        data_path (str): Path to the directory containing 'ops.npy' and 'data.bin'.
        model_dirs (list, optional): List of paths to the Cellpose model directories. Defaults to predefined model directories.
    """

    io.logger_setup()

    try:
        ops = np.load(data_path + "/ops.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("Ops file not found.")
        raise SystemExit

    # Select the appropriate image based on image_type parameter
    available_images = {
        "meanImg": "meanImg",
        "meanImg_chan2": "meanImg_chan2",
        "meanImgE": "meanImgE",
        "max_proj": "max_proj",
    }

    if image_type not in available_images:
        raise ValueError(
            f"Invalid image_type '{image_type}'. Available options: {list(available_images.keys())}"
        )

    image_key = available_images[image_type]

    if image_key not in ops:
        raise KeyError(
            f"Image type '{image_type}' (key: '{image_key}') not found in ops file. Available keys: {list(ops.keys())}"
        )

    mean_image = ops[image_key]

    print(f"Using {image_type} for segmentation on channel {segmentation_channel}")
    print(f"Image shape: {mean_image.shape}")

    for model_dir in model_dirs:
        print(f"Segmenting using {model_dir}")
        model = models.CellposeModel(pretrained_model=model_dir)
        # Save mean_image as PNG for compatibility with Cellpose
        model_name = os.path.basename(model_dir)
        output_filename = f"{model_name}_{image_type}_ch{segmentation_channel}"
        plt.imsave(data_path + f"/{output_filename}.png", mean_image, cmap="gray")

        flows = cellpose_segmentation(
            mean_image,
            model,
            file_name=data_path + f"/{output_filename}",
        )

    # Combine the masks from the three models using their saved files _seg.npy files in data_path
    masks = combine_masks(data_path)

    # Writes the combined masks to a _seg.npy file
    # Note the flows here are from the last model used in the loop above
    io.masks_flows_to_seg(
        mean_image,
        masks,
        flows=flows,
        diams=30.0,
        file_names=data_path + f"/{combined_filename}",
    )
