import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
import os
from .combination import combine_masks


base_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(base_dir, "../../data")

# Construct the absolute paths for the model directories
model_dir1 = os.path.join(base_dir, "models/CP2_s3_039234")  # processes
model_dir2 = os.path.join(base_dir, "models/CP2_s2_039234")  # body
model_dir3 = os.path.join(
    base_dir, "models/CP2_s1_039189"
)  # complete cell
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
        mean_image, masks, flows, model.diam_labels, file_names=file_name
    )

    return flows


def segment_cells(data_path, model_dirs=model_dirs):
    io.logger_setup()


    try:
        ops = np.load(data_path + "/ops.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("Ops file not found.")
        raise SystemExit
    mean_image = ops["meanImg"]

    for model_dir in model_dirs:
        print(f"Segmenting using {model_dir}")
        model = models.CellposeModel(pretrained_model=model_dir)
        # Save mean_image as PNG for compatibility with Cellpose
        model_name = os.path.basename(model_dir)
        plt.imsave(data_path + f"/{model_name}_mean_image.png", mean_image, cmap="gray")

        flows = cellpose_segmentation(
            mean_image,
            model,
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
