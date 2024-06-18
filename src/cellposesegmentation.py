import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
import os
from .combination import combine_masks

io.logger_setup()


base_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.join(base_dir, "../data")

# Construct the absolute paths for the model directories
model_dir1 = os.path.join(base_dir, "../models/CP2_s3_039234")  # processes
model_dir2 = os.path.join(base_dir, "../models/CP2_s2_039234")  # body
model_dir3 = os.path.join(
    base_dir, "../models/CP2_s1_039189"
)  # body plus processes in one model
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

    masks, flows, styles = model.eval(
        mean_image, channels=[0, 0], flow_threshold=1.0, cellprob_threshold=0.0
    )
    # fig = plt.figure(figsize=(15, 10))
    # plot.show_segmentation(fig, mean_image, masks, flows[0], channels=[0, 0])

    # plt.tight_layout()
    # plt.show()
    # mask_overlay = plot.mask_overlay(im0, masks)
    # plt.imshow(mask_overlay)git status
    # plt.show()
    io.masks_flows_to_seg(
        mean_image, masks, flows, model.diam_labels, file_names=file_name
    )

    return flows


def segment_cells(data_path, model_dirs=model_dirs):
    ops = np.load(data_path + "/ops.npy", allow_pickle=True).item()
    mean_image = ops["meanImg"]

    for model_dir in model_dirs:
        print(f"Segmenting using {model_dir}")
        model = models.CellposeModel(pretrained_model=model_dir)
        # Save mean_image as PNG for compatibility with Cellpose
        plt.imsave(
            data_path + f"/{model_dir[14:16]}_mean_image.png", mean_image, cmap="gray"
        )

        flows = cellpose_segmentation(
            mean_image,
            model,
            file_name=data_path + f"/{model_dir[14:16]}_mean_image",
        )

    masks = combine_masks(data_path)

    io.masks_flows_to_seg(
        mean_image,
        masks,
        flows=flows,
        diams=30.0,
        file_names=data_path + "/combined_mean_image",
    )
