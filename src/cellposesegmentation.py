import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt
from combination import extend_and_merge_masks


io.logger_setup()


data_path = "../data"
model_dir1 = "../models/CP2_s3_039234"
model_dir2 = "../models/CP2_s2_039234"
model_dir3 = "../models/CP2_s1_039189"
model_dirs = [model_dir1, model_dir2, model_dir3]
flowss = 0


def cellpose_segmentation(mean_image, model, file_name="mean_image"):

    masks, flows, styles = model.eval(
        mean_image, channels=[0, 0], flow_threshold=0.9, cellprob_threshold=0.0
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


def segment_cells(data_path, model_dirs):
    ops = np.load(data_path + "/ops.npy", allow_pickle=True).item()
    mean_image = ops["meanImg"]

    for model_dir in model_dirs:
        model = models.CellposeModel(pretrained_model=model_dir)
        # Save mean_image as PNG
        plt.imsave(
            "../output/" + f"{model_dir[14:16]}_mean_image.png", mean_image, cmap="gray"
        )

        flows = cellpose_segmentation(
            mean_image,
            model,
            file_name="../output/" + f"{model_dir[14:16]}_mean_image",
        )

    # Load the masks and outlines
    body_mask = np.load("../output/s2_mean_image_seg.npy", allow_pickle=True).item()[
        "masks"
    ]
    outflow_mask = np.load("../output/s3_mean_image_seg.npy", allow_pickle=True).item()[
        "masks"
    ]
    combined_mask = np.load(
        "../output/s1_mean_image_seg.npy", allow_pickle=True
    ).item()["masks"]

    overlap_threshold = 0.25
    masks = extend_and_merge_masks(combined_mask, outflow_mask, overlap_threshold)
    masks = extend_and_merge_masks(masks, body_mask, overlap_threshold)
    io.masks_flows_to_seg(
        mean_image,
        masks,
        flows=flows,
        diams=30.0,
        file_names="../output/combined_mean_image",
    )


segment_cells(data_path, model_dirs)
