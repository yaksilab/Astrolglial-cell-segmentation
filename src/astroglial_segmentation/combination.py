import numpy as np
import glob
from .utils import reassign_consecutive_labels


def extend_and_merge_masks(mask1, mask2, overlap_threshold):
    """
    extends mask1 using mask2 by merging regions that overlap more than overlap_threshold

    Args:
        mask1 (numpy.ndarray): first mask
        mask2 (numpy.ndarray): second mask
        overlap_threshold (float):  relativ overlap threshold

    Returns:
       merged masks (numpy.ndarray): extended and merged mask
    """

    labels1 = np.unique(mask1)
    labels2 = np.unique(mask2)
    labels1 = labels1[labels1 != 0]  # excluding background
    labels2 = labels2[labels2 != 0]

    extended_mask = mask1.copy()

    new_label = max(labels1, default=0) + 1

    total_overlaps = {}

    for label2 in labels2:
        mask2_region = mask2 == label2
        mask2_area = np.sum(mask2_region)
        overlaps = []

        for label1 in labels1:
            mask1_region = mask1 == label1

            overlap = np.sum(mask2_region & mask1_region)
            mask1_area = np.sum(mask1_region)

            if (
                overlap / mask2_area > overlap_threshold
                or overlap / mask1_area > overlap_threshold
            ):
                overlaps.append(label1)

        if overlaps:
            total_overlaps[label2] = overlaps
        else:
            extended_mask[mask2 == label2] = new_label
            new_label += 1
    # Here i devide the important overlapping regions into two groups. These two groups contains two critical overlap cases that appear. one is that when several masks from
    # From mask2 is overlapping with one mask from mask1. The other case is that when one mask from mask2 is overlapping with several masks from mask1.
    # This makes it easier to merge related masks with each other for better segmentation.
    overlaps_label1 = (
        {}
    )  # overlapping masks from mask1 to mask2, keys here are the labels from mask1
    overlaps_label2 = (
        {}
    )  # overlapping masks from mask2 to mask1, keys here are the labels from mask2

    for label2, label1_overlaps in total_overlaps.items():
        if len(label1_overlaps) > 1:
            overlaps_label2[label2] = label1_overlaps
        else:
            if label1_overlaps[0] in overlaps_label1:
                overlaps_label1[label1_overlaps[0]].append(label2)
            else:
                overlaps_label1[label1_overlaps[0]] = [label2]

    for label1, label2_overlaps in overlaps_label1.items():
        extended_mask[mask1 == label1] = new_label
        for label2 in label2_overlaps:
            extended_mask[mask2 == label2] = new_label
        new_label += 1

    for label2, label1_overlaps in overlaps_label2.items():
        extended_mask[mask2 == label2] = new_label
        for label1 in label1_overlaps:
            extended_mask[mask1 == label1] = new_label
        new_label += 1

    return extended_mask


def combine_masks(
    data_path,
    image_type="meanImg",
    channel=1,
    overlap_threshold_processes=0.15,
    overlap_threshold_body=0.35,
):
    """
    Combines the masks from the three cellpose models

    Args:
        data_path (str): path to the data folder
        image_type (str): type of image used for segmentation
        channel (int): channel used for segmentation
        overlap_threshold_processes (float, optional): overlap threshold for processes. Defaults to 0.15.
        overlap_threshold_body (float, optional): overlap threshold for body. Defaults to 0.35.

    Returns:
        numpy.ndarray: combined masks
    """

    try:
        # Update glob patterns to match new naming convention
        pattern_s1 = f"*s1*{image_type}_ch{channel}*.npy"
        pattern_s2 = f"*s2*{image_type}_ch{channel}*.npy"
        pattern_s3 = f"*s3*{image_type}_ch{channel}*.npy"

        combined_mask_file = glob.glob(data_path + "/" + pattern_s1)[0]
        body_mask_file = glob.glob(data_path + "/" + pattern_s2)[0]
        outflow_mask_file = glob.glob(data_path + "/" + pattern_s3)[0]

        print(f"Found mask files:")
        print(f"  Combined (s1): {combined_mask_file}")
        print(f"  Body (s2): {body_mask_file}")
        print(f"  Outflow (s3): {outflow_mask_file}")

    except IndexError:
        # Fallback to old naming convention for backward compatibility
        try:
            combined_mask_file = glob.glob(data_path + "/*s1*.npy")[0]
            body_mask_file = glob.glob(data_path + "/*s2*.npy")[0]
            outflow_mask_file = glob.glob(data_path + "/*s3*.npy")[0]
            print("Using legacy mask file naming convention")
        except IndexError:
            raise FileNotFoundError(
                f"One or more mask files not found in the specified data path. "
                f"Looking for files matching patterns: {pattern_s1}, {pattern_s2}, {pattern_s3}"
            )

    body_mask = np.load(body_mask_file, allow_pickle=True).item()["masks"]
    outflow_mask = np.load(outflow_mask_file, allow_pickle=True).item()["masks"]
    combined_mask = np.load(combined_mask_file, allow_pickle=True).item()["masks"]

    masks = extend_and_merge_masks(
        combined_mask, outflow_mask, overlap_threshold_processes
    )
    masks = extend_and_merge_masks(masks, body_mask, overlap_threshold_body)
    masks = reassign_consecutive_labels(masks)

    return masks
