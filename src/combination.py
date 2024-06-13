import numpy as np


def extend_and_merge_masks(mask1, mask2, overlap_threshold):
    labels1 = np.unique(mask1)
    labels2 = np.unique(mask2)
    labels1 = labels1[labels1 != 0]  # excluding background
    labels2 = labels2[labels2 != 0]

    extended_mask = mask1.copy()

    new_label = max(labels1, default=0) + 1

    for label2 in labels2:
        mask2_region = mask2 == label2
        overlapping_labels = []

        for label1 in labels1:
            mask1_region = mask1 == label1

            overlap = np.sum(mask2_region & mask1_region)
            mask2_area = np.sum(mask2_region)

            if overlap / mask2_area > overlap_threshold:
                overlapping_labels.append(label1)
                # print(
                #     f"Overlap detected between {label1} and {label2}: {overlap / mask2_area}"
                # )

        if overlapping_labels:
            for ol_label in overlapping_labels:
                extended_mask[extended_mask == ol_label] = new_label
            extended_mask[mask2_region] = new_label
            new_label += 1
        else:
            extended_mask[mask2_region] = new_label
            new_label += 1

    return extended_mask


def combine_masks(
    data_path, overlap_threshold_processes=0.15, overlap_threshold_body=0.35
):
    body_mask = np.load(data_path + "/s2_mean_image_seg.npy", allow_pickle=True).item()[
        "masks"
    ]
    outflow_mask = np.load(
        data_path + "/s3_mean_image_seg.npy", allow_pickle=True
    ).item()["masks"]
    combined_mask = np.load(
        data_path + "/s1_mean_image_seg.npy", allow_pickle=True
    ).item()["masks"]

    masks = extend_and_merge_masks(
        combined_mask, outflow_mask, overlap_threshold_processes
    )
    masks = extend_and_merge_masks(masks, body_mask, overlap_threshold_body)

    return masks
