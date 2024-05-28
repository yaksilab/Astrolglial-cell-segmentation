import numpy as np
import matplotlib.pyplot as plt
import cv2


from scipy.ndimage import binary_closing, binary_opening

# Load masks and outlines
body_mask = np.load("../data/im9body_seg.npy", allow_pickle=True).item()["masks"]
outflow_mask = np.load("../data/im9outflows_seg.npy", allow_pickle=True).item()["masks"]
combined_mask = np.load("../data/im9combi_seg.npy", allow_pickle=True).item()["masks"]

body_outlines = np.load("../data/im9body_seg.npy", allow_pickle=True).item()["outlines"]
outflow_outlines = np.load("../data/im9outflows_seg.npy", allow_pickle=True).item()[
    "outlines"
]
combined_outlines = np.load("../data/im9combi_seg.npy", allow_pickle=True).item()[
    "outlines"
]

# Initialize combined mask and outlines
combined_result_mask = np.zeros_like(combined_mask)
combined_result_outlines = np.zeros_like(combined_outlines)


# Function to reassign labels
def reassign_labels(mask, offset):
    unique_labels = np.unique(mask)
    new_mask = np.zeros_like(mask)
    for label in unique_labels:
        if label != 0:
            new_mask[mask == label] = label + offset
    return new_mask


# Assign unique labels
offset_body = np.max(body_mask)
offset_outflow = offset_body + np.max(outflow_mask)
offset_combined = offset_outflow + np.max(combined_mask)

body_mask = reassign_labels(body_mask, 0)
outflow_mask = reassign_labels(outflow_mask, offset_body)
combined_mask = reassign_labels(combined_mask, offset_outflow)

# Combine masks
combined_result_mask = np.maximum(body_mask, np.maximum(outflow_mask, combined_mask))

# Assign unique labels to outlines
body_outlines = reassign_labels(body_outlines, 0)
outflow_outlines = reassign_labels(outflow_outlines, offset_body)
combined_outlines = reassign_labels(combined_outlines, offset_outflow)

# Combine outlines
combined_result_outlines = np.maximum(
    body_outlines, np.maximum(outflow_outlines, combined_outlines)
)


# Save combined masks and outlines to a new *_seg.npy file
combined_seg = {
    "outlines": combined_result_outlines,
    "colors": np.zeros((1, 3)),  # Placeholder for colors, can be modified if needed
    "masks": combined_result_mask,
    "chan_choose": [],
    "filename": "",
    "flows": [],
    "ismanual": np.zeros((1,)),
    "manual_changes": [],
    "model_path": "",
    "flow_threshold": 0.0,
    "cellprob_threshold": 0.0,
}

np.save("../data/imcombined_seg.npy", combined_seg)
# # Post-process combined mask and outlines
# combined_result_mask = binary_closing(combined_result_mask).astype(np.int)
# combined_result_outlines = binary_opening(combined_result_outlines).astype(np.int)

# Plot the results

plt.imshow(combined_result_mask, cmap="gray")
plt.title("Combined Mask")
plt.show()


final_mask = np.zeros_like(body_mask)
final_mask = np.maximum(final_mask, body_mask)

plt.subplot(2, 3, 1)
plt.imshow(final_mask.astype(np.uint8), cmap="gray")
plt.title("Final Mask")

final_mask = np.maximum(final_mask, outflow_mask)
plt.subplot(2, 3, 2)
plt.imshow(final_mask.astype(np.uint8), cmap="gray")
plt.title("Final Mask with Outflow Mask")

connection_regions = (combined_mask > 0) & ((body_mask == 0) & (outflow_mask == 0))
plt.subplot(2, 3, 3)
plt.imshow(connection_regions.astype(np.uint8), cmap="gray")
plt.title("Connection Regions")
final_mask[connection_regions] = 100


plt.subplot(2, 3, 4)
plt.imshow(final_mask.astype(np.uint8), cmap="gray")
plt.title("Final Mask with Connection Regions")


# kernel = np.ones((5, 5), np.uint8)
# final_mask = final_mask.astype(np.uint8)  # Convert final_mask to uint8
# final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
# final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

# plt.subplot(2, 3, 5)
# plt.imshow(final_mask.astype(np.uint8), cmap="gray")
# plt.title("Final Mask with Morphological Operations")

# plt.subplot(2, 2, 4)
# plt.imshow(labels_im)
# plt.title("Connected Components")

plt.tight_layout()
plt.show()

num_labels, labels_im = cv2.connectedComponents(final_mask.astype(np.uint8))

labels_im = cv2.convertScaleAbs(labels_im)  # convert to 8-bit unsigned integer

if len(labels_im.shape) == 2:  # if the image is grayscale
    labels_im = cv2.cvtColor(labels_im, cv2.COLOR_GRAY2BGR)  # convert to BGR

labels_im = cv2.cvtColor(labels_im, cv2.COLOR_HSV2BGR)  # convert from HSV to BGR

labels_im[labels_im == 0] = 0

print("Shape: ", labels_im.shape)

plt.imshow(labels_im)
plt.show()
