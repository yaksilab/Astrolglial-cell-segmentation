from matplotlib import pyplot as plt
import numpy as np
from cellpose import plot, utils, io


data = np.load("../data/im1_seg.npy", allow_pickle=True).item()

# Plotting the masks
masks = data["masks"]
plt.imshow(masks, cmap="rainbow")
plt.show()


outlines = data["outlines"]
plt.imshow(outlines, cmap="gray")
plt.show()
