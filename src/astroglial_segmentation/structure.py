import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


# data = np.load("../data/im1_seg.npy", allow_pickle=True).item()
# print("Keys in data: \n", data.keys())


def print_data_structure(data, indent=""):
    for key in data.keys():
        pprint(f"{indent}Key: {key}")

        pprint(f"{indent}Type: {type(data[key])}")
        if isinstance(data[key], np.ndarray):
            pprint(f"{indent}Shape: {data[key].shape}")
        elif isinstance(data[key], list):
            pprint(f"{indent}Outer list length: {len(data[key])}")
            if len(data[key]) > 0 and isinstance(data[key][0], list):
                pprint(f"{indent}Inner list length: {len(data[key][0])}")
            else:
                pprint(f"{indent}This is a 1-dimensional list.")
        else:
            pprint(f"{indent}Shape: Not applicable")
        pprint("\n")
        if isinstance(data[key], dict):
            print_data_structure(data[key], indent + "  ")


# print_data_structure(data)

print("ops['meanImg'] shape: ", data["meanImg"].shape)
# pprint("Colors: \n", data["colors"][0:2])
# pprint("chan_choose: \n", data["chan_choose"])

img = data["meanImg_chan2_en"]

plt.imshow(img, cmap="gray")
plt.title("Mean Image Channel 2")
plt.axis("off")
plt.show()
