import numpy as np
import pprint
import matplotlib.pyplot as plt


data = np.load("../data/im1_seg.npy", allow_pickle=True).item()
print(type(data))

print("Keys in data: \n", data.keys())


def print_data_structure(data, indent=""):
    for key in data.keys():
        print(f"{indent}Key: {key}")
        print(f"{indent}Type: {type(data[key])}")
        if isinstance(data[key], np.ndarray):
            print(f"{indent}Shape: {data[key].shape}")
        elif isinstance(data[key], list):
            print(f"{indent}Outer list length: {len(data[key])}")
            if len(data[key]) > 0 and isinstance(data[key][0], list):
                print(f"{indent}Inner list length: {len(data[key][0])}")
            else:
                print(f"{indent}This is a 1-dimensional list.")
        else:
            print(f"{indent}Shape: Not applicable")
        print("\n")
        if isinstance(data[key], dict):
            print_data_structure(data[key], indent + "  ")


print_data_structure(data)

print("Colors: \n", data["colors"][0:2])
print("chan_choose: \n", data["chan_choose"])
