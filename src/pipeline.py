from .cellposesegmentation import segment_cells
from .extraction import create_suite2p_masks_extract_traces
import argparse


def main_pipeline(data_path):

    # Do cellpose segmentation and mask combination
    segment_cells(data_path)

    # Use cobined masks from previous step to create suite2p masks and extract traces
    create_suite2p_masks_extract_traces(data_path)

    print("\n Pipeline complete! \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Astroglial segmentation and suite2p trace extraction pipeline"
    )
    parser.add_argument(
        "data_path", type=str, help="Path to data folder, it sutie2p output folder"
    )
    args = parser.parse_args()
    data_path = args.data_path

    main_pipeline(data_path)
