from .cellposesegmentation import segment_cells
from .extraction import create_suite2p_masks_extract_traces
import argparse


def main_pipeline(data_path, extraction=True):

    # Do cellpose segmentation and mask combination
    print("\n Starting cellpose segmentation and mask combination... \n")
    segment_cells(data_path)
    print("\n Cellpose segmentation and mask combination complete! \n")

    # Use combined masks from previous step to create suite2p masks and extract traces
    if extraction:
        print("\n Starting suite2p mask creation and trace extraction")
        create_suite2p_masks_extract_traces(data_path)
        print("\n Pipeline complete! You can now investigate masks in suite2p \n")
    else:
        print("\n Extraction step skipped (extraction=False). \n")


if __name__ == "__main__":

    """
    To run the pipeline, run the following command in the terminal:
    python -m astroglial_segmentation <suite2p_output_folder> [--no-extraction]
    """

    parser = argparse.ArgumentParser(
        description="Run Astroglial segmentation and suite2p trace extraction pipeline"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to data folder, it is the suite2p output folder",
    )
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Skip Suite2p mask creation and trace extraction",
    )
    args = parser.parse_args()
    data_path = args.data_path
    extraction = not args.no_extraction

    main_pipeline(data_path, extraction=extraction)
