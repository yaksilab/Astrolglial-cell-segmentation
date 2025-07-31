from .cellposesegmentation import segment_cells
from .extraction import create_suite2p_masks_extract_traces
import argparse


def main_pipeline(
    data_path,
    segmentation_channel=1,
    extraction_channel=1,
    image_type="meanImg",
    run_extraction=True,
):
    """
    Main pipeline for astroglial cell segmentation and trace extraction.

    Args:
        data_path (str): Path to Suite2p output folder
        segmentation_channel (int): Channel to use for segmentation (1 or 2)
        extraction_channel (int): Channel to use for extraction (1 or 2)
        image_type (str): Type of image to use for segmentation
                         ('meanImg', 'meanImg_chan2', 'meanImgE', 'max_proj')
        run_extraction (bool): Whether to run extraction after segmentation
    """

    # Do cellpose segmentation and mask combination
    print(
        f"\n Starting cellpose segmentation and mask combination on {image_type}... \n"
    )
    segment_cells(
        data_path, segmentation_channel=segmentation_channel, image_type=image_type
    )
    print("\n Cellpose segmentation and mask combination complete! \n")

    # Use combined masks from previous step to create suite2p masks and extract traces
    if run_extraction:
        print(
            f"\n Starting suite2p mask creation and trace extraction on channel {extraction_channel}"
        )
        create_suite2p_masks_extract_traces(
            data_path, extraction_channel=extraction_channel
        )
        print("\n Pipeline complete! You can now investigate masks in suite2p \n")
    else:
        print("\n Segmentation complete! Skipping extraction as requested. \n")


if __name__ == "__main__":

    """
    To run the pipeline, run the following command in the terminal:
    python -m astroglial_segmentation data_path [options]

    Examples:
    python -m astroglial_segmentation /path/to/suite2p/output
    python -m astroglial_segmentation /path/to/suite2p/output --segmentation-channel 2 --image-type max_proj
    python -m astroglial_segmentation /path/to/suite2p/output --no-extraction
    """

    parser = argparse.ArgumentParser(
        description="Run Astroglial segmentation and suite2p trace extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Image types available:
  meanImg        - Mean image from channel 1 (default)
  meanImg_chan2  - Mean image from channel 2  
  meanImgE       - Enhanced mean image
  max_proj       - Maximum projection image

Examples:
  %(prog)s /path/to/suite2p/output
  %(prog)s /path/to/suite2p/output --segmentation-channel 2 --image-type max_proj
  %(prog)s /path/to/suite2p/output --extraction-channel 2
  %(prog)s /path/to/suite2p/output --no-extraction
        """,
    )

    parser.add_argument("data_path", type=str, help="Path to Suite2p output folder")

    parser.add_argument(
        "--segmentation-channel",
        type=int,
        choices=[1, 2],
        default=1,
        help="Channel to use for segmentation (1 or 2, default: 1)",
    )

    parser.add_argument(
        "--extraction-channel",
        type=int,
        choices=[1, 2],
        default=1,
        help="Channel to use for extraction (1 or 2, default: 1)",
    )

    parser.add_argument(
        "--image-type",
        type=str,
        choices=["meanImg", "meanImg_chan2", "meanImgE", "max_proj"],
        default="meanImg",
        help="Type of image to use for segmentation (default: meanImg)",
    )

    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Skip extraction step, only perform segmentation",
    )

    args = parser.parse_args()

    main_pipeline(
        data_path=args.data_path,
        segmentation_channel=args.segmentation_channel,
        extraction_channel=args.extraction_channel,
        image_type=args.image_type,
        run_extraction=not args.no_extraction,
    )
