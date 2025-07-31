"""
Taken from issue #292 in cellpose gitHub repo, written by @neurochatter, check it out here:
https://github.com/MouseLand/suite2p/issues/292#issuecomment-1539041902
"""

import numpy as np
import suite2p
from suite2p.detection import roi_stats
from suite2p.extraction.masks import (
    create_masks,
)
from suite2p.io.binary import BinaryFile
from pathlib import Path
import os


def create_suite2p_masks_extract_traces(
    working_dir, extraction_channel=1, cp_seg_file=None
):
    """
    Create Suite2p compatible masks and extract traces.

    Args:
        working_dir (str): Path to Suite2p output folder
        extraction_channel (int): Channel to use for extraction (1 or 2)
        cp_seg_file (str, optional): Specific segmentation file to use.
                                   If None, will look for combined mask file.
    """
    wd = Path(working_dir)
    ops_file = wd / "ops.npy"
    if not ops_file.exists():
        raise FileNotFoundError(f"Ops file not found: {ops_file}")

    ops = np.load(ops_file, allow_pickle=True).item()
    Lx = ops["Lx"]
    Ly = ops["Ly"]

    # Handle channel selection for extraction
    f_reg = BinaryFile(Ly, Lx, str(wd / "data.bin"))
    f_reg_chan2 = None

    if extraction_channel == 2:
        # Check if channel 2 data exists
        chan2_file = wd / "data_chan2.bin"
        if chan2_file.exists():
            f_reg_chan2 = BinaryFile(Ly, Lx, str(chan2_file))
            print(f"Using channel {extraction_channel} for extraction")
        else:
            print(f"Warning: Channel 2 data file not found, falling back to channel 1")
            extraction_channel = 1

    # Auto-detect combined segmentation file if not specified
    if cp_seg_file is None:
        # Look for combined mask files with new naming convention first
        import glob

        combined_files = glob.glob(str(wd / "combined_*_seg.npy"))
        if combined_files:
            cp_seg_file = Path(combined_files[0]).name
            print(f"Auto-detected segmentation file: {cp_seg_file}")
        else:
            # Fallback to legacy naming
            cp_seg_file = "combined_mean_image_seg.npy"
            print(f"Using default segmentation file: {cp_seg_file}")

    cellpose_fpath = wd / cp_seg_file
    if not cellpose_fpath.exists():
        raise FileNotFoundError(
            f"The combined mask file is not found: {cellpose_fpath}"
        )

    cellpose_masks = np.load(cellpose_fpath, allow_pickle=True).item()
    masks = cellpose_masks["masks"]

    stat = []
    for u_ix, u in enumerate(np.unique(masks)[1:]):
        ypix, xpix = np.nonzero(masks == u)
        npix = len(ypix)
        stat.append(
            {
                "ypix": ypix,
                "xpix": xpix,
                "npix": npix,
                "lam": np.ones(npix, np.float32),
                "med": [np.mean(ypix), np.mean(xpix)],
            }
        )
    stat = np.array(stat)
    stat = roi_stats(
        stat, Ly, Lx
    )  # This function fills in remaining roi properties to make it compatible with the rest of the suite2p pipeline/GUI

    # Using the constructed stat file, get masks
    cell_masks, neuropil_masks = create_masks(stat, Ly, Lx, ops)

    # Feed these values into the wrapper functions
    stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(
        stat, f_reg, f_reg_chan2=f_reg_chan2, ops=ops
    )

    # Do cell classification
    classfile = suite2p.classification.builtin_classfile
    iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

    # Apply preprocessing step for deconvolution
    # Use the appropriate fluorescence data based on extraction channel
    if extraction_channel == 2 and F_chan2 is not None:
        F_extract = F_chan2.copy()
        Fneu_extract = Fneu_chan2.copy()
        print("Using channel 2 fluorescence data for preprocessing")
    else:
        F_extract = F.copy()
        Fneu_extract = Fneu.copy()
        print("Using channel 1 fluorescence data for preprocessing")

    dF = F_extract - ops["neucoeff"] * Fneu_extract
    dF = suite2p.extraction.preprocess(
        F=dF,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
    )
    # spikes
    spks = suite2p.extraction.oasis(
        F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"]
    )

    new_dir = wd / f"cellpose_suite2p_output_ch{extraction_channel}"

    os.makedirs(new_dir, exist_ok=True)

    # Save both channel data if available
    np.save(new_dir / "F.npy", F)
    np.save(new_dir / "Fneu.npy", Fneu)
    if F_chan2 is not None:
        np.save(new_dir / "F_chan2.npy", F_chan2)
        np.save(new_dir / "Fneu_chan2.npy", Fneu_chan2)

    # Save the processed data from the selected extraction channel
    np.save(new_dir / f"F_processed_ch{extraction_channel}.npy", F_extract)
    np.save(new_dir / f"Fneu_processed_ch{extraction_channel}.npy", Fneu_extract)

    np.save(new_dir / "iscell.npy", iscell)
    np.save(new_dir / "ops.npy", ops)
    np.save(new_dir / "spks.npy", spks)
    np.save(new_dir / "stat.npy", stat)

    print(f"Extraction complete! Results saved to: {new_dir}")
    print(f"Extracted {len(stat)} ROIs using channel {extraction_channel}")
