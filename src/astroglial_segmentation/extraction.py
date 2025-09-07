"""
Taken from issue #292 in cellpose gitHub repo, written by @neurochatter, check it out here:
https://github.com/MouseLand/suite2p/issues/292#issuecomment-1539041902
"""

import numpy as np

from suite2p import extraction_wrapper, classify
from suite2p.io import BinaryFile
from suite2p.detection import roi_stats
from suite2p.extraction import preprocess, oasis
from suite2p.classification import builtin_classfile
from suite2p.extraction.masks import create_masks
from pathlib import Path
import os


def create_suite2p_masks_extract_traces(
    working_dir, cp_seg_file="combined_mean_image_seg.npy", new_output_dir: bool = True
):
    """
    Create suite2p masks from cellpose segmentation and extract traces.

    Args:
        working_dir (str):
            Path to the working directory containing 'ops.npy' and 'data.bin'.
        cp_seg_file (str, optional):
            Name of the combined mask file from cellpose segmentation.
            Defaults to "combined_mean_image_seg.npy".
        new_output_dir (bool, optional):
            Whether to save outputs in a new directory named 'cellpose_suite2p_output'.
            Defaults to True. If False, overwrites existing suite2p result files in working_dir.
    """

    wd = Path(working_dir)
    ops_file = wd / "ops.npy"
    if not ops_file.exists():
        raise FileNotFoundError(f"Ops file not found: {ops_file}")

    ops = np.load(ops_file, allow_pickle=True).item()
    Lx = ops["Lx"]
    Ly = ops["Ly"]
    f_reg = BinaryFile(Ly, Lx, str(wd / "data.bin"))

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
    stat_after_extraction, F, Fneu, _, _ = extraction_wrapper(
        stat, f_reg, f_reg_chan2=None, ops=ops
    )

    # Do cell classification
    classfile = builtin_classfile
    iscell = classify(stat=stat_after_extraction, classfile=classfile)

    # Apply preprocessing step for deconvolution
    dF = F.copy() - ops["neucoeff"] * Fneu
    dF = preprocess(
        F=dF,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
    )
    # spikes
    spks = oasis(F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])

    if new_output_dir:
        new_dir = wd / "cellpose_suite2p_output"
    else:
        new_dir = wd

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
