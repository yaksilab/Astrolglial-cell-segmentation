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
from pathlib import Path
import os


def create_suite2p_masks_extract_traces(
    working_dir, cp_seg_file="combined_mean_image_seg.npy"
):
    wd = Path(working_dir)
    ops_file = wd / "ops.npy"
    if not ops_file.exists():
        raise FileNotFoundError(f"Ops file not found: {ops_file}")

    ops = np.load(ops_file, allow_pickle=True).item()
    Lx = ops["Lx"]
    Ly = ops["Ly"]
    f_reg = suite2p.io.BinaryFile(Ly, Lx, wd / "data.bin")

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
        stat, f_reg, f_reg_chan2=None, ops=ops
    )

    # Do cell classification
    classfile = suite2p.classification.builtin_classfile
    iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

    # Apply preprocessing step for deconvolution
    dF = F.copy() - ops["neucoeff"] * Fneu
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

    new_dir = wd / "cellpose_suite2p_output"

    os.makedirs(new_dir, exist_ok=True)

    np.save(new_dir / "F.npy", F)
    np.save(new_dir / "Fneu.npy", Fneu)
    np.save(new_dir / "iscell.npy", iscell)
    np.save(new_dir / "ops.npy", ops)
    np.save(new_dir / "spks.npy", spks)
    np.save(new_dir / "stat.npy", stat)
