import numpy as np


import suite2p
from scipy import stats
from suite2p.extraction.extract import create_masks_and_extract
from combination import extend_and_merge_masks




stat = []


def extract_trace(masks, datapath):
    for u in np.unique(masks):
        ypix, xpix = np.nonzero(masks == u)
        npix = len(ypix)
        stat.append({ypix:ypix, xpix:xpix, npix:npix, 'lam':np.ones(npix,npix,np.float32)})

    stat = np.array(stat)

    ops = np.load(datapath, allow_pickle=True).item()
    F, Fneu,_,_ = create_masks_and_extract(ops, stat)

    df = F-Fneu*0.3

    # compute activity statistics for classifier
    sk = stats.skew(df, axis=1)
    sd = np.std(df, axis=1)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]
        stat[k]['std'] = sd[k]

    fpath = ops['save_path0']
    np.save(fpath + 'stat.npy', stat)
    np.save(fpath + 'F.npy', F)
    np.save(fpath + 'Fneu.npy', Fneu)
    
