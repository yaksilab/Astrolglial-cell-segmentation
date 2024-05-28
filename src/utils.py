import numpy as np
from scipy import signal


def enhanced_mean_image(ops):
    """ computes enhanced mean image and adds it to ops

    Median filters ops["meanImg"] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops["meanImgE"]

    Parameters
    ----------
    ops : dictionary
        uses "meanImg", "aspect", "spatscale_pix", "yrange" and "xrange"

    Returns
    -------
        ops : dictionary
            "meanImgE" field added

    """

    I = ops["meanImg"].astype(np.float32)
    if "spatscale_pix" not in ops:
        if isinstance(ops["diameter"], int):
            diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            diameter = np.array(ops["diameter"])
        if diameter[0] == 0:
            diameter[:] = 12
        ops["spatscale_pix"] = diameter[1]
        ops["aspect"] = diameter[0] / diameter[1]

    diameter = 4 * np.ceil(
        np.array([ops["spatscale_pix"] * ops["aspect"], ops["spatscale_pix"]])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[ops["yrange"][0]:ops["yrange"][1], ops["xrange"][0]:ops["xrange"][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0, np.minimum(1, mimg0))
    mimg = mimg0.min() * np.ones((ops["Ly"], ops["Lx"]), np.float32)
    mimg[ops["yrange"][0]:ops["yrange"][1], ops["xrange"][0]:ops["xrange"][1]] = mimg0
    ops["meanImgE"] = mimg
    print("added enhanced mean image")
    return mimg

