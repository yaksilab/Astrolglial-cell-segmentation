import numpy as np
from scipy import signal
from cellpose import plot


def enhanced_mean_image(ops):
    """computes enhanced mean image and adds it to ops

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

    diameter = (
        4
        * np.ceil(
            np.array([ops["spatscale_pix"] * ops["aspect"], ops["spatscale_pix"]])
        )
        + 1
    )
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[
        ops["yrange"][0] : ops["yrange"][1], ops["xrange"][0] : ops["xrange"][1]
    ]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0, np.minimum(1, mimg0))
    mimg = mimg0.min() * np.ones((ops["Ly"], ops["Lx"]), np.float32)
    mimg[ops["yrange"][0] : ops["yrange"][1], ops["xrange"][0] : ops["xrange"][1]] = (
        mimg0
    )
    ops["meanImgE"] = mimg
    print("added enhanced mean image")
    return mimg


def transform_mean_image_for_plotting(mean_image):
    """Transforms the mean image for plotting

    Parameters
    ----------
    mean_image : np.ndarray
        Mean image to be transformed same from suite2p

    Returns
    -------
    im0: NDArray[unsignedinteger[_8Bit]] | uint8 | NDArray | Any
        Transformed mean image
    """
    im0 = mean_image.copy()
    if im0.shape[0] < 4:
        im0 = np.transpose(im0, (1, 2, 0))
    if im0.shape[-1] < 3 or im0.ndim < 3:
        im0 = plot.image_to_rgb(im0, channels=[0, 0])
    else:
        if im0.max() <= 50.0:
            im0 = np.uint8(np.clip(im0, 0, 1) * 255)

    return im0
