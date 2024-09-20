import cv2
import numpy as np
from napari.layers import Image
import dask.array as da
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def f(lam, b):
    return np.exp(-1 * (np.power(lam, 2)) / (np.power(b, 2)))


def process_slice(slice_data, iteration, b):
    # Handles processing for Dask arrays
    if isinstance(slice_data, da.Array):
        slice_data = slice_data.compute()  # Converts Dask array into Numpy array
        
    # Pads original image
    pad_width = 400
    img_pad = np.pad(slice_data, ((pad_width, pad_width), (pad_width, pad_width)), mode="reflect")

    # Normalizes pixel intensity values to range 0-1
    img_norm = cv2.normalize(img_pad, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    img_new = np.zeros(img_norm.shape, dtype=img_norm.dtype)

    # Applies anisotropic diffusion
    for t in range(iteration):
        dn = img_norm[:-2, 1:-1] - img_norm[1:-1, 1:-1]  # north
        ds = img_norm[2:, 1:-1] - img_norm[1:-1, 1:-1]  # south
        de = img_norm[1:-1, 2:] - img_norm[1:-1, 1:-1]  # east
        dw = img_norm[1:-1, :-2] - img_norm[1:-1, 1:-1]  # west
        dnw = img_norm[:-2, :-2] - img_norm[1:-1, 1:-1]  # northwest
        dne = img_norm[:-2, 2:] - img_norm[1:-1, 1:-1]  # northeast
        dsw = img_norm[2:, :-2] - img_norm[1:-1, 1:-1]  # southwest
        dse = img_norm[2:, 2:] - img_norm[1:-1, 1:-1]  # southeast

        lam = 1 / 8

        img_new[1:-1, 1:-1] = img_norm[1:-1, 1:-1] + \
                              lam * (f(dn, b) * dn + f(ds, b) * ds +
                                     f(de, b) * de + f(dw, b) * dw +
                                     f(dnw, b) * dnw + f(dne, b) * dne +
                                     f(dsw, b) * dsw + f(dse, b) * dse)
        img_norm = img_new

    # Removes padding
    img_unpad = img_norm[pad_width:pad_width + slice_data.shape[0], pad_width:pad_width + slice_data.shape[1]]

    # Converts and normalizes range to original 8 or 16 bit unsigned integers
    processed_slice_uint = None
    if slice_data.dtype == "uint16":
        processed_slice_uint = cv2.normalize(img_unpad, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_16U)
    elif slice_data.dtype == "uint8":
        processed_slice_uint = cv2.normalize(img_unpad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8U)

    return processed_slice_uint


@magic_factory(
    call_button="Denoise",
    image={"label": "Input Image"},
    iteration={"label": "Iteration"},
    b={"label": "Gradient Threshold",
       "choices": ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.10"]}
)
def denoise(
        image: Image,
        iteration: int = 1,
        b: str = "0.05"
) -> Image:
    """
    This widget denoises the image based on the anisotropic diffusion algorithm
    (Perona and Malik 1990).

    Parameters
    ----------
    Image : "Image"
        Image to be processed

    Iteration : int
        Number of iterations

    Gradient Threshold : int
        Threshold defining gradients corresponding to true edges or to noise

    Returns
    -------
        napari Image layer containing the decurtained image
    """
    if image is None:  # Handles null cases
        print("Please select an image layer.")
        return

    b_float = float(b)

    if len(image.data.shape) > 2:
        stack = image.data
        processed_slices = []
        slice_order = []  # To keep track of slice order

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {executor.submit(process_slice, stack[slice_idx], iteration, b_float):
                                   slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())

        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]
        processed_stack = np.stack(processed_slices)

    else:
        processed_stack = process_slice(image.data, iteration, b_float)

    image_name = f"Dnoi_iter{iteration}_gradthr{b}"

    print(f"\nImage or Stack denoised successfully!\n{image_name} added to Layer List.")

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=image_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return denoise
