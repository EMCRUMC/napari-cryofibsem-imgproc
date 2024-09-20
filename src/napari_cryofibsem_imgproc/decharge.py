import cv2
import numpy as np
from napari.layers import Image
import dask.array as da
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def process_slice(slice_data, sigma_x, sigma_y):
    # Handles processing for Dask arrays
    if isinstance(slice_data, da.Array):
        slice_data = slice_data.compute()  # Converts Dask array into Numpy array
        
    # Pads original image
    pad_width = 400
    img_ori_pad = np.pad(slice_data, ((pad_width, pad_width), (pad_width, pad_width)), mode="reflect")

    # Applies Gaussian blur to original image
    gauss_filt = cv2.GaussianBlur(img_ori_pad, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

    # Subtracts Gaussian blur image from original image
    img_pro = np.float64(img_ori_pad) - np.float64(gauss_filt)

    # Removes padding
    img_pro_unpad = img_pro[pad_width:pad_width + slice_data.shape[0], pad_width:pad_width + slice_data.shape[1]]

    # Converts and normalizes range to original 8 or 16 bit unsigned integers
    processed_slice_uint = None
    if slice_data.dtype == "uint16":
        processed_slice_uint = cv2.normalize(img_pro_unpad, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_16U)
    elif slice_data.dtype == "uint8":
        processed_slice_uint = cv2.normalize(img_pro_unpad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8U)

    return processed_slice_uint


@magic_factory(
    call_button="Decharge",
    image={"label": "Input Image"},
    sigma_x={"label": "Sigma X"},
    sigma_y={"label": "Sigma Y"}
)
def decharge(
        image: Image,
        sigma_x: int = 100,
        sigma_y: int = 1
) -> Image:
    """
    This widget removes the bright areas and evens out the brightness of the image.
    It utilizes a gaussian blur to get the mask of large bright and dark regions.
    Then the mask is simply subtracted from the original image.

    Parameters
    ----------
    Image : "Image"
        Image to be processed

    Sigma X : int
        Width of the filter for the horizontal direction

    Sigma Y : int
        Width of the filter for the vertical direction

    Returns
    -------
        napari Image layer containing the decurtained image
    """
    if image is None:  # Handles null cases
        print("Please select an image layer.")
        return

    if len(image.data.shape) > 2:
        stack = image.data
        processed_slices = []
        slice_order = []  # To keep track of slice order

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {executor.submit(process_slice, stack[slice_idx], sigma_x, sigma_y):
                                   slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())

        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]
        processed_stack = np.stack(processed_slices)

    else:
        processed_stack = process_slice(image.data, sigma_x, sigma_y)

    image_name = f"Dcha_sigx{sigma_x}_sigy{sigma_y}"

    print(f"\nImage or Stack decharged successfully!\n{image_name} added to Layer List.")

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=image_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return decharge
