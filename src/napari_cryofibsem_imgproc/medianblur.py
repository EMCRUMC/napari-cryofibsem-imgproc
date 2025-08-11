import cv2
import numpy as np
from napari.layers import Image
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def process_slice(slice_data, kernel):
    clean = cv2.medianBlur(slice_data, kernel)
    return clean


@magic_factory(
    call_button="Median Blur",
    image={"label": "Input Image"},
    kernel={"label": "Kernel"}
)
def medianblur(
        image: Image,
        kernel: int = 3
) -> Image:
    """
    This widget is for median blur. It is for removing the salt-and-pepper noise
    left after denoising.

    Parameters
    ----------
    Image : "Image"
        Image to be processed

    Kernel : int
        Size of the kernel

    Returns
    -------
        napari Image layer containing the median-blurred image
    """
    if image is None:  # Handles null cases
        print("Please select an image layer.")
        return

    if len(image.data.shape) > 2:
        stack = image.data
        processed_slices = []
        slice_order = []  # To keep track of slice order

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {executor.submit(process_slice, stack[slice_idx], kernel):
                                   slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())

        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]
        processed_stack = np.stack(processed_slices)

    else:
        processed_stack = process_slice(image.data, kernel)

    image_name = f"MedBlur_kernel{kernel}"

    print(f"\nImage or Stack smoothened successfully!\n{image_name} added to Layer List.")

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=image_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return medianblur
