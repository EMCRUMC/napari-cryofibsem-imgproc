import cv2
import numpy as np
from napari.layers import Image
import dask.array as da
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def process_slice(slice_data, central_tendency, average_intensity, datatype):
    current_intensity = None
    if central_tendency == "median":
        current_intensity = np.median(slice_data)
    elif central_tendency == "mean":
        current_intensity = np.mean(slice_data)

    ratio = average_intensity / current_intensity

    slice_adjusted = None
    if datatype == "uint16":
        slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=65535).astype(np.uint16)
    elif datatype == "uint8":
        slice_adjusted = np.clip(slice_data * ratio, a_min=0, a_max=255).astype(np.uint8)

    return slice_adjusted


@magic_factory(
    call_button="Correct Brightness Variation",
    stack={"label": "Input Stack"},
    central_tendency={"label": "Central Tendency", "choices": ["median", "mean"]},
    pixel_value_adjust={"label": "Average Intensity Adjustment", "widget_type": "SpinBox", "min": -10000, "max": 10000}
)
def brightness(
        stack: Image,
        central_tendency: str = "median",
        pixel_value_adjust: int = 0
) -> Image:
    """
    This widget corrects the global brightness variations across the slices of a stack.
    It simply obtains the average pixel intensity value of all the slices, calculates a
    ratio between the average to the current slice, and adjusts the pixel intensity values
    by multiplying to the ratio.

    Parameters
    ----------
    Stack : "Image"
        Stack to be processed

    Central Tendency : str
        Chosen measure of central tendency

    Average Intensity Adjustment : int
        Value that may be added or subtracted to the average intensity value

    Returns
    -------
        napari Image layer containing the decurtained image

    """
    if stack is None:  # Handles null cases
        print("Please select a stack.")
        return

    stack_data = stack.data
    is_dask = isinstance(stack_data, da.Array)
    datatype = stack_data.dtype
    average_intensity = None

    # Handle processing for Dask arrays
    if is_dask:
        if central_tendency == "median":
            median_intensities = da.median(stack_data, axis=(1, 2)).compute()
            average_intensity = np.median(median_intensities)
        elif central_tendency == "mean":
            mean_intensities = da.mean(stack_data, axis=(1, 2)).compute()
            average_intensity = np.mean(mean_intensities)
        average_intensity += pixel_value_adjust

        # Define function to process slices
        def dask_process_slice(slice_idx):
            slice_data = stack_data[slice_idx].compute()
            return process_slice(slice_data, central_tendency, average_intensity, datatype)

        # Process slices using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Track order with slice indices
            future_to_slice = {executor.submit(dask_process_slice, i): i for i in range(stack_data.shape[0])}
            processed_slices = [None] * stack_data.shape[0]
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                processed_slices[slice_idx] = future.result()

        processed_stack = np.stack(processed_slices)

    # Handle processing for NumPy arrays
    else:
        if central_tendency == "median":
            median_intensities = np.median(stack_data, axis=(1, 2))
            average_intensity = np.median(median_intensities)
        elif central_tendency == "mean":
            mean_intensities = np.mean(stack_data, axis=(1, 2))
            average_intensity = np.mean(mean_intensities)
        average_intensity += pixel_value_adjust

        # Process slices
        processed_slices = [None] * stack_data.shape[0]
        for i in range(stack_data.shape[0]):
            processed_slices[i] = process_slice(stack_data[i], central_tendency, average_intensity, datatype)

        processed_stack = np.stack(processed_slices)

    stack_name = f"Brightness_corrected_{central_tendency}_{pixel_value_adjust}"

    print(f"\nStack brightness corrected successfully!\n{stack_name} added to Layer List.")

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=stack_name)

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return brightness
