import cv2
import numpy as np
import pywt
from napari.layers import Image
import concurrent.futures
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation


def process_slice(slice_data, dec_num, sigma, wname):
    slice_data_dtype = slice_data.dtype

    # Decomposes image into details
    Ch, Cv, Cd = [], [], []
    for ii in range(dec_num):
        slice_data, (ch, cv, cd) = pywt.dwt2(slice_data, wname)
        Ch.append(ch)
        Cv.append(cv)
        Cd.append(cd)

    # Applies damping to vertical detail coefficient at each decomposition level
    for ii in range(dec_num):
        fCv = np.fft.fftshift(np.fft.fft2(Cv[ii]))
        my, mx = fCv.shape

        damp = 1 - np.exp(-np.square(np.arange(-my // 2, my // 2)) / (2 * sigma ** 2))
        fCv *= damp[:, np.newaxis]

        Cv[ii] = np.fft.ifft2(np.fft.ifftshift(fCv))

    img_ori_recon = slice_data

    # Reconstructs details into image
    for ii in range(dec_num - 1, -1, -1):
        img_ori_recon = img_ori_recon[:Ch[ii].shape[0], :Ch[ii].shape[1]]
        img_ori_recon = pywt.idwt2((img_ori_recon, (Ch[ii], Cv[ii], Cd[ii])), wname)

    # Converts complex128 into float64
    img_ori_float = np.abs(img_ori_recon).astype(np.float64)

    # Converts and normalizes range to original 8 or 16 bit unsigned integers
    processed_slice_uint = None
    if slice_data_dtype == "uint16":
        processed_slice_uint = cv2.normalize(img_ori_float, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_16U)
    elif slice_data_dtype == "uint8":
        processed_slice_uint = cv2.normalize(img_ori_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8U)

    return processed_slice_uint


@magic_factory(
    call_button="Decurtain",
    image={"label": "Input Image"},
    dec_num={"label": "Decomposition level"},
    sigma={"label": "Sigma"},
    wname={"label": "Wavelet", "choices": ["coif1", "coif3", "coif5"]}
)
def decurtain(
    image: Image, 
    dec_num: int = 6, 
    sigma: int = 4,
    wname: str = "coif5"
) -> Image:
    """
    This widget removes the vertical stripes or the "curtain" artefacts due to FIB milling. 
    The algorithm is based on the combined wavelet-Fourier (Münch et al. 2009). It utilizes 
    wavelet decomposition, FFT transform, damping of vertical details, and wavelet reconstruction.

    Parameters
    ----------
    Image : "Image"
        Image to be processed

    Decomposition level : int
        Number of decomposition levels of features in the image

    Sigma : int
        Width of the damping filter for the destriping
    
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
            future_to_slice = {executor.submit(process_slice, stack[slice_idx], dec_num, sigma, wname): 
                               slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())
                
        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]  
        processed_stack = np.stack(processed_slices)
        
    else: 
        processed_stack = process_slice(image.data, dec_num, sigma, wname)

    image_name = f"Dcur_dec{dec_num}_sig{sigma}_{wname}"

    print(f"\nImage or Stack decurtained successfully!\n{image_name} added to Layer List.")
    
    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=image_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return decurtain
