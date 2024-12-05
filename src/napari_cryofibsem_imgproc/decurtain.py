import napari
import cv2
import numpy as np
import pywt
from qtpy.QtWidgets import QHBoxLayout, QWidget, QSlider, QLabel, QFormLayout, QComboBox
from qtpy.QtCore import Qt


class Decurtain(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.dec_num = 20  # Default decomposition level
        self.sigma = 20  # Default sigma value

        # Layout setup
        layout = QFormLayout()
        self.setLayout(layout)

        # First Slider and Label
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setSingleStep(1)
        self.slider1.setMinimum(1)
        self.slider1.setMaximum(15)
        self.slider1.setValue(6)
        # self.slider1.setTickPosition(QSlider.TicksBelow)
        # self.slider1.setTickInterval(5)
        self.slider1.valueChanged.connect(self.on_slider1_change)

        self.label1 = QLabel(f"Decomposition Level: {self.slider1.value()}", self)
        layout.addRow("Decomposition Level", self.slider1)
        layout.addRow(self.label1)

        # Second Slider and Label
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setSingleStep(1)
        self.slider2.setMinimum(1)
        self.slider2.setMaximum(15)
        self.slider2.setValue(4)
        # self.slider2.setTickPosition(QSlider.TicksBelow)
        # self.slider2.setTickInterval(5)
        self.slider2.valueChanged.connect(self.on_slider2_change)

        self.label2 = QLabel(f"Sigma: {self.slider2.value()}", self)
        layout.addRow("Sigma", self.slider2)
        layout.addRow(self.label2)

        # Combo Box
        self.combobox = QComboBox(self)
        self.combobox.addItems(["Coiflet 1", "Coiflet 2", "Coiflet 3"])
        # self.combobox.currentTextChanged.connect(self.update_wavelet_selection)

        layout.addRow("Wavelet", self.combobox)

    # Handle slider updates
    def on_slider1_change(self, value):
        self.label1.setText(f'Decomposition Level Value: {value}')
        self.on_slider_change(value, "Decomposition Level")
        self.last_slider = "Decomposition Level"
        self.dec_num = value

    def on_slider2_change(self, value):
        self.label2.setText(f'Sigma: {value}')
        self.on_slider_change(value, "Sigma")
        self.last_slider = "Sigma"
        self.sigma = value

    def on_slider_change(self, value, slider_type):
        print(f"Slider {slider_type} changed to {value}")
        self.process_slice()

    # Decurtaining function
    def process_slice(self):
        slice_data = self.viewer.layers[0].data

        if len(self.viewer.layers) == 1:
            self.viewer.add_image(self.viewer.layers[0].data, name="copy")

        slice_data_dtype = slice_data.dtype
        slice_data_shape = slice_data.shape

        dec_num = self.dec_num
        sigma = self.sigma
        wname = "coif3"

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
        print(img_ori_recon.shape)

        # Reconstructs details into image
        for ii in range(dec_num - 1, -1, -1):
            img_ori_recon = img_ori_recon[:Ch[ii].shape[0], :Ch[ii].shape[1]]
            img_ori_recon = pywt.idwt2((img_ori_recon, (Ch[ii], Cv[ii], Cd[ii])), wname)

        # Crops back to original size
        img_ori_crop = img_ori_recon[:slice_data_shape[0], :slice_data_shape[1]]

        # Converts complex128 into float64
        img_ori_float = np.abs(img_ori_crop).astype(np.float64)

        # Converts and normalizes range to original 8 or 16 bit unsigned integers
        processed_slice_uint = None
        if slice_data_dtype == "uint16":
            processed_slice_uint = cv2.normalize(img_ori_float, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_16U)
        elif slice_data_dtype == "uint8":
            processed_slice_uint = cv2.normalize(img_ori_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8U)
        print(slice_data_dtype)

        self.viewer.layers["copy"].data = processed_slice_uint