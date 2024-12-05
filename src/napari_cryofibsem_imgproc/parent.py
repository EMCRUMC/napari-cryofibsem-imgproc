import napari
from napari.layers import Image
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QTabWidget
from .decurtain import Decurtain


class Parent(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        # Set up layout
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        # IMAGE SELECTION
        # Create a Label for Image Selection
        self._img_label = QLabel("Select Image")
        vbox.addWidget(self._img_label)

        # Create a QComboBox and populate it with Image layers
        self._layer_combo = QComboBox()
        self.update_layer_combo()
        vbox.addWidget(self._layer_combo)

        # Connect Napari events to update the combobox
        viewer.layers.events.inserted.connect(self.update_layer_combo)
        viewer.layers.events.removed.connect(self.update_layer_combo)

        # Connect combobox selection change to layer activation
        self._layer_combo.currentIndexChanged.connect(self.activate_selected_layer)

        # FUNCTION TAB GROUP
        # Create a QTabWidget for tab management
        self._tab_widget = QTabWidget()
        vbox.addWidget(self._tab_widget)

        # Add tabs using QTabWidget's addTab method
        self._tab_widget.addTab(Decurtain(viewer), "Tab 1")
        self._tab_widget.addTab(QLabel("Content of Tab 2"), "Tab 2")

    def update_layer_combo(self):
        # Clear and repopulate the combobox with only Image layers
        self._layer_combo.clear()
        image_layer_names = [
            layer.name for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        self._layer_combo.addItems(image_layer_names)

    def activate_selected_layer(self):
        # Get the current layer name from the combobox
        selected_layer_name = self._layer_combo.currentText()

        if selected_layer_name in self.viewer.layers:
            # Set the active layer
            selected_layer = self.viewer.layers[selected_layer_name]
            self.viewer.layers.selection.active = selected_layer

            # Ensure the layer is visible
            selected_layer.visible = True

            # Optionally, hide all other layers if you want to show only this layer
            for layer in self.viewer.layers:
                if layer.name != selected_layer_name:
                    layer.visible = False
