__version__ = "0.0.3"
from ._sample_data import make_sample_data
from ._widget import ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget
from .brightness import brightness
from .contrast import contrast
from .decharge import decharge
from .decurtain import Decurtain
from .denoise import denoise
from .parent import Parent

__all__ = (
    "make_sample_data",
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
    "brightness"
    "contrast",
    "decharge",
    "Decurtain",
    "denoise",
    "Parent"
)
