# launch_napari.py
from napari import Viewer, run
from napari_cryofibsem_imgproc import parent

viewer = Viewer()
my_widg = parent.Parent(viewer)
viewer.window.add_dock_widget(my_widg)

run()
