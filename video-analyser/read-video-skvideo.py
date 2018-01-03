import skvideo.io
import skvideo.utils
import skvideo.datasets


import numpy as np

filename = skvideo.datasets.bigbuckbunny()
filename_yuv = "video/outpy.avi"

vid = skvideo.io.vread(filename)
