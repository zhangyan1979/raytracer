import numpy as np
import sys, struct

def read_raw_image(raw_img_fn):
    with open(raw_img_fn, 'rb') as fid:
        width, height = struct.unpack('II', fid.read(8))
        img = np.frombuffer(fid.read(), dtype=np.float32).reshape((height, width, 3))
    return img