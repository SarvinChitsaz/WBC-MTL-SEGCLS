import numpy as np

def normalize_id(x):
    return str(x).zfill(3)


def convert_mask(mask_array):
    """
    Converts grayscale mask to class indices.
    Mapping:
    0   -> Background
    128 -> Nucleus
    255 -> Cytoplasm
    """
    out = np.zeros_like(mask_array, dtype=np.int64)
    out[mask_array == 128] = 1
    out[mask_array == 255] = 2
    return out
