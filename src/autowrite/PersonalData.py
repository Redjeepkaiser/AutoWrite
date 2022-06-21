import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

class PersonalData:
    """
    Class containing functions to collect samples from the personal dataset.
    """
    def __init__(self, path_to_data):
        self.path_to_data = Path(path_to_data)

    def get_samples(self):
        """ Returns all samples in the personal dataset. """
        files = [filenames for (_, _, filenames) in
                 os.walk(self.path_to_data)][0]
        files = [Path(file) for file in files]

        txt_files = sorted([elem for elem in files if elem.suffix == ".txt"])
        npy_files = sorted([elem for elem in files if elem.suffix == ".npy"])

        strokes = [np.load(self.path_to_data / filename)
                   for filename in npy_files]
        lines = [read_file(self.path_to_data / filename)
                 for filename in txt_files]

        return (strokes, lines)

def read_file(path):
    """ Reads the contents of a file. """
    contents = None

    with open(path) as f:
        contents = f.read()

    return contents
