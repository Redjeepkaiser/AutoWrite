import os
from pathlib import Path

class Iamondb:
    def __init__(self, path_to_root):
        self.path_to_root = Path(path_to_root)
        self.path_to_text_files = path_to_root / "text_files"
        self.path_to_stroke_files = path_to_root / "stroke_files"

    def __get_text_file_paths(self):
        return [Path(dirpath + "/" + filenames[0])
                for (dirpath, dirnames, filenames) in os.walk(self.path_to_text_files)
                if filenames != []] 

    def get_samples(self):
