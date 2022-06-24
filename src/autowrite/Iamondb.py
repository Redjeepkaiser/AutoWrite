import numpy as np
import os
import re
import xml.etree.ElementTree as ET

from pathlib import Path

class Iamondb:
    def __init__(self, path_to_root):
        self.path_to_root = Path(path_to_root)
        self.path_to_text_files = self.path_to_root / "text_files"
        assert self.path_to_text_files.is_dir()

        self.path_to_stroke_files = self.path_to_root / "stroke_files"
        assert self.path_to_stroke_files.is_dir()

    def get_text_file_paths(self):
        return [Path(dirpath + "/" + filenames[0])
                for (dirpath, dirnames, filenames) in os.walk(self.path_to_text_files)
                if filenames != []]

    def get_stroke_file_paths_for_text_file(self, text_file_path):
        stroke_dir = self.path_to_stroke_files / text_file_path.parts[-3] / text_file_path.parts[-2]

        if not stroke_dir.is_dir():
            return None

        m = re.search("(.*?)-(.*)", text_file_path.stem)

        res = [stroke_dir / filename for filename in sorted(os.listdir(stroke_dir))
               if re.search("(.*?)-(.*?)-.*", filename).groups() == m.groups()]

        if len(res) == 0:
            return None

        return res

    def parse_stroke_file(self, stroke_file_path):
        root = ET.parse(stroke_file_path).getroot()

        strokes = [[[point.attrib["x"], point.attrib["y"], point.attrib["time"]]
                    for point in stroke.findall("./Point")]
                   for stroke in root.findall("./StrokeSet/Stroke")]

        max_stroke_len = max(len(r) for r in strokes)

        strokes_padded = np.zeros((len(strokes), max_stroke_len, 3))
        strokes_padded[:, :, 2] -= 1

        for i, row in enumerate(strokes):
            strokes_padded[i, :len(row)] = row

        return strokes_padded

    def get_file_lines(self, textfile_path):
        f = open(textfile_path)
        content = f.read()
        lines = re.search("CSR:\s*([^~]*)", content).group(1).strip().split("\n")
        return lines

    def get_samples(self):
        strokes = []
        lines = []
        for text_file_path in self.get_text_file_paths()[:10]:
            stroke_file_paths = self.get_stroke_file_paths_for_text_file(text_file_path)

            if not stroke_file_paths:
                continue

            strokes.extend([self.parse_stroke_file(stroke_file_path) for stroke_file_path in stroke_file_paths])
            lines.extend([line for line in self.get_file_lines(text_file_path)])

        return (strokes, lines)

