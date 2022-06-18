import numpy as np
import tensorflow as tf

class Preprocessor:
    def normalize(self, data):
        ustrokes = [np.unique(stroke[:, :2], return_index=True, axis=0)[1]
                for stroke in data]

        max_stroke_len = max(len(r) for r in ustrokes)
        unique_strokes = np.zeros((len(ustrokes), max_stroke_len, 3))
        unique_strokes[:, :, 2] -= 1

        for i, row in enumerate(ustrokes):
            unique_strokes[i, :len(row)] = data[i, np.sort(row)]

        non_ragged = unique_strokes[:, :, 2] >= 0
        normalized_strokes = unique_strokes.copy()

        normalized_strokes[non_ragged] -= [
                normalized_strokes[0, 0, 0],
                np.amax(normalized_strokes[:, :, 1]),
                normalized_strokes[0, 0, 2]
                ]

        normalized_strokes[non_ragged] /= [
                -np.amin(normalized_strokes[:, :, 1]),
                np.amin(normalized_strokes[:, :, 1]),
                1
                ]
        
        return normalized_strokes
