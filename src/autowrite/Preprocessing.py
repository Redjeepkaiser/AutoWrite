import numpy as np
import tensorflow as tf

def preprocess_to_rtp(data):
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

    rs = [resampleStroke(stroke) for stroke in normalized_strokes]
    max_stroke_len = max(len(r) for r in rs)

    resampled_strokes = np.zeros((len(rs), max_stroke_len, 3))
    resampled_strokes[:, :, 2] -= 1

    for i, row in enumerate(rs):
        resampled_strokes[i, :len(row)] = row

    directions = np.apply_along_axis(
            lambda x: int(x[x >= 0][0] < x[x >= 0][-1]),
            1,
            resampled_strokes[:,:,1])
    directions = np.tile(np.expand_dims(directions, axis=0).transpose(),
            (1, resampled_strokes.shape[1]))

    rtps = np.append(
            resampled_strokes,
            np.expand_dims(directions, axis=2),
            axis=2)
    rtps = np.append(
            rtps,
            np.zeros((resampled_strokes.shape[0], resampled_strokes.shape[1], 1)),
            axis=2)

    rtps[:, 0, 4] = 1

    rtps = rtps[rtps[:, :, 2] >= 0].tolist()
    return np.expand_dims(np.array(rtps), 0)

def sampleLine(p0, p1, delta=0.05):
    l = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5
    num = int(l/delta)

    if num == 0:
        return [p0]

    sampled_xs = np.linspace(p0[0], p1[0], num)
    sampled_ys = np.linspace(p0[1], p1[1], num)
    sampled_timestamps = np.linspace(p0[2], p1[2], num)

    return np.stack((sampled_xs, sampled_ys, sampled_timestamps), axis=1).tolist()

def resampleStroke(stroke):
    resampled_stroke = []

    for i, _ in enumerate(stroke[stroke[:, 2] >= 0][:-1]):
        resampled_stroke.extend(sampleLine(stroke[i], stroke[i+1]))

    return resampled_stroke



