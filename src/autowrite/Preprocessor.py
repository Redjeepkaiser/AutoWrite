import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle

class Preprocessor:
    def __init__(self, path_to_alphabet):
        path_to_alphabet = Path(path_to_alphabet)
        assert path_to_alphabet.is_file()

        with open(path_to_alphabet, "rb") as f:
            self.alphabet = pickle.load(f)

    def normalize_strokes(self, data):
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

    def strokes_to_bezier(self, strokes, precision=0.005, debug=False):
        normalized_strokes = scale_timestamps(self.normalize_strokes(strokes))
        points = []

        res = convert_stroke_to_bezier_curves(normalized_strokes[0], precision=precision, debug=debug)

        if res:
            res[0][9] = 1 # Indicate start of stroke
            points.extend(res)

        first = normalized_strokes[0]
        last_x = first[first[:, 2] >= 0][-1, 0]
        last_y = first[first[:, 2] >= 0][-1, 1]

        for stroke in normalized_strokes[1:]:
            res = convert_stroke_to_bezier_curves(stroke, precision=precision)

            current_x = stroke[stroke[:, 2] >= 0][0, 0]
            current_y = stroke[stroke[:, 2] >= 0][0, 1]

            dx = current_x - last_x
            dy = current_y - last_y

            last_x = stroke[stroke[:, 2] >= 0][-1, 0]
            last_y = stroke[stroke[:, 2] >= 0][-1, 1]

            if res:
                points.extend([[dx, dy, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # New starting point
                res[0][9] = 1 # Indicate start of stroke
                points.extend(res)

        return np.array(points)

    def encode_sample(self, text):
        return [self.alphabet.index(c) for c in text]

    def decode_sample(self, encoded_text):
        return [self.alphabet[int(v)] for v in encoded_text]

    def pad_data(self, l, value=0, width=None):
        max_len = max(len(item) for item in l)

        padded_numpy_array = None

        if width:
            padded_numpy_array = np.full((len(l), max_len, width), value, dtype=np.float32)
        else:
            padded_numpy_array = np.full((len(l), max_len), value, dtype=np.float32)

        for i, row in enumerate(l):
            padded_numpy_array[i, :len(row)] = row

        return padded_numpy_array

def scale_timestamps(strokes):
    squared_differences = np.diff(strokes, axis=1)[:, :, :2]**2
    distances = np.sum(squared_differences, axis=2)**(1/2)
    total_length = np.sum(distances[(strokes[:, :, 2] >= 0)[:, 1:]])
    scaling_factor = strokes[strokes[:, :, 2] >= 0][-1, 2] / total_length
    normalized = strokes.copy()
    normalized[:,:,2][strokes[:, :, 2] >= 0] /= scaling_factor
    return normalized

def makeSMatrix(s, width):
    return np.column_stack([s**p for p in range(width)])

def SSE(data, P, s):
    D = data[data[:, 2] >= 0]
    S = makeSMatrix(s, 4)
    return np.sum(np.sum((D - (S@P))**2, axis=1), axis=0)

def newton_step(data, P, s):
    D = data[data[:, 2] >= 0]
    S = makeSMatrix(s, 4)
    C = S@P

    P1d = P[1:, :] * [[1], [2], [3]]
    C1d = makeSMatrix(s, 3)@P1d # First derivates

    P2d = P1d[1:, :] * [[1], [2]]
    C2d = makeSMatrix(s, 2)@P2d # Second derivates

    P3d = P2d[1:, :]
    C3d = makeSMatrix(s, 1)@P3d # Third derivates

    N1 = (D[:, 0] - C[:, 0])*C2d[:, 0] + (D[:, 1] - C[:, 1])*C2d[:, 1]\
            - C2d[:, 0]**2 - C2d[:, 1]**2

    N2 = (D[:, 0] - C[:, 0])*C3d[:, 0] + (D[:, 1] - C[:, 1])*C3d[:, 1]\
            - 2*C3d[:, 0]*C2d[:, 0] - C1d[:, 0]*C2d[:, 0]\
            - 2*C3d[:, 1]*C2d[:, 1] - C1d[:, 1]*C2d[:, 1]

    s_new = np.copy(s)
    s_new[1:-1] -= (N1/N2)[1:-1] # Keep s=0 and s=1 in place.
    return s_new


def get_control_points(P):
    C = makeSMatrix(np.array([0, 1]), 4)@P
    p0 = C[0, :2]
    p3 = C[-1, :2]

    P1d = P[1:, :] * [[1], [2], [3]]
    C1d = makeSMatrix(np.array([0, 1]), 3)@P1d # First derivates

    p1 = p0 + (1/3) * C1d[0, :2]
    p2 = p3 - (1/3) * C1d[-1, :2]

    return [p0, p1, p2, p3]

def parameterize_curve(P, p):
    p0, p1, p2, p3 = get_control_points(P)

    vec_14 = p3 - p0 # Vec from controlpoint 1 to control point 4
    vec_41 = p0 - p3 # Vec from controlpoint 4 to control point 1

    distance_endpoints = np.sum((p3 - p0)**2)**(1/2)

    if distance_endpoints == 0:
        distance_endpoints = 0.0001

    control_vec1 = p1 - p0
    control_vec2 = p2 - p3

    d1 = np.sum(control_vec1**2)**(1/2) / distance_endpoints
    d2 = np.sum(control_vec2**2)**(1/2) / distance_endpoints

    a1 = np.arctan2(
        control_vec1[0] * vec_14[1] - control_vec1[1] * vec_14[0],
        np.dot(vec_14, control_vec1)
    )

    a2 = np.arctan2(
        control_vec2[0] * vec_41[1] - control_vec2[1] * vec_41[0],
        np.dot(vec_41, control_vec2) # switched
    )

    return [vec_14[0], vec_14[1], d1, d2, a1, a2, P[1, 2], P[2, 2], P[3, 2], p, p]

def get_relative_distances(data):
    if len(data) < 2:
        return 0

    diffs = (data[1:, :] - data[:-1, :])
    distances = np.insert((diffs[:, 0]**2 + diffs[:, 1]**2)**(1/2), 0, 0)
    cummulative_distances = np.cumsum(distances)
    return cummulative_distances/cummulative_distances[-1]

def fit_curve_newton_step(data, delta=0.05, precision=0.05, maxiter=10):
    D = data[data[:, 2] >= 0]
    s = get_relative_distances(D)
    S = makeSMatrix(s, 4)
    PE = np.linalg.lstsq(S, D, rcond=None)[0]

    prev_error = SSE(D, PE, s)

    if prev_error < precision:
        return PE, s, prev_error

    for value in range(maxiter):
        s = newton_step(D, PE, s)
        S = makeSMatrix(s, 4)
        PE = np.linalg.lstsq(S, D, rcond=None)[0]

        error = SSE(D, PE, s)

        if abs(error - prev_error) < delta:
            break

        prev_error = error

    return PE, s, prev_error

def length_vecs(vec):
    return (vec[:, 0]**2 + vec[:, 1]**2)**(1/2)

def dot_vecs(vec1, vec2):
    return vec1[:, 0]*vec2[:, 0] + vec1[:, 1]*vec2[:, 1]

def calc_angles(stroke):
    D = stroke[stroke[:, 2] >= 0]
    vecs_back = D[1:, :] - D[:-1, :]
    vecs_forward = D[:-1, :] - D[1:, :]

    frac = dot_vecs(vecs_forward[1:], vecs_back[:-1]) / (length_vecs(vecs_forward[1:]) * length_vecs(vecs_back[:-1]))

    # Prevent numerical errors
    frac[frac < -1] = -1
    frac[frac > 1] = 1

    return np.arccos(frac)

def split_datapoints(stroke):
    angles = calc_angles(stroke)
    indices = np.argsort(angles) + 1

    for index in indices:
        if 3 <= index and\
           index <= len(stroke[stroke[:,2]>=0]) - 3 and\
           len(stroke[stroke[:,2]>=0]) > 6: # Make sure there are enough datapoints to make the fit.
            return stroke[:index+1], stroke[index:]

    return None

def fit_datapoints(datapoints, precision=0.001):
    res = fit_curve_newton_step(datapoints)
    PE, s, error = res
    stdev = (error/len(datapoints))**(1/2)
    curves = []

    curve_diffs = (datapoints[1:, :] - datapoints[:-1, :]) # Smarter way to do this?
    distances = (curve_diffs[:, 0]**2 + curve_diffs[:, 1]**2)**(1/2)

    abs_diffs = datapoints[0] - datapoints[-1]
    abs_dist = (abs_diffs[0]**2 + abs_diffs[1]**2)**(1/2)

    if abs_dist != 0 and (stdev > precision or (np.sum(distances) / abs_dist) > 3):
        split = split_datapoints(datapoints)

        if split:
            first_h, second_h = split
            res_f = fit_datapoints(first_h, precision)
            res_s = fit_datapoints(second_h, precision)

            if res_f:
                curves.extend(res_f)
            if res_s:
                curves.extend(res_s)
        else:
            curves.append([PE, s, stdev, datapoints])
    else:
        curves.append([PE, s, stdev, datapoints])

    return curves

def stitch_curves(curves, precision=0.001):
    fitted_curves = curves.copy()
    curves = []

    if len(fitted_curves) == 1:
        return fitted_curves

    for i, _ in enumerate(fitted_curves[:-1]):
        d = np.vstack((fitted_curves[i][3][:-1], fitted_curves[i+1][3]))
        PE, s, error = fit_curve_newton_step(d)
        stdev = (error/len(d))**(1/2)

        curve_diffs = (d[1:, :] - d[:-1, :])
        distances = (curve_diffs[:, 0]**2 + curve_diffs[:, 1]**2)**(1/2)

        abs_diffs = d[0] - d[-1]
        abs_dist = (abs_diffs[0]**2 + abs_diffs[1]**2)**(1/2)

        if stdev > precision or (np.sum(distances) / abs_dist) > 3:
            curves.append(fitted_curves[i])
            if i == len(fitted_curves) - 2:
                curves.append(fitted_curves[i+1])
        else:
            if i == len(fitted_curves) - 2:
                curves.append([PE, s, stdev, d])
            else:
                fitted_curves[i+1] = [PE, s, stdev, d]

    return curves

def convert_stroke_to_bezier_curves(datapoints, precision=0.001, debug=False):
    if len(datapoints[datapoints[:, 2]>=0]) < 1:
        return None

    fitted_curves = fit_datapoints(datapoints)
    stiched_curves = stitch_curves(fitted_curves, precision=precision)

    parameters = [parameterize_curve(PE, 0) for (PE, s, stdev, d) in stiched_curves]
    return parameters
