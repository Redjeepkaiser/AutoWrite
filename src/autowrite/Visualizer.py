import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, axes=None):
        self.axes = axes

    def plot_raw_sample(self, raw_data):
        plt.title("raw sample")

        for i, stroke in enumerate(raw_data):
            plt.plot(stroke[:, 0][stroke[:, 2] >= 0], stroke[:, 1][stroke[:, 2] >= 0], label=f"stroke {i}")

        plt.xlabel("x")
        plt.ylabel("y")

    def plot_bezier_curves(self, data, control_points=False):
        current_x = 0
        current_y = 0

        for i, (dx, dy, d1, d2, a1, a2, g1, g2, g3, p1, p2) in enumerate(data):
            if p2:
                current_x += dx
                current_y += dy
                continue

            p0 = np.array([current_x, current_y])
            p3 = np.array([current_x + dx, current_y + dy])

            p1 = make_rotation_matrix(a1) @ np.array([dx, dy]) * d1 + np.array([current_x, current_y])
            p2 = make_rotation_matrix(a2) @ np.array([-dx, -dy]) * d2 + np.array([current_x + dx, current_y + dy])

            if control_points:
                plt.scatter(p0[0], p0[1])
                plt.scatter(p1[0], p1[1])
                plt.scatter(p2[0], p2[1])
                plt.scatter(p3[0], p3[1])
                plt.plot([p3[0], p2[0]], [p3[1], p2[1]])
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]])

            ts = np.linspace(0, 1, 100)
            xs = []
            ys = []

            for t in ts:
                (x, y) = bezier_curve(p0, p1, p2, p3, t)
                xs.append(x)
                ys.append(y)

            current_x = p3[0]
            current_y = p3[1]

            if self.axes:
                self.axes.plot(xs, ys, label=f"Bezier {i}")
            else:
                plt.plot(xs, ys, label=f"Bezier {i}")

        if self.axes:
            self.axes.axis('off')

def bezier_curve(p0, p1, p2, p3, t):
    return ((1-t)**3 * p0) + (3*(1-t)**2*t * p1) + (3*(1-t)*t**2 * p2) + (t**3 * p3)

def make_rotation_matrix(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
