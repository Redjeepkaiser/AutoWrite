import matplotlib.pyplot as plt

class Visualizer:
    def plot_raw_sample(self, raw_data):
        plt.title("raw sample")
        
        for i, stroke in enumerate(raw_data):
            plt.plot(stroke[:, 0][stroke[:, 2] >= 0], stroke[:, 1][stroke[:, 2] >= 0], label=f"stroke {i}")
        
        plt.xlabel("x")
        plt.ylabel("y")
