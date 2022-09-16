import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy


class PredictionEntropyPlot():

    def __init__(self, members, non_members, plot_class="scatter2",
                 member_color="red", non_member_color="blue"):

        assert members.ndim == 2
        assert non_members.ndim == 2
        assert members.shape[1] == non_members.shape[1]

        number_of_classes = members.shape[1]
        member_entropy = entropy(members, base=number_of_classes, axis=1)
        non_member_entropy = entropy(
            non_members, base=number_of_classes, axis=1)

        labels = ["Members", "Non-members"]

        if plot_class == "bars":

            self.f = plt.figure(figsize=(6, 4))

            bins = [10 ** i for i in range(-10, 1, 1)]
            hist_m, _ = np.histogram(member_entropy, bins=bins)
            hist_nm, _ = np.histogram(non_member_entropy, bins=bins)
            hist_m = hist_m / hist_m.sum()
            hist_nm = hist_nm / hist_nm.sum()

            x = np.arange(0.05, 1, 0.1)
            plt.bar(
                x - 0.02, hist_m,
                width=0.04, color=member_color, align='center', zorder=2)
            plt.bar(
                x + 0.02, hist_nm,
                width=0.04, color=non_member_color, align='center', zorder=2)

            plt.xlim([0.0, 1.0])
            plt.xticks(
                np.arange(0, 1.01, 0.1),
                [f"$10^{{{e}}}$" for e in range(-10, 1, 1)])
            plt.ylim([0.0, 1.0])
            plt.ylabel("Density")

            plt.grid(zorder=1)

            plt.legend(
                [plt.Rectangle((0, 0), 1, 1, color=member_color),
                 plt.Rectangle((0, 0), 1, 1, color=non_member_color)],
                labels,
                framealpha=1.0)

        elif plot_class == "scatter1":

            self.f = plt.figure(figsize=(6, 3))

            plt.plot(
                member_entropy, np.zeros_like(member_entropy),
                'o', color=member_color)
            plt.plot(
                non_member_entropy, np.zeros_like(non_member_entropy),
                'o', color=non_member_color)

            plt.xscale("log")
            plt.yticks([])

            plt.legend(
                [plt.Rectangle((0, 0), 1, 1, color=member_color),
                 plt.Rectangle((0, 0), 1, 1, color=non_member_color)],
                labels,
                framealpha=1.0)

        elif plot_class == "scatter2":

            self.f = plt.figure(figsize=(6, 2.5))
            DELTA = 0.12
            LINEWIDTH = 0.5

            plt.axhline(y=DELTA, color="black", linewidth=LINEWIDTH)
            plt.axhline(y=-DELTA, color="black", linewidth=LINEWIDTH)

            plt.plot(
                member_entropy, np.zeros_like(member_entropy) + DELTA,
                'o', color=member_color)
            plt.plot(
                non_member_entropy, np.zeros_like(non_member_entropy) - DELTA,
                'o', color=non_member_color)

            plt.xscale("log")
            plt.yticks([DELTA, -DELTA], labels)
            plt.ylim([-1, 1])

        else:
            raise ValueError(f"plot_class \"{plot_class}\" is invalid")

        plt.xlabel("Prediction entropy")
        plt.tight_layout()

    def show(self):
        self.f.show()

    def save(self, outpath):
        self.f.savefig(outpath)


def main():

    from pathlib import Path

    import tensorflow as tf
    from repkd.data_utils.load_data import load_data

    # Load model
    print("Loading model")
    model_path = Path("./test/MNIST_MLP/models/d100/"
                      "model0_d100_oadam_lcc_b10_a30-10")
    model = tf.keras.models.load_model(model_path)

    # Load data
    print("Loading data")
    train_size = 100
    training_dataset_path = f"data/mnist_digits/train_subset_{train_size}.npy"
    test_dataset_path = "data/mnist_digits/test_subset_10000.npy"
    train_x, train_y = load_data(training_dataset_path)
    test_x, test_y = load_data(test_dataset_path)
    test_x, test_y = test_x[:train_size], test_y[:train_size]

    # Predict
    train_prediction = model.predict(train_x)   # members
    test_prediction = model.predict(test_x)     # non-members

    # Split predictions by class
    members = [train_prediction[train_y == i] for i in range(10)]
    non_members = [test_prediction[test_y == i] for i in range(10)]

    # Plot
    print("Creating plots")
    CLASS = 8
    plot1 = PredictionEntropyPlot(
        members[CLASS], non_members[CLASS], plot_class="bars")
    plot2 = PredictionEntropyPlot(
        members[CLASS], non_members[CLASS], plot_class="scatter1")
    plot3 = PredictionEntropyPlot(
        members[CLASS], non_members[CLASS], plot_class="scatter2")
    plot1.show()
    plot2.show()
    plot3.show()
    input()
    # plot1.save("plot_bars.png")
    # plot2.save("plot_scatter1.jpg")
    # plot3.save("plot_scatter2.svg")


if __name__ == "__main__":
    main()
