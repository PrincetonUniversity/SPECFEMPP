import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.figure(figsize=(10, 8))
    for ifg, comp in enumerate(["x", "z"]):
        trace_ref = np.load("reference/v" + comp + ".npy")
        trace = np.zeros(trace_ref.shape)

        for i in range(trace_ref.shape[0]):
            trace[i, :] = np.loadtxt(
                f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
            )[:, 1]

        max1 = abs(trace_ref).max()
        max2 = abs(trace).max()

        for i in range(trace_ref.shape[0]):
            plt.subplot(trace_ref.shape[0], 2, i + 1 + ifg * trace_ref.shape[0])
            plt.ylim(-1.1, 1.1)
            plt.plot(trace_ref[i, :] / max1)
            plt.plot(trace[i, :] / max2)
            # plt.gca().get_xaxis().set_visible(i==trace_ref.shape[0]-1)
            plt.gca().get_xaxis().set_visible(False)

        plt.tight_layout()

    plt.savefig("OUTPUT_FILES/seis.png")
