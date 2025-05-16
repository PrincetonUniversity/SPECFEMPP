import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.figure(figsize=(15, 12))

    plt.title("Displacement field, no spin (orange: specfem, blue: finite difference)")

    for ifg, comp in enumerate(["x", "z"]):
        trace_ref = np.load("reference/traces_fd/u" + comp + "_no_spin.npy")
        trace = np.zeros(trace_ref.shape)

        for i in range(trace_ref.shape[0]):
            trace[i, :] = np.loadtxt(
                f"reference/traces_no_spin/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
            )[:, 1]

        max1 = abs(trace_ref).max()
        max2 = abs(trace).max()

        for i in range(trace_ref.shape[0]):
            plt.subplot(trace_ref.shape[0], 2, i * 2 + 1 + ifg)
            plt.ylim(-1.1, 1.1)
            plt.plot(trace_ref[i, :] / max1)
            plt.plot(trace[i, :] / max2)

            plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/u_no_spin.png")

    plt.figure(figsize=(15, 12))
    plt.title(
        "Displacement field, with spin (orange: specfem, blue: finite difference)"
    )

    for ifg, comp in enumerate(["x", "z"]):
        trace_ref = np.load("reference/traces_fd/u" + comp + ".npy")
        trace = np.zeros(trace_ref.shape)

        for i in range(trace_ref.shape[0]):
            trace[i, :] = np.loadtxt(
                f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
            )[:, 1]

        max1 = abs(trace_ref).max()
        max2 = abs(trace).max()

        for i in range(trace_ref.shape[0]):
            plt.subplot(trace_ref.shape[0], 2, i * 2 + 1 + ifg)
            plt.ylim(-1.1, 1.1)
            plt.plot(trace_ref[i, :] / max1)
            plt.plot(trace[i, :] / max2)

            plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/u.png")

    plt.figure(figsize=(8, 12))
    plt.title("Spin field (orange: specfem, blue: finite difference)")

    trace_ref = np.load("reference/traces_fd/ry.npy")
    trace = np.zeros(trace_ref.shape)

    for i in range(trace_ref.shape[0]):
        trace[i, :] = np.loadtxt(f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BXT.semr")[
            :, 1
        ]

    max1 = abs(trace_ref).max()
    max2 = abs(trace).max()

    for i in range(trace_ref.shape[0]):
        plt.subplot(trace_ref.shape[0], 1, i + 1)
        plt.ylim(-1.1, 1.1)
        plt.plot(trace_ref[i, :] / max1)
        plt.plot(trace[i, :] / max2)

        plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/spin.png")
