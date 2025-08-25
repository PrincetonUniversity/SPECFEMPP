import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # plt.figure(figsize=(15, 12))

    # plt.title("Displacement field, no spin (orange: specfem, blue: finite difference)")

    # for ifg, comp in enumerate(["x", "z"]):
    #     trace_ref = np.load("reference/traces_fd/u" + comp + "_no_spin.npy")[:, ::5]
    #     trace = np.zeros(trace_ref.shape)

    #     for i in range(trace_ref.shape[0]):
    #         trace[i, :] = np.loadtxt(
    #             f"reference/traces_no_spin/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
    #         )[:, 1]

    #     max1 = abs(trace_ref).max()
    #     max2 = abs(trace).max()

    #     for i in range(trace_ref.shape[0]):
    #         plt.subplot(trace_ref.shape[0], 2, i * 2 + 1 + ifg)
    #         plt.ylim(-1.1, 1.1)
    #         plt.plot(trace_ref[i, :] / max1)
    #         plt.plot(trace[i, :] / max2, "--")

    #         plt.gca().get_xaxis().set_visible(False)

    # plt.tight_layout()
    # plt.savefig("OUTPUT_FILES/u_no_spin.png")

    plt.figure(figsize=(15, 12))
    plt.title(
        "Displacement field, with spin (orange: specfem, blue: finite difference)"
    )

    for ifg, comp in enumerate(["x", "z"]):
        trace_ref = np.load(
            "/home/ccui/seispie/examples/spin2/output/u" + comp + "_000000.npy"
        )[:, ::3]
        trace = np.zeros(trace_ref.shape)

        for i in range(trace_ref.shape[0]):
            tr = np.loadtxt(
                f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
            )
            trace[i, :] = np.loadtxt(
                f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BX{comp.upper()}.semd"
            )[:, 1]
            tr[:, 1] = trace_ref[i, :]
            np.savetxt(
                f"../tests/unit-tests/displacement_tests/Newmark/serial/HomogeneousIsotropicCosseratDomainHighFrequency/traces/AA.S000{i + 1}.S2.BX{comp.upper()}.semd",
                tr,
            )

        max1 = abs(trace_ref).max()
        max2 = abs(trace_ref).max()

        for i in range(trace_ref.shape[0]):
            plt.subplot(trace_ref.shape[0], 2, i * 2 + 1 + ifg)
            plt.ylim(-1.1, 1.1)
            plt.plot(trace_ref[i, :] / max1)
            plt.plot(trace[i, :] / max2, "--")

            plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/u.png")

    plt.figure(figsize=(8, 12))
    plt.title("Spin field (orange: specfem, blue: finite difference)")

    trace_ref = np.load("/home/ccui/seispie/examples/spin2/output/ry_000000.npy")[
        :, ::3
    ]
    trace = np.zeros(trace_ref.shape)

    for i in range(trace_ref.shape[0]):
        tr = np.loadtxt(f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BXY.semr")
        trace[i, :] = np.loadtxt(f"OUTPUT_FILES/results/AA.S000{i + 1}.S2.BXY.semr")[
            :, 1
        ]
        tr[:, 1] = trace_ref[i, :]
        np.savetxt(
            f"../tests/unit-tests/displacement_tests/Newmark/serial/HomogeneousIsotropicCosseratDomainHighFrequency/traces/AA.S000{i + 1}.S2.BXY.semr",
            tr,
        )

    max1 = abs(trace_ref).max()
    max2 = abs(trace_ref).max()

    for i in range(trace_ref.shape[0]):
        plt.subplot(trace_ref.shape[0], 1, i + 1)
        plt.ylim(-1.1, 1.1)
        plt.plot(trace_ref[i, :] / max1)
        plt.plot(trace[i, :] / max2, "--")

        plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/spin.png")
