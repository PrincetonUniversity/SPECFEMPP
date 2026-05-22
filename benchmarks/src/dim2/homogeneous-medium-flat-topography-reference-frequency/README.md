# Reference-Frequency Sensitivity Benchmark

This benchmark demonstrates the effect of the `reference-frequency` parameter on
viscoelastic attenuation in SPECFEM++. Nine simulations are run on the same
homogeneous attenuating medium (QKappa=100, Qmu=50), varying only the
reference frequency logarithmically from F0/10 to F0*10 (3–300 Hz for F0=30 Hz,
the default from the attenuation benchmark). The 9 log-spaced points give 4
frequencies below, the exact center (30 Hz), and 4 above. The resulting
seismograms are compared in a rainbow colormap plot, with the central frequency
(30 Hz) highlighted in black.
