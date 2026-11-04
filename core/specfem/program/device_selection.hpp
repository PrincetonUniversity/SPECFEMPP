#pragma once

#include <string>

namespace specfem {
namespace program_impl {

/**
 * @brief Choose the default Kokkos device before Kokkos is initialized.
 *
 * On AdaptiveCpp/Metal builds (`SPECFEM_ENABLE_METAL`), Kokkos enumerates
 * every SYCL device and defaults to device 0, which is AdaptiveCpp's OpenMP
 * host device. This function sets `KOKKOS_DEVICE_ID` to the index of the
 * first Metal GPU unless the user already chose a device through
 * `KOKKOS_DEVICE_ID`, `KOKKOS_VISIBLE_DEVICES` or `KOKKOS_MAP_DEVICE_ID_BY`.
 * A `--kokkos-device-id` command-line argument still takes precedence.
 *
 * No-op on all other builds.
 *
 * @throws std::runtime_error if no Metal GPU is visible (Metal builds only)
 */
void select_default_device();

/**
 * @brief Verify the device Kokkos selected, after Kokkos is initialized.
 *
 * On AdaptiveCpp/Metal builds, throws unless the default SYCL execution space
 * runs on a Metal GPU. A generic AdaptiveCpp build otherwise falls back to the
 * CPU device silently.
 *
 * @return Human-readable description of the selected device, or an empty
 *         string on builds without a device check
 * @throws std::runtime_error if the selected device is not a Metal GPU
 */
std::string check_execution_device();

} // namespace program_impl
} // namespace specfem
