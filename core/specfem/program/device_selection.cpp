#include "specfem/program/device_selection.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(SPECFEM_ENABLE_METAL)
#include <sycl/sycl.hpp>
#include <vector>
#endif

void specfem::program_impl::select_default_device() {
#if defined(SPECFEM_ENABLE_METAL)
  if (std::getenv("KOKKOS_DEVICE_ID") != nullptr ||
      std::getenv("KOKKOS_VISIBLE_DEVICES") != nullptr ||
      std::getenv("KOKKOS_MAP_DEVICE_ID_BY") != nullptr) {
    return; // explicit user choice; verified by check_execution_device()
  }
  // Same enumeration Kokkos uses when no KOKKOS_ARCH_* GPU is configured
  // (Kokkos::Impl::get_sycl_devices): all SYCL devices, unfiltered.
  const std::vector<sycl::device> devices = sycl::device::get_devices();
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].get_backend() == sycl::backend::metal &&
        devices[i].is_gpu()) {
      // overwrite = 0: never replace a value set concurrently by the user.
      ::setenv("KOKKOS_DEVICE_ID", std::to_string(i).c_str(), 0);
      return;
    }
  }
  throw std::runtime_error(
      "SPECFEM++ was built for Apple GPUs (SPECFEM_ENABLE_METAL), but "
      "AdaptiveCpp reports no Metal GPU. Check ACPP_VISIBILITY_MASK.");
#endif
}

std::string specfem::program_impl::check_execution_device() {
#if defined(SPECFEM_ENABLE_METAL)
  const sycl::device device = Kokkos::SYCL().sycl_queue().get_device();
  const std::string name = device.get_info<sycl::info::device::name>();
  if (device.get_backend() != sycl::backend::metal || !device.is_gpu()) {
    // Diagnostics only: run the SYCL (GPU) code path on AdaptiveCpp's host
    // device to separate Metal-specific defects from GPU-path defects.
    if (std::getenv("SPECFEM_METAL_ALLOW_HOST_DEVICE") != nullptr) {
      return "SYCL device: " + name +
             " (NOT Metal; allowed by SPECFEM_METAL_ALLOW_HOST_DEVICE)";
    }
    throw std::runtime_error(
        "SPECFEM++ was built for Apple GPUs (SPECFEM_ENABLE_METAL), but Kokkos "
        "selected '" +
        name +
        "', which is not a Metal GPU. Remove --kokkos-device-id / "
        "KOKKOS_DEVICE_ID or point it at the Metal device.");
  }
  return "SYCL device: " + name + " (AdaptiveCpp Metal backend)";
#else
  return {};
#endif
}
