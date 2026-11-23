# ==============================================================================
# SPECFEM++ Build Configuration Variables
# ==============================================================================
# This file contains boolean variables that indicate the type of build being
# configured based on Kokkos backend settings.

# Recompute derived flags even when reusing a cache for another backend.
set(SPECFEM_CPU_BUILD FALSE CACHE BOOL "SPECFEM++ CPU build flag" FORCE)
set(SPECFEM_GPU_BUILD FALSE CACHE BOOL "SPECFEM++ GPU build flag" FORCE)

# Determine build type based on Kokkos configuration
# GPU build: if CUDA, HIP, or SYCL is enabled
if(DEFINED Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
    set(SPECFEM_GPU_BUILD TRUE CACHE BOOL "SPECFEM++ GPU build flag" FORCE)
elseif(DEFINED Kokkos_ENABLE_HIP AND Kokkos_ENABLE_HIP)
    set(SPECFEM_GPU_BUILD TRUE CACHE BOOL "SPECFEM++ GPU build flag" FORCE)
elseif(DEFINED Kokkos_ENABLE_SYCL AND Kokkos_ENABLE_SYCL)
    set(SPECFEM_GPU_BUILD TRUE CACHE BOOL "SPECFEM++ GPU build flag" FORCE)
else()
    # CPU build: when CUDA, HIP, and SYCL are disabled (defaults to OpenMP/Serial)
    set(SPECFEM_CPU_BUILD TRUE CACHE BOOL "SPECFEM++ CPU build flag" FORCE)
endif()

# Ensure exactly one build type is active
if(SPECFEM_CPU_BUILD AND SPECFEM_GPU_BUILD)
    message(FATAL_ERROR "SPECFEM++ Configuration Error: Both CPU and GPU build flags are enabled simultaneously.\n"
                        "  This indicates a logic error in the Kokkos backend detection.\n"
                        "  Please check your Kokkos configuration and ensure only one backend is enabled.")
elseif(NOT SPECFEM_CPU_BUILD AND NOT SPECFEM_GPU_BUILD)
    message(FATAL_ERROR "SPECFEM++ Configuration Error: Neither CPU nor GPU build flags are enabled.\n"
                        "  This indicates a failure in Kokkos backend detection.\n"
                        "  Please verify your Kokkos installation and configuration settings.")
endif()

# Display comprehensive configuration summary
message(STATUS "")
message(STATUS "SPECFEM++ Build Configuration Summary:")
message(STATUS "  CPU Build Enabled: ${SPECFEM_CPU_BUILD}")
message(STATUS "  GPU Build Enabled: ${SPECFEM_GPU_BUILD}")
if(SPECFEM_GPU_BUILD)
    if(DEFINED Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_CUDA)
        message(STATUS "  Active GPU Backend: NVIDIA CUDA")
    elseif(DEFINED Kokkos_ENABLE_HIP AND Kokkos_ENABLE_HIP)
        message(STATUS "  Active GPU Backend: AMD ROCm/HIP")
    elseif(DEFINED Kokkos_ENABLE_SYCL AND Kokkos_ENABLE_SYCL)
        message(STATUS "  Active GPU Backend: Intel SYCL")
    endif()
else()
    message(STATUS "  Active CPU Backend: OpenMP/Serial (Kokkos default)")
endif()
message(STATUS "")
