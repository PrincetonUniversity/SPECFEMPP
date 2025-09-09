#pragma once

#include <iostream>

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else // NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#endif

#ifndef NDEBUG
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define DEVICE_ASSERT(condition, message)                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("Assertion `%s` failed in %s line %d: %s\n", #condition,          \
             __FILE__, __LINE__, message);                                     \
      assert(false);                                                           \
    }                                                                          \
  } while (false)
#else // KOKKOS_ENABLE_CUDA
#define DEVICE_ASSERT(condition, message)                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("Assertion `%s` failed in %s line %d: %s\n", #condition,          \
             __FILE__, __LINE__, message);                                     \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#endif // KOKKOS_ENABLE_CUDA
#else  // NDEBUG
#define DEVICE_ASSERT(condition, message)                                      \
  do {                                                                         \
  } while (false)
#endif // NDEBUG

// if intel compiler
#ifndef SPECFEM_ENABLE_PROFILING
#ifdef __INTEL_COMPILER
#define NOINLINE __declspec(noinline)
#elif __GNUC__
#define NOINLINE __attribute__((noinline))
#elif __CUDA__
#define NOINLINE __noinline__
#elif __HIP__
#define NOINLINE __noinline__
#endif
#else
#define NOINLINE
#endif

#define DEPRECATE(msg) [[deprecated(msg)]]
