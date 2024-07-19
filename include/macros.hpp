#ifndef MACROS_HPP
#define MACROS_HPP

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
#ifdef KOKKOS_ENABLE_CUDA
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

#endif /* MACROS_HPP */
