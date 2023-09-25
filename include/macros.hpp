#ifndef MACROS_HPP
#define MACROS_HPP

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#endif

// #ifndef NDEBUG
// #define DEVICE_ASSERT(condition) assert(condition)
// #else
// #define DEVICE_ASSERT(condition)
// #endif

#endif /* MACROS_HPP */
