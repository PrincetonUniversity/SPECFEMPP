#pragma once

/**
 * @file suppress_warnings.hpp
 * @brief Macros to suppress compiler warnings for different compilers
 */

#if defined(SPECFEMPP_MSVC_BUILD)
// mscv (microsoft)
// this is untested.

/**
 * @brief Pop warning state (MSVC)
 */
#define _SUPPRESS_WARNING_END __pragma("warning( pop )")

/**
 * @brief Disable warning 4172: returning address of local variable or temporary
 * (MSVC)
 */
#define _SUPPRESS_TEMPORARY_RETURN_CODE __pragma("warning( disable : 4172 )")

// https://learn.microsoft.com/en-us/cpp/preprocessor/warning?view=msvc-170#push-and-pop
/**
 * @brief Push warning state (MSVC)
 */
#define _SUPPRESS_WARNING_START __pragma("warning( push )")
#endif

#if defined(SPECFEMPP_GNU_BUILD)

/**
 * @brief Pop warning state (GCC)
 */
#define _SUPPRESS_WARNING_END _Pragma("GCC diagnostic pop")

/**
 * @brief Disable -Wreturn-local-addr (GCC)
 */
#define _SUPPRESS_TEMPORARY_RETURN_CODE                                        \
  _Pragma("GCC diagnostic ignored \"-Wreturn-local-addr\"")

// gcc
// https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html

// clang should also be reading these (first paragraph of
// https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas
//)
/**
 * @brief Push warning state (GCC)
 */
#define _SUPPRESS_WARNING_START _Pragma("GCC diagnostic push")

#endif

#if defined(SPECFEMPP_APPLE_BUILD) || defined(SPECFEMPP_INTEL_LLVM_BUILD)
// apple clang
// https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via
// pragmas

/**
 * @brief Pop warning state (Clang)
 */
#define _SUPPRESS_WARNING_END _Pragma("clang diagnostic pop")

/**
 * @brief Disable -Wreturn-stack-address and -Wreturn-local-addr (Clang)
 */
#define _SUPPRESS_TEMPORARY_RETURN_CODE                                        \
  _Pragma("clang diagnostic ignored \"-Wreturn-stack-address\"")               \
      _Pragma("clang diagnostic ignored \"-Wreturn-local-addr\"")

/**
 * @brief Push warning state (Clang)
 */
#define _SUPPRESS_WARNING_START _Pragma("clang diagnostic push")

#endif

// Fallback definitions for unsupported compilers
#if !defined(_SUPPRESS_WARNING_START) || !defined(_SUPPRESS_WARNING_END)
/**
 * @brief Fallback for _SUPPRESS_WARNING_START (no-op)
 */
#define _SUPPRESS_WARNING_START
/**
 * @brief Fallback for _SUPPRESS_WARNING_END (no-op)
 */
#define _SUPPRESS_WARNING_END
#endif

#if !defined(_SUPPRESS_TEMPORARY_RETURN_CODE)
/**
 * @brief Fallback for _SUPPRESS_TEMPORARY_RETURN_CODE (no-op)
 */
#define _SUPPRESS_TEMPORARY_RETURN_CODE
#endif

/**
 * @brief Suppress warnings about returning reference to temporary.
 *
 * This macro wraps a code block to suppress compiler warnings about returning
 * the address of a local variable or temporary. It handles compiler-specific
 * pragmas for MSVC, GCC, Clang, and Intel LLVM.
 *
 * @param CODEBLOCK The code to be executed with the warning suppressed.
 */
#define SUPPRESS_TEMPORARY_REF(CODEBLOCK)                                      \
  _SUPPRESS_WARNING_START                                                      \
  _SUPPRESS_TEMPORARY_RETURN_CODE                                              \
  CODEBLOCK                                                                    \
  _SUPPRESS_WARNING_END
