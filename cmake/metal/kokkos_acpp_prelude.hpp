// Forced-include prelude for building Kokkos (and code using Kokkos) with
// AdaptiveCpp's generic single-pass (SSCP) compiler. Injected with
// `-include kokkos_acpp_prelude.hpp`; makes no change to Kokkos sources.
//
// Kokkos and desul pick host vs device code at preprocessing time with
// __SYCL_DEVICE_ONLY__. acpp's generic flow compiles host and device code in
// one pass and never defines that macro, so without this prelude every
// KOKKOS_IF_ON_DEVICE branch is dropped and every KOKKOS_IF_ON_HOST branch
// (View refcounting, std::abort, host atomics) is outlined into kernels.
//
// Both libraries only define their macros if neither is already defined
// (Kokkos_Macros.hpp: `#if !defined(KOKKOS_IF_ON_HOST) &&
// !defined(KOKKOS_IF_ON_DEVICE)`; desul/atomics/Macros.hpp: same for
// DESUL_IF_ON_*), so defining them here takes precedence. The replacement
// uses __acpp_if_target_host/device, which AdaptiveCpp resolves when the IR
// is specialised for the host or for a device target. Validated in Stage 1
// (probes/host_device_split.cpp).
#pragma once

#if defined(__ACPP__) && defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)

// Defines __acpp_if_target_host/device (sscp_backend.hpp) and declares the
// __acpp_sscp_is_host/device IR constants they test.
#include <hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp>
#include <hipSYCL/sycl/libkernel/backend.hpp>

// Same argument convention as Kokkos/desul: CODE may or may not be wrapped
// in one extra pair of parentheses, e.g. both KOKKOS_IF_ON_HOST((foo();)) and
// KOKKOS_IF_ON_HOST(foo();) occur. This is a copy of Kokkos'
// KOKKOS_IMPL_STRIP_PARENS ("remove surrounding parentheses if present"),
// under prelude-private names because desul headers do not include
// Kokkos_Macros.hpp.
#define KOKKOS_ACPP_IMPL_STRIP_PARENS(X)                                       \
  KOKKOS_ACPP_IMPL_ESC(KOKKOS_ACPP_IMPL_ISH X)
#define KOKKOS_ACPP_IMPL_ISH(...) KOKKOS_ACPP_IMPL_ISH __VA_ARGS__
#define KOKKOS_ACPP_IMPL_ESC(...) KOKKOS_ACPP_IMPL_ESC_(__VA_ARGS__)
#define KOKKOS_ACPP_IMPL_ESC_(...) KOKKOS_ACPP_IMPL_VAN_##__VA_ARGS__
#define KOKKOS_ACPP_IMPL_VAN_KOKKOS_ACPP_IMPL_ISH

#define KOKKOS_IF_ON_DEVICE(CODE)                                              \
  __acpp_if_target_device(KOKKOS_ACPP_IMPL_STRIP_PARENS(CODE))
#define KOKKOS_IF_ON_HOST(CODE)                                                \
  __acpp_if_target_host(KOKKOS_ACPP_IMPL_STRIP_PARENS(CODE))

#define DESUL_IF_ON_DEVICE(CODE)                                               \
  __acpp_if_target_device(KOKKOS_ACPP_IMPL_STRIP_PARENS(CODE))
#define DESUL_IF_ON_HOST(CODE)                                                 \
  __acpp_if_target_host(KOKKOS_ACPP_IMPL_STRIP_PARENS(CODE))

#endif
