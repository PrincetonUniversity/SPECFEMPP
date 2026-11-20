// Root-constructor compile-fail matrix.
//
// tests.cmake compiles this file once for every pair of a LAYOUT_BAD row and a
// LAYOUT_ENTRY constructor, in debug and release, and requires each compile to
// fail with the row's "expect" message. Without LAYOUT_BAD it compiles a valid
// descriptor, which is each entry's control build.
//
// Every row is a fully static descriptor that no runtime value can make valid,
// so every public root constructor must reject it at compile time. Add a row
// for each new type-known failure and an entry for each new public root
// constructor; tests.cmake picks both up and tests every pair.
//
// Overrides NDEBUG for this file only. Keep each target single-source and
// NO_UNITY, because other files would see a different layout.hpp (ODR).
#if LAYOUT_COMPILE_DEBUG
#undef NDEBUG
#else
#ifndef NDEBUG
#define NDEBUG
#endif
#endif
#include "specfem/experimental/layout/layout.hpp"
#include <limits>

namespace hl = specfem::experimental::layout;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();

// Logical bounds for the bounds entries, by shape rank.
#define LAYOUT_BOUNDS_RUNTIME_1 hl::make_tuple(1)
#define LAYOUT_BOUNDS_RUNTIME_2 hl::make_tuple(1, 1)
#define LAYOUT_BOUNDS_EMPTY_1 hl::make_tuple(hl::c<0>)
#define LAYOUT_BOUNDS_EMPTY_2 hl::make_tuple(hl::c<0>, hl::c<0>)
#define LAYOUT_CAT_IMPL(a, b) a##b
#define LAYOUT_CAT(a, b) LAYOUT_CAT_IMPL(a, b)

// clang-format off
#if !defined(LAYOUT_BAD)
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<4>, hl::c<3>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>, hl::c<4>)
#elif LAYOUT_BAD == 1 // expect: topology
#define LAYOUT_RANK 1
#define LAYOUT_SHAPE hl::make_tuple(hl::make_tuple(hl::c<2>, hl::c<3>))
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>)
#elif LAYOUT_BAD == 2 // expect: nonnegative
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<-1>, hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>, hl::c<2>)
#elif LAYOUT_BAD == 3 // expect: nonnegative
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<2>, hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<-1>, hl::c<2>)
#elif LAYOUT_BAD == 4 // expect: overflow
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<maximum>, hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>, hl::c<maximum>)
#elif LAYOUT_BAD == 5 // expect: overflow
#define LAYOUT_RANK 1
#define LAYOUT_SHAPE hl::make_tuple(hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<maximum>)
#elif LAYOUT_BAD == 6 // expect: overflow
#define LAYOUT_RANK 1
#define LAYOUT_SHAPE hl::make_tuple(hl::c<3>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<maximum>)
#elif LAYOUT_BAD == 7 // expect: layout broadcast
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<3>, hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>, hl::c<0>)
#elif LAYOUT_BAD == 8 // expect: uniqueness not certified
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::c<2>, hl::c<2>)
#define LAYOUT_STRIDES hl::make_tuple(hl::c<1>, hl::c<1>)
#elif LAYOUT_BAD == 9 // expect: uniqueness not certified
#define LAYOUT_RANK 2
#define LAYOUT_SHAPE hl::make_tuple(hl::make_tuple(hl::c<4>, hl::make_tuple(hl::c<2>, hl::c<2>)), hl::c<3>)
#define LAYOUT_STRIDES hl::make_tuple(hl::make_tuple(hl::c<1>, hl::make_tuple(hl::c<8>, hl::c<8>)), hl::c<32>)
#endif
// clang-format on

void construct_layout() {
  [[maybe_unused]] bool runtime_true = true;
  // clang-format off
#if LAYOUT_ENTRY == 1 // make_layout(shape, strides)
  [[maybe_unused]] const auto mapping =
      hl::make_layout(LAYOUT_SHAPE, LAYOUT_STRIDES);
#elif LAYOUT_ENTRY == 2 // make_layout(shape, strides, false)
  [[maybe_unused]] const auto mapping =
      hl::make_layout(LAYOUT_SHAPE, LAYOUT_STRIDES, false);
#elif LAYOUT_ENTRY == 3 // make_layout(shape, strides, runtime true)
  [[maybe_unused]] const auto mapping =
      hl::make_layout(LAYOUT_SHAPE, LAYOUT_STRIDES, runtime_true);
#elif LAYOUT_ENTRY == 4 // make_layout(shape, strides, runtime logical bounds)
  [[maybe_unused]] const auto mapping = hl::make_layout(
      LAYOUT_SHAPE, LAYOUT_STRIDES,
      LAYOUT_CAT(LAYOUT_BOUNDS_RUNTIME_, LAYOUT_RANK));
#elif LAYOUT_ENTRY == 5 // make_layout(shape, strides, static empty bounds)
  [[maybe_unused]] const auto mapping = hl::make_layout(
      LAYOUT_SHAPE, LAYOUT_STRIDES,
      LAYOUT_CAT(LAYOUT_BOUNDS_EMPTY_, LAYOUT_RANK));
#endif
  // clang-format on
}
