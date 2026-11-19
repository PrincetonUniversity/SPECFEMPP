// Overrides NDEBUG for this file only. Keep each target single-source and
// NO_UNITY, because other files would see a different layout.hpp (ODR).
#if LAYOUT_COMPILE_DEBUG
#undef NDEBUG
#else
#ifndef NDEBUG
#define NDEBUG
#endif
#endif
#include "specfem/experimental/layout/view.hpp"
#include <limits>

namespace hl = specfem::experimental::layout;
constexpr auto maximum = std::numeric_limits<std::size_t>::max();

void construct_layout() {
#if LAYOUT_COMPILE_CASE == 1
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::make_tuple(2, 3)), hl::make_tuple(1));
#elif LAYOUT_COMPILE_CASE == 2
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<-1>, 2), hl::make_tuple(1, 2));
#elif LAYOUT_COMPILE_CASE == 3
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(hl::c<-1>, 2));
#elif LAYOUT_COMPILE_CASE == 4
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<maximum>, hl::c<2>),
                      hl::make_tuple(hl::c<1>, hl::c<1>));
#elif LAYOUT_COMPILE_CASE == 5
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<2>), hl::make_tuple(hl::c<maximum>));
#elif LAYOUT_COMPILE_CASE == 6
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<3>), hl::make_tuple(hl::c<maximum>));
#elif LAYOUT_COMPILE_CASE == 7
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<4>), hl::make_tuple(hl::c<1>));
  [[maybe_unused]] const auto offset = mapping.offset(hl::make_tuple(hl::c<4>));
#elif LAYOUT_COMPILE_CASE == 8
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(4), hl::make_tuple(1));
  [[maybe_unused]] const auto offset = mapping.offset(hl::make_tuple(1, 2));
#elif LAYOUT_COMPILE_CASE == 9
  [[maybe_unused]] constexpr auto mapping =
      hl::make_layout(hl::make_tuple(2), hl::make_tuple(maximum));
#elif LAYOUT_COMPILE_CASE == 10
  [[maybe_unused]] constexpr auto mapping =
      hl::make_layout(hl::make_tuple(2), hl::make_tuple(1));
  [[maybe_unused]] constexpr auto view =
      hl::make_view(static_cast<int *>(nullptr), 2, mapping);
#elif LAYOUT_COMPILE_CASE == 11
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<2>, hl::c<2>), hl::make_tuple(hl::c<maximum>, 1));

#elif LAYOUT_COMPILE_CASE == 12
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<4>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto offset =
      mapping.offset(hl::make_tuple(hl::c<4>, 0));
#elif LAYOUT_COMPILE_CASE == 13
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(4), hl::make_tuple(1));
  [[maybe_unused]] const auto offset =
      mapping.offset(hl::make_tuple(hl::c<-1>));

#elif LAYOUT_COMPILE_CASE == 14
  [[maybe_unused]] constexpr auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<2>), hl::make_tuple(hl::c<maximum>), true);
#elif LAYOUT_COMPILE_CASE == 15
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<2>), hl::make_tuple(hl::c<maximum>), false);
#elif LAYOUT_COMPILE_CASE == 16
  bool enabled = true;
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<2>), hl::make_tuple(hl::c<maximum>), enabled);

#else
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<2>, 3)),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 2)));
  [[maybe_unused]] const auto offset = mapping.offset(hl::make_tuple(5));
#endif
}
