// Overrides NDEBUG for this file only. Keep each target single-source and
// NO_UNITY, because other files would see a different layout.hpp (ODR).
#if LAYOUT_COMPILE_DEBUG
#undef NDEBUG
#else
#ifndef NDEBUG
#define NDEBUG
#endif
#endif
#include "specfem/experimental/layout/kokkos.hpp"
#include "specfem/experimental/layout/subview.hpp"
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

#elif LAYOUT_COMPILE_CASE == 17
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<3>, hl::c<2>), hl::make_tuple(hl::c<1>, hl::c<0>));
#elif LAYOUT_COMPILE_CASE == 18
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(hl::c<4>, hl::c<2>), hl::make_tuple(hl::c<1>, hl::c<2>));
#elif LAYOUT_COMPILE_CASE == 19
  [[maybe_unused]] const auto mapping = hl::make_layout(
      hl::make_tuple(
          hl::make_tuple(hl::c<4>, hl::make_tuple(hl::c<2>, hl::c<2>)),
          hl::c<3>),
      hl::make_tuple(
          hl::make_tuple(hl::c<1>, hl::make_tuple(hl::c<8>, hl::c<8>)),
          hl::c<32>));
#elif LAYOUT_COMPILE_CASE == 20
  [[maybe_unused]] constexpr auto mapping =
      hl::make_layout(hl::make_tuple(4, 2), hl::make_tuple(1, 2));
#elif LAYOUT_COMPILE_CASE == 21
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<2>, hl::c<2>),
                      hl::make_tuple(hl::c<1>, hl::c<1>), false);
#elif LAYOUT_COMPILE_CASE == 22
  bool enabled = true;
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::c<3>, hl::c<2>),
                      hl::make_tuple(hl::c<1>, hl::c<0>), enabled);

#elif LAYOUT_COMPILE_CASE == 23
  // Kokkos cannot allocate const elements, so convert a mutable view.
  const Kokkos::View<int *, Kokkos::HostSpace> storage("storage", 4);
  const Kokkos::View<const int *, Kokkos::HostSpace> native = storage;
  const auto view = hl::make_view(native);
  view(hl::make_tuple(0)) = 1;
#elif LAYOUT_COMPILE_CASE == 24
  const Kokkos::View<int *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Atomic>>
      native("atomic", 4);
  [[maybe_unused]] const auto view = hl::make_view(native);
#elif LAYOUT_COMPILE_CASE == 25
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::permute<hl::ModePath<>, 0, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 26
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::permute<hl::ModePath<>, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 27
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::permute<hl::ModePath<>, 0, 2>(parent);

#elif LAYOUT_COMPILE_CASE == 28
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::permute<hl::ModePath<2>, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 29
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::permute<hl::ModePath<0>, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 30
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::group<hl::ModePath<>, 0, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 31
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::group<hl::ModePath<>, 1, maximum>(parent);

#elif LAYOUT_COMPILE_CASE == 32
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::group<hl::ModePath<>, maximum, 1>(parent);

#elif LAYOUT_COMPILE_CASE == 33
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::ungroup<hl::ModePath<>>(parent);

#elif LAYOUT_COMPILE_CASE == 34
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::ungroup<hl::ModePath<0>>(parent);

#elif LAYOUT_COMPILE_CASE == 35
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::permute<hl::ModePath<0, 0>, 0>(parent);

#elif LAYOUT_COMPILE_CASE == 36
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(parent);
  [[maybe_unused]] const auto offset = p.offset(hl::make_tuple(0, hl::c<2>));
#elif LAYOUT_COMPILE_CASE == 37
  // Kokkos cannot allocate const elements, so convert a mutable view.
  const Kokkos::View<int **, Kokkos::HostSpace> storage("storage", 2, 3);
  const Kokkos::View<const int **, Kokkos::HostSpace> native = storage;
  const auto g = hl::group<hl::ModePath<>, 0, 2>(hl::make_view(native));
  g(hl::make_tuple(5)) = 1;

#elif LAYOUT_COMPILE_CASE == 38
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::subview(parent, hl::all);

#elif LAYOUT_COMPILE_CASE == 39
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::make_tuple(hl::all), hl::all);

#elif LAYOUT_COMPILE_CASE == 40
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<4>, 3)),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 8)));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::make_tuple(hl::all, hl::all, hl::all));

#elif LAYOUT_COMPILE_CASE == 41
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 4));
  [[maybe_unused]] const auto result = hl::subview(parent, 1.5, hl::all);

#elif LAYOUT_COMPILE_CASE == 42
  const auto parent = hl::make_layout(hl::make_tuple(hl::c<2>, hl::c<3>),
                                      hl::make_tuple(hl::c<1>, hl::c<2>));
  [[maybe_unused]] const auto result = hl::subview(parent, hl::c<2>, hl::all);

#elif LAYOUT_COMPILE_CASE == 43
  const auto parent = hl::make_layout(hl::make_tuple(hl::c<2>, hl::c<3>),
                                      hl::make_tuple(hl::c<1>, hl::c<2>));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::range(hl::c<0>, hl::c<3>), hl::all);

#elif LAYOUT_COMPILE_CASE == 44
  const auto parent =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<4>, hl::c<3>)),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::c<4>)));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::range(hl::c<1>, hl::c<5>));

#elif LAYOUT_COMPILE_CASE == 45
  // Kokkos cannot allocate const elements, so convert a mutable view.
  const Kokkos::View<int **, Kokkos::HostSpace> storage("storage", 2, 3);
  const Kokkos::View<const int **, Kokkos::HostSpace> native = storage;
  const auto s = hl::subview(hl::make_view(native), hl::all, 1);
  s(hl::make_tuple(0)) = 1;

#elif LAYOUT_COMPILE_CASE == 46
  [[maybe_unused]] const auto selector = hl::range(true, 1);

#elif LAYOUT_COMPILE_CASE == 47
  const auto parent = hl::make_layout(hl::make_tuple(hl::c<2>, hl::c<3>),
                                      hl::make_tuple(hl::c<1>, hl::c<2>));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::range(hl::c<2>, hl::c<1>), hl::all);

#elif LAYOUT_COMPILE_CASE == 48
  // Static bounds are checked even when the selected extent is dynamic.
  const auto parent = hl::make_layout(hl::make_tuple(3), hl::make_tuple(1));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::range(hl::c<2>, hl::c<1>));

#elif LAYOUT_COMPILE_CASE == 49
  // A Range built without range() still needs integer bounds.
  const auto parent = hl::make_layout(hl::make_tuple(3), hl::make_tuple(1));
  [[maybe_unused]] const auto result =
      hl::subview(parent, hl::Range<double, double>{ 0.0, 1.0 });

#elif LAYOUT_COMPILE_CASE == 50
  // A boolean constant is not an index, as a plain bool is not.
  const auto parent =
      hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(1, 2));
  [[maybe_unused]] const auto result =
      hl::subview(parent, std::true_type{}, hl::all);

#else
  [[maybe_unused]] const auto mapping =
      hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<2>, 3)),
                      hl::make_tuple(hl::make_tuple(hl::c<1>, 2)));
  [[maybe_unused]] const auto offset = mapping.offset(hl::make_tuple(5));
#endif
}
