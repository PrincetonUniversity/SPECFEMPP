#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define MANAGED_SUITE ManagedDebug
#else
#define MANAGED_SUITE Managed
#endif
#include "specfem/experimental/layout/domain_layout.hpp"
#include "specfem/experimental/layout/kokkos.hpp"
#include "specfem/experimental/layout/tile.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <mdspan/mdspan.hpp>

namespace hl = specfem::experimental::layout;

TEST(MANAGED_SUITE, DomainPaddingAndTiling) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2, 3));
  EXPECT_EQ(mapping.template extent<0>(), 7u);
  EXPECT_EQ(mapping.physical_span_size(), 192u);
  EXPECT_TRUE(mapping.valid(hl::make_tuple(6, 1, 2)));
  EXPECT_FALSE(mapping.valid(hl::make_tuple(7, 1, 2)));
  EXPECT_EQ(mapping.offset(hl::make_tuple(6, 1, 2)), 166u);
  const auto tiled = hl::tile(mapping, hl::c<4>, hl::all, hl::all);
  EXPECT_TRUE(tiled.valid(hl::make_tuple(6, 1, 2)));
  EXPECT_FALSE(tiled.valid(hl::make_tuple(7, 1, 2)));
  EXPECT_EQ(tiled.offset(hl::make_tuple(6, 1, 2)), 166u);
}

TEST(MANAGED_SUITE, ManagedConstructionAndLifetime) {
  using M = decltype(hl::make_domain_layout<32>(hl::make_tuple(7, 2, 3)));
  using V = hl::ManagedView<int, M, Kokkos::HostSpace>;
  V field("field", 7, 2, 3);
  EXPECT_EQ(field.rank(), 3u);
  EXPECT_EQ(field.extent(0), 7u);
  EXPECT_EQ(field.storage_span_size(), 192u);
  EXPECT_EQ(field.physical_span_size(), 192u);
  field(6, 1, 2) = 17;
  EXPECT_EQ(field(hl::make_tuple(6, 1, 2)), 17);
  V clone("clone", field.layout());
  EXPECT_EQ(clone.extent(0), 7u);
  EXPECT_EQ(clone.storage_span_size(), 192u);
  const auto retained = [&]() {
    V local("local", 7, 2, 3);
    local(6, 1, 2) = 23;
    return hl::subview(local, hl::all, 1, hl::all);
  }();
  EXPECT_EQ(retained(6, 2), 23);
  static_assert(V::is_physical_root);
  static_assert(!decltype(retained)::is_physical_root);
  const auto identity = hl::permute<hl::ModePath<>, 0, 1, 2>(field);
  static_assert(!decltype(identity)::is_physical_root);
  EXPECT_EQ(identity(6, 1, 2), 17);
  V assigned;
  assigned = field;
  assigned(6, 1, 2) = 29;
  EXPECT_EQ(field(6, 1, 2), 29);
}

TEST(MANAGED_SUITE, MutableViewsConvertToReadOnlyViews) {
  using Mapping = decltype(hl::make_domain_layout<32>(hl::make_tuple(7)));
  using Mutable = hl::ManagedView<int, Mapping, Kokkos::HostSpace>;
  using Readonly = hl::ManagedView<const int, Mapping, Kokkos::HostSpace>;
  static_assert(std::is_convertible_v<Mutable, Readonly>);
  static_assert(!std::is_convertible_v<Readonly, Mutable>);
  Mutable field("const_conversion", 7);
  field(6) = 23;
  const Readonly readonly = field;
  EXPECT_TRUE(readonly.valid(6));
  static_assert(
      std::is_same_v<decltype(readonly.flat_storage()(0)), const int &>);
  const auto slice = hl::subview(field, hl::all);
  using ConstStorage = specfem::experimental::layout_impl::KokkosStorage<
      Kokkos::View<const int *, Kokkos::HostSpace>>;
  using ConstSlice = hl::View<const int, typename decltype(slice)::mapping_type,
                              ConstStorage, true>;
  const ConstSlice const_slice = slice;
  static_assert(!ConstSlice::is_physical_root);
  static_assert(!std::is_constructible_v<
                hl::View<const int, typename decltype(slice)::mapping_type,
                         ConstStorage>,
                decltype(slice)>);
  field = {};
  EXPECT_EQ(readonly(6), 23);
  EXPECT_EQ(const_slice(6), 23);
  Mutable empty;
  const Readonly const_empty = empty;
  EXPECT_FALSE(const_empty.valid(0));
  EXPECT_EQ(const_empty.physical_span_size(), 0u);
  int data[7]{};
  const auto borrowed =
      hl::make_view(data, 7, hl::make_domain_layout<1>(hl::make_tuple(7)));
  using BorrowedMapping = typename decltype(borrowed)::mapping_type;
  const hl::View<const int, BorrowedMapping> const_borrowed = borrowed;
  static_assert(
      std::is_same_v<decltype(const_borrowed.flat_storage()(0)), const int &>);
  EXPECT_TRUE(const_borrowed.valid(6));
}

namespace specfem::testing::managed_impl {
template <std::size_t B> void check_domain_family();
void check_device_capture();
} // namespace specfem::testing::managed_impl

template <std::size_t B>
void specfem::testing::managed_impl::check_domain_family() {
  for (const std::size_t n :
       { std::size_t{ 0 }, std::size_t{ 1 }, B - 1, B, B + 1 }) {
    const auto m = hl::make_domain_layout<B>(hl::make_tuple(n, 2, 3));
    std::size_t expected = 0;
    if constexpr (B == 1) {
      for (int ix = 0; ix < 3; ++ix) {
        for (int iz = 0; iz < 2; ++iz) {
          for (std::size_t element = 0; element < n; ++element, ++expected) {
            EXPECT_EQ(m.offset(hl::make_tuple(element, iz, ix)), expected);
          }
        }
      }
    } else {
      for (std::size_t block = 0; block < n; block += B) {
        for (int ix = 0; ix < 3; ++ix) {
          for (int iz = 0; iz < 2; ++iz) {
            for (std::size_t lane = 0; lane < B; ++lane, ++expected) {
              const auto coord = hl::make_tuple(block + lane, iz, ix);
              if (block + lane < n) {
                EXPECT_TRUE(m.valid(coord));
                EXPECT_EQ(m.offset(coord), expected);
              } else {
                EXPECT_FALSE(m.valid(coord));
              }
            }
          }
        }
      }
    }
    EXPECT_EQ(m.physical_span_size(), expected);
  }
}

TEST(MANAGED_SUITE, DomainOffsetsAndBoundarySpans) {
  specfem::testing::managed_impl::check_domain_family<1>();
  specfem::testing::managed_impl::check_domain_family<32>();
  specfem::testing::managed_impl::check_domain_family<64>();
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(hl::make_domain_layout<32>(hl::make_tuple(maximum - 31))
                .physical_span_size(),
            maximum - 31);
  EXPECT_EQ(hl::make_domain_layout<64>(hl::make_tuple(maximum - 63))
                .physical_span_size(),
            maximum - 63);
}

TEST(MANAGED_SUITE, BoundedConstructionRequiresLogicalBounds) {
  const auto original = hl::make_domain_layout<32>(hl::make_tuple(7));
  using M = std::remove_cv_t<decltype(original)>;
  using Shape = typename M::shape_type;
  using Strides = typename M::strides_type;
  static_assert(!std::is_constructible_v<M, Shape, Strides>);
  static_assert(!std::is_constructible_v<M, Shape, Strides, bool>);
  static_assert(std::is_default_constructible_v<M>);
  const M bounded(original.shape(), original.strides(), hl::make_tuple(7));
  EXPECT_FALSE(bounded.empty());
  EXPECT_EQ(bounded.extent(0), 7u);
  EXPECT_TRUE(bounded.valid(hl::make_tuple(6)));
  EXPECT_FALSE(bounded.valid(hl::make_tuple(7)));
  const M empty(original.shape(), original.strides(), hl::make_tuple(0));
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.physical_span_size(), 0u);
  EXPECT_TRUE(M{}.empty());
  hl::ManagedView<int, M, Kokkos::HostSpace> clone("bounded_clone", bounded);
  EXPECT_EQ(clone.extent(0), 7u);
  EXPECT_EQ(clone.storage_span_size(), 32u);
}

TEST(MANAGED_SUITE, FactoryAndAliasShareBackendDefault) {
  using E = Kokkos::dextents<std::size_t, 3>;
  const auto mapping = hl::make_domain_layout(E{ 7, 2, 3 });
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(mapping)>, hl::DomainLayout<E>>);
#if defined(KOKKOS_ENABLE_CUDA)
  constexpr std::size_t expected_span = 192;
#elif defined(KOKKOS_ENABLE_HIP)
  constexpr std::size_t expected_span = 384;
#else
  constexpr std::size_t expected_span = 42;
#endif
  EXPECT_EQ(mapping.physical_span_size(), expected_span);
  hl::ManagedView<int, hl::DomainLayout<E>, Kokkos::HostSpace> field("default",
                                                                     mapping);
  EXPECT_EQ(field.storage_span_size(), expected_span);
  EXPECT_EQ(field.extent(0), 7u);
}

TEST(MANAGED_SUITE, CpuLabelledConstruction) {
  using M = decltype(hl::make_domain_layout<1>(hl::make_tuple(7, 2, 3)));
  hl::ManagedView<int, M, Kokkos::HostSpace> field("cpu", 7, 2, 3);
  field(6, 1, 2) = 19;
  EXPECT_EQ(field.offset(6, 1, 2), 41u);
  EXPECT_EQ(field.storage_span_size(), 42u);
  EXPECT_EQ(field.flat_storage()(41), 19);
}

TEST(MANAGED_SUITE, StaticMetadataAndNativeExtents) {
  using E = Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 2, 3>;
  using M = hl::DomainLayout<E, 64>;
  using V = hl::ManagedView<int, M, Kokkos::HostSpace>;
  V field("static", 7, 2, 3);
  static_assert(decltype(field.template extent<1>())::value == 2);
  EXPECT_EQ(field.storage_span_size(), 384u);
  EXPECT_EQ(field.offset(6, 1, 2), 326u);
  V clone("clone", field.layout());
  EXPECT_EQ(clone.extent(0), 7u);
  const auto m =
      hl::make_domain_layout<32>(hl::make_tuple(hl::c<7>, hl::c<2>, hl::c<3>));
  static_assert(decltype(m.template extent<0>())::value == 7);
  EXPECT_FALSE(m.valid(hl::make_tuple(hl::make_tuple(7, 0), 0, 0)));
  const auto mixed = hl::make_domain_layout<32>(hl::make_tuple(hl::c<7>, 2));
  EXPECT_EQ(mixed.offset(hl::make_tuple(hl::c<6>, hl::c<1>)), 38u);
  EXPECT_EQ(mixed.offset(
                hl::make_tuple(hl::make_tuple(hl::c<6>, hl::c<0>), hl::c<1>)),
            38u);
}

TEST(MANAGED_SUITE, BorrowedSpansAndEscapedStorage) {
  const auto m = hl::make_domain_layout<32>(hl::make_tuple(7));
  int buffer[64]{};
  const auto borrowed = hl::make_view(buffer, 64, m);
  EXPECT_EQ(borrowed.storage_span_size(), 64u);
  EXPECT_EQ(borrowed.physical_span_size(), 32u);
  borrowed(6) = 9;
  EXPECT_EQ(buffer[6], 9);
  const auto flat = [&]() {
    hl::ManagedView<int, decltype(m), Kokkos::HostSpace> local("flat", m);
    local(6) = 37;
    return local.flat_storage();
  }();
  EXPECT_EQ(flat(6), 37);
  Kokkos::View<const int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      native(buffer, 64);
  const auto readonly = hl::make_view(native, m);
  static_assert(std::is_same_v<decltype(readonly(6)), const int &>);
  EXPECT_EQ(readonly(6), 9);
  const auto identity = hl::subview(borrowed, hl::all);
  const auto rewrapped = hl::make_view(buffer, 64, identity.layout());
  static_assert(!decltype(rewrapped)::is_physical_root);
}

TEST(MANAGED_SUITE, NativeFlatLifetimeAndTensorAxes) {
  const auto flat = [] {
    Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> native(
        "native", 2, 3, 4);
    const auto adapted = hl::make_view(native);
    adapted(1, 2, 3) = 53;
    return adapted.flat_storage();
  }();
  EXPECT_EQ(flat(23), 53);
  const auto mapping =
      hl::make_domain_layout<64>(hl::make_tuple(7, 2, 3, 2, 2, 2));
  using V = hl::ManagedView<int, decltype(mapping), Kokkos::HostSpace>;
  V tensor("tensor", mapping);
  tensor(6, 1, 2, 1, 1, 1) = 61;
  EXPECT_EQ(tensor.flat_storage()(3014), 61);
  EXPECT_EQ(tensor.storage_span_size(), 3072u);
}

void specfem::testing::managed_impl::check_device_capture() {
  const auto m = hl::make_domain_layout<32>(hl::make_tuple(7, 2, 3));
  hl::ManagedView<int, decltype(m)> view("device", m);
  using Readonly = hl::ManagedView<const int, decltype(m)>;
  Kokkos::View<int *> output("result", 1);
  Kokkos::parallel_for(
      "managed_view_device", 1, KOKKOS_LAMBDA(int) {
        view(6, 1, 2) = 41;
        const Readonly readonly = view;
        const auto t = hl::tile(readonly, hl::c<4>, hl::all, hl::all);
        const auto s = hl::subview(t, hl::all, 1, 2);
        output(0) =
            s.valid(6) && !s.valid(7) &&
                    readonly.flat_storage()(readonly.offset(6, 1, 2)) == 41
                ? s(6)
                : -1;
      });
  auto host = Kokkos::create_mirror_view(output);
  Kokkos::deep_copy(host, output);
  EXPECT_EQ(host(0), 41);
}

TEST(MANAGED_SUITE, DeviceCaptureAndInheritedValidity) {
  specfem::testing::managed_impl::check_device_capture();
}

TEST(MANAGED_SUITE, DefaultEmptyAndRankZero) {
  using M = decltype(hl::make_domain_layout<32>(hl::make_tuple(7)));
  hl::ManagedView<int, M, Kokkos::HostSpace> empty;
  EXPECT_FALSE(empty.valid(0));
  EXPECT_EQ(empty.storage_span_size(), 0u);
  for (const auto axes :
       { std::array<int, 3>{ 0, 2, 3 }, std::array<int, 3>{ 7, 0, 3 },
         std::array<int, 3>{ 7, 2, 0 } }) {
    const auto m =
        hl::make_domain_layout<64>(hl::make_tuple(axes[0], axes[1], axes[2]));
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.physical_span_size(), 0u);
  }
  const auto huge_empty = hl::make_domain_layout<64>(
      hl::make_tuple(std::size_t{ 0 }, std::numeric_limits<std::size_t>::max(),
                     std::size_t{ 2 }));
  EXPECT_EQ(huge_empty.physical_span_size(), 0u);
  const auto scalar = hl::make_layout(hl::make_tuple(), hl::make_tuple());
  hl::ManagedView<int, decltype(scalar), Kokkos::HostSpace> v("scalar", scalar);
  v() = 5;
  EXPECT_EQ(v(), 5);
}

#ifndef NDEBUG
TEST(MANAGED_SUITE, RuntimeDiagnostics) {
  EXPECT_DEATH((hl::make_domain_layout<32>(hl::make_tuple(-1))), "nonnegative");
  EXPECT_DEATH((hl::make_domain_layout<32>(
                   hl::make_tuple(std::numeric_limits<std::size_t>::max()))),
               "overflow");
  const auto m = hl::make_domain_layout<32>(hl::make_tuple(7));
  int buffer[32]{};
  const auto v = hl::make_view(buffer, 32, m);
  EXPECT_DEATH(v(7), "out of bounds");
  EXPECT_DEATH((void)hl::tile(m, 3), "divide the fastest leaf");
  using MutableMapping = std::remove_cv_t<decltype(m)>;
  using NarrowView = hl::ManagedView<int, MutableMapping, Kokkos::HostSpace>;
  EXPECT_DEATH(
      (void)NarrowView("range", std::numeric_limits<std::size_t>::max()),
      "leaf range");
  EXPECT_DEATH((void)hl::make_view(buffer, 6, m), "backing capacity");
  using V = hl::ManagedView<int, decltype(m), Kokkos::HostSpace>;
  V empty;
  EXPECT_DEATH(empty(0), "uninitialized");
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  if (argc == 2 && std::strcmp(argv[1], "--invalid-managed-index") == 0) {
    {
      const auto m = hl::make_domain_layout<32>(hl::make_tuple(7));
      hl::ManagedView<int, decltype(m)> view("invalid_device_index", m);
      Kokkos::parallel_for(
          "invalid_managed_index", 1, KOKKOS_LAMBDA(int) { view(7) = 1; });
      Kokkos::fence();
    }
    Kokkos::finalize();
    return 0;
  }
  const int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
