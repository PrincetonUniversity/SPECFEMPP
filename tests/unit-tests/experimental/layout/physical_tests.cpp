#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define PHYSICAL_SUITE PhysicalDebug
#else
#define PHYSICAL_SUITE Physical
#endif
#include "specfem/experimental/layout/physical.hpp"
#include "specfem/experimental/layout/tile.hpp"
#include <array>
#include <gtest/gtest.h>

namespace hl = specfem::experimental::layout;

TEST(PHYSICAL_SUITE, MirrorSharing) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping), Kokkos::HostSpace>;
  const auto retained = [&]() {
    const V source("source", mapping);
    auto mirror = hl::create_mirror_view(source);
    static_assert(
        std::is_same_v<decltype(mirror), typename V::host_mirror_type>);
    EXPECT_EQ(mirror.flat_storage().data(), source.flat_storage().data());
    mirror(6, 1) = 17;
    EXPECT_EQ(source(6, 1), 17);
    return mirror;
  }();
  EXPECT_EQ(retained(6, 1), 17);
}

TEST(PHYSICAL_SUITE, SeparateMutableMirrorHost) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping), Kokkos::HostSpace>;
  V original("original", mapping);
  original(6, 1) = 23;
  const hl::ManagedView<const int, decltype(mapping), Kokkos::HostSpace> source(
      original);
  auto mirror = hl::create_mirror_view(source);
  static_assert(std::is_same_v<decltype(mirror),
                               typename decltype(source)::host_mirror_type>);
  EXPECT_NE(mirror.flat_storage().data(), source.flat_storage().data());
  EXPECT_EQ(mirror.physical_span_size(), 64u);
  mirror(6, 1) = 41;
  EXPECT_EQ(original(6, 1), 23);
  const auto borrowed = hl::make_view(
      static_cast<const int *>(original.flat_storage().data()), 64, mapping);
  auto borrowed_mirror = hl::create_mirror_view(borrowed);
  EXPECT_NE(borrowed_mirror.flat_storage().data(),
            borrowed.flat_storage().data());
  borrowed_mirror(6, 1) = 57;
  EXPECT_EQ(original(6, 1), 23);
}

TEST(PHYSICAL_SUITE, PhysicalPaddingRoundTrip) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping)>;
  const V source("source", mapping);
  const V saved("saved", mapping);
  auto host = hl::create_mirror_view(source);
  hl::fill(host, -19);
  for (int e = 0; e < 7; ++e) {
    host(e, 1) = 100 + e;
  }
  hl::deep_copy(source, host);
  hl::deep_copy(saved, source);
  hl::fill(source, 0);
  hl::deep_copy(source, saved);
  hl::deep_copy(host, source);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(host.flat_storage()(i), i >= 32 && i < 39 ? 100 + i - 32 : -19);
  }
}

#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
TEST(PHYSICAL_SUITE, DescriptorMismatchDebug) {
  const auto a = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  const auto b = hl::make_domain_layout<32>(hl::make_tuple(8, 2));
  hl::ManagedView<int, decltype(a), Kokkos::HostSpace> dst("dst", a);
  hl::ManagedView<int, decltype(b), Kokkos::HostSpace> src("src", b);
  EXPECT_DEATH(hl::deep_copy(dst, src), "physical descriptor mismatch");
}
#endif

namespace specfem::testing::physical_impl {
template <std::size_t B, std::size_t... I>
void check_padding(std::index_sequence<I...>);
template <std::size_t B> void check_ranks();
} // namespace specfem::testing::physical_impl

template <std::size_t B, std::size_t... I>
void specfem::testing::physical_impl::check_padding(std::index_sequence<I...>) {
  for (const std::size_t n :
       { std::size_t{ 0 }, std::size_t{ 1 }, B - 1, B, B + 1 }) {
    const auto mapping = hl::make_domain_layout<B>(
        hl::make_tuple(n, (static_cast<void>(I), 2)...));
    using V = hl::ManagedView<int, decltype(mapping)>;
    V source("source", mapping), copy("copy", mapping);
    auto host = hl::create_mirror_view(source);
    hl::fill(host, -19);
    for (std::size_t e = 0; e < n; ++e) {
      host(e, (static_cast<void>(I), 1)...) = static_cast<int>(100 + e);
    }
    hl::deep_copy(source, host);
    hl::deep_copy(copy, source);
    hl::fill(source, 0);
    hl::deep_copy(host, copy);
    constexpr std::size_t p = std::size_t{ 1 } << sizeof...(I);
    EXPECT_EQ(host.storage_span_size(), (n / B + (n % B != 0)) * B * p);
    for (std::size_t slot = 0; slot < host.storage_span_size(); ++slot) {
      int expected = -19;
      for (std::size_t e = 0; e < n; ++e) {
        const auto offset =
            B == 1 ? e + n * (p - 1) : (e / B) * B * p + e % B + B * (p - 1);
        if (slot == offset) {
          expected = static_cast<int>(100 + e);
        }
      }
      EXPECT_EQ(host.flat_storage()(slot), expected);
    }
  }
}

template <std::size_t B> void specfem::testing::physical_impl::check_ranks() {
  check_padding<B>(std::make_index_sequence<2>{});
  check_padding<B>(std::make_index_sequence<3>{});
  check_padding<B>(std::make_index_sequence<4>{});
  check_padding<B>(std::make_index_sequence<5>{});
}

TEST(PHYSICAL_SUITE, PaddingAcrossDomainFamilies) {
  specfem::testing::physical_impl::check_ranks<1>();
  specfem::testing::physical_impl::check_ranks<32>();
  specfem::testing::physical_impl::check_ranks<64>();
}

TEST(PHYSICAL_SUITE, DescriptorCompatibility) {
  const auto fixed =
      hl::make_domain_layout<32>(hl::make_tuple(hl::c<7>, hl::c<2>));
  const auto dynamic = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  hl::ManagedView<int, decltype(fixed), Kokkos::HostSpace> source("fixed",
                                                                  fixed);
  const hl::ManagedView<int, decltype(dynamic), Kokkos::HostSpace> destination(
      "dynamic", dynamic);
  hl::fill(source, 73);
  const auto readonly = hl::make_view(
      static_cast<const int *>(source.flat_storage().data()), 64, fixed);
  hl::deep_copy(destination, readonly);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(destination.flat_storage()(i), 73);
  }
}

TEST(PHYSICAL_SUITE, BorrowedRegion) {
  std::array<int, 68> allocation;
  allocation.fill(-7);
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  const auto root = hl::make_view(allocation.data() + 2, 64, mapping);
  auto mirror = hl::create_mirror_view(root);
  EXPECT_EQ(mirror.flat_storage().data(), allocation.data() + 2);
  hl::fill(mirror, 31);
  for (int i = 0; i < 68; ++i) {
    EXPECT_EQ(allocation[i], i >= 2 && i < 66 ? 31 : -7);
  }
  using Native = Kokkos::View<int *, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const auto unmanaged =
      hl::make_view(Native(allocation.data() + 2, 64), mapping);
  auto shared = hl::create_mirror_view(unmanaged);
  EXPECT_EQ(shared.flat_storage().data(), allocation.data() + 2);
  std::array<int, 68> target;
  target.fill(-9);
  const auto copy = hl::make_view(Native(target.data() + 2, 64), mapping);
  hl::deep_copy(copy, root);
  for (int i = 0; i < 68; ++i) {
    EXPECT_EQ(target[i], i >= 2 && i < 66 ? 31 : -9);
  }
}

namespace specfem::testing::physical_impl {
template <typename V> void initialize_device(const V &root);
} // namespace specfem::testing::physical_impl

template <typename V>
void specfem::testing::physical_impl::initialize_device(const V &root) {
  Kokkos::parallel_for(
      "initialize_physical", 128,
      KOKKOS_LAMBDA(int i) { root.flat_storage()(i) = 200 + i; });
  Kokkos::fence();
}

TEST(PHYSICAL_SUITE, DeviceMirrorsAndBorrowing) {
  const auto mapping = hl::make_domain_layout<64>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping)>;
  const V root("device", mapping);
  specfem::testing::physical_impl::initialize_device(root);
  const hl::ManagedView<const int, decltype(mapping)> readonly(root);
  auto host = hl::create_mirror_view(readonly);
  EXPECT_NE(host.flat_storage().data(), root.flat_storage().data());
  hl::deep_copy(host, readonly);
  EXPECT_EQ(host.flat_storage()(127), 327);
  hl::fill(host, -1);
  auto check = hl::create_mirror_view(root);
  hl::deep_copy(check, root);
  EXPECT_EQ(check(6, 1), 270);
  if constexpr (std::is_same_v<typename V::memory_space, Kokkos::HostSpace>) {
    EXPECT_EQ(check.flat_storage().data(), root.flat_storage().data());
  } else {
    EXPECT_NE(check.flat_storage().data(), root.flat_storage().data());
  }
  using Native = Kokkos::View<int *, typename V::memory_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const auto borrowed =
      hl::make_view(Native(root.flat_storage().data(), 128), mapping);
  hl::fill(borrowed, 39);
  hl::deep_copy(host, borrowed);
  EXPECT_EQ(host.flat_storage()(127), 39);
}

#ifdef KOKKOS_ENABLE_CUDA
TEST(PHYSICAL_SUITE, HostAccessibleDeviceMemoryMirrors) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7));
  using V = hl::ManagedView<int, decltype(mapping), Kokkos::CudaUVMSpace>;
  V root("uvm", mapping);
  auto shared = hl::create_mirror_view(root);
  using NativeMirror =
      typename Kokkos::View<int *, Kokkos::CudaUVMSpace>::host_mirror_type;
  static_assert(std::is_same_v<typename decltype(shared)::memory_space,
                               typename NativeMirror::memory_space>);
  EXPECT_EQ(shared.flat_storage().data(), root.flat_storage().data());
  root(6) = 81;
  const hl::ManagedView<const int, decltype(mapping), Kokkos::CudaUVMSpace>
      readonly(root);
  auto separate = hl::create_mirror_view(readonly);
  EXPECT_NE(separate.flat_storage().data(), root.flat_storage().data());
  separate(6) = 42;
  EXPECT_EQ(root(6), 81);
  hl::deep_copy(separate, readonly);
  EXPECT_EQ(separate(6), 81);
  using P =
      hl::ManagedView<int, decltype(mapping), Kokkos::CudaHostPinnedSpace>;
  P pinned("pinned", mapping);
  auto pinned_mirror = hl::create_mirror_view(pinned);
  EXPECT_EQ(pinned_mirror.flat_storage().data(), pinned.flat_storage().data());
}
#endif

TEST(PHYSICAL_SUITE, EmptyScalarAndStridedRoots) {
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping)>;
  const V empty;
  auto mirror = hl::create_mirror_view(empty);
  hl::fill(empty, 3);
  hl::deep_copy(mirror, empty);
  EXPECT_EQ(mirror.physical_span_size(), 0u);
  const hl::ManagedView<const int, decltype(mapping)> readonly_empty(empty);
  auto allocated_empty = hl::create_mirror_view(readonly_empty);
  hl::deep_copy(allocated_empty, readonly_empty);
  EXPECT_EQ(allocated_empty.storage_span_size(), 0u);
  const auto fixed =
      hl::make_domain_layout<32>(hl::make_tuple(hl::c<7>, hl::c<2>));
  const hl::ManagedView<const int, decltype(fixed)> fixed_empty;
  auto fixed_mirror = hl::create_mirror_view(fixed_empty);
  EXPECT_EQ(fixed_mirror.storage_span_size(), 0u);
  EXPECT_EQ(fixed_mirror.physical_span_size(), 0u);
  hl::deep_copy(fixed_mirror, fixed_empty);
  const auto zero = hl::make_domain_layout<64>(hl::make_tuple(7, 0));
  const auto borrowed = hl::make_view(static_cast<int *>(nullptr), 0, zero);
  hl::fill(borrowed, 3);
  auto zero_mirror = hl::create_mirror_view(borrowed);
  hl::deep_copy(zero_mirror, borrowed);
  const auto scalar = hl::make_layout(hl::make_tuple(), hl::make_tuple());
  hl::ManagedView<int, decltype(scalar), Kokkos::HostSpace> one("scalar",
                                                                scalar);
  hl::fill(one, 91);
  EXPECT_EQ(one(hl::make_tuple()), 91);
  const auto strided =
      hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(1, 4));
  hl::ManagedView<int, decltype(strided), Kokkos::HostSpace> holes("holes",
                                                                   strided);
  hl::fill(holes, 53);
  EXPECT_EQ(holes.physical_span_size(), 6u);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(holes.flat_storage()(i), 53);
  }
}

namespace specfem::testing::physical_impl {
template <typename V>
concept CanMirror = requires(const V &v) { hl::create_mirror_view(v); };
template <typename V>
concept CanFill = requires(const V &v) { hl::fill(v, 0); };
template <typename D, typename S>
concept CanCopy = requires(const D &d, const S &s) { hl::deep_copy(d, s); };

} // namespace specfem::testing::physical_impl

TEST(PHYSICAL_SUITE, PhysicalOperationConstraints) {
  namespace pi = specfem::testing::physical_impl;
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  using V = hl::ManagedView<int, decltype(mapping), Kokkos::HostSpace>;
  using C = hl::ManagedView<const int, decltype(mapping), Kokkos::HostSpace>;
  using F = hl::ManagedView<float, decltype(mapping), Kokkos::HostSpace>;
  V root("root", mapping);
  const auto identity = hl::permute<hl::ModePath<>, 0, 1>(root);
  const auto full = hl::subview(root, hl::all, hl::all);
  using I = decltype(identity);
  using S = decltype(full);
  static_assert(pi::CanMirror<V> && pi::CanMirror<C> && pi::CanFill<V> &&
                !pi::CanFill<C>);
  static_assert(pi::CanCopy<V, C> && !pi::CanCopy<C, V> && !pi::CanCopy<V, F>);
  static_assert(!pi::CanMirror<I> && !pi::CanMirror<S> && !pi::CanFill<I> &&
                !pi::CanFill<S>);
  static_assert(!pi::CanCopy<V, I> && !pi::CanCopy<I, V> &&
                !pi::CanCopy<V, S> && !pi::CanCopy<S, V>);
  const auto flat = hl::make_layout(hl::make_tuple(4), hl::make_tuple(1));
  const auto nested = hl::make_layout(hl::make_tuple(hl::make_tuple(2, 2)),
                                      hl::make_tuple(hl::make_tuple(1, 2)));
  using A = hl::ManagedView<int, decltype(flat)>;
  using B = hl::ManagedView<int, decltype(nested)>;
  static_assert(!pi::CanCopy<A, B>);
}

#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
TEST(PHYSICAL_SUITE, BorrowedSpanMismatchDebug) {
  std::array<int, 65> allocation{};
  const auto mapping = hl::make_domain_layout<32>(hl::make_tuple(7, 2));
  const auto root = hl::make_view(allocation.data(), 64, mapping);
  const auto oversized = hl::make_view(allocation.data(), 65, mapping);
  EXPECT_DEATH((void)hl::create_mirror_view(oversized), "exact storage span");
  EXPECT_DEATH(hl::fill(oversized, 1), "exact storage span");
  EXPECT_DEATH(hl::deep_copy(root, oversized), "exact storage span");
  EXPECT_DEATH(hl::deep_copy(oversized, root), "exact storage span");
}

TEST(PHYSICAL_SUITE, ShapeStrideAndSpanMismatchDebug) {
  const auto a = hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(1, 3));
  const auto b = hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(3, 1));
  const auto c = hl::make_layout(hl::make_tuple(1, 5), hl::make_tuple(1, 1));
  const auto d = hl::make_layout(hl::make_tuple(2, 2), hl::make_tuple(1, 4));
  hl::ManagedView<int, decltype(a), Kokkos::HostSpace> dst("dst", a);
  hl::ManagedView<int, decltype(b), Kokkos::HostSpace> stride("stride", b);
  hl::ManagedView<int, decltype(c), Kokkos::HostSpace> shape("shape", c);
  hl::ManagedView<int, decltype(d), Kokkos::HostSpace> span("span", d);
  EXPECT_DEATH(hl::deep_copy(dst, stride), "physical descriptor mismatch");
  EXPECT_DEATH(hl::deep_copy(dst, shape), "physical descriptor mismatch");
  EXPECT_DEATH(hl::deep_copy(dst, span), "physical descriptor mismatch");
}
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
