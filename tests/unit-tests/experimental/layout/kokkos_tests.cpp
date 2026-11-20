#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define LAYOUT_KOKKOS_SUITE HierarchicalKokkosDebug
#else
#define LAYOUT_KOKKOS_SUITE HierarchicalKokkos
#endif
#include "specfem/datatype/chunk_element_view.hpp"
#include "specfem/datatype/element_view.hpp"
#include "specfem/experimental/layout/kokkos.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace hl = specfem::experimental::layout;

namespace specfem::testing::layout_kokkos_impl {
template <typename NativeView, std::size_t... I>
void expect_native_writes(const NativeView &native, std::index_sequence<I...>);
template <typename NativeView>
void expect_native_writes(const NativeView &native);
template <typename ArrayLayout> void expect_native_families();
template <typename ScratchView> void expect_scratch_writes();
void expect_device_capture();
} // namespace specfem::testing::layout_kokkos_impl

template <typename NativeView, std::size_t... I>
void specfem::testing::layout_kokkos_impl::expect_native_writes(
    const NativeView &native, std::index_sequence<I...>) {
  using execution_space = typename NativeView::execution_space;
  const auto view = hl::make_view(native);
  int errors = 0;
  Kokkos::parallel_reduce(
      "native_adapter_oracle",
      Kokkos::RangePolicy<execution_space>(0, native.size()),
      KOKKOS_LAMBDA(const int linear, int &failures) {
        Kokkos::Array<std::size_t, NativeView::rank()> indices{};
        std::size_t remaining = linear;
        for (int axis = static_cast<int>(NativeView::rank()) - 1; axis >= 0;
             --axis) {
          indices[axis] = remaining % native.extent(axis);
          remaining /= native.extent(axis);
        }
        const auto coordinate = hl::make_tuple(indices[I]...);
        if (&view(coordinate) != &native(indices[I]...)) {
          ++failures;
        }
        view(coordinate) = linear + 17;
        if (native(indices[I]...) != linear + 17) {
          ++failures;
        }
      },
      errors);
  EXPECT_EQ(errors, 0);
}

template <typename NativeView>
void specfem::testing::layout_kokkos_impl::expect_native_writes(
    const NativeView &native) {
  expect_native_writes(native, std::make_index_sequence<NativeView::rank()>{});
}

template <typename ArrayLayout>
void specfem::testing::layout_kokkos_impl::expect_native_families() {
  using space = typename Kokkos::DefaultExecutionSpace::memory_space;
  expect_native_writes(Kokkos::View<int, ArrayLayout, space>("scalar"));
  expect_native_writes(Kokkos::View<int *, ArrayLayout, space>("rank1", 7));
  expect_native_writes(Kokkos::View<int **, ArrayLayout, space>("rank2", 5, 3));
  expect_native_writes(
      Kokkos::View<int ***, ArrayLayout, space>("rank3", 4, 1, 3));
  expect_native_writes(
      Kokkos::View<int ****, ArrayLayout, space>("rank4", 3, 2, 1, 4));
  expect_native_writes(
      Kokkos::View<int *****, ArrayLayout, space>("rank5", 2, 3, 2, 1, 3));
  expect_native_writes(
      Kokkos::View<int ******, ArrayLayout, space>("rank6", 3, 2, 2, 1, 2, 3));
  // Large enough for Kokkos to add native pitch padding where enabled.
  expect_native_writes(Kokkos::View<int **, ArrayLayout, space>(
      Kokkos::view_alloc("padded", Kokkos::AllowPadding), 129, 17));
  const Kokkos::View<int ***, ArrayLayout, space> buffer("buffer", 3, 8, 5);
  expect_native_writes(Kokkos::subview(buffer, 1, Kokkos::ALL, Kokkos::ALL));
  const Kokkos::View<int ****, ArrayLayout, space> field2("field2", 10, 5, 5,
                                                          2);
  expect_native_writes(Kokkos::subview(field2, Kokkos::make_pair(2, 7),
                                       Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  const Kokkos::View<int *****, ArrayLayout, space> field3("field3", 9, 3, 3, 3,
                                                           3);
  expect_native_writes(Kokkos::subview(field3, Kokkos::make_pair(4, 9),
                                       Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                       Kokkos::ALL));
  const Kokkos::View<int ***, ArrayLayout, space> faces("faces", 6, 4, 4);
  expect_native_writes(Kokkos::subview(faces, 2, Kokkos::ALL, Kokkos::ALL));
  expect_native_writes(Kokkos::subview(faces, Kokkos::make_pair(1, 4),
                                       Kokkos::ALL, Kokkos::ALL));
  const Kokkos::View<int *, ArrayLayout, space> indices("indices", 20);
  expect_native_writes(Kokkos::subview(indices, Kokkos::make_pair(3, 11)));
}

TEST(LAYOUT_KOKKOS_SUITE, NativeOffsetsAndWrites) {
  specfem::testing::layout_kokkos_impl::expect_native_families<
      Kokkos::LayoutLeft>();
  specfem::testing::layout_kokkos_impl::expect_native_families<
      Kokkos::LayoutRight>();
}

TEST(LAYOUT_KOKKOS_SUITE, StaticAndMixedExtents) {
  Kokkos::View<int *[3], Kokkos::HostSpace> mixed("mixed", 5);
  const auto view = hl::make_view(mixed);
  static_assert(decltype(view.layout().template extent<1>())::value == 3);
  specfem::testing::layout_kokkos_impl::expect_native_writes(mixed);
  Kokkos::View<int[5][3], Kokkos::LayoutLeft, Kokkos::HostSpace> fixed("fixed");
  specfem::testing::layout_kokkos_impl::expect_native_writes(fixed);
}

TEST(LAYOUT_KOKKOS_SUITE, RetainsManagedSubview) {
  const auto view = [] {
    Kokkos::View<int ***, Kokkos::HostSpace> parent("parent", 3, 8, 5);
    const auto slot = Kokkos::subview(parent, 1, Kokkos::ALL, Kokkos::ALL);
    const auto owners_before = slot.use_count();
    const auto adapted = hl::make_view(slot);
    EXPECT_GT(slot.use_count(), owners_before);
    adapted(hl::make_tuple(2, 3)) = 73;
    return adapted;
  }();
  Kokkos::View<int ***, Kokkos::HostSpace> distractor("distractor", 3, 8, 5);
  Kokkos::deep_copy(distractor, -1);
  EXPECT_EQ(view(hl::make_tuple(2, 3)), 73);
}

TEST(LAYOUT_KOKKOS_SUITE, ConstReferencesAndBorrowedWrites) {
  int storage[4]{};
  using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  const Kokkos::View<int *, Kokkos::HostSpace, unmanaged> native(storage, 4);
  const auto mutable_view = hl::make_view(native);
  static_assert(
      std::is_same_v<decltype(mutable_view(hl::make_tuple(0))), int &>);
  mutable_view(hl::make_tuple(2)) = 31;
  EXPECT_EQ(storage[2], 31);
  const Kokkos::View<const int *, Kokkos::HostSpace, unmanaged> readonly(
      native);
  const auto const_view = hl::make_view(readonly);
  static_assert(
      std::is_same_v<decltype(const_view(hl::make_tuple(0))), const int &>);
  EXPECT_EQ(const_view(hl::make_tuple(2)), 31);
  const auto raw = hl::make_view(storage, 4, mutable_view.layout());
  raw(hl::make_tuple(1)) = 23;
  EXPECT_EQ(mutable_view(hl::make_tuple(1)), 23);
}

TEST(LAYOUT_KOKKOS_SUITE, EmptyAndScalar) {
  using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  const Kokkos::View<int *, Kokkos::HostSpace, unmanaged> empty(nullptr, 0);
  EXPECT_TRUE(hl::make_view(empty).layout().empty());
  Kokkos::View<int *, Kokkos::HostSpace> native("native", 5);
  const auto range = Kokkos::subview(native, Kokkos::make_pair(2, 2));
  EXPECT_TRUE(hl::make_view(range).layout().empty());
  const auto disabled =
      hl::make_layout(hl::make_tuple(), hl::make_tuple(), false);
  EXPECT_TRUE(hl::make_view(empty, disabled).layout().empty());
  int value = 9;
  const Kokkos::View<int, Kokkos::HostSpace, unmanaged> scalar(&value);
  const auto view = hl::make_view(scalar);
  EXPECT_EQ(view(hl::make_tuple()), 9);
  view(hl::make_tuple()) = 11;
  EXPECT_EQ(value, 11);
}

void specfem::testing::layout_kokkos_impl::expect_device_capture() {
  const auto captured = [] {
    Kokkos::View<int **, Kokkos::LayoutLeft> native("device_retained", 3, 4);
    return hl::make_view(native);
  }();
  Kokkos::View<int **, Kokkos::LayoutLeft> native("device_native", 3, 4);
  Kokkos::View<int *> result("result", 4);
  Kokkos::parallel_for(
      "adapter_capture", 4, KOKKOS_LAMBDA(const int j) {
        captured(hl::make_tuple(2, j)) = 41 + j;
        const auto local = hl::make_view(native);
        local(hl::make_tuple(1, j)) = 19 + j;
        const auto slot = Kokkos::subview(native, 1, Kokkos::ALL);
        const auto subview = hl::make_view(slot);
        result(j) = captured(hl::make_tuple(2, j)) + subview(hl::make_tuple(j));
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(host(j), 60 + 2 * j);
  }
  Kokkos::fence();
}

TEST(LAYOUT_KOKKOS_SUITE, DeviceCaptureAndConstruction) {
  specfem::testing::layout_kokkos_impl::expect_device_capture();
}

template <typename ScratchView>
void specfem::testing::layout_kokkos_impl::expect_scratch_writes() {
  using policy = Kokkos::TeamPolicy<>;
  using member = typename policy::member_type;
  Kokkos::View<int *> result("scratch_result", 2);
  Kokkos::parallel_for(
      "scratch_adapter",
      policy(2, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(ScratchView::shmem_size())),
      KOKKOS_LAMBDA(const member &team) {
        const ScratchView native(team.team_scratch(0));
        const auto view = hl::make_view(native);
        if (team.team_rank() == 0) {
          // The final corner exercises every imported scratch stride.
          if constexpr (ScratchView::rank() == 2) {
            view(hl::make_tuple(native.extent(0) - 1, native.extent(1) - 1)) =
                83;
          } else if constexpr (ScratchView::rank() == 3) {
            view(hl::make_tuple(native.extent(0) - 1, native.extent(1) - 1,
                                native.extent(2) - 1)) = 83;
          } else if constexpr (ScratchView::rank() == 4) {
            view(hl::make_tuple(native.extent(0) - 1, native.extent(1) - 1,
                                native.extent(2) - 1, native.extent(3) - 1)) =
                83;
          } else {
            view(hl::make_tuple(native.extent(0) - 1, native.extent(1) - 1,
                                native.extent(2) - 1, native.extent(3) - 1,
                                native.extent(4) - 1)) = 83;
          }
        }
        team.team_barrier();
        if (team.team_rank() == team.team_size() - 1) {
          result(team.league_rank()) = native.data()[native.span() - 1];
        }
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  EXPECT_EQ(host(0), 83);
  EXPECT_EQ(host(1), 83);
}

TEST(LAYOUT_KOKKOS_SUITE, ScratchWrites) {
  using specfem::element::dimension_tag;
  using space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  using element2 = typename specfem::datatype::ScalarElementViewType<
      int, dimension_tag::dim2, 5, space, unmanaged>::type;
  using element3 = typename specfem::datatype::ScalarElementViewType<
      int, dimension_tag::dim3, 5, space, unmanaged>::type;
  using chunk2_left = typename specfem::datatype::VectorChunkElementViewType<
      int, dimension_tag::dim2, 32, 5, 2, false, space, unmanaged,
      Kokkos::LayoutLeft>::type;
  using chunk2_right = typename specfem::datatype::VectorChunkElementViewType<
      int, dimension_tag::dim2, 32, 5, 2, false, space, unmanaged,
      Kokkos::LayoutRight>::type;
  using chunk3 = typename specfem::datatype::VectorChunkElementViewType<
      int, dimension_tag::dim3, 8, 5, 3, false, space, unmanaged,
      Kokkos::LayoutLeft>::type;
  specfem::testing::layout_kokkos_impl::expect_scratch_writes<element2>();
  specfem::testing::layout_kokkos_impl::expect_scratch_writes<element3>();
  specfem::testing::layout_kokkos_impl::expect_scratch_writes<chunk2_left>();
  specfem::testing::layout_kokkos_impl::expect_scratch_writes<chunk2_right>();
  specfem::testing::layout_kokkos_impl::expect_scratch_writes<chunk3>();
}

TEST(LAYOUT_KOKKOS_SUITE, ExplicitMappingRetainsAllocationAndAliases) {
  const auto view = [] {
    Kokkos::View<int *, Kokkos::HostSpace> native("retained", 15);
    Kokkos::deep_copy(native, -1);
    const auto mapping =
        hl::make_layout(hl::make_tuple(hl::make_tuple(hl::c<5>, 3)),
                        hl::make_tuple(hl::make_tuple(hl::c<1>, 5)));
    const auto owners_before = native.use_count();
    const auto adapted = hl::make_view(native, mapping);
    EXPECT_GT(native.use_count(), owners_before);
    adapted(hl::make_tuple(7)) = 42;
    EXPECT_EQ(native(7), 42);
    return adapted;
  }();
  Kokkos::View<int *, Kokkos::HostSpace> distractor("distractor", 15);
  Kokkos::deep_copy(distractor, -5);
  EXPECT_EQ(view(hl::make_tuple(7)), 42);
  view(hl::make_tuple(hl::make_tuple(2, 1))) = 19;
  EXPECT_EQ(view(hl::make_tuple(7)), 19);
}

TEST(LAYOUT_KOKKOS_SUITE, ImportsNativeStrides) {
  Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> parent("parent",
                                                                      3, 8, 5);
  Kokkos::deep_copy(parent, -1);
  const auto slot = Kokkos::subview(parent, 1, Kokkos::ALL, Kokkos::ALL);
  const auto view = hl::make_view(slot);
  view(hl::make_tuple(2, 3)) = 73;
  EXPECT_EQ(parent(1, 2, 3), 73);
  EXPECT_EQ(parent(0, 2, 3), -1);
  EXPECT_EQ(parent(2, 2, 3), -1);
  EXPECT_EQ(&view(hl::make_tuple(2, 3)), &slot(2, 3));
}

namespace specfem::testing::layout_kokkos_impl {
class LayoutEnvironment : public ::testing::Environment {
public:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};
} // namespace specfem::testing::layout_kokkos_impl

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ::testing::AddGlobalTestEnvironment(
      new specfem::testing::layout_kokkos_impl::LayoutEnvironment);
  return RUN_ALL_TESTS();
}

TEST(LAYOUT_KOKKOS_SUITE, MutableViewsConvertToReadOnlyViews) {
  Kokkos::View<int *, Kokkos::HostSpace> native("conversion", 4);
  const auto mutable_view = hl::make_view(native);
  using MutableView = std::remove_cvref_t<decltype(mutable_view)>;
  using Mapping = std::remove_cvref_t<decltype(mutable_view.layout())>;
  using Storage =
      specfem::experimental::layout_impl::KokkosStorage<decltype(native)>;
  using ReadonlyView = hl::View<const int, Mapping, Storage>;
  static_assert(std::is_convertible_v<MutableView, ReadonlyView>);
  static_assert(!std::is_convertible_v<ReadonlyView, MutableView>);
  const ReadonlyView readonly = mutable_view;
  mutable_view(hl::make_tuple(2)) = 17;
  native = {};
  EXPECT_EQ(readonly(hl::make_tuple(2)), 17);
}
