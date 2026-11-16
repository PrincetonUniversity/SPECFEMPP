#ifdef NDEBUG
#undef NDEBUG
#endif
#include "specfem/experimental/layout/kokkos.hpp"
#include <gtest/gtest.h>

namespace hl = specfem::experimental::layout;

TEST(HierarchicalKokkosDiagnostic, RejectsCapacity) {
  const Kokkos::View<int *, Kokkos::HostSpace> native("small", 3);
  const auto mapping = hl::make_layout(hl::make_tuple(4), hl::make_tuple(1));
  EXPECT_DEATH((void)hl::make_view(native, mapping),
               "exceeds backing capacity");
}

TEST(HierarchicalKokkosDiagnostic, RejectsNullBacking) {
  using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  const Kokkos::View<int *, Kokkos::HostSpace, unmanaged> native(nullptr, 4);
  EXPECT_DEATH((void)hl::make_view(native), "nonnull backing");
}

TEST(HierarchicalKokkosDiagnostic, MemorySpaceAccess) {
  using native_type = Kokkos::View<int *>;
  const native_type native("device", 3);
  const auto view = hl::make_view(native);
  // Metadata queries do not access device storage.
  EXPECT_EQ(view.layout().template extent<0>(), 3);
  if constexpr (!Kokkos::SpaceAccessibility<
                    Kokkos::DefaultHostExecutionSpace,
                    typename native_type::memory_space>::accessible) {
    EXPECT_DEATH((void)view(hl::make_tuple(0)), "inaccessible memory space");
  } else {
    view(hl::make_tuple(0)) = 37;
    EXPECT_EQ(native(0), 37);
  }
}
