#include "specfem/datatype/domain_view.hpp"
#include "specfem/datatype/simd.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

namespace specfem::testing::domain_view_impl {

class DomainViewEnvironment : public ::testing::Environment {
public:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};

template <specfem::element::dimension_tag DimensionTag>
void check_unaligned_simd_access();

} // namespace specfem::testing::domain_view_impl

template <specfem::element::dimension_tag DimensionTag>
void specfem::testing::domain_view_impl::check_unaligned_simd_access() {
  using simd = specfem::datatype::simd<float, true>;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;
  constexpr int width = simd::size();
  constexpr int rank =
      DimensionTag == specfem::element::dimension_tag::dim2 ? 3 : 4;
  using view_type = specfem::datatype::DomainView<
      DimensionTag, float, rank, Kokkos::DefaultExecutionSpace::memory_space>;
  const int nelements = 2 * width;
  auto view = [&]() {
    if constexpr (rank == 3) {
      return view_type("unaligned", nelements, 3, 2);
    } else {
      return view_type("unaligned", nelements, 3, 2, 2);
    }
  }();
  auto host = specfem::datatype::create_mirror_view(view);
  auto value = [&](int e, int z) -> float & {
    if constexpr (rank == 3) {
      return host(e, z, 0);
    } else {
      return host(e, z, 0, 0);
    }
  };
  for (int e = 0; e < nelements; ++e) {
    for (int z = 0; z < 3; ++z) {
      value(e, z) = 100 * z + e;
    }
  }

  // Start inside a former storage tile, then exercise a partial final vector.
  for (int first = 1; first < nelements; first += width) {
    const mask_type mask(
        [&](std::size_t lane) { return first + lane < nelements; });
    auto lanes = Kokkos::Experimental::simd_partial_load(&value(first, 1), mask,
                                                         tag_type());
    for (int lane = 0; lane < width && first + lane < nelements; ++lane) {
      EXPECT_NEAR(lanes[lane], 100 + first + lane, 1.e-6);
    }
    lanes += typename simd::datatype(1000);
    Kokkos::Experimental::simd_partial_store(lanes, &value(first, 1), mask,
                                             tag_type());
  }

  specfem::datatype::deep_copy(view, host);
  view_type clone("snapshot", view.layout());
  specfem::datatype::deep_copy(clone, view);
  auto result = specfem::datatype::create_mirror_view(clone);
  specfem::datatype::deep_copy(result, clone);
  for (int e = 0; e < nelements; ++e) {
    for (int z = 0; z < 3; ++z) {
      const auto &mapping = result.mapping();
      const auto offset = [&]() {
        if constexpr (rank == 3) {
          return mapping(e, z, 0);
        } else {
          return mapping(e, z, 0, 0);
        }
      }();
      const float expected = 100 * z + e + (z == 1 && e > 0 ? 1000 : 0);
      EXPECT_NEAR(result.data()[offset], expected, 1.e-6);
    }
  }
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
TEST(DomainView, CpuAliasesAreNativeLayoutLeft) {
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using view2d = specfem::datatype::DomainView2d<float, 3, memory_space>;
  using view3d = specfem::datatype::DomainView3d<float, 4, memory_space>;
  using vector_view = specfem::datatype::DomainView3d<float, 5, memory_space>;
  using tensor_view = specfem::datatype::DomainView3d<float, 6, memory_space>;
  static_assert(Kokkos::is_view_v<view2d> && Kokkos::is_view_v<view3d>);
  static_assert(Kokkos::is_view_v<vector_view> &&
                Kokkos::is_view_v<tensor_view>);
  static_assert(
      std::is_same_v<typename view2d::array_layout, Kokkos::LayoutLeft>);
  static_assert(
      std::is_same_v<typename view3d::array_layout, Kokkos::LayoutLeft>);
  vector_view vectors("vectors", 7, 3, 2, 2, 3);
  tensor_view tensors("tensors", 7, 3, 2, 2, 3, 3);
  EXPECT_EQ(vectors.stride(0), 1);
  EXPECT_EQ(tensors.stride(0), 1);
}

TEST(DomainView, UnalignedSimdAccess2d) {
  specfem::testing::domain_view_impl::check_unaligned_simd_access<
      specfem::element::dimension_tag::dim2>();
}

TEST(DomainView, UnalignedSimdAccess3d) {
  specfem::testing::domain_view_impl::check_unaligned_simd_access<
      specfem::element::dimension_tag::dim3>();
}
#else
TEST(DomainView, GpuAliasesRetainTiling) {
  using view_type = specfem::datatype::DomainView3d<
      float, 4, Kokkos::DefaultExecutionSpace::memory_space>;
  using mirror_type = typename view_type::host_mirror_type;
  static_assert(!Kokkos::is_view_v<view_type>);
  static_assert(std::is_same_v<typename view_type::layout_type,
                               typename mirror_type::layout_type>);
}
#endif

TEST(DomainView, TiledMappingMirrorsAndSnapshots) {
  using extents = Kokkos::dextents<std::size_t, 3>;
  using layout = specfem::datatype::DomainViewMapping<
      specfem::element::dimension_tag::dim2, 4>;
  using static_extents = Kokkos::extents<std::size_t, 7, 3, 2>;
  const layout::mapping<static_extents> static_mapping(static_extents{});
  const layout::mapping<extents> dynamic_mapping(static_mapping);
  EXPECT_EQ(dynamic_mapping.required_span_size(),
            static_mapping.required_span_size());
  using view_type =
      specfem::datatype::View<float, extents, layout,
                              Kokkos::DefaultExecutionSpace::memory_space>;
  view_type view("tiled", 7, 3, 2);
  auto host = specfem::datatype::create_mirror_view(view);
  for (int e = 0; e < 7; ++e) {
    for (int z = 0; z < 3; ++z) {
      for (int x = 0; x < 2; ++x) {
        host(e, z, x) = 100 * e + 10 * z + x;
      }
    }
  }
  specfem::datatype::deep_copy(view, host);
  view_type clone("snapshot", view.layout());
  specfem::datatype::deep_copy(clone, view);
  auto result = specfem::datatype::create_mirror_view(clone);
  specfem::datatype::deep_copy(result, clone);
  for (int e = 0; e < 7; ++e) {
    for (int z = 0; z < 3; ++z) {
      for (int x = 0; x < 2; ++x) {
        EXPECT_NEAR(result(e, z, x), 100 * e + 10 * z + x, 1.e-6);
        EXPECT_NEAR(result.data()[result.mapping()(e, z, x)], result(e, z, x),
                    1.e-6);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new specfem::testing::domain_view_impl::DomainViewEnvironment);
  return RUN_ALL_TESTS();
}
