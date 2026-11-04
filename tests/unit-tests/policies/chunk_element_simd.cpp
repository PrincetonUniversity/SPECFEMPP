#include "specfem/execution/chunked_domain_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace specfem::testing::chunk_iterator_impl {

// Exercise multi-lane indexing even when native SIMD is scalar or disabled.
struct FourLaneSimd {
  static constexpr bool using_simd = true;
  static constexpr int size() { return 4; }
};

template <specfem::element::dimension_tag DimensionTag, typename GridTag>
void check_simd_visits(
    const specfem::mesh_entity::element_grid<DimensionTag, GridTag> &grid) {
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  using Elements = Kokkos::View<int *>;
  using Iterator = specfem::execution::ChunkElementIterator<
      DimensionTag, FourLaneSimd, Elements, TeamMember, GridTag, 4>;

  for (const int nelements : { 0, 1, 3, 4, 5, 15, 16 }) {
    SCOPED_TRACE(nelements);
    Elements elements("elements", nelements);
    auto host_elements = Kokkos::create_mirror_view(elements);
    for (int element = 0; element < nelements; ++element) {
      host_elements(element) = 100 + element;
    }
    Kokkos::deep_copy(elements, host_elements);
    Kokkos::View<int **> visits("visits", nelements, grid.size);
    Kokkos::View<int> errors("errors");

    Kokkos::parallel_for(
        "check_simd_visits", TeamPolicy(1, 1),
        KOKKOS_LAMBDA(const TeamMember &team) {
          const Iterator iterator(team, elements, grid);
          for (int i = 0; i < 4 * grid.size; ++i) {
            const auto result = iterator(i);
            if (result.is_end()) {
              continue;
            }
            const auto point = result.get_index();
            const auto local = result.get_local_index();
            const int element = point.ispec - 100;
            if (element != local.ispec * FourLaneSimd::size()) {
              ++errors();
            }
            int ipoint = point.iz * grid.ngllx + point.ix;
            if constexpr (DimensionTag ==
                          specfem::element::dimension_tag::dim3) {
              ipoint =
                  (point.iz * grid.nglly + point.iy) * grid.ngllx + point.ix;
            }
            for (int lane = 0; lane < FourLaneSimd::size(); ++lane) {
              if (point.mask(lane)) {
                if (element + lane < 0 || element + lane >= nelements) {
                  ++errors();
                } else {
                  ++visits(element + lane, ipoint);
                }
              }
            }
          }
        });
    const auto host_visits =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), visits);
    const auto host_errors =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), errors);
    EXPECT_EQ(host_errors(), 0);
    for (int element = 0; element < nelements; ++element) {
      for (int ipoint = 0; ipoint < grid.size; ++ipoint) {
        EXPECT_EQ(host_visits(element, ipoint), 1)
            << "element=" << element << ", point=" << ipoint;
      }
    }
  }
}

} // namespace specfem::testing::chunk_iterator_impl

TEST(ChunkElementSimd, Dynamic2D) {
  constexpr auto dimension = specfem::element::dimension_tag::dim2;
  specfem::testing::chunk_iterator_impl::check_simd_visits(
      specfem::mesh_entity::element_grid<dimension>(3));
}

TEST(ChunkElementSimd, Fixed2D) {
  constexpr auto dimension = specfem::element::dimension_tag::dim2;
  specfem::testing::chunk_iterator_impl::check_simd_visits(
      specfem::mesh_entity::element_grid<dimension,
                                         specfem::mesh_entity::Grid<3, 3>>{});
}

TEST(ChunkElementSimd, Dynamic3D) {
  constexpr auto dimension = specfem::element::dimension_tag::dim3;
  specfem::testing::chunk_iterator_impl::check_simd_visits(
      specfem::mesh_entity::element_grid<dimension>(3));
}

TEST(ChunkElementSimd, Fixed3D) {
  constexpr auto dimension = specfem::element::dimension_tag::dim3;
  specfem::testing::chunk_iterator_impl::check_simd_visits(
      specfem::mesh_entity::element_grid<
          dimension, specfem::mesh_entity::Grid<3, 3, 3>>{});
}
