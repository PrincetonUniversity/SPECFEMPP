#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/datatype/point_view.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point/acceleration.hpp"
#include "specfem/point/face_index.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

namespace specfem::testing::chunk_face_view_impl {

class ChunkFaceViewEnvironment : public ::testing::Environment {
public:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};

constexpr int chunk_size = 4;
constexpr int ngll = 3;
constexpr int ncomp = 3;
constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
constexpr type_real tolerance = 1.e-6;

using VectorStorage = Kokkos::View<type_real[chunk_size][ngll][ngll][ncomp],
                                   Kokkos::LayoutRight, Kokkos::HostSpace>;
using VectorView = specfem::datatype::VectorChunkFaceViewType<
    type_real, dimension_tag, chunk_size, ngll, ncomp, false /*UseSIMD*/,
    Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<>,
    Kokkos::LayoutRight>;
using ScalarStorage = Kokkos::View<type_real[chunk_size][ngll][ngll],
                                   Kokkos::LayoutRight, Kokkos::HostSpace>;
using ScalarView = specfem::datatype::ScalarChunkFaceViewType<
    type_real, dimension_tag, chunk_size, ngll, false /*UseSIMD*/,
    Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<>,
    Kokkos::LayoutRight>;
using PointView =
    specfem::datatype::VectorPointViewType<type_real, ncomp, false>;
using PointField = specfem::point::acceleration<specfem::tags::Tags<
    dimension_tag, specfem::element::medium_tag::elastic, false /*UseSIMD*/>>;

// Chunk-local index of face 2, point (1, 2). The element index is that of a
// mesh element and lies outside the chunk on purpose: it must not be read.
inline specfem::point::face_index<dimension_tag> local_index() {
  return { 1000 /*ispec*/,
           2 /*iface*/,
           1 /*ipoint_i*/,
           2 /*ipoint_j*/,
           0,
           0,
           0,
           specfem::mesh_entity::dim3::type::bottom };
}

} // namespace specfem::testing::chunk_face_view_impl

TEST(ChunkFaceView, VectorSubviewStoresPointViewAtLocalFace) {
  using namespace specfem::testing::chunk_face_view_impl;

  VectorStorage storage("storage");
  Kokkos::deep_copy(storage, 0);
  VectorView view(storage);

  const auto index = local_index();
  PointView point;
  point(0) = 1.5;
  point(1) = -2.5;
  point(2) = 4.0;

  view(index) = point;

  type_real sum_of_others = 0;
  for (int iface = 0; iface < chunk_size; ++iface) {
    for (int i = 0; i < ngll; ++i) {
      for (int j = 0; j < ngll; ++j) {
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          if (iface == 2 && i == 1 && j == 2) {
            EXPECT_NEAR(storage(iface, i, j, icomp), point(icomp), tolerance);
          } else {
            sum_of_others += std::abs(storage(iface, i, j, icomp));
          }
        }
      }
    }
  }
  EXPECT_NEAR(sum_of_others, 0, tolerance);
}

TEST(ChunkFaceView, VectorSubviewStoresPointField) {
  using namespace specfem::testing::chunk_face_view_impl;

  VectorStorage storage("storage");
  Kokkos::deep_copy(storage, 0);
  VectorView view(storage);

  PointField field;
  field(0) = 0.5;
  field(1) = 6.0;
  field(2) = -3.0;

  view(local_index()) = field;

  type_real sum = 0;
  for (int iface = 0; iface < chunk_size; ++iface) {
    for (int i = 0; i < ngll; ++i) {
      for (int j = 0; j < ngll; ++j) {
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          sum += std::abs(storage(iface, i, j, icomp));
        }
      }
    }
  }
  EXPECT_NEAR(storage(2, 1, 2, 0), 0.5, tolerance);
  EXPECT_NEAR(storage(2, 1, 2, 1), 6.0, tolerance);
  EXPECT_NEAR(storage(2, 1, 2, 2), -3.0, tolerance);
  EXPECT_NEAR(sum, 9.5, tolerance);
}

TEST(ChunkFaceView, VectorSubviewReadsComponents) {
  using namespace specfem::testing::chunk_face_view_impl;

  VectorStorage storage("storage");
  Kokkos::deep_copy(storage, 0);
  storage(2, 1, 2, 0) = 3.0;
  storage(2, 1, 2, 1) = 5.0;
  storage(2, 1, 2, 2) = 7.0;
  VectorView view(storage);

  const auto index = local_index();
  const auto &const_subview = view(index);
  EXPECT_NEAR(const_subview(0), 3.0, tolerance);
  EXPECT_NEAR(const_subview(1), 5.0, tolerance);
  EXPECT_NEAR(const_subview(2), 7.0, tolerance);

  // component access through the index, without a subview
  EXPECT_NEAR(view(index, 1), 5.0, tolerance);
}

TEST(ChunkFaceView, ScalarAccessByLocalIndex) {
  using namespace specfem::testing::chunk_face_view_impl;

  ScalarStorage storage("storage");
  Kokkos::deep_copy(storage, 0);
  ScalarView view(storage);

  view(local_index()) = 9.0;

  EXPECT_NEAR(storage(2, 1, 2), 9.0, tolerance);
  type_real sum = 0;
  for (int iface = 0; iface < chunk_size; ++iface) {
    for (int i = 0; i < ngll; ++i) {
      for (int j = 0; j < ngll; ++j) {
        sum += std::abs(storage(iface, i, j));
      }
    }
  }
  EXPECT_NEAR(sum, 9.0, tolerance);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new specfem::testing::chunk_face_view_impl::ChunkFaceViewEnvironment);
  return RUN_ALL_TESTS();
}
