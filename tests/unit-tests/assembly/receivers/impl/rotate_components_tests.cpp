#include "specfem/assembly/receivers/impl/rotate_components.hpp"
#include "specfem/assembly/receivers/impl/seismogram_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <gtest/gtest.h>

namespace rotate_components_test_impl {

using RotationView = Kokkos::View<type_real ***, Kokkos::HostSpace>;

RotationView make_identity(const int nrec, const int ncomp) {
  RotationView view("rotation", nrec, ncomp, ncomp);
  for (int irec = 0; irec < nrec; ++irec) {
    for (int i = 0; i < ncomp; ++i) {
      for (int j = 0; j < ncomp; ++j) {
        view(irec, i, j) = (i == j) ? 1.0 : 0.0;
      }
    }
  }
  return view;
}

} // namespace rotate_components_test_impl

TEST(RotateComponents, IdentityIsNoOp3d) {
  const auto view = rotate_components_test_impl::make_identity(2, 3);
  const type_real in[3] = { 1, 2, 3 };
  for (int irec = 0; irec < 2; ++irec) {
    type_real out[3];
    specfem::assembly::receivers_impl::rotate_components<3>(view, irec, in,
                                                            out);
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(out[i], in[i], 1e-6);
    }
  }
}

TEST(RotateComponents, PermutationApplied3d) {
  auto view = rotate_components_test_impl::make_identity(2, 3);
  const type_real permutation[3][3] = { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } };
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      view(1, i, j) = permutation[i][j];
    }
  }
  const type_real in[3] = { 1, 2, 3 };
  type_real out[3];
  specfem::assembly::receivers_impl::rotate_components<3>(view, 1, in, out);
  EXPECT_NEAR(out[0], 2, 1e-6);
  EXPECT_NEAR(out[1], 3, 1e-6);
  EXPECT_NEAR(out[2], 1, 1e-6);

  specfem::assembly::receivers_impl::rotate_components<3>(view, 0, in, out);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(out[i], in[i], 1e-6);
  }
}

TEST(RotateComponents, Rotation2d) {
  auto view = rotate_components_test_impl::make_identity(1, 2);
  view(0, 0, 0) = 0;
  view(0, 0, 1) = -1;
  view(0, 1, 0) = 1;
  view(0, 1, 1) = 0;
  const type_real in[2] = { 1, 2 };
  type_real out[2];
  specfem::assembly::receivers_impl::rotate_components<2>(view, 0, in, out);
  EXPECT_NEAR(out[0], -2, 1e-6);
  EXPECT_NEAR(out[1], 1, 1e-6);
}

TEST(RotateComponents, SeismogramIteratorRoundTripsMatrix) {
  constexpr auto dim3 = specfem::element::dimension_tag::dim3;
  specfem::assembly::receivers_impl::SeismogramIterator<dim3> iterator(
      2,
      specfem::assembly::receivers_impl::SeismogramTypes(
          { specfem::enums::wavefield::displacement }),
      10, 0.01, 0.0, 1);
  const std::array<std::array<double, 3>, 3> permutation = {
    { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } }
  };
  iterator.set_rotation_matrix(1, permutation);

  const auto matrix = iterator.get_rotation_matrix(1);
  const auto identity = iterator.get_rotation_matrix(0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(matrix[i][j], permutation[i][j], 1e-6);
      EXPECT_NEAR(identity[i][j], (i == j) ? 1.0 : 0.0, 1e-6);
    }
  }
  EXPECT_NO_THROW(iterator.sync_rotation_matrices());
}
