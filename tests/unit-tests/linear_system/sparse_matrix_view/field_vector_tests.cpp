#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/sparse_matrix_view/fe_assembly.hpp"
#include "specfem/linear_system/sparse_matrix_view/field_vector.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace sparse_matrix_view_field_vector_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using MappingType = specfem::linear_system::FEMapping<dim3_tag, elastic_tag>;
using FEAssemblyType = specfem::linear_system::FEAssembly<MappingType>;
using VectorType = specfem::linear_system::vector_type;
using scalar_type = specfem::linear_system::scalar_type;

/// Field storage as SPECFEM++ allocates it
using FieldType =
    Kokkos::View<scalar_type **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Build a full assembly from a Newmark displacement-test dataset. Paths are
// relative to TEST_OUTPUT_DIR, where the displacement_tests data tree is
// linked (see SERIAL_LINK_DIRS in serial.cmake).
std::unique_ptr<AssemblyType> build_assembly_3d(const std::string &test_name) {
  const std::string test_path =
      "displacement_tests/Newmark/serial/dim3/" + test_name;

  specfem::runtime_configuration::setup setup(test_path +
                                              "/specfem_config.yaml");

  const auto database_filename = setup.get_databases();
  const auto &source_entries = setup.get_source_entries();
  const auto stations_node = setup.get_stations();
  const auto quadratures = setup.instantiate_quadrature();

  auto mesh = specfem::io::read_3d_mesh(database_filename,
                                        setup.get_attenuation_setup());

  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  auto [sources, t0, starttime] = specfem::io::read_sources<dim3_tag>(
      source_entries, nsteps, setup.get_t0(), dt, setup.get_simulation_type());
  (void)starttime;
  setup.update_t0(t0);

  auto receivers = specfem::io::read_3d_receivers(stations_node);

  return std::make_unique<AssemblyType>(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, setup.get_max_seismogram_step(),
      setup.get_nstep_between_samples(), setup.get_simulation_type(),
      setup.allocate_boundary_values(), setup.instantiate_property_reader());
}

/// A distinct value per (iglob, icomp), so a permuted copy cannot pass
scalar_type field_value(const int iglob, const int icomp) {
  return static_cast<scalar_type>(iglob) +
         static_cast<scalar_type>(0.125) * static_cast<scalar_type>(1 + icomp);
}

class FieldVector3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    fe_.reset();
    delete assembly_;
    assembly_ = nullptr;
  }

  static AssemblyType &assembly() {
    if (assembly_ == nullptr) {
      assembly_ = build_assembly_3d("HomogeneousHalfspaceSmallNoABCForceSource")
                      .release();
    }
    return *assembly_;
  }

  static FEAssemblyType &fe() {
    if (!fe_) {
      fe_ = std::make_unique<FEAssemblyType>(MappingType(assembly()));
    }
    return *fe_;
  }

  /// A field filled with @ref field_value
  static FieldType make_field() {
    const auto &mapping = fe().mapping();
    FieldType field("field", mapping.nglob(), mapping.ncomp());
    for (int icomp = 0; icomp < mapping.ncomp(); ++icomp) {
      for (int iglob = 0; iglob < mapping.nglob(); ++iglob) {
        field(iglob, icomp) = field_value(iglob, icomp);
      }
    }
    return field;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<FEAssemblyType> fe_;
};

AssemblyType *FieldVector3D::assembly_ = nullptr;
std::unique_ptr<FEAssemblyType> FieldVector3D::fe_;

// The contract the copy exists to exploit: entry gid(iglob, icomp) of the
// vector is the field's (iglob, icomp), whichever path the copy took.
TEST_F(FieldVector3D, VectorEntryAtGidHoldsTheFieldValue) {
  const auto &mapping = fe().mapping();
  const auto field = make_field();

  VectorType vector(fe().owned_map());
  specfem::linear_system::copy_field_to_vector(mapping, field, vector);

  const auto view = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (int icomp = 0; icomp < mapping.ncomp(); ++icomp) {
    for (int iglob = 0; iglob < mapping.nglob(); ++iglob) {
      ASSERT_EQ(view(static_cast<std::size_t>(mapping(iglob, icomp)), 0),
                field_value(iglob, icomp))
          << "at (iglob=" << iglob << ", icomp=" << icomp << ")";
    }
  }
}

TEST_F(FieldVector3D, RoundTripIsTheIdentity) {
  const auto &mapping = fe().mapping();
  const auto field = make_field();

  VectorType vector(fe().owned_map());
  specfem::linear_system::copy_field_to_vector(mapping, field, vector);

  FieldType back("back", mapping.nglob(), mapping.ncomp());
  specfem::linear_system::copy_vector_to_field(mapping, vector, back);

  for (int icomp = 0; icomp < mapping.ncomp(); ++icomp) {
    for (int iglob = 0; iglob < mapping.nglob(); ++iglob) {
      ASSERT_EQ(back(iglob, icomp), field(iglob, icomp))
          << "at (iglob=" << iglob << ", icomp=" << icomp << ")";
    }
  }
}

// The fast path is an optimization, not a second definition. A field view that
// is not contiguous -- rows sliced out of a taller allocation, so the stride
// between components is not nglob -- forces the indexed fallback, which must
// produce exactly the same vector.
TEST_F(FieldVector3D, FallbackAgreesWithFastPath) {
  const auto &mapping = fe().mapping();
  const int nglob = mapping.nglob();
  const int ncomp = mapping.ncomp();

  const auto contiguous = make_field();

  FieldType padded("padded", nglob + 1, ncomp);
  const auto strided =
      Kokkos::subview(padded, Kokkos::pair<int, int>(0, nglob), Kokkos::ALL);
  for (int icomp = 0; icomp < ncomp; ++icomp) {
    for (int iglob = 0; iglob < nglob; ++iglob) {
      strided(iglob, icomp) = field_value(iglob, icomp);
    }
  }
  ASSERT_FALSE(strided.span_is_contiguous())
      << "the fallback is not being exercised: this view is contiguous";

  VectorType from_contiguous(fe().owned_map());
  VectorType from_strided(fe().owned_map());
  specfem::linear_system::copy_field_to_vector(mapping, contiguous,
                                               from_contiguous);
  specfem::linear_system::copy_field_to_vector(mapping, strided, from_strided);

  const auto fast = from_contiguous.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto slow = from_strided.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (std::size_t dof = 0; dof < fast.extent(0); ++dof) {
    ASSERT_EQ(fast(dof, 0), slow(dof, 0)) << "at dof " << dof;
  }
}

TEST_F(FieldVector3D, MismatchedExtentsThrow) {
  const auto &mapping = fe().mapping();
  VectorType vector(fe().owned_map());

  FieldType wrong_points("wrong_points", mapping.nglob() + 1, mapping.ncomp());
  EXPECT_THROW(specfem::linear_system::copy_field_to_vector(
                   mapping, wrong_points, vector),
               std::runtime_error);

  FieldType wrong_components("wrong_components", mapping.nglob(),
                             mapping.ncomp() + 1);
  EXPECT_THROW(specfem::linear_system::copy_field_to_vector(
                   mapping, wrong_components, vector),
               std::runtime_error);
}

} // namespace sparse_matrix_view_field_vector_test

#else

TEST(FieldVector3D, SkippedWithoutTrilinos) {
  GTEST_SKIP() << "SPECFEM_ENABLE_TRILINOS is off; the field/vector copies are "
                  "not built.";
}

#endif // SPECFEM_ENABLE_TRILINOS
