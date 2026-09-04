#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/linear_system/dof_numbering.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/linear_system/sparse_matrix_view/mapping.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/tags.hpp"
#include <memory>
#include <string>

namespace sparse_matrix_view_mapping_test {

constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto elastic_tag = specfem::element::medium_tag::elastic;
constexpr int NGLL = 5;

// The Trilinos-free half names its own ordinal: there is no FEMapping alias
// without Tpetra. Any integer type works here; the graphs built in
// fe_assembly_tests use Tpetra's.
using global_ordinal_type = long long;

using AssemblyType = specfem::assembly::assembly<dim3_tag>;
using MappingType =
    specfem::linear_system::Mapping<dim3_tag, elastic_tag, global_ordinal_type>;
using NumberingType = specfem::linear_system::DofNumbering<global_ordinal_type>;
using MediumTags =
    specfem::tags::Tags<dim3_tag, elastic_tag,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;

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

// Mesh with natural boundaries only: nothing absorbs.
class MappingNaturalBoundary3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    mapping_.reset();
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

  static MappingType &mapping() {
    if (!mapping_) {
      mapping_ = std::make_unique<MappingType>(assembly());
    }
    return *mapping_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<MappingType> mapping_;
};

AssemblyType *MappingNaturalBoundary3D::assembly_ = nullptr;
std::unique_ptr<MappingType> MappingNaturalBoundary3D::mapping_;

TEST_F(MappingNaturalBoundary3D, DofIdsMatchDofNumbering) {
  const auto &map = mapping();
  const NumberingType numbering(assembly(), MediumTags{});

  ASSERT_EQ(map.nglob(), numbering.nglob());
  ASSERT_EQ(map.ncomp(), numbering.ncomp());
  ASSERT_EQ(map.num_global_dofs(), numbering.num_global_dofs());

  // The layout_left mapping must reproduce DofNumbering::gid exactly, or a
  // solver vector stops aliasing field memory.
  for (int iglob = 0; iglob < map.nglob(); ++iglob) {
    for (int icomp = 0; icomp < map.ncomp(); ++icomp) {
      ASSERT_EQ(map(iglob, icomp), numbering.gid(iglob, icomp))
          << "dof id layout diverged at (iglob=" << iglob << ", icomp=" << icomp
          << ")";
    }
  }
}

// The five-argument operator must agree with the field's own get_iglob path,
// which is what the stiffness assembler uses to place element blocks.
TEST_F(MappingNaturalBoundary3D, ElementDofIdsMatchFieldIndexMapping) {
  const auto &map = mapping();
  const NumberingType numbering(assembly(), MediumTags{});
  const auto &field =
      assembly()
          .fields
          .get_simulation_field<specfem::simulation::field_type::forward>();

  const auto elements = map.elements();

  for (int i = 0; i < elements.size(); ++i) {
    const int ispec = elements(i);
    for (int iz = 0; iz < map.ngllz(); ++iz) {
      for (int iy = 0; iy < map.nglly(); ++iy) {
        for (int ix = 0; ix < map.ngllx(); ++ix) {
          const int iglob =
              field.get_iglob<false, elastic_tag>(ispec, iz, iy, ix);
          for (int icomp = 0; icomp < map.ncomp(); ++icomp) {
            ASSERT_EQ(map(ispec, iz, iy, ix, icomp),
                      numbering.gid(iglob, icomp))
                << "dof id diverged at (ispec=" << ispec << ", iz=" << iz
                << ", iy=" << iy << ", ix=" << ix << ", icomp=" << icomp << ")";
          }
        }
      }
    }
  }
}

// element_dofs is the shared ldof <-> gid ordering: a dense element block is
// scattered through it, so entry local_dof_index(ldof) must be that dof.
TEST_F(MappingNaturalBoundary3D, ElementDofsFollowLocalDofIndexOrder) {
  const auto &map = mapping();

  ASSERT_EQ(map.ngllz(), NGLL);
  ASSERT_EQ(map.nglly(), NGLL);
  ASSERT_EQ(map.ngllx(), NGLL);

  const auto elements = map.elements();
  ASSERT_GT(elements.size(), 0);

  for (int i = 0; i < elements.size(); ++i) {
    const int ispec = elements(i);
    const auto dofs = map.element_dofs(ispec);
    ASSERT_EQ(dofs.size(),
              static_cast<std::size_t>(map.ncomp()) * NGLL * NGLL * NGLL);

    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          for (int icomp = 0; icomp < map.ncomp(); ++icomp) {
            const int ldof = specfem::linear_system::local_dof_index<NGLL>(
                icomp, iz, iy, ix);
            ASSERT_EQ(dofs[ldof], map(ispec, iz, iy, ix, icomp))
                << "element " << ispec << " local dof " << ldof
                << " is not the dof local_dof_index numbers it";
          }
        }
      }
    }
  }
}

TEST_F(MappingNaturalBoundary3D, DampingMaskEmpty) {
  const auto &map = mapping();
  for (int iglob = 0; iglob < map.nglob(); ++iglob) {
    ASSERT_FALSE(map.is_damping_point(iglob))
        << "point " << iglob << " damps on a mesh with no absorbing boundary";
  }
}

// Mesh with Stacey absorbing boundaries: part of the domain damps.
class MappingStacey3D : public ::testing::Test {
protected:
  static void TearDownTestSuite() {
    mapping_.reset();
    delete assembly_;
    assembly_ = nullptr;
  }

  static AssemblyType &assembly() {
    if (assembly_ == nullptr) {
      assembly_ = build_assembly_3d("HomogeneousHalfSpaceStacey").release();
    }
    return *assembly_;
  }

  static MappingType &mapping() {
    if (!mapping_) {
      mapping_ = std::make_unique<MappingType>(assembly());
    }
    return *mapping_;
  }

  static AssemblyType *assembly_;
  static std::unique_ptr<MappingType> mapping_;
};

AssemblyType *MappingStacey3D::assembly_ = nullptr;
std::unique_ptr<MappingType> MappingStacey3D::mapping_;

TEST_F(MappingStacey3D, DampingMaskMatchesBoundaryTags) {
  const auto &map = mapping();
  const auto &field =
      assembly()
          .fields
          .get_simulation_field<specfem::simulation::field_type::forward>();
  const auto elements = map.elements();

  // Recompute the mask independently, straight from the boundary tags.
  std::vector<bool> expected(map.nglob(), false);
  for (int i = 0; i < elements.size(); ++i) {
    const int ispec = elements(i);
    for (int iz = 0; iz < map.ngllz(); ++iz) {
      for (int iy = 0; iy < map.nglly(); ++iy) {
        for (int ix = 0; ix < map.ngllx(); ++ix) {
          const auto tag =
              assembly().boundaries.get_boundary_tag_on_host(ispec, iz, iy, ix);
          if (tag == specfem::element::boundary_tag::stacey ||
              tag ==
                  specfem::element::boundary_tag::composite_stacey_dirichlet) {
            expected[field.get_iglob<false, elastic_tag>(ispec, iz, iy, ix)] =
                true;
          }
        }
      }
    }
  }

  int damping_points = 0;
  for (int iglob = 0; iglob < map.nglob(); ++iglob) {
    ASSERT_EQ(map.is_damping_point(iglob), expected[iglob])
        << "mask disagrees with the boundary tags at iglob=" << iglob;
    damping_points += expected[iglob] ? 1 : 0;
  }

  // A Stacey mesh must damp somewhere, and must not damp everywhere -- either
  // extreme would mean the tag query, not the mesh, is degenerate.
  EXPECT_GT(damping_points, 0);
  EXPECT_LT(damping_points, map.nglob());
}

} // namespace sparse_matrix_view_mapping_test

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
