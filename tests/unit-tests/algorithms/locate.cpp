#include "../SPECFEM_Environment.hpp"
#include "algorithms/locate_point.hpp"
#include "io/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "specfem/assembly.hpp"
#include <Kokkos_Core.hpp>

TEST(ALGORITHMS, locate_point) {

  std::string database_file = "algorithms/serial/database.bin";

  // Read Mesh database
  specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(
      database_file, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te);

  // Quadratures
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  // Assemble
  specfem::assembly::mesh assembly(mesh.tags, mesh.control_nodes, quadratures);

  specfem::kokkos::HostView1d<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> >
      coordinates_ref("coordinates_ref", 5);
  specfem::kokkos::HostView1d<
      specfem::point::local_coordinates<specfem::dimension::type::dim2> >
      lcoord_ref("lcoord_ref", 5);
  specfem::kokkos::HostView1d<
      specfem::point::local_coordinates<specfem::dimension::type::dim2> >
      lcoord("lcoord", 5);
  specfem::kokkos::HostView1d<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> >
      gcoord("gcoord", 5);

  coordinates_ref(0) = { 606.313, 957.341 };
  coordinates_ref(1) = { 1835.050, 146.444 };
  coordinates_ref(2) = { 1539.578, 242.743 };
  coordinates_ref(3) = { 3497.449, 2865.483 };
  coordinates_ref(4) = { 3018.139, 530.283 };

  lcoord_ref(0) = { 1452, -0.7475e+00, 0.2905e+00 };
  lcoord_ref(1) = { 196, 0.4020E+00, 0.5832E-01 };
  lcoord_ref(2) = { 350, 0.5831E+00, -0.2247E+00 };
  lcoord_ref(3) = { 4549, 0.8980E+00, 0.4248E+00 };
  lcoord_ref(4) = { 700, -0.2744E+00, 0.3945E+00 };

  for (int i = 0; i < 5; ++i) {
    lcoord_ref(i).ispec = assembly.mapping.mesh_to_compute(lcoord_ref(i).ispec);
  }

  // Test Serial implementations

  for (int i = 0; i < 5; ++i) {
    lcoord(i) = specfem::algorithms::locate_point(coordinates_ref(i), assembly);
    gcoord(i) = specfem::algorithms::locate_point(lcoord_ref(i), assembly);

    EXPECT_EQ(lcoord(i).ispec, lcoord_ref(i).ispec);
    EXPECT_NEAR(lcoord(i).xi, lcoord_ref(i).xi, 1e-4);
    EXPECT_NEAR(lcoord(i).gamma, lcoord_ref(i).gamma, 1e-4);

    EXPECT_NEAR(gcoord(i).x, coordinates_ref(i).x, 1e-2);
    EXPECT_NEAR(gcoord(i).z, coordinates_ref(i).z, 1e-2);
  }

  // Test Team Parallel implementations

  const int ngnod = assembly.control_nodes.ngnod;

  int scratch_size =
      specfem::kokkos::HostScratchView2d<type_real>::shmem_size(ndim, ngnod);

  Kokkos::parallel_for(
      specfem::kokkos::HostTeam(5, Kokkos::AUTO, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      [=](const specfem::kokkos::HostTeam::member_type &team_member) {
        const int i = team_member.league_rank();

        gcoord(i) = specfem::algorithms::locate_point(team_member,
                                                      lcoord_ref(i), assembly);

        team_member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          EXPECT_EQ(lcoord(i).ispec, lcoord_ref(i).ispec);
          EXPECT_NEAR(lcoord(i).xi, lcoord_ref(i).xi, 1e-4);
          EXPECT_NEAR(lcoord(i).gamma, lcoord_ref(i).gamma, 1e-4);

          EXPECT_NEAR(gcoord(i).x, coordinates_ref(i).x, 1e-2);
          EXPECT_NEAR(gcoord(i).z, coordinates_ref(i).z, 1e-2);
        });
      });

  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
