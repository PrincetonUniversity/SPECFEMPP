#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"

#include "compute/interface.hpp"
// #include "kokkos_abstractions.h"
#include "IO/interface.hpp"
#include "parameter_parser/interface.hpp"

#include "chunk_edge.hpp"
#include <memory>

std::shared_ptr<specfem::compute::assembly>
init_assembly(const std::string &parameter_file) {
  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto database_file = setup.get_databases();
  const auto source_node = setup.get_sources();

  // Set up GLL quadrature points
  const auto quadratures = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  specfem::mesh::mesh mesh = specfem::IO::read_mesh(database_file, mpi);
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] = specfem::IO::read_sources(
      source_node, nsteps, setup.get_t0(), dt, setup.get_simulation_type());

  for (auto &source : sources) {
    if (mpi->main_proc())
      std::cout << source->print() << std::endl;
  }

  setup.update_t0(t0);

  // Instantiate the solver and timescheme
  auto it = setup.instantiate_timescheme();

  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::IO::read_receivers(stations_node, angle);

  std::cout << "  Receiver information\n";
  std::cout << "------------------------------" << std::endl;
  for (auto &receiver : receivers) {
    if (mpi->main_proc())
      std::cout << receiver->print() << std::endl;
  }

  const auto seismogram_types = setup.get_seismogram_types();
  const int max_sig_step = it->get_max_seismogram_step();

  return std::make_shared<specfem::compute::assembly>(
      mesh, quadratures, sources, receivers, seismogram_types, t0,
      setup.get_dt(), nsteps, max_sig_step, it->get_nstep_between_samples(),
      setup.get_simulation_type(), nullptr);
}

template <specfem::dimension::type DimensionType, typename ViewType,
          typename FieldValFunction>
void reset_fields(std::shared_ptr<specfem::compute::assembly> assembly,
                  ViewType &view, FieldValFunction &fieldval) {

  const auto element_type = assembly->element_types.medium_tags;

  auto &simfield = assembly->fields.forward;
  const int nspec = simfield.nspec;
  const int ngllx = simfield.ngllx;
  const int ngllz = simfield.ngllz;
  const auto index_mapping = simfield.h_index_mapping;

  for (int ispec = 0; ispec < nspec; ispec++) {
    switch (element_type(ispec)) {
    case specfem::element::medium_tag::acoustic: {
      constexpr auto medium = specfem::element::medium_tag::acoustic;
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          int iglob = index_mapping(ispec, iz, ix);
          int field_iglob = simfield.h_assembly_index_mapping(
              iglob, static_cast<int>(medium));
          for (int icomp = 0;
               icomp < specfem::element::attributes<DimensionType,
                                                    medium>::components();
               icomp++) {
            simfield.acoustic.h_field(field_iglob, icomp) =
                fieldval(iglob, icomp, 0);
            view(iglob, icomp, 0) = fieldval(iglob, icomp, 0);
            simfield.acoustic.h_field_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 1);
            view(iglob, icomp, 1) = fieldval(iglob, icomp, 1);
            simfield.acoustic.h_field_dot_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 2);
            view(iglob, icomp, 2) = fieldval(iglob, icomp, 2);
            simfield.acoustic.h_mass_inverse(field_iglob, icomp) =
                fieldval(iglob, icomp, 3);
            view(iglob, icomp, 3) = fieldval(iglob, icomp, 3);
          }
        }
      }
      break;
    }
    case specfem::element::medium_tag::elastic: {
      constexpr auto medium = specfem::element::medium_tag::elastic;
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          int iglob = index_mapping(ispec, iz, ix);
          int field_iglob = simfield.h_assembly_index_mapping(
              iglob, static_cast<int>(medium));
          for (int icomp = 0;
               icomp < specfem::element::attributes<DimensionType,
                                                    medium>::components();
               icomp++) {
            simfield.elastic.h_field(field_iglob, icomp) =
                fieldval(iglob, icomp, 0);
            view(iglob, icomp, 0) = fieldval(iglob, icomp, 0);
            simfield.elastic.h_field_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 1);
            view(iglob, icomp, 1) = fieldval(iglob, icomp, 1);
            simfield.elastic.h_field_dot_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 2);
            view(iglob, icomp, 2) = fieldval(iglob, icomp, 2);
            simfield.elastic.h_mass_inverse(field_iglob, icomp) =
                fieldval(iglob, icomp, 3);
            view(iglob, icomp, 3) = fieldval(iglob, icomp, 3);
          }
        }
      }
      break;
    }
    }
  }
  simfield.copy_to_device();
}

TEST(accessor_tests, ACCESSOR_TESTS) {
  constexpr auto DimensionType = specfem::dimension::type::dim2;
  constexpr int NGLL = 5;
  constexpr int max_components = std::max(
      specfem::element::attributes<
          DimensionType, specfem::element::medium_tag::elastic>::components(),
      specfem::element::attributes<DimensionType,
                                   specfem::element::medium_tag::acoustic>::
          components()); // number of components to reserve in fieldval
  constexpr bool USE_SIMD = false;
  using SIMD = specfem::datatype::simd<type_real, USE_SIMD>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, SIMD, Kokkos::DefaultExecutionSpace>;
  constexpr int CHUNK_SIZE = ParallelConfig::chunk_size;

  std::shared_ptr<specfem::compute::assembly> assembly =
      init_assembly("../../../tests/unit-tests/accessor/databases/fluid_solid/"
                    "specfem_config.yaml");
  assert(assembly->fields.forward.ngllx == NGLL &&
         assembly->fields.forward.ngllz == NGLL);

  const auto fieldval = [](int iglob, int icomp, int ideriv) {
    return (type_real)(iglob + icomp * (1.0 / 7.0) + ideriv * (1.0 / 5.0));
  };
  Kokkos::View<type_real ***,
               typename ParallelConfig::execution_space::memory_space>
      fieldval_ref("fieldval_ref", assembly->fields.forward.nglob,
                   max_components, 4);
  auto h_fieldval_ref = Kokkos::create_mirror_view(fieldval_ref);

  reset_fields<DimensionType>(assembly, h_fieldval_ref, fieldval);

  Kokkos::deep_copy(fieldval_ref, h_fieldval_ref);

  //============[ check pointwise accessors ]============
  // TODO actually fill out, or get rid of?
  // TODO should we try each combination of bools?
  using PointAcoustic =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, true, false,
                            false, false, USE_SIMD>;
  using PointElastic =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::elastic, true, false,
                            false, false, USE_SIMD>;

  //=====================================================
  verify_chunk_edges<CHUNK_SIZE, NGLL, DimensionType,
                     specfem::element::medium_tag::acoustic, USE_SIMD>(
      assembly, fieldval_ref);
  verify_chunk_edges<CHUNK_SIZE, NGLL, DimensionType,
                     specfem::element::medium_tag::elastic, USE_SIMD>(
      assembly, fieldval_ref);

  // /**
  //  *  This test checks if compute_lagrange_interpolants and
  //  * compute_lagrange_derivatives_GLL give the same value at GLL points
  //  *
  //  */
  // int ngll = 5;
  // type_real degpoly = ngll - 1;
  // type_real tol = 1e-6;

  // auto [h_z1, h_w1] =
  //     specfem::quadrature::gll::gll_library::zwgljd(ngll, 0.0, 0.0);
  // auto h_hprime_xx =
  //     specfem::quadrature::gll::Lagrange::compute_lagrange_derivatives_GLL(
  //         h_z1, ngll);

  // for (int i = 0; i < ngll; i++) {
  //   auto [h_h1, h_h1_prime] =
  //       specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
  //           h_z1(i), ngll, h_z1);
  //   for (int j = 0; j < ngll; j++) {
  //     EXPECT_NEAR(h_hprime_xx(j, i), h_h1_prime(j), tol);
  //     if (i == j) {
  //       EXPECT_NEAR(h_h1(j), 1.0, tol);
  //       if (i == 0) {
  //         type_real result = -1.0 * static_cast<type_real>(degpoly) *
  //                            (static_cast<type_real>(degpoly) + 1.0) * 0.25;
  //         EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
  //       } else if (i == degpoly) {
  //         type_real result = 1.0 * static_cast<type_real>(degpoly) *
  //                            (static_cast<type_real>(degpoly) + 1.0) * 0.25;
  //         EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
  //       } else {
  //         type_real result = 0.0;
  //         EXPECT_NEAR(h_h1_prime(j), result, tol) << i;
  //       }
  //     } else {
  //       EXPECT_NEAR(h_h1(j), 0.0, tol);
  //     }
  //   }
  // }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
