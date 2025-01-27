#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"

#include "build_demo_assembly.hpp"

#include "chunk_edge.hpp"

template <specfem::dimension::type DimensionType, typename FieldValFunction>
void reset_fields(std::shared_ptr<specfem::compute::assembly> assembly,
                  FieldValFunction &fieldval) {

  const auto element_type = assembly->properties.h_element_types;

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
            simfield.acoustic.h_field_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 1);
            simfield.acoustic.h_field_dot_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 2);
            simfield.acoustic.h_mass_inverse(field_iglob, icomp) =
                fieldval(iglob, icomp, 3);
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
            simfield.elastic.h_field_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 1);
            simfield.elastic.h_field_dot_dot(field_iglob, icomp) =
                fieldval(iglob, icomp, 2);
            simfield.elastic.h_mass_inverse(field_iglob, icomp) =
                fieldval(iglob, icomp, 3);
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
  constexpr bool USE_SIMD = false;
  using SIMD = specfem::datatype::simd<type_real, USE_SIMD>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionType, SIMD, Kokkos::DefaultExecutionSpace>;
  constexpr int CHUNK_SIZE = ParallelConfig::chunk_size;

  // initialize assembly; TODO: decide if we use demo_assembly, or load from
  // meshfem

  _util::demo_assembly::simulation_params params =
      _util::demo_assembly::simulation_params().use_demo_mesh(0b1000);
  std::shared_ptr<specfem::compute::assembly> assembly = params.get_assembly();
  assert(assembly->fields.forward.ngllx == NGLL &&
         assembly->fields.forward.ngllz == NGLL);

  const auto fieldval = [](int iglob, int icomp, int ideriv) {
    return (type_real)(iglob + icomp * (1.0 / 7.0) + ideriv * (1.0 / 5.0));
  };
  reset_fields<DimensionType>(assembly, fieldval);

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
  verify_chunk_edges<CHUNK_SIZE, NGLL, DimensionType, USE_SIMD, Kokkos::Serial>(
      assembly, fieldval);

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
