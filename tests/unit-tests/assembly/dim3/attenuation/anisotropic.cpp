#include "globe_model_evaluator.h"
#include "specfem/assembly/assembly/dim3/globe_properties.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium/dim3/unrelaxed_tensor.hpp"
#include "specfem/globe/model_evaluator.hpp"
#include "specfem/globe/region_codes.hpp"
#include "specfem/io.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/read_materials.hpp"
#include "specfem/medium/dim3/elastic/isotropic/attenuation.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

namespace specfem::assembly::anisotropic_attenuation_test_impl {

using Cartesian = specfem::coordinate_systems::CartesianFrame;
using Geocentric = specfem::coordinate_systems::GeocentricFrame;
using Tensor = specfem::medium_physics::ElasticityTensor<Cartesian, double>;
using Radial = specfem::medium_physics::ElasticityTensor<Geocentric, double>;
using Medium = specfem::element::medium_tag;
using Property = specfem::element::property_tag;
constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Assembly = specfem::assembly::assembly<dimension>;
using Attenuation = specfem::assembly::Attenuation<dimension>;
static_assert(
    Attenuation::has_attenuation<Medium::elastic, Property::anisotropic>());
static_assert(
    !Attenuation::has_attenuation<Medium::acoustic, Property::isotropic>());

struct ZeroGradientPack {
  template <typename Tags>
  KOKKOS_INLINE_FUNCTION specfem::point::field_derivatives<Tags>
  get_du() const {
    specfem::point::field_derivatives<Tags> derivatives;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        derivatives.du(i, j) = 0;
    return derivatives;
  }
  template <typename Tags> KOKKOS_INLINE_FUNCTION auto get_dv() const {
    return get_du<Tags>();
  }
};

Tensor global_tensor(const Radial &radial, double theta, double phi) {
  return specfem::medium_physics::rotate(
      radial,
      specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(theta,
                                                                          phi));
}

void expect_tensor_near(const Tensor &actual, const Tensor &expected,
                        double tolerance = 1.e-10) {
  double scale = 1;
  for (int k = 0; k < 21; ++k)
    scale = std::max(scale, std::abs(expected[k]));
  for (int k = 0; k < 21; ++k) {
    EXPECT_NEAR(actual[k], expected[k], tolerance * scale) << "component " << k;
  }
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(actual(i, j), actual(j, i), 0.0);
}

} // namespace specfem::assembly::anisotropic_attenuation_test_impl

TEST(AnisotropicAttenuation, IdentityAndOrientationIncludingPoles) {
  namespace test = specfem::assembly::anisotropic_attenuation_test_impl;
  const test::Radial radial(3300., 8000., 8200., 4400., 4600., 0.9);
  // A radial shift followed by a rotation is an independent TISO expectation.
  const double increment = 0.04;
  const double a = 3300. * 8200. * 8200.;
  const double l = 3300. * 4400. * 4400.;
  const double n = 3300. * 4600. * 4600.;
  auto expected_radial = radial;
  expected_radial(0, 0) += 4. / 3. * n * increment;
  expected_radial(1, 1) = expected_radial(0, 0);
  expected_radial(0, 1) = expected_radial(0, 0) - 2 * n * (1 + increment);
  expected_radial(0, 2) = expected_radial(1, 2) =
      0.9 * (a + 4. / 3. * n * increment - 2 * l * (1 + increment));
  expected_radial(2, 2) += 4. / 3. * l * increment;
  expected_radial(3, 3) = expected_radial(4, 4) = l * (1 + increment);
  expected_radial(5, 5) = n * (1 + increment);
  for (double theta : { 0., 0.4, 1.7, std::numbers::pi }) {
    for (double phi : { 0., 0.7, 4.2 }) {
      const auto global = test::global_tensor(radial, theta, phi);
      const auto identity = specfem::assembly::impl::shift_unrelaxed_tensor(
          global, theta, phi, 0.);
      EXPECT_EQ(std::memcmp(&global, &identity, sizeof(global)), 0);
      const auto shifted = specfem::assembly::impl::shift_unrelaxed_tensor(
          global, theta, phi, increment);
      test::expect_tensor_near(
          shifted, test::global_tensor(expected_radial, theta, phi));
      double input[21], output[21];
      for (int k = 0; k < 21; ++k)
        input[k] = global[k];
      globe_evaluator_shift_cij(theta, phi, increment, input, output);
      test::expect_tensor_near(shifted, test::Tensor(output));
    }
  }
}

TEST(AnisotropicAttenuation, GlobeGllModelShiftAndPhysicalModelRecovery) {
  namespace test = specfem::assembly::anisotropic_attenuation_test_impl;
  using specfem::units::unit_symbols::Hz;
  const std::string path = "data/dim3_globe/GlobalSmallMeshAttenuation/"
                           "DATABASES_MPI/proc000000_specfempp_database.bin";
  auto input =
      specfem::io::read_globe_mesh(path, specfem::attenuation::Setup{});
  const auto &config = input.globe.model_config;
  input = specfem::io::read_globe_mesh(
      path, { true, 1.0 * Hz,
              specfem::utilities::Band<specfem::units::Hertz>(
                  (1. / config.max_attenuation_period) * Hz,
                  (1. / config.min_attenuation_period) * Hz) });
  input.globe.model_config.model_name = "1D_transversely_isotropic_prem";
  input.globe.model_verification.codes.clear();
  input.globe.model_verification.flags.clear();
  std::vector<test::Medium> media;
  std::vector<test::Property> properties;
  for (int i = 0; i < input.nspec; ++i) {
    auto &tags = input.tags.tags_container(i);
    if (input.globe.element_context[i].region ==
        specfem::element::region_tag::crust_mantle) {
      tags.property_tag = test::Property::anisotropic;
    }
    media.push_back(tags.medium_tag);
    properties.push_back(tags.property_tag);
  }
  input.materials =
      specfem::io::mesh::impl::fortran::dim3_globe::make_materials(
          media, properties, true);
  test::Assembly assembly;
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  assembly.mesh = { input.nspec,
                    input.control_nodes.ngnod,
                    input.element_grid.ngllz,
                    input.element_grid.nglly,
                    input.element_grid.ngllx,
                    input.tags,
                    input.adjacency_graph,
                    input.control_nodes,
                    quadrature,
                    input.globe.reference_coordinates };
  assembly.mesh.compute_spherical_coordinates();
  assembly.element_types = { input.nspec, assembly.mesh.element_grid,
                             assembly.mesh, input.tags,
                             input.globe.element_context };
  assembly.properties = { assembly.element_types, assembly.mesh,
                          input.materials, true };
  assembly.attenuation = { input.attenuation, 0.19, assembly.mesh,
                           assembly.element_types, input.materials };
  specfem::assembly::dim3_impl::read_globe_properties(input, assembly);
  const auto &att =
      assembly.attenuation
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  const auto &props =
      assembly.properties
          .get_container<test::Medium::elastic, test::Property::anisotropic>();
  ASSERT_GT(att.element_range.size(), 0);
  EXPECT_THROW(
      att.recompute(props, assembly.attenuation.fc, assembly.attenuation.f0,
                    assembly.attenuation.band, assembly.attenuation.tau_sigma,
                    assembly.mesh.h_spherical_coord,
                    specfem::assembly::impl::AssemblyAttenuationContext{
                        assembly.element_types }),
      std::runtime_error);
  const std::array views = { props.h_c11, props.h_c12, props.h_c13, props.h_c14,
                             props.h_c15, props.h_c16, props.h_c22, props.h_c23,
                             props.h_c24, props.h_c25, props.h_c26, props.h_c33,
                             props.h_c34, props.h_c35, props.h_c36, props.h_c44,
                             props.h_c45, props.h_c46, props.h_c55, props.h_c56,
                             props.h_c66 };
  const std::array<std::string, 21> names = {
    "c11", "c12", "c13", "c14", "c15", "c16", "c22", "c23", "c24", "c25", "c26",
    "c33", "c34", "c35", "c36", "c44", "c45", "c46", "c55", "c56", "c66"
  };
  std::array<decltype(att.h_Qmu), 21> physical;
  for (int k = 0; k < 21; ++k) {
    physical[k] = decltype(att.h_Qmu)("physical", att.h_Qmu.extent(0),
                                      att.h_Qmu.extent(1), att.h_Qmu.extent(2),
                                      att.h_Qmu.extent(3));
    const int offset =
        att.element_range.begin_index() - props.element_range.begin_index();
    for (int i = 0; i < static_cast<int>(att.h_Qmu.extent(0)); ++i)
      for (int z = 0; z < static_cast<int>(att.h_Qmu.extent(1)); ++z)
        for (int y = 0; y < static_cast<int>(att.h_Qmu.extent(2)); ++y)
          for (int x = 0; x < static_cast<int>(att.h_Qmu.extent(3)); ++x)
            physical[k](i, z, y, x) = views[k](offset + i, z, y, x);
    att.to_physical(physical[k], names[k]);
  }
  const specfem::globe::ModelEvaluator oracle(input.globe.model_config);
  int checked = 0;
  for (int i = 0; i < att.element_range.size(); i += 7) {
    const int ispec = att.element_range(i);
    const int offset = ispec - props.element_range.begin_index();
    std::vector<double> xyz;
    for (int z = 0; z < assembly.mesh.element_grid.ngllz; ++z)
      for (int y = 0; y < assembly.mesh.element_grid.nglly; ++y)
        for (int x = 0; x < assembly.mesh.element_grid.ngllx; ++x)
          for (int k = 0; k < 3; ++k)
            xyz.push_back(assembly.mesh.h_reference_coord(ispec, z, y, x, k));
    const auto &types = assembly.element_types;
    const auto model = oracle.evaluate_element(
        specfem::globe::to_region_code(types.get_region_tag(ispec)),
        types.idoubling(ispec), types.rmin(ispec), types.rmax(ispec),
        types.elem_in_crust(ispec), types.elem_in_mantle(ispec), xyz);
    for (int z = 0; z < assembly.mesh.element_grid.ngllz; ++z) {
      const int y = 0, x = 0;
      double relaxed[21], expected[21], actual[21];
      for (int k = 0; k < 21; ++k) {
        relaxed[k] = physical[k](i, z, y, x);
        actual[k] = views[k](offset, z, y, x);
      }
      const double theta = assembly.mesh.h_spherical_coord(ispec, z, y, x, 1);
      const double phi = assembly.mesh.h_spherical_coord(ispec, z, y, x, 2);
      const int point = z * assembly.mesh.element_grid.nglly *
                        assembly.mesh.element_grid.ngllx;
      double original[21];
      globe_evaluator_rotate_tiso_to_cij(
          theta, phi, model.rho[point], model.vpv[point], model.vph[point],
          model.vsv[point], model.vsh[point], model.eta[point], original);
      const double tolerance =
          std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;
      test::expect_tensor_near(test::Tensor(relaxed), test::Tensor(original),
                               tolerance);
      EXPECT_NEAR(att.h_Qmu(i, z, y, x), model.qmu[point],
                  tolerance * model.qmu[point]);
      constexpr int n = specfem::constants::N_SLS;
      double expected_muv, expected_rates[n];
      // No SPECFEM++ attenuation coefficient enters this reference pipeline.
      globe_evaluator_attenuate_cij(
          theta, phi, assembly.attenuation.band.min.raw(),
          assembly.attenuation.band.max.raw(), model.qmu[point],
          assembly.attenuation.fc.raw(), assembly.attenuation.f0.raw(),
          original, expected, &expected_muv, expected_rates);
      test::expect_tensor_near(test::Tensor(actual), test::Tensor(expected),
                               tolerance);
      const double rate_tolerance =
          std::is_same_v<type_real, double> ? 1.e-10 : 2.e-3;
      for (int j = 0; j < n; ++j) {
        EXPECT_NEAR(att.h_mu_relaxation_rate(i, z, y, x, j), expected_rates[j],
                    rate_tolerance * std::abs(expected_rates[j]));
      }
      ++checked;
    }
  }
  EXPECT_GE(checked, 50);

  // Exercise the actual stress/memory update: the point type is shared with
  // isotropic media, but must load/store this anisotropic container's state.
  Kokkos::deep_copy(att.memory_variable_Rxx, 1.0);
  const auto attenuation = assembly.attenuation;
  const int ispec = att.element_range(0);
  Kokkos::View<type_real[2]> result("attenuating_stress");
  Kokkos::parallel_for(
      "anisotropic_attenuation_update", 1, KOKKOS_LAMBDA(int) {
        using Tags = specfem::tags::Tags<
            test::dimension, test::Medium::elastic, test::Property::anisotropic,
            specfem::element::attenuation_tag::constant_isotropic, false>;
        const specfem::point::index<test::dimension, false> index(ispec, 0, 0,
                                                                  0);
        specfem::point::stress<Tags> stress;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            stress.T(i, j) = 0;
        specfem::medium_physics::impl_compute_attenuation<Tags>(
            index, stress, test::ZeroGradientPack{}, attenuation);
        result(0) = stress.T(0, 0);
        result(1) = stress.T(2, 2);
      });
  const auto h_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  EXPECT_NEAR(h_result(0), -specfem::constants::N_SLS, 0.);
  EXPECT_NEAR(h_result(1), specfem::constants::N_SLS, 0.);
  specfem::datatype::deep_copy(att.h_memory_variable_Rxx,
                               att.memory_variable_Rxx);
  EXPECT_NEAR(att.h_memory_variable_Rxx(0, 0, 0, 0, 0),
              assembly.attenuation.alpha_rk(0), 1.e-6);
}

TEST(AnisotropicAttenuation, FullAnisotropyMatchesGlobe) {
  namespace test = specfem::assembly::anisotropic_attenuation_test_impl;
  test::Radial radial(3300., 8000., 8200., 4400., 4600., 0.9);
  // Break transverse isotropy, including normal/shear and cross-shear terms.
  radial(0, 0) *= 1.03;
  radial(4, 4) *= 1.02;
  radial(0, 3) = 1.1e9;
  radial(1, 4) = -2.e9;
  radial(2, 5) = 1.3e9;
  radial(3, 4) = 0.4e9;
  radial(4, 5) = -0.2e9;
  for (double theta : { 0., 0.6, 2.2, std::numbers::pi }) {
    for (double phi : { 0., 1.4, 5.1 }) {
      const auto global = test::global_tensor(radial, theta, phi);
      double input[21], expected[21];
      for (int k = 0; k < 21; ++k)
        input[k] = global[k];
      globe_evaluator_shift_cij(theta, phi, -0.03, input, expected);
      const auto shifted = specfem::assembly::impl::shift_unrelaxed_tensor(
          global, theta, phi, -0.03);
      test::expect_tensor_near(shifted, test::Tensor(expected));
      const auto recovered = specfem::medium_physics::rotate(
          shifted, specfem::coordinate_systems::rotation_matrix<
                       test::Geocentric, test::Cartesian>(theta, phi));
      EXPECT_NEAR(recovered(0, 3), radial(0, 3), 1.e-3);
      EXPECT_NEAR(recovered(3, 4), radial(3, 4), 1.e-3);
    }
  }
}
