#include "specfem/assembly/assembly.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/enums.hpp"
#include "specfem/globe/model_evaluator.hpp"
#include "specfem/globe/region_codes.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/simulation.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace globe_attenuation_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Medium = specfem::element::medium_tag;
using Property = specfem::element::property_tag;
using Attenuation = specfem::element::attenuation_tag;
using Assembly = specfem::assembly::assembly<dimension>;

const std::string database_path =
    "data/dim3_globe/GlobalSmallMeshAttenuation/DATABASES_MPI/"
    "proc000000_specfempp_database.bin";

/**
 * @brief Runtime attenuation setup whose band is the database's band, read
 *        back from the database itself.
 */
specfem::attenuation::Setup matching_attenuation_setup() {
  using specfem::units::unit_symbols::Hz;
  const auto config =
      specfem::io::read_globe_mesh(database_path, specfem::attenuation::Setup{})
          .globe.model_config;
  return { true, 1.0 * Hz,
           specfem::utilities::Band<specfem::units::Hertz>(
               (1.0 / config.max_attenuation_period) * Hz,
               (1.0 / config.min_attenuation_period) * Hz) };
}

/** @brief Full globe assembly, built the way a simulation builds it. */
std::unique_ptr<Assembly>
build_assembly(const specfem::attenuation::Setup &setup) {
  const auto mesh = specfem::io::read_globe_mesh(database_path, setup);
  specfem::quadrature::gll::gll gll{};
  const specfem::quadrature::quadratures quadratures(gll);
  std::vector<std::shared_ptr<specfem::sources::source<dimension>>> sources;
  const std::vector<std::shared_ptr<specfem::receivers::receiver<dimension>>>
      receivers;
  const std::vector<specfem::enums::wavefield> stypes;
  return std::make_unique<Assembly>(
      mesh, quadratures, sources, receivers, stypes, 0.0, 0.19, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr);
}

} // namespace globe_attenuation_test_impl

/**
 * @brief Builds the attenuating and the elastic assembly of the same globe
 *        mesh once and shares them across the tests.
 */
class GlobeAttenuation : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    namespace test = globe_attenuation_test_impl;
    attenuating = test::build_assembly(test::matching_attenuation_setup());
    elastic = test::build_assembly(specfem::attenuation::Setup{});
  }

  static void TearDownTestSuite() {
    attenuating.reset();
    elastic.reset();
  }

  static std::unique_ptr<globe_attenuation_test_impl::Assembly> attenuating;
  static std::unique_ptr<globe_attenuation_test_impl::Assembly> elastic;
};

std::unique_ptr<globe_attenuation_test_impl::Assembly>
    GlobeAttenuation::attenuating;
std::unique_ptr<globe_attenuation_test_impl::Assembly>
    GlobeAttenuation::elastic;

TEST_F(GlobeAttenuation, AttenuatesExactlyTheElasticElements) {
  namespace test = globe_attenuation_test_impl;
  const auto &element_types = attenuating->element_types;
  const auto &container =
      attenuating->attenuation
          .get_container<test::Medium::elastic, test::Property::isotropic>();

  const int nelastic = element_types.get_number_of_elements(
      test::Medium::elastic, test::Property::isotropic);
  ASSERT_GT(nelastic, 0);
  EXPECT_EQ(container.element_range.size(), nelastic);

  // The outer core is acoustic: its elements carry no attenuation container.
  ASSERT_GT(element_types.get_number_of_elements(test::Medium::acoustic), 0);
  EXPECT_EQ(element_types.get_number_of_elements(
                test::Medium::acoustic, test::Property::isotropic,
                test::Attenuation::constant_isotropic),
            0);
  for (int i = 0; i < container.element_range.size(); ++i) {
    EXPECT_EQ(element_types.get_medium_tag(container.element_range(i)),
              test::Medium::elastic);
  }
}

TEST_F(GlobeAttenuation, StoresOracleQAtEveryGllPoint) {
  namespace test = globe_attenuation_test_impl;
  const auto &element_types = attenuating->element_types;
  const auto &mesh = attenuating->mesh;
  const auto &container =
      attenuating->attenuation
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  const int ngllz = mesh.element_grid.ngllz;
  const int nglly = mesh.element_grid.nglly;
  const int ngllx = mesh.element_grid.ngllx;

  const auto globe_mesh = specfem::io::read_globe_mesh(
      test::database_path, specfem::attenuation::Setup{});
  const specfem::globe::ModelEvaluator evaluator(globe_mesh.globe.model_config);

  int npoints_checked = 0;
  for (int i = 0; i < container.element_range.size(); i += 10) {
    const int ispec = container.element_range(i);
    std::vector<double> xyz;
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          for (int icomp = 0; icomp < 3; ++icomp) {
            xyz.push_back(mesh.h_reference_coord(ispec, iz, iy, ix, icomp));
          }
        }
      }
    }
    const auto values = evaluator.evaluate_element(
        specfem::globe::to_region_code(element_types.get_region_tag(ispec)),
        element_types.idoubling(ispec), element_types.rmin(ispec),
        element_types.rmax(ispec), element_types.elem_in_crust(ispec),
        element_types.elem_in_mantle(ispec), xyz);

    std::size_t ipoint = 0;
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix, ++ipoint) {
          const type_real expected_Qmu = values.qmu[ipoint];
          const type_real expected_Qkappa = values.qkappa[ipoint];
          EXPECT_NEAR(container.h_Qmu(i, iz, iy, ix), expected_Qmu,
                      1.0e-6 * expected_Qmu);
          EXPECT_NEAR(container.h_Qkappa(i, iz, iy, ix), expected_Qkappa,
                      1.0e-6 * expected_Qkappa);
          ++npoints_checked;
        }
      }
    }
  }
  EXPECT_GE(npoints_checked, 50);
}

TEST_F(GlobeAttenuation, ShiftsOnlyTheShearModulus) {
  namespace test = globe_attenuation_test_impl;
  const auto &container =
      attenuating->attenuation
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  const auto &properties =
      attenuating->properties
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  const auto &elastic_properties =
      elastic->properties
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  ASSERT_EQ(properties.element_range.begin_index(),
            elastic_properties.element_range.begin_index());
  const int offset = container.element_range.begin_index() -
                     properties.element_range.begin_index();
  const int ngllz = container.h_Qmu.extent(1);
  const int nglly = container.h_Qmu.extent(2);
  const int ngllx = container.h_Qmu.extent(3);

  for (int i = 0; i < container.element_range.size(); ++i) {
    const int ip = offset + i;
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          // Kappa is bit-identical to the elastic run: SPECFEM3D_GLOBE has no
          // bulk attenuation.
          ASSERT_NEAR(properties.h_kappa(ip, iz, iy, ix),
                      elastic_properties.h_kappa(ip, iz, iy, ix), 0.0);
          ASSERT_NEAR(container.h_kappa_scale(i, iz, iy, ix), 1.0, 0.0);
          for (int j = 0; j < specfem::constants::N_SLS; ++j) {
            ASSERT_NEAR(container.h_kappa_relaxation_rate(i, iz, iy, ix, j),
                        0.0, 0.0);
          }

          // Mu is the physical modulus scaled to unrelaxed by the stored
          // per-GLL factor, which lies inside the range both SPECFEM3D codes
          // accept.
          const type_real scale = container.h_mu_scale(i, iz, iy, ix);
          const type_real physical_mu = elastic_properties.h_mu(ip, iz, iy, ix);
          ASSERT_NEAR(properties.h_mu(ip, iz, iy, ix), physical_mu * scale,
                      1.0e-6 * physical_mu);
          ASSERT_GE(scale, 0.5);
          ASSERT_LE(scale, 1.5);
        }
      }
    }
  }
}

TEST_F(GlobeAttenuation, AllocatesNothingWhenAttenuationIsOff) {
  namespace test = globe_attenuation_test_impl;
  const auto &container =
      elastic->attenuation
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  EXPECT_EQ(container.element_range.size(), 0);
  EXPECT_EQ(container.h_Qmu.size(), 0);
  EXPECT_EQ(container.h_Qkappa.size(), 0);
  EXPECT_EQ(container.mu_relaxation_rate.size(), 0);
  EXPECT_EQ(elastic->element_types.get_number_of_elements(
                test::Medium::elastic, test::Property::isotropic,
                test::Attenuation::constant_isotropic),
            0);
}

TEST_F(GlobeAttenuation, RejectsASecondRecompute) {
  namespace test = globe_attenuation_test_impl;
  const auto &attenuation = attenuating->attenuation;
  const auto &container =
      attenuation
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  const auto &properties =
      attenuating->properties
          .get_container<test::Medium::elastic, test::Property::isotropic>();
  EXPECT_THROW(container.recompute(
                   properties, attenuation.fc, attenuation.f0, attenuation.band,
                   attenuation.tau_sigma, attenuating->mesh.h_spherical_coord,
                   specfem::assembly::impl::AssemblyAttenuationContext{
                       attenuating->element_types }),
               std::runtime_error);
}
