#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "specfem/utilities/logarithmic_center.hpp"

#include "globe_model_evaluator.h"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace attenuation_recompute_test_impl {

using Medium = specfem::assembly::impl::attenuation_medium<
    specfem::element::dimension_tag::dim3,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic,
    specfem::element::attenuation_tag::constant_isotropic>;
using HostView =
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;

using SphericalView = Kokkos::View<double *****, Kokkos::HostSpace>;

/** @brief Globe crust/mantle metadata for every element. */
const auto mantle = [](int) {
  return specfem::assembly::impl::AttenuationElementContext{};
};

/** @brief Cartesian (non-globe) metadata for every element. */
const auto cartesian = [](int) {
  specfem::assembly::impl::AttenuationElementContext context;
  context.cartesian = true;
  return context;
};

constexpr int nspec = 2;
constexpr int ngll = 3;
constexpr int N_SLS = specfem::constants::N_SLS;

/** @brief Minimal property container accepted by recompute(). */
struct Props {
  HostView h_kappa;
  HostView h_mu;
  specfem::datatype::ElementIndexRange element_range;
};

/** @brief Attenuation inputs shared by every test: a global-scale band. */
struct Inputs {
  specfem::utilities::Band<specfem::units::Hertz> band;
  specfem::units::Hertz fc;
  specfem::units::Hertz f0;
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma;

  Inputs() {
    using specfem::units::unit_symbols::Hz;
    band = specfem::utilities::Band<specfem::units::Hertz>(
        (1.0 / 15632.376) * Hz, (1.0 / 277.987305) * Hz);
    fc =
        specfem::utilities::logarithmic_center(band.min.raw(), band.max.raw()) *
        Hz;
    f0 = 1.0 * Hz;
    tau_sigma = specfem::attenuation::compute_tau_sigma<N_SLS>(band);
  }
};

/** @brief Flat GLL-point index used to vary values within an element. */
int point_index(const int iz, const int iy, const int ix) {
  return (iz * ngll + iy) * ngll + ix;
}

/** @brief Physical kappa at a GLL point. */
type_real physical_kappa(const int i, const int iz, const int iy,
                         const int ix) {
  return 1.0e11 * (1.0 + 0.01 * point_index(iz, iy, ix) + 0.1 * i);
}

/** @brief Physical mu at a GLL point. */
type_real physical_mu(const int i, const int iz, const int iy, const int ix) {
  return 5.0e10 * (1.0 + 0.01 * point_index(iz, iy, ix) + 0.1 * i);
}

/**
 * @brief An attenuation container whose Q varies at every GLL point, laid
 *        out the way the constructor allocates it.
 */
Medium make_medium(const bool bulk_attenuation) {
  Medium medium;
  medium.element_range = specfem::datatype::ElementIndexRange(0, nspec);
  medium.bulk_attenuation = bulk_attenuation;
  medium.h_recomputed = Kokkos::View<bool, Kokkos::HostSpace>("recomputed");
  medium.h_Qkappa = Medium::q_view_type("Qkappa", nspec, ngll, ngll, ngll);
  medium.h_Qmu = Medium::q_view_type("Qmu", nspec, ngll, ngll, ngll);
  medium.h_kappa_scale =
      Medium::q_view_type("kappa_scale", nspec, ngll, ngll, ngll);
  medium.h_mu_scale = Medium::q_view_type("mu_scale", nspec, ngll, ngll, ngll);
  medium.kappa_relaxation_rate =
      Medium::view_type("kappa_rr", nspec, ngll, ngll, ngll, N_SLS);
  medium.h_kappa_relaxation_rate =
      specfem::datatype::create_mirror_view(medium.kappa_relaxation_rate);
  medium.mu_relaxation_rate =
      Medium::view_type("mu_rr", nspec, ngll, ngll, ngll, N_SLS);
  medium.h_mu_relaxation_rate =
      specfem::datatype::create_mirror_view(medium.mu_relaxation_rate);
  for (int i = 0; i < nspec; ++i) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int ix = 0; ix < ngll; ++ix) {
          const int ipoint = point_index(iz, iy, ix);
          medium.h_Qmu(i, iz, iy, ix) = 80.0 + 10.0 * ipoint + 100.0 * i;
          medium.h_Qkappa(i, iz, iy, ix) = 500.0 + 20.0 * ipoint;
        }
      }
    }
  }
  return medium;
}

/** @brief A property container holding the physical model. */
Props make_props() {
  Props props{ HostView("kappa", nspec, ngll, ngll, ngll),
               HostView("mu", nspec, ngll, ngll, ngll),
               specfem::datatype::ElementIndexRange(0, nspec) };
  for (int i = 0; i < nspec; ++i) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int ix = 0; ix < ngll; ++ix) {
          props.h_kappa(i, iz, iy, ix) = physical_kappa(i, iz, iy, ix);
          props.h_mu(i, iz, iy, ix) = physical_mu(i, iz, iy, ix);
        }
      }
    }
  }
  return props;
}

/** @brief Deep copy of a host property view. */
HostView copy_of(const HostView &view) {
  HostView copy("copy", view.extent(0), view.extent(1), view.extent(2),
                view.extent(3));
  Kokkos::deep_copy(copy, view);
  return copy;
}

} // namespace attenuation_recompute_test_impl

TEST(AttenuationRecompute, WriterRecoversPhysicalModelUnderGllVaryingQ) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_medium(true);
  const auto props = test::make_props();

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::mantle);

  // Q differs at every GLL point, so the scale must as well; one value per
  // element could not undo the shift below.
  EXPECT_NE(medium.h_mu_scale(0, 0, 0, 0), medium.h_mu_scale(0, 2, 2, 2));

  const auto kappa = test::copy_of(props.h_kappa);
  const auto mu = test::copy_of(props.h_mu);
  medium.to_physical(kappa, "kappa");
  medium.to_physical(mu, "mu");

  for (int i = 0; i < test::nspec; ++i) {
    for (int iz = 0; iz < test::ngll; ++iz) {
      for (int iy = 0; iy < test::ngll; ++iy) {
        for (int ix = 0; ix < test::ngll; ++ix) {
          const type_real expected_kappa = test::physical_kappa(i, iz, iy, ix);
          const type_real expected_mu = test::physical_mu(i, iz, iy, ix);
          EXPECT_NEAR(kappa(i, iz, iy, ix), expected_kappa,
                      1.0e-5 * expected_kappa);
          EXPECT_NEAR(mu(i, iz, iy, ix), expected_mu, 1.0e-5 * expected_mu);
          // Outside this range the band or f0 is misconfigured.
          EXPECT_GE(medium.h_mu_scale(i, iz, iy, ix), 0.5);
          EXPECT_LE(medium.h_mu_scale(i, iz, iy, ix), 1.5);
        }
      }
    }
  }
}

TEST(AttenuationRecompute, ShearOnlyLeavesKappaPhysicalAndUnrelaxed) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_medium(false);
  const auto props = test::make_props();

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::mantle);

  for (int i = 0; i < test::nspec; ++i) {
    for (int iz = 0; iz < test::ngll; ++iz) {
      for (int iy = 0; iy < test::ngll; ++iy) {
        for (int ix = 0; ix < test::ngll; ++ix) {
          // Exact: kappa must not be touched at all.
          EXPECT_NEAR(props.h_kappa(i, iz, iy, ix),
                      test::physical_kappa(i, iz, iy, ix), 0.0);
          EXPECT_NEAR(medium.h_kappa_scale(i, iz, iy, ix), 1.0, 0.0);
          EXPECT_NE(medium.h_mu_scale(i, iz, iy, ix), 1.0);
          for (int j = 0; j < test::N_SLS; ++j) {
            EXPECT_NEAR(medium.h_kappa_relaxation_rate(i, iz, iy, ix, j), 0.0,
                        0.0);
            EXPECT_NE(medium.h_mu_relaxation_rate(i, iz, iy, ix, j), 0.0);
          }
        }
      }
    }
  }
}

TEST(AttenuationRecompute, RejectsASecondCall) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_medium(true);
  const auto props = test::make_props();

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::mantle);
  EXPECT_THROW(medium.recompute(props, inputs.fc, inputs.f0, inputs.band,
                                inputs.tau_sigma, test::SphericalView{},
                                test::mantle),
               std::runtime_error);

  // A copy shares the guard, so it cannot be used to scale twice either.
  const auto copy = medium;
  EXPECT_THROW(copy.recompute(props, inputs.fc, inputs.f0, inputs.band,
                              inputs.tau_sigma, test::SphericalView{},
                              test::mantle),
               std::runtime_error);
}

namespace attenuation_recompute_test_impl {
using AnisotropicMedium = specfem::assembly::impl::attenuation_medium<
    specfem::element::dimension_tag::dim3,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::anisotropic,
    specfem::element::attenuation_tag::constant_isotropic>;
using Cartesian = specfem::coordinate_systems::CartesianFrame;
using Geocentric = specfem::coordinate_systems::GeocentricFrame;
using Tensor = specfem::medium_physics::ElasticityTensor<Cartesian, double>;
using Radial = specfem::medium_physics::ElasticityTensor<Geocentric, double>;

struct AnisotropicProps {
  HostView h_c11{ "c11", nspec, ngll, ngll, ngll };
  HostView h_c12{ "c12", nspec, ngll, ngll, ngll };
  HostView h_c13{ "c13", nspec, ngll, ngll, ngll };
  HostView h_c14{ "c14", nspec, ngll, ngll, ngll };
  HostView h_c15{ "c15", nspec, ngll, ngll, ngll };
  HostView h_c16{ "c16", nspec, ngll, ngll, ngll };
  HostView h_c22{ "c22", nspec, ngll, ngll, ngll };
  HostView h_c23{ "c23", nspec, ngll, ngll, ngll };
  HostView h_c24{ "c24", nspec, ngll, ngll, ngll };
  HostView h_c25{ "c25", nspec, ngll, ngll, ngll };
  HostView h_c26{ "c26", nspec, ngll, ngll, ngll };
  HostView h_c33{ "c33", nspec, ngll, ngll, ngll };
  HostView h_c34{ "c34", nspec, ngll, ngll, ngll };
  HostView h_c35{ "c35", nspec, ngll, ngll, ngll };
  HostView h_c36{ "c36", nspec, ngll, ngll, ngll };
  HostView h_c44{ "c44", nspec, ngll, ngll, ngll };
  HostView h_c45{ "c45", nspec, ngll, ngll, ngll };
  HostView h_c46{ "c46", nspec, ngll, ngll, ngll };
  HostView h_c55{ "c55", nspec, ngll, ngll, ngll };
  HostView h_c56{ "c56", nspec, ngll, ngll, ngll };
  HostView h_c66{ "c66", nspec, ngll, ngll, ngll };
  // Nonzero global indices exercise spherical-cache indexing.
  specfem::datatype::ElementIndexRange element_range{ 7, 7 + nspec };
  auto views() const {
    return std::array{ h_c11, h_c12, h_c13, h_c14, h_c15, h_c16, h_c22,
                       h_c23, h_c24, h_c25, h_c26, h_c33, h_c34, h_c35,
                       h_c36, h_c44, h_c45, h_c46, h_c55, h_c56, h_c66 };
  }
};

AnisotropicMedium make_anisotropic_medium() {
  const auto iso = make_medium(false);
  AnisotropicMedium medium;
  medium.bulk_attenuation = iso.bulk_attenuation;
  medium.h_recomputed = iso.h_recomputed;
  medium.h_Qkappa = iso.h_Qkappa;
  medium.h_Qmu = iso.h_Qmu;
  medium.h_kappa_scale = iso.h_kappa_scale;
  medium.h_mu_scale = iso.h_mu_scale;
  medium.kappa_relaxation_rate = iso.kappa_relaxation_rate;
  medium.h_kappa_relaxation_rate = iso.h_kappa_relaxation_rate;
  medium.mu_relaxation_rate = iso.mu_relaxation_rate;
  medium.h_mu_relaxation_rate = iso.h_mu_relaxation_rate;
  medium.element_range = specfem::datatype::ElementIndexRange(7, 7 + nspec);
  for (auto &view : medium.h_cij_correction) {
    view = HostView("correction", nspec, ngll, ngll, ngll);
  }
  return medium;
}
} // namespace attenuation_recompute_test_impl

TEST(AnisotropicAttenuationRecompute, IsotropicLimitAndGllVaryingQ) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto iso = test::make_medium(false);
  const auto aniso = test::make_anisotropic_medium();
  const auto iso_props = test::make_props();
  const test::AnisotropicProps props;
  Kokkos::View<double *****, Kokkos::HostSpace> spherical(
      "spherical", 7 + test::nspec, test::ngll, test::ngll, test::ngll, 3);
  const auto views = props.views();
  for (int i = 0; i < test::nspec; ++i) {
    for (int z = 0; z < test::ngll; ++z) {
      for (int y = 0; y < test::ngll; ++y) {
        for (int x = 0; x < test::ngll; ++x) {
          const double kappa = iso_props.h_kappa(i, z, y, x);
          const double mu = iso_props.h_mu(i, z, y, x);
          const double vp = std::sqrt(kappa + 4. / 3. * mu);
          const double vs = std::sqrt(mu);
          const test::Radial radial(1., vp, vp, vs, vs, 1.);
          for (int k = 0; k < 21; ++k)
            views[k](i, z, y, x) = radial[k];
          spherical(7 + i, z, y, x, 1) = 0.1 * test::point_index(z, y, x);
          spherical(7 + i, z, y, x, 2) = 0.7 * i;
        }
      }
    }
  }
  iso.recompute(iso_props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                test::SphericalView{}, test::mantle);
  aniso.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                  spherical, test::mantle);
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-12 : 3.e-6;
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          const double mu = iso_props.h_mu(i, z, y, x);
          const double kappa = iso_props.h_kappa(i, z, y, x);
          EXPECT_NEAR(props.h_c44(i, z, y, x), mu, tolerance * mu);
          EXPECT_NEAR(props.h_c11(i, z, y, x), kappa + 4. / 3. * mu,
                      tolerance * kappa);
          EXPECT_NEAR(props.h_c12(i, z, y, x), kappa - 2. / 3. * mu,
                      tolerance * kappa);
          for (int j = 0; j < test::N_SLS; ++j) {
            const double expected = iso.h_mu_relaxation_rate(i, z, y, x, j);
            EXPECT_NEAR(aniso.h_mu_relaxation_rate(i, z, y, x, j), expected,
                        tolerance * std::abs(expected));
            EXPECT_NEAR(aniso.h_kappa_relaxation_rate(i, z, y, x, j), 0., 0.);
          }
        }
  EXPECT_THROW(aniso.recompute(props, inputs.fc, inputs.f0, inputs.band,
                               inputs.tau_sigma, spherical, test::mantle),
               std::runtime_error);
  const auto copy = aniso;
  EXPECT_THROW(copy.recompute(props, inputs.fc, inputs.f0, inputs.band,
                              inputs.tau_sigma, spherical, test::mantle),
               std::runtime_error);
}

TEST(AnisotropicAttenuationRecompute, InfiniteQUsesGlobeLimit) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto infinite = test::make_anisotropic_medium();
  const auto limited = test::make_anisotropic_medium();
  Kokkos::deep_copy(infinite.h_Qmu, std::numeric_limits<type_real>::infinity());
  Kokkos::deep_copy(limited.h_Qmu, 9000.0);
  const test::AnisotropicProps actual, expected;
  const auto a = actual.views();
  const auto e = expected.views();
  for (int k = 0; k < 21; ++k) {
    Kokkos::deep_copy(a[k], static_cast<type_real>(k + 1));
    Kokkos::deep_copy(e[k], static_cast<type_real>(k + 1));
  }
  Kokkos::View<double *****, Kokkos::HostSpace> spherical(
      "spherical", 7 + test::nspec, test::ngll, test::ngll, test::ngll, 3);
  Kokkos::deep_copy(spherical, 0.7);
  EXPECT_THROW(infinite.recompute(actual, inputs.fc, inputs.f0, inputs.band,
                                  inputs.tau_sigma, test::SphericalView{},
                                  test::mantle),
               std::runtime_error);
  EXPECT_FALSE(infinite.h_recomputed());
  infinite.recompute(actual, inputs.fc, inputs.f0, inputs.band,
                     inputs.tau_sigma, spherical, test::mantle);
  limited.recompute(expected, inputs.fc, inputs.f0, inputs.band,
                    inputs.tau_sigma, spherical, test::mantle);
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          EXPECT_NEAR(infinite.h_Qmu(i, z, y, x), 9000., 0.);
          for (int k = 0; k < 21; ++k)
            EXPECT_NEAR(a[k](i, z, y, x), e[k](i, z, y, x), 0.);
          for (int j = 0; j < test::N_SLS; ++j)
            EXPECT_NEAR(infinite.h_mu_relaxation_rate(i, z, y, x, j),
                        limited.h_mu_relaxation_rate(i, z, y, x, j), 0.);
        }
}

TEST(AnisotropicAttenuationRecompute, FullAnisotropyUsesFinalRadialC44) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_anisotropic_medium();
  const test::AnisotropicProps props;
  const auto views = props.views();
  constexpr double theta = 0.7, phi = 1.1;
  test::Radial radial(3300., 8000., 8200., 4400., 4600., 0.9);
  radial(4, 4) *= 1.2; // c44 differs from L=(c44+c55)/2.
  radial(0, 3) = 1.1e9;
  radial(3, 4) = 0.4e9;
  const auto global = specfem::medium_physics::rotate(
      radial, specfem::coordinate_systems::rotation_matrix<test::Cartesian,
                                                           test::Geocentric>(
                  theta, phi));
  double original[21];
  for (int k = 0; k < 21; ++k) {
    const type_real value = global[k];
    original[k] = value;
    Kokkos::deep_copy(views[k], value);
  }
  Kokkos::View<double *****, Kokkos::HostSpace> spherical(
      "spherical", 7 + test::nspec, test::ngll, test::ngll, test::ngll, 3);
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          spherical(7 + i, z, y, x, 1) = theta;
          spherical(7 + i, z, y, x, 2) = phi;
        }
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   spherical, test::mantle);
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;
  const double rate_tolerance =
      std::is_same_v<type_real, double> ? 1.e-10 : 2.e-3;
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          double expected[21], muv, rates[test::N_SLS];
          globe_evaluator_attenuate_cij(
              theta, phi, inputs.band.min.raw(), inputs.band.max.raw(),
              medium.h_Qmu(i, z, y, x), inputs.fc.raw(), inputs.f0.raw(),
              original, expected, &muv, rates);
          for (int k = 0; k < 21; ++k)
            EXPECT_NEAR(views[k](i, z, y, x), expected[k],
                        tolerance * radial(0, 0));
          for (int j = 0; j < test::N_SLS; ++j)
            EXPECT_NEAR(medium.h_mu_relaxation_rate(i, z, y, x, j), rates[j],
                        rate_tolerance * std::abs(rates[j]));
        }
}

TEST(AttenuationRecompute, GlobeRatesUseRelaxedShearModulus) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_medium(false);
  const auto props = test::make_props();
  Kokkos::deep_copy(medium.h_Qmu, 80.0);
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::mantle);
  double sigma[test::N_SLS], epsilon[test::N_SLS], rates[test::N_SLS],
      relaxed_scale, ratio;
  globe_evaluator_attenuation_reference(
      inputs.band.min.raw(), inputs.band.max.raw(), 80., inputs.fc.raw(),
      inputs.f0.raw(), sigma, epsilon, &relaxed_scale, &ratio, rates);
  ASSERT_GT(ratio - 1, 0.01);
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-4;
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x)
          for (int j = 0; j < test::N_SLS; ++j) {
            // Expected = 2*mu_relaxed*(tau_e/tau_s-1)/tau_s,
            // with the relaxed modulus and factors independently from globe.
            const double expected =
                test::physical_mu(i, z, y, x) * relaxed_scale * rates[j];
            EXPECT_NEAR(medium.h_mu_relaxation_rate(i, z, y, x, j), expected,
                        tolerance * std::abs(expected));
          }
}

TEST(AttenuationRecompute, QCacheRetainsFirstActualValueInBucket) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_medium(false);
  const auto props = test::make_props();
  Kokkos::deep_copy(medium.h_Qmu, 80.09);
  medium.h_Qmu(0, 0, 0, 0) = 80.01;
  medium.h_Qmu(0, 0, 0, 1) = 80.11;
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::mantle);
  const type_real first = 80.01, second = 80.11;
  EXPECT_NEAR(medium.h_Qmu(0, 0, 0, 0), first, 0.);
  EXPECT_NEAR(medium.h_Qmu(0, 0, 0, 1), second, 0.);
  EXPECT_NEAR(medium.h_Qmu(1, 2, 2, 2), first, 0.);
  EXPECT_NEAR(medium.h_mu_scale(1, 2, 2, 2), medium.h_mu_scale(0, 0, 0, 0), 0.);
}

TEST(AnisotropicAttenuationRecompute, InnerCoreUsesUnrotatedHexagonalUpdate) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_anisotropic_medium();
  const test::AnisotropicProps props;
  const auto views = props.views();
  const test::Radial hexagonal(13000., 11000., 11200., 3500., 3700., 0.9);
  double original[21];
  for (int k = 0; k < 21; ++k) {
    const type_real value = hexagonal[k];
    original[k] = value;
    Kokkos::deep_copy(views[k], value);
  }
  // Inner-core attenuation must work without any spherical coordinate cache.
  Kokkos::View<double *****, Kokkos::HostSpace> spherical;
  const auto context = [](int) {
    return specfem::assembly::impl::AttenuationElementContext{
      specfem::element::region_tag::inner_core,
      static_cast<int>(specfem::globe::radial_flag::inner_core_normal)
    };
  };
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   spherical, context);
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;
  const double rate_tolerance =
      std::is_same_v<type_real, double> ? 1.e-10 : 2.e-3;
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          double expected[21], muv, rates[test::N_SLS];
          globe_evaluator_attenuate_inner_core(
              inputs.band.min.raw(), inputs.band.max.raw(),
              medium.h_Qmu(i, z, y, x), inputs.fc.raw(), inputs.f0.raw(),
              original, expected, &muv, rates);
          for (int k = 0; k < 21; ++k)
            EXPECT_NEAR(views[k](i, z, y, x), expected[k],
                        tolerance * hexagonal(0, 0));
          for (int j = 0; j < test::N_SLS; ++j)
            EXPECT_NEAR(medium.h_mu_relaxation_rate(i, z, y, x, j), rates[j],
                        rate_tolerance * std::abs(rates[j]));
          EXPECT_NEAR(props.h_c22(i, z, y, x), props.h_c11(i, z, y, x), 0.);
          EXPECT_NEAR(props.h_c23(i, z, y, x), props.h_c13(i, z, y, x), 0.);
          EXPECT_NEAR(props.h_c55(i, z, y, x), props.h_c44(i, z, y, x), 0.);
        }
}

TEST(AnisotropicAttenuationRecompute, SkipsFictitiousInnerCoreElements) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_anisotropic_medium();
  const test::AnisotropicProps props;
  const test::Radial hexagonal(13000., 11000., 11200., 3500., 3700., 0.9);
  const auto views = props.views();
  for (int k = 0; k < 21; ++k)
    Kokkos::deep_copy(views[k], static_cast<type_real>(hexagonal[k]));
  // A skipped element must not even fit its invalid Q.
  medium.h_Qmu(0, 0, 0, 0) = std::numeric_limits<type_real>::quiet_NaN();
  Kokkos::View<double *****, Kokkos::HostSpace> spherical;
  const auto context = [](int ispec) {
    return specfem::assembly::impl::AttenuationElementContext{
      specfem::element::region_tag::inner_core,
      static_cast<int>(ispec == 7
                           ? specfem::globe::radial_flag::fictitious_cube
                           : specfem::globe::radial_flag::inner_core_normal)
    };
  };
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   spherical, context);
  for (int k = 0; k < 21; ++k)
    EXPECT_NEAR(views[k](0, 0, 0, 0), static_cast<type_real>(hexagonal[k]), 0.);
  for (int j = 0; j < test::N_SLS; ++j)
    EXPECT_NEAR(medium.h_mu_relaxation_rate(0, 0, 0, 0, j), 0., 0.);
  EXPECT_GT(medium.h_mu_relaxation_rate(1, 0, 0, 0, 0), 0.);
}

TEST(AnisotropicAttenuationRecompute, CartesianMatchesGlobeAtNorthPole) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto medium = test::make_anisotropic_medium();
  const test::AnisotropicProps props;
  const auto views = props.views();
  test::Radial radial(3300., 8000., 8200., 4400., 4600., 0.9);
  radial(4, 4) *= 1.2;
  radial(0, 3) = 1.1e9;
  radial(3, 4) = 0.4e9;
  double original[21];
  for (int k = 0; k < 21; ++k) {
    const type_real value = radial[k];
    original[k] = value;
    Kokkos::deep_copy(views[k], value);
  }
  // No spherical coordinates: Cartesian z is the radial axis.
  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                   test::SphericalView{}, test::cartesian);
  const double tolerance = std::is_same_v<type_real, double> ? 1.e-10 : 2.e-6;
  const double rate_tolerance =
      std::is_same_v<type_real, double> ? 1.e-10 : 2.e-3;
  for (int i = 0; i < test::nspec; ++i)
    for (int z = 0; z < test::ngll; ++z)
      for (int y = 0; y < test::ngll; ++y)
        for (int x = 0; x < test::ngll; ++x) {
          // At the north pole globe's radial frame is the Cartesian frame.
          double expected[21], muv, rates[test::N_SLS];
          globe_evaluator_attenuate_cij(
              0., 0., inputs.band.min.raw(), inputs.band.max.raw(),
              medium.h_Qmu(i, z, y, x), inputs.fc.raw(), inputs.f0.raw(),
              original, expected, &muv, rates);
          for (int k = 0; k < 21; ++k)
            EXPECT_NEAR(views[k](i, z, y, x), expected[k],
                        tolerance * radial(0, 0));
          EXPECT_NEAR(medium.h_kappa_scale(i, z, y, x), 1., 0.);
          for (int j = 0; j < test::N_SLS; ++j) {
            EXPECT_NEAR(medium.h_mu_relaxation_rate(i, z, y, x, j), rates[j],
                        rate_tolerance * std::abs(rates[j]));
            EXPECT_NEAR(medium.h_kappa_relaxation_rate(i, z, y, x, j), 0., 0.);
          }
        }
}

TEST(AttenuationRecompute, PerGllQ9999IsLimitedOnCartesianMeshes) {
  namespace test = attenuation_recompute_test_impl;
  const test::Inputs inputs;
  const auto iso = test::make_medium(true);
  const auto iso_props = test::make_props();
  const auto aniso = test::make_anisotropic_medium();
  const test::AnisotropicProps aniso_props;
  const auto views = aniso_props.views();
  for (int k = 0; k < 21; ++k)
    Kokkos::deep_copy(views[k], static_cast<type_real>(1.e9 * (k + 1)));
  // As in SPECFEM3D's get_attenuation_model, 9999 is not a per-GLL flag: it
  // is limited to 9000 like any other Q above the maximum. Only materials
  // with both Q at 9999 skip attenuation, in the mesh reader.
  for (const auto &q : { iso.h_Qkappa, iso.h_Qmu, aniso.h_Qmu })
    q(0, 0, 0, 0) = 9999.;
  iso.recompute(iso_props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma,
                test::SphericalView{}, test::cartesian);
  aniso.recompute(aniso_props, inputs.fc, inputs.f0, inputs.band,
                  inputs.tau_sigma, test::SphericalView{}, test::cartesian);

  EXPECT_NEAR(iso.h_Qkappa(0, 0, 0, 0), 9000., 0.);
  EXPECT_NEAR(iso.h_Qmu(0, 0, 0, 0), 9000., 0.);
  EXPECT_NEAR(aniso.h_Qmu(0, 0, 0, 0), 9000., 0.);
  EXPECT_NE(iso.h_mu_scale(0, 0, 0, 0), 1.);
  EXPECT_NE(views[0](0, 0, 0, 0), static_cast<type_real>(1.e9));
  for (int j = 0; j < test::N_SLS; ++j) {
    EXPECT_NE(iso.h_kappa_relaxation_rate(0, 0, 0, 0, j), 0.);
    EXPECT_NE(iso.h_mu_relaxation_rate(0, 0, 0, 0, j), 0.);
    EXPECT_NE(aniso.h_mu_relaxation_rate(0, 0, 0, 0, j), 0.);
  }
}
