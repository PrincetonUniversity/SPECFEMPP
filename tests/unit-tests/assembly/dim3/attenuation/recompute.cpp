#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "specfem/utilities/logarithmic_center.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace attenuation_recompute_test_impl {

using Medium = specfem::assembly::impl::attenuation_medium<
    specfem::element::dimension_tag::dim3,
    specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic,
    specfem::element::attenuation_tag::constant_isotropic>;
using HostView =
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;

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

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma);

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

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma);

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

  medium.recompute(props, inputs.fc, inputs.f0, inputs.band, inputs.tau_sigma);
  EXPECT_THROW(medium.recompute(props, inputs.fc, inputs.f0, inputs.band,
                                inputs.tau_sigma),
               std::runtime_error);

  // A copy shares the guard, so it cannot be used to scale twice either.
  const auto copy = medium;
  EXPECT_THROW(copy.recompute(props, inputs.fc, inputs.f0, inputs.band,
                              inputs.tau_sigma),
               std::runtime_error);
}
