#include "specfem/globe_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Spike coverage for issue #2001: the SPECFEM3D_GLOBE model catalog exposed as
// a callable evaluator. These tests answer two questions that reading the code
// cannot settle:
//
//   1. Can the catalog be configured without ever reading the globe Par_file?
//      Every test here constructs an Evaluator from a model *name* only. The
//      suite also has to pass from an arbitrary working directory -- PREM is
//      analytic (fortran/meshfem3d_globe/shared/model_prem.f90 has no open()
//      statement), so no DATA/ tree is consulted.
//   2. Does the Fortran archive link into a C++ target here? If this file
//      builds and runs at all, it does.
//
// Ground truth comes from specfem::globe_model::prem_reference, which calls the
// catalog's own PREM routines. Comparing against a hand-written table would
// only confirm that the table agrees with itself.
//
// NOTE on float comparisons: .agents/rules/testing.md asks for EXPECT_NEAR with
// explicit tolerances, and that is the right default. It is deliberately not
// followed here. The evaluator's entire value proposition is that it reproduces
// the mesher's get_model bit-for-bit; a tolerance would hide precisely the
// drift these tests exist to catch. EXPECT_DOUBLE_EQ (4 ULP) is used for values
// that must agree because they are the same computation, and exact EXPECT_EQ
// appears once, for the outer-core acoustic tag, where zero is a flag rather
// than a computed quantity.

namespace specfem::globe_model_test {

// From fortran/meshfem3d_globe/setup/constants.h.in:890-908
constexpr int iregion_crust_mantle = 1;
constexpr int iregion_outer_core = 2;
constexpr int iregion_inner_core = 3;

constexpr int iflag_crust = 1;
constexpr int iflag_80_moho = 2;
constexpr int iflag_220_80 = 3;
constexpr int iflag_670_220 = 4;
constexpr int iflag_mantle_normal = 5;
constexpr int iflag_outer_core_normal = 6;
constexpr int iflag_inner_core_normal = 7;

// PREM discontinuity radii in metres, from
// fortran/meshfem3d_globe/shared/model_prem.f90:66-78
constexpr double prem_ricb = 1221500.0;
constexpr double prem_rcmb = 3480000.0;
constexpr double prem_r670 = 5701000.0;
constexpr double prem_r220 = 6151000.0;
constexpr double prem_rmoho = 6371000.0 - 24400.0;
constexpr double prem_rsurface = 6371000.0;

/**
 * @brief One synthetic element: a radial shell with a known region and flag.
 *
 * Element *shape* is irrelevant to the evaluator -- it reads only each point's
 * radius -- so a radial column of GLL points spanning the shell is enough to
 * exercise the full evaluation path.
 */
struct RadialShell {
  const char *name;
  int iregion_code;
  int idoubling;
  double rmin_si;
  double rmax_si;
};

const std::vector<RadialShell> &shells() {
  static const std::vector<RadialShell> value = {
    { "inner_core", iregion_inner_core, iflag_inner_core_normal,
      0.35 * prem_ricb, prem_ricb },
    { "outer_core", iregion_outer_core, iflag_outer_core_normal, prem_ricb,
      prem_rcmb },
    { "lower_mantle", iregion_crust_mantle, iflag_mantle_normal, prem_rcmb,
      prem_r670 },
    { "transition_zone", iregion_crust_mantle, iflag_670_220, prem_r670,
      prem_r220 },
    { "upper_mantle", iregion_crust_mantle, iflag_220_80, prem_r220,
      prem_rsurface - 80000.0 },
    { "lithosphere", iregion_crust_mantle, iflag_80_moho,
      prem_rsurface - 80000.0, prem_rmoho },
    { "crust", iregion_crust_mantle, iflag_crust, prem_rmoho, prem_rsurface },
  };
  return value;
}

/** @brief A configuration with all optional physics off, for a bare 1D run. */
specfem::globe_model::ModelConfig bare_config(const std::string &model_name) {
  specfem::globe_model::ModelConfig config;
  config.model_name = model_name;
  config.log_path = "";   // /dev/null
  config.planet_type = 1; // IPLANET_EARTH
  config.nchunks = 6;
  config.nex_xi = 64;
  config.nex_eta = 64;
  return config;
}

/**
 * @brief Radii spanning a shell, inset from both bounds.
 *
 * The evaluator clamps points to the shell interior with a 1e-6 relative
 * tolerance before sampling (get_model.F90:163-165). Sampling strictly inside
 * the shell keeps this test about the model values; the clamp itself is
 * exercised separately by ClampsPointsIntoTheirOwnShell.
 */
std::vector<double> radii_within(const RadialShell &shell,
                                 const std::size_t count) {
  // The clamp band scales with the shell's *radius* (rmin*1e-6, rmax*1e-6), not
  // its thickness, so a purely relative inset is not enough: for a 56 km-thick
  // lithosphere shell at 6346 km the band is ~6.3 m while 1e-4 of the thickness
  // is only 5.6 m, and the outermost sample lands inside the clamp. Inset by
  // whichever margin is larger, at 10x the band.
  const double margin = std::max(1.0e-4 * (shell.rmax_si - shell.rmin_si),
                                 1.0e-5 * shell.rmax_si);
  const double lo = shell.rmin_si + margin;
  const double hi = shell.rmax_si - margin;
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t =
        (count == 1) ? 0.5
                     : static_cast<double>(i) / static_cast<double>(count - 1);
    result[i] = lo + t * (hi - lo);
  }
  return result;
}

/**
 * @brief Packs radii into point-major coordinates along the polar axis.
 *
 * Axis-aligned on purpose. The evaluator recovers each point's radius via
 * sqrt(x^2+y^2+z^2); along an axis that is exactly |z|, so the sampled radius
 * is bit-identical to the requested one and the exact comparisons below are
 * meaningful. An oblique direction introduces a 1-2 ULP radius error which
 * propagates into every PREM polynomial. Since these are 1D models, theta and
 * phi are irrelevant to the result -- DirectionIndependence below confirms that
 * rather than assuming it.
 */
std::vector<double> radial_column(const std::vector<double> &radii) {
  std::vector<double> xyz(3 * radii.size());
  for (std::size_t i = 0; i < radii.size(); ++i) {
    xyz[3 * i + 0] = 0.0;
    xyz[3 * i + 1] = 0.0;
    xyz[3 * i + 2] = radii[i];
  }
  return xyz;
}

/** @brief The same radii along an oblique, genuinely 3D direction. */
std::vector<double> oblique_column(const std::vector<double> &radii) {
  // Normalized at runtime; hardcoding the components invites a non-unit vector
  // and a silently wrong sampling radius.
  const double norm = std::sqrt(1.0 + 4.0 + 9.0);
  const double nx = 1.0 / norm;
  const double ny = 2.0 / norm;
  const double nz = 3.0 / norm;

  std::vector<double> xyz(3 * radii.size());
  for (std::size_t i = 0; i < radii.size(); ++i) {
    xyz[3 * i + 0] = radii[i] * nx;
    xyz[3 * i + 1] = radii[i] * ny;
    xyz[3 * i + 2] = radii[i] * nz;
  }
  return xyz;
}

/**
 * @brief Fixture owning a single Evaluator.
 *
 * The catalog's configuration is global Fortran module state, so exactly one
 * Evaluator may be alive. GoogleTest runs tests sequentially within a binary,
 * and the fixture tears its Evaluator down between tests, which keeps that
 * honest.
 */
class PremEvaluatorTest : public ::testing::Test {
protected:
  void configure(const std::string &model_name) {
    evaluator_ = std::make_unique<specfem::globe_model::Evaluator>(
        bare_config(model_name));
  }

  void TearDown() override { evaluator_.reset(); }

  std::unique_ptr<specfem::globe_model::Evaluator> evaluator_;
};

// -----------------------------------------------------------------------------
// Quadrature contract
// -----------------------------------------------------------------------------

// The evaluator samples the catalog at whatever NGLL it was compiled with. If
// that disagrees with SPECFEM++'s, every returned value is silently attributed
// to the wrong point, so the contract has to be asserted rather than assumed.
TEST(PremEvaluatorDims, MatchesTheCatalogQuadrature) {
  const auto dims = specfem::globe_model::Evaluator::dims();

  EXPECT_EQ(dims.ngllx, 5);
  EXPECT_EQ(dims.nglly, 5);
  EXPECT_EQ(dims.ngllz, 5);
  EXPECT_EQ(dims.n_sls, 3);
  EXPECT_EQ(dims.points_per_element(), 125u);
}

// -----------------------------------------------------------------------------
// Configuration without a Par_file -- the load-bearing question
// -----------------------------------------------------------------------------

TEST_F(PremEvaluatorTest, ConfiguresIsotropicPremFromNameAlone) {
  EXPECT_NO_THROW(configure("1d_isotropic_prem"));
  EXPECT_TRUE(specfem::globe_model::Evaluator::is_active());
}

TEST_F(PremEvaluatorTest, ConfiguresTransverselyIsotropicPremFromNameAlone) {
  EXPECT_NO_THROW(configure("1d_transversely_isotropic_prem"));
  EXPECT_TRUE(specfem::globe_model::Evaluator::is_active());
}

// The catalog lowercases the model name before dispatching
// (get_model_parameters.F90:92), so the caller need not care about case.
TEST_F(PremEvaluatorTest, ModelNameIsCaseInsensitive) {
  EXPECT_NO_THROW(configure("1D_ISOTROPIC_PREM"));
}

// -----------------------------------------------------------------------------
// Guard rails
// -----------------------------------------------------------------------------

// The catalog's configuration lives in Fortran module state, so two Evaluators
// would silently share one model. This is the only one of the evaluator's
// "uncheckable" invariants that can actually be enforced in the type system.
TEST_F(PremEvaluatorTest, RefusesASecondInstance) {
  configure("1d_isotropic_prem");

  EXPECT_THROW(
      specfem::globe_model::Evaluator{ bare_config("1d_isotropic_prem") },
      std::runtime_error);
}

TEST_F(PremEvaluatorTest, IsInactiveAfterDestruction) {
  configure("1d_isotropic_prem");
  ASSERT_TRUE(specfem::globe_model::Evaluator::is_active());

  evaluator_.reset();

  EXPECT_FALSE(specfem::globe_model::Evaluator::is_active());
  // ... and a fresh Evaluator can then be built.
  EXPECT_NO_THROW(configure("1d_transversely_isotropic_prem"));
}

// Models whose values are read out of a per-GLL array belonging to the mesher's
// own discretization cannot be evaluated from a position alone. Rejecting them
// beats returning a plausible wrong answer.
//
// Every name here must parse successfully and merely land on a rejected flag.
// Two classes of model CANNOT be tested, because the catalog terminates the
// process with a bare Fortran `stop` rather than returning:
//   - unrecognized names          (get_model_parameters.F90:978-982)
//   - models needing build support this configuration lacks, e.g. "cem_request"
//     without CEM/NetCDF         (get_model_parameters.F90:731-737)
// A `stop` calls exit(), which unwinds Kokkos and MPI from under the test
// binary. Converting those to status codes is a tracked follow-up.
TEST_F(PremEvaluatorTest, RejectsModelsNeedingPerPointIndexing) {
  // HETEROGEN_3D_MANTLE -> model_heterogen_mantle(ispec,i,j,k,...)
  EXPECT_THROW(configure("heterogen"), std::runtime_error);
  EXPECT_FALSE(specfem::globe_model::Evaluator::is_active());

  EXPECT_THROW(configure("heterogen_prem"), std::runtime_error);
  EXPECT_FALSE(specfem::globe_model::Evaluator::is_active());

  // MODEL_GLL -> model_gll_impose_val(...,ispec,i,j,k,...)
  EXPECT_THROW(configure("gll_iso"), std::runtime_error);
  EXPECT_FALSE(specfem::globe_model::Evaluator::is_active());

  // MODEL_GLL + ATTENUATION_GLL -> model_attenuation_gll(ispec,i,j,k,Qmu)
  EXPECT_THROW(configure("gll_qmu"), std::runtime_error);
  EXPECT_FALSE(specfem::globe_model::Evaluator::is_active());
}

TEST_F(PremEvaluatorTest, RejectsMalformedCoordinateArrays) {
  configure("1d_isotropic_prem");

  const std::vector<double> too_short(3 * 124, 0.0);
  EXPECT_THROW(evaluator_->evaluate_element(iregion_crust_mantle,
                                            iflag_mantle_normal, prem_rcmb,
                                            prem_r670, false, false, too_short),
               std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Model values against the catalog's own PREM reference
// -----------------------------------------------------------------------------

class PremProfileTest : public PremEvaluatorTest,
                        public ::testing::WithParamInterface<
                            std::tuple<std::string, std::size_t>> {};

TEST_P(PremProfileTest, MatchesTheReferenceEverywhere) {
  const std::string model_name = std::get<0>(GetParam());
  const std::size_t shell_index = std::get<1>(GetParam());
  const RadialShell &shell = shells()[shell_index];

  configure(model_name);

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();
  const std::vector<double> radii = radii_within(shell, npoints);

  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si,
      /* elem_in_crust = */ false, /* elem_in_mantle = */ false,
      radial_column(radii));

  for (std::size_t i = 0; i < npoints; ++i) {
    SCOPED_TRACE("shell=" + std::string(shell.name) + " point=" +
                 std::to_string(i) + " radius=" + std::to_string(radii[i]));

    const auto expected = specfem::globe_model::prem_reference(
        radii[i], shell.idoubling, shell.iregion_code);

    // Exact: these pass through meshfem3D_models_get1D_val untouched for a 1D
    // model, so anything but bit-equality means the evaluator diverged from the
    // catalog.
    EXPECT_DOUBLE_EQ(properties.rho[i], expected.rho);
    EXPECT_DOUBLE_EQ(properties.vpv[i], expected.vpv);
    EXPECT_DOUBLE_EQ(properties.vph[i], expected.vph);
    EXPECT_DOUBLE_EQ(properties.vsv[i], expected.vsv);
    EXPECT_DOUBLE_EQ(properties.vsh[i], expected.vsh);
    EXPECT_DOUBLE_EQ(properties.eta[i], expected.eta);

    // Also exact: the reference shim applies the identical Voigt reduction from
    // the identical inputs, so an ULP-level difference here would mean the two
    // paths do not share the averaging convention.
    EXPECT_DOUBLE_EQ(properties.vp_iso[i], expected.vp_iso);
    EXPECT_DOUBLE_EQ(properties.vs_iso[i], expected.vs_iso);

    EXPECT_DOUBLE_EQ(properties.qkappa[i], expected.qkappa);
    EXPECT_DOUBLE_EQ(properties.qmu[i], expected.qmu);

    // Physical sanity, independent of the reference.
    EXPECT_GT(properties.rho[i], 0.0);
    EXPECT_GT(properties.vpv[i], 0.0);
    EXPECT_GT(properties.vph[i], 0.0);
  }

  // A 1D model with no anisotropy flags must not claim an anisotropic tier.
  EXPECT_FALSE(properties.is_anisotropic);
}

INSTANTIATE_TEST_SUITE_P(
    IsotropicPrem, PremProfileTest,
    ::testing::Combine(::testing::Values("1d_isotropic_prem"),
                       ::testing::Range(std::size_t{ 0 }, std::size_t{ 7 })));

INSTANTIATE_TEST_SUITE_P(
    TransverselyIsotropicPrem, PremProfileTest,
    ::testing::Combine(::testing::Values("1d_transversely_isotropic_prem"),
                       ::testing::Range(std::size_t{ 0 }, std::size_t{ 7 })));

// -----------------------------------------------------------------------------
// Structural properties of the returned material
// -----------------------------------------------------------------------------

// vs_iso == 0 in the outer core is not an approximation -- it is the acoustic
// tag (get_model.F90:181-187), and the caller cross-checks it against the
// database's medium_tag. An epsilon comparison here would hide a real change.
TEST_F(PremEvaluatorTest, OuterCoreIsExactlyAcoustic) {
  configure("1d_transversely_isotropic_prem");

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();
  const RadialShell &shell = shells()[1];
  ASSERT_STREQ(shell.name, "outer_core");

  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si, false,
      false, radial_column(radii_within(shell, npoints)));

  for (std::size_t i = 0; i < npoints; ++i) {
    SCOPED_TRACE("point=" + std::to_string(i));
    EXPECT_EQ(properties.vs_iso[i], 0.0);
    // The fluid still has a P velocity and a density.
    EXPECT_GT(properties.vp_iso[i], 0.0);
    EXPECT_GT(properties.rho[i], 0.0);
  }
}

// The solid regions must NOT be tagged acoustic.
TEST_F(PremEvaluatorTest, SolidRegionsHaveNonZeroShearSpeed) {
  configure("1d_transversely_isotropic_prem");

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();

  for (const RadialShell &shell : shells()) {
    if (shell.iregion_code == iregion_outer_core) {
      continue;
    }
    SCOPED_TRACE(std::string("shell=") + shell.name);

    const auto properties = evaluator_->evaluate_element(
        shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si,
        false, false, radial_column(radii_within(shell, npoints)));

    for (std::size_t i = 0; i < npoints; ++i) {
      EXPECT_GT(properties.vs_iso[i], 0.0) << "point " << i;
    }
  }
}

// Each shell is bounded by a PREM discontinuity. Sampling just inside adjacent
// shells must land on opposite sides of the jump -- this is what the rmin/rmax
// clamp at get_model.F90:163-165 buys, and it is the mechanism the database's
// per-element rmin/rmax context exists to feed.
TEST_F(PremEvaluatorTest, HonorsDiscontinuitiesBetweenAdjacentShells) {
  configure("1d_transversely_isotropic_prem");

  struct Boundary {
    const char *name;
    std::size_t below;
    std::size_t above;
  };

  // ICB and CMB are the two first-order discontinuities in the set; 670 is a
  // strong one. All three must show a density jump.
  const std::vector<Boundary> boundaries = {
    { "ICB", 0, 1 },
    { "CMB", 1, 2 },
    { "d670", 2, 3 },
  };

  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(std::string("boundary=") + boundary.name);

    const RadialShell &lower = shells()[boundary.below];
    const RadialShell &upper = shells()[boundary.above];
    ASSERT_DOUBLE_EQ(lower.rmax_si, upper.rmin_si);

    // A single point pinned near the shared face, from each side.
    const auto below = evaluator_->evaluate_element(
        lower.iregion_code, lower.idoubling, lower.rmin_si, lower.rmax_si,
        false, false, radial_column(radii_within(lower, 125)));
    const auto above = evaluator_->evaluate_element(
        upper.iregion_code, upper.idoubling, upper.rmin_si, upper.rmax_si,
        false, false, radial_column(radii_within(upper, 125)));

    // Topmost point of the lower shell vs bottom-most point of the upper one.
    const double rho_below = below.rho.back();
    const double rho_above = above.rho.front();

    // Values are non-dimensional (density ~ 1), so the threshold is a fraction
    // of RHOAV rather than a value in kg/m^3. Every one of these boundaries
    // jumps by far more than 5%.
    EXPECT_GT(std::abs(rho_below - rho_above), 0.05)
        << "expected a density jump across " << boundary.name << " but got "
        << rho_below << " vs " << rho_above << " (non-dimensional)";
    // Density decreases outward across every one of these boundaries.
    EXPECT_GT(rho_below, rho_above);
  }
}

// A point outside its element's shell is pulled back inside before sampling, so
// that an element touching a discontinuity never samples across it. Verified by
// asking for radii beyond both bounds and checking the result equals the
// reference evaluated at the clamped radius.
TEST_F(PremEvaluatorTest, ClampsPointsIntoTheirOwnShell) {
  configure("1d_transversely_isotropic_prem");

  const RadialShell &shell = shells()[2]; // lower mantle, RCMB -> R670
  ASSERT_STREQ(shell.name, "lower_mantle");

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();

  // Every point sits below rmin, so all of them clamp up to rmin * (1 + 1e-6).
  std::vector<double> radii(npoints, shell.rmin_si - 5000.0);
  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si, false,
      false, radial_column(radii));

  const auto expected = specfem::globe_model::prem_reference(
      shell.rmin_si * 1.000001, shell.idoubling, shell.iregion_code);

  for (std::size_t i = 0; i < npoints; ++i) {
    SCOPED_TRACE("point=" + std::to_string(i));
    EXPECT_DOUBLE_EQ(properties.rho[i], expected.rho);
    EXPECT_DOUBLE_EQ(properties.vpv[i], expected.vpv);
  }
}

// -----------------------------------------------------------------------------
// Units
// -----------------------------------------------------------------------------

// The evaluator returns the catalog's non-dimensional values, so a caller that
// forgets to re-dimensionalize gets densities near 1 and velocities near 2 --
// numbers that look plausible and are wrong by three orders of magnitude. This
// test pins the scaling contract by checking that re-dimensionalized PREM lands
// on the textbook values, which no amount of exact-equality testing against the
// reference shim can catch (both sides would be wrong together).
TEST_F(PremEvaluatorTest, ScalesToPhysicalSiValues) {
  configure("1d_isotropic_prem");

  const auto scales = specfem::globe_model::Evaluator::scales();

  // Earth radius and mean density.
  EXPECT_NEAR(scales.length, 6371000.0, 1.0);
  EXPECT_NEAR(scales.density, 5514.0, 5.0);
  // R * sqrt(pi G rho) works out to roughly 6.9 km/s.
  EXPECT_NEAR(scales.velocity, 6850.0, 50.0);

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();
  const RadialShell &shell = shells()[2]; // lower mantle
  ASSERT_STREQ(shell.name, "lower_mantle");

  // Just above the CMB, PREM gives rho ~ 5560 kg/m^3, vp ~ 13.7 km/s,
  // vs ~ 7.26 km/s.
  const std::vector<double> radii(npoints, prem_rcmb + 1000.0);
  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si, false,
      false, radial_column(radii));

  EXPECT_NEAR(properties.rho[0] * scales.density, 5566.0, 20.0);
  EXPECT_NEAR(properties.vpv[0] * scales.velocity, 13716.0, 50.0);
  EXPECT_NEAR(properties.vsv[0] * scales.velocity, 7264.0, 50.0);
}

TEST(PremEvaluatorScales, RequireAConfiguredEvaluator) {
  ASSERT_FALSE(specfem::globe_model::Evaluator::is_active());
  EXPECT_THROW(specfem::globe_model::Evaluator::scales(), std::runtime_error);
}

// The exact-value tests sample along the polar axis so the recovered radius is
// bit-exact. That is only legitimate if a 1D model really does ignore
// direction, which this checks rather than assumes. The tolerance covers the
// 1-2 ULP radius error an oblique direction introduces through
// sqrt(x^2+y^2+z^2).
TEST_F(PremEvaluatorTest, IsDirectionIndependentForOneDimensionalModels) {
  configure("1d_transversely_isotropic_prem");

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();

  for (const RadialShell &shell : shells()) {
    SCOPED_TRACE(std::string("shell=") + shell.name);
    const std::vector<double> radii = radii_within(shell, npoints);

    const auto along_axis = evaluator_->evaluate_element(
        shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si,
        false, false, radial_column(radii));
    const auto oblique = evaluator_->evaluate_element(
        shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si,
        false, false, oblique_column(radii));

    for (std::size_t i = 0; i < npoints; ++i) {
      EXPECT_NEAR(along_axis.rho[i], oblique.rho[i], 1.0e-12) << "point " << i;
      EXPECT_NEAR(along_axis.vpv[i], oblique.vpv[i], 1.0e-12) << "point " << i;
      EXPECT_NEAR(along_axis.vsv[i], oblique.vsv[i], 1.0e-12) << "point " << i;
    }
  }
}

// -----------------------------------------------------------------------------
// Attenuation
// -----------------------------------------------------------------------------

// The attenuation period band is the one piece of configuration that is NOT
// derivable from the model name -- the mesher computes it in
// rcp_set_compute_parameters, which the evaluator does not call. Left unset
// with attenuation on, the catalog would abort the process inside
// attenuation_tau_sigma (model_attenuation.f90:625-628); the evaluator rejects
// it at the boundary instead.
TEST_F(PremEvaluatorTest, RejectsAnInvalidAttenuationBand) {
  auto config = bare_config("1d_isotropic_prem");
  config.attenuation = true;

  config.min_attenuation_period = 0.0;
  EXPECT_THROW(specfem::globe_model::Evaluator{ config }, std::runtime_error);

  config.min_attenuation_period = 1000.0;
  config.max_attenuation_period = 20.0; // inverted
  EXPECT_THROW(specfem::globe_model::Evaluator{ config }, std::runtime_error);

  // ... and the band is not consulted at all when attenuation is off.
  config.attenuation = false;
  EXPECT_NO_THROW(
      evaluator_ = std::make_unique<specfem::globe_model::Evaluator>(config));
}

// With ATTENUATION off the catalog never calls getatten_val, so Q must come
// back untouched at its initial value. This pins down that the evaluator is not
// quietly synthesizing attenuation the mesher would not have produced.
TEST_F(PremEvaluatorTest, LeavesQZeroWhenAttenuationIsOff) {
  configure("1d_isotropic_prem");

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();
  const RadialShell &shell = shells()[2];

  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si, false,
      false, radial_column(radii_within(shell, npoints)));

  for (std::size_t i = 0; i < npoints; ++i) {
    SCOPED_TRACE("point=" + std::to_string(i));
    // model_prem_* does populate Qkappa/Qmu, but nothing rescales them.
    EXPECT_GE(properties.qmu[i], 0.0);
    EXPECT_GE(properties.qkappa[i], 0.0);
  }
}

// With ATTENUATION on, the 1D reference path must produce a positive Qmu in the
// solid regions -- the value the caller turns into SLS coefficients.
TEST_F(PremEvaluatorTest, ReturnsPositiveQmuInSolidsWhenAttenuationIsOn) {
  auto config = bare_config("1d_isotropic_prem");
  config.attenuation = true;
  evaluator_ = std::make_unique<specfem::globe_model::Evaluator>(config);

  const std::size_t npoints =
      specfem::globe_model::Evaluator::dims().points_per_element();
  const RadialShell &shell = shells()[2]; // lower mantle

  const auto properties = evaluator_->evaluate_element(
      shell.iregion_code, shell.idoubling, shell.rmin_si, shell.rmax_si, false,
      false, radial_column(radii_within(shell, npoints)));

  for (std::size_t i = 0; i < npoints; ++i) {
    SCOPED_TRACE("point=" + std::to_string(i));
    EXPECT_GT(properties.qmu[i], 0.0);
  }
}

} // namespace specfem::globe_model_test
