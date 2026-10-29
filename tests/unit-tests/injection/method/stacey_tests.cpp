// Tests of StaceyInjection through specfem::injection::evaluate_point: the
// injected force (t* + Z(v*)) dS, and its pairing with Stacey's -Z(v) dS.

#include "specfem/injection/method/point_harness.hpp"
#include "specfem/injection/method/requirements.hpp"
#include "specfem/injection/method/stacey_injection.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace specfem::injection_method_test {

using specfem::element::medium_tag;
using specfem::element::property_tag;
using specfem::injection::StaceyInjection;

static_assert(
    specfem::injection_impl::PointwiseInjectionMethod<StaceyInjection>);
static_assert(specfem::injection_impl::ComputesAt<
              StaceyInjection, medium_tag::elastic, property_tag::isotropic>);
static_assert(specfem::injection_impl::ComputesAt<
              StaceyInjection, medium_tag::elastic, property_tag::anisotropic>);
static_assert(!specfem::injection_impl::ComputesAt<
              StaceyInjection, medium_tag::acoustic, property_tag::isotropic>);

template <property_tag PropertyTag>
using Properties =
    typename StaceyInjection::Context<medium_tag::elastic,
                                      PropertyTag>::properties_type;

using Vector = std::array<type_real, 3>;

/// An incident field at one point, in the method's slot order
struct Incident {
  Vector velocity;
  Vector traction;

  std::vector<type_real> slots() const {
    return { velocity[0], velocity[1], velocity[2],
             traction[0], traction[1], traction[2] };
  }
};

type_real dot(const Vector &a, const Vector &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Z(v) = (rho_vp - rho_vs)(v.n)n + rho_vs v, written out independently
Vector impedance(const type_real rho_vp, const type_real rho_vs,
                 const Vector &normal, const Vector &velocity) {
  const type_real vn = dot(velocity, normal);
  Vector z;
  for (int i = 0; i < 3; ++i)
    z[i] = (rho_vp - rho_vs) * vn * normal[i] + rho_vs * velocity[i];
  return z;
}

template <property_tag PropertyTag>
Vector inject(const Properties<PropertyTag> &property, const Vector &normal,
              const type_real dS, const Incident &incident) {
  const auto acceleration =
      specfem::injection::evaluate_point<StaceyInjection, medium_tag::elastic,
                                         PropertyTag>(property, normal, dS,
                                                      incident.slots());
  return { acceleration(0), acceleration(1), acceleration(2) };
}

// The tolerance is relative to the size of the terms that were summed, which
// can be far larger than the result when they cancel.
void expect_near(const Vector &actual, const Vector &expected,
                 const type_real magnitude = 0) {
  const type_real scale =
      std::max<type_real>({ 1, std::sqrt(dot(expected, expected)), magnitude });
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-5 * scale) << "component " << i;
}

class StaceyInjectionTest : public ::testing::Test {
protected:
  Vector random_unit_normal() {
    Vector n{ uniform(-1, 1), uniform(-1, 1), uniform(-1, 1) };
    const type_real length = std::sqrt(dot(n, n));
    for (auto &component : n)
      component /= length;
    return n;
  }

  Vector random_vector() {
    return { uniform(-1, 1), uniform(-1, 1), uniform(-1, 1) };
  }

  Properties<property_tag::isotropic> random_isotropic() {
    const type_real rho = uniform(2000, 3500);
    const type_real mu = rho * std::pow(uniform(2000, 4000), 2);
    const type_real kappa = uniform(1.5, 3) * mu;
    return { kappa, mu, rho };
  }

  // A stiffness matrix with all 21 entries set, so the Voigt averages differ
  // from any single entry.
  Properties<property_tag::anisotropic> random_anisotropic() {
    std::array<type_real, 21> c;
    for (auto &entry : c)
      entry = uniform(0.1e10, 0.5e10);
    for (const int diagonal : { 0, 6, 11, 15, 18, 20 })
      c[diagonal] = uniform(5e10, 9e10);
    return { c[0],  c[1],
             c[2],  c[3],
             c[4],  c[5],
             c[6],  c[7],
             c[8],  c[9],
             c[10], c[11],
             c[12], c[13],
             c[14], c[15],
             c[16], c[17],
             c[18], c[19],
             c[20], uniform(2000, 3500) };
  }

  type_real uniform(const type_real low, const type_real high) {
    return std::uniform_real_distribution<type_real>(low, high)(engine_);
  }

private:
  std::mt19937 engine_{ 2156 };
};

template <property_tag PropertyTag>
void expect_formula(const Properties<PropertyTag> &property,
                    const Vector &normal, const type_real dS,
                    const Incident &incident) {
  const Vector z = impedance(property.rho_vp(), property.rho_vs(), normal,
                             incident.velocity);
  Vector expected;
  for (int i = 0; i < 3; ++i)
    expected[i] = (incident.traction[i] + z[i]) * dS;
  expect_near(inject<PropertyTag>(property, normal, dS, incident), expected);
}

TEST_F(StaceyInjectionTest, IsotropicForceIsTractionPlusImpedance) {
  for (int icase = 0; icase < 8; ++icase)
    expect_formula<property_tag::isotropic>(
        random_isotropic(), random_unit_normal(), uniform(0.1, 2),
        { random_vector(), random_vector() });
}

TEST_F(StaceyInjectionTest, AnisotropicForceUsesVoigtImpedances) {
  for (int icase = 0; icase < 8; ++icase)
    expect_formula<property_tag::anisotropic>(
        random_anisotropic(), random_unit_normal(), uniform(0.1, 2),
        { random_vector(), random_vector() });
}

TEST_F(StaceyInjectionTest, ZeroIncidentFieldGivesZero) {
  expect_near(inject<property_tag::isotropic>(random_isotropic(),
                                              random_unit_normal(), 1.5,
                                              { { 0, 0, 0 }, { 0, 0, 0 } }),
              { 0, 0, 0 });
}

// Stacey's absorbing term adds -Z(v) dS; when the total velocity equals the
// incident one and there is no incident traction, the boundary force is zero.
TEST_F(StaceyInjectionTest, CancelsStaceyWhenFieldIsIncident) {
  for (int icase = 0; icase < 4; ++icase) {
    const auto property = random_isotropic();
    const Vector normal = random_unit_normal();
    const type_real dS = uniform(0.1, 2);
    const Vector v = random_vector();

    const Vector injected =
        inject<property_tag::isotropic>(property, normal, dS, { v, {} });
    const Vector z = impedance(property.rho_vp(), property.rho_vs(), normal, v);
    Vector net;
    for (int i = 0; i < 3; ++i)
      net[i] = injected[i] - z[i] * dS;
    expect_near(net, { 0, 0, 0 }, std::sqrt(dot(z, z)) * dS);
  }
}

TEST_F(StaceyInjectionTest, ImpedanceDoesNotDependOnNormalDirection) {
  const auto property = random_isotropic();
  const Vector normal = random_unit_normal();
  const Vector flipped{ -normal[0], -normal[1], -normal[2] };
  const Incident incident{ random_vector(), {} };
  expect_near(inject<property_tag::isotropic>(property, flipped, 0.7, incident),
              inject<property_tag::isotropic>(property, normal, 0.7, incident));
}

TEST_F(StaceyInjectionTest, TractionAloneIsScaledBySurfaceWeight) {
  const Vector t = random_vector();
  expect_near(inject<property_tag::isotropic>(
                  random_isotropic(), random_unit_normal(), 0.25, { {}, t }),
              { t[0] / 4, t[1] / 4, t[2] / 4 });
}

} // namespace specfem::injection_method_test
