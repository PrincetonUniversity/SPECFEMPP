// Tests of the injection kernel run by PointwiseSurfaceMethod::apply: the
// ConstantForce test method, fed by the session's mock plugin, on a mesh whose
// Stacey faces are injection faces.

#include "../session/mock_plugin.hpp"
#include "constant_force.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/injection.hpp"
#include "specfem/injection/points/points.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace specfem::injection_method_test {

constexpr auto dim3 = specfem::element::dimension_tag::dim3;
constexpr auto elastic = specfem::element::medium_tag::elastic;
constexpr auto forward = specfem::simulation::field_type::forward;
constexpr int nstep = 6;
constexpr int istep = 2;
// Velocity follows the three displacement slots.
constexpr int velocity_slot = 3;

class InjectionKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto mesh = specfem::io::read_3d_mesh(
        "data/dim3/EightNodeElasticInjectionStacey/database.bin",
        specfem::attenuation::Setup{});
    const auto quadratures =
        specfem::quadrature::quadratures(specfem::quadrature::gll::gll{});
    std::vector<std::shared_ptr<specfem::sources::source<dim3>>> sources;
    const std::vector<std::shared_ptr<specfem::receivers::receiver<dim3>>>
        receivers;
    const std::vector<specfem::enums::wavefield> stypes;
    assembly = specfem::assembly::assembly<dim3>(
        mesh, quadratures, sources, receivers, stypes, /* t0 */ -1.5,
        /* dt */ 0.25, nstep, /* max_sig_step */ 1,
        /* nsteps_between_samples */ 1, specfem::simulation::type::forward,
        /* allocate_boundary_values */ false, /* property_reader */ nullptr);
  }

  void setup_method() {
    specfem::injection::ProviderSession session(
        specfem::injection::PluginLibrary::open(
            specfem::injection_session_test::mock_plugin_spec()),
        "ok", specfem::injection::make_simulation_parameters(assembly, nstep));
    session.setup(specfem::injection::make_injection_points(
        assembly, method.quantities(), method.name()));
    method.setup(assembly, std::move(session));
  }

  // Expected acceleration per global point: the sum over the unmasked entries
  // at the point of scale * velocity * dS.
  std::map<int, std::array<double, 3>>
  expected(const std::vector<bool> &masked) const {
    const auto &surface = assembly.injection_surface;
    const type_real scale = ConstantForce::Parameters{}.scale;
    std::map<int, std::array<double, 3>> sums;
    for (int entry = 0; entry < surface.nentries; ++entry) {
      if (masked[entry])
        continue;
      const int iglob = assembly.mesh.h_index_mapping(
          surface.h_ispec(entry), surface.h_iz(entry), surface.h_iy(entry),
          surface.h_ix(entry));
      const int row = surface.h_row(entry);
      auto &sum = sums[iglob];
      for (int icomp = 0; icomp < 3; ++icomp)
        sum[icomp] +=
            scale * surface.h_weight(entry) *
            static_cast<type_real>(specfem::injection_session_test::mock_value(
                surface.x(row), velocity_slot + icomp, istep));
    }
    return sums;
  }

  // Compare the forward acceleration at every GLL point with the expected
  // sums; points without a sum must be zero. Returns the GLL points that
  // carry a sum.
  int expect_acceleration(const std::map<int, std::array<double, 3>> &sums) {
    assembly.fields.copy_to_host();
    const auto &field = assembly.fields.get_simulation_field<forward>();
    const int ngll = 5;
    int checked = 0;
    for (int ispec = 0; ispec < assembly.mesh.nspec; ++ispec)
      for (int iz = 0; iz < ngll; ++iz)
        for (int iy = 0; iy < ngll; ++iy)
          for (int ix = 0; ix < ngll; ++ix) {
            const specfem::point::index<dim3, false> index(ispec, iz, iy, ix);
            specfem::point::acceleration<specfem::tags::Tags<
                dim3, elastic, specfem::element::property_tag::isotropic,
                false>>
                acceleration;
            specfem::assembly::load_on_host(index, field, acceleration);
            const auto found =
                sums.find(assembly.mesh.h_index_mapping(ispec, iz, iy, ix));
            for (int icomp = 0; icomp < 3; ++icomp) {
              const double value =
                  found == sums.end() ? 0.0 : found->second[icomp];
              EXPECT_NEAR(acceleration(icomp), value,
                          1e-5 * std::abs(value) + 1e-6)
                  << "ispec " << ispec << " (" << iz << ", " << iy << ", " << ix
                  << ") component " << icomp;
            }
            if (found != sums.end())
              ++checked;
          }
    return checked;
  }

  specfem::assembly::assembly<dim3> assembly;
  specfem::injection::PointwiseSurfaceMethod<ConstantForce> method;
};

TEST_F(InjectionKernelTest, ForceLandsOnTheInjectionPoints) {
  setup_method();
  method.prepare_step(istep);
  method.apply(assembly, elastic, forward, istep);
  EXPECT_GT(expect_acceleration(expected(
                std::vector<bool>(assembly.injection_surface.nentries, false))),
            0);
}

TEST_F(InjectionKernelTest, DirichletPointsReceiveNothing) {
  auto &surface = assembly.injection_surface;
  std::vector<bool> masked(surface.nentries, false);
  for (int entry = 0; entry < surface.nentries; entry += 2) {
    masked[entry] = true;
    surface.h_boundary_tag(entry) =
        specfem::element::boundary_tag::acoustic_free_surface;
  }
  Kokkos::deep_copy(surface.boundary_tag, surface.h_boundary_tag);

  setup_method();
  method.prepare_step(istep);
  method.apply(assembly, elastic, forward, istep);
  EXPECT_GT(expect_acceleration(expected(masked)), 0);
}

TEST_F(InjectionKernelTest, MediumWithoutInjectionPointsIsUntouched) {
  setup_method();
  method.prepare_step(istep);
  method.apply(assembly, specfem::element::medium_tag::acoustic, forward,
               istep);
  expect_acceleration({});
}

TEST_F(InjectionKernelTest, NonForwardWavefieldIsRejected) {
  setup_method();
  method.prepare_step(istep);
  EXPECT_THROW(method.apply(assembly, elastic,
                            specfem::simulation::field_type::adjoint, istep),
               std::runtime_error);
}

TEST_F(InjectionKernelTest, UseBeforeSetupIsRejected) {
  EXPECT_THROW(method.prepare_step(istep), std::logic_error);
}

} // namespace specfem::injection_method_test
