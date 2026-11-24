#include "../acoustic_elastic.hpp"
#include "specfem/compute/impl/compute_coupling.hpp"
#include "specfem/compute/impl/compute_coupling.tpp" // so that we don't need to load the entire solver
#include "specfem/element/attributes.hpp"
#include "specfem/element/dimension.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/io.hpp"
#include "utilities/include/fieldmanip/fieldgetter.hpp"
#include "utilities/include/fieldmanip/fieldsetter.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * Host dispatch of the nonconforming coupling kernels.
 *
 * `highres_host` integrates every pair on the faces of the finer side, using
 * the self kernel on that side and the conjugate kernel on the other. The
 * checkers of kerneltest.cpp cannot be reused for it (see the test comment
 * below), so this file compares a `highres_host` assembly against a
 * `self_host` assembly of the same mesh with the same fields.
 */
namespace specfem::nonconforming_test::host_dispatch_impl {

using assembly_type =
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>;
using meshing_type = specfem::element_coupling::interfacial_meshing_type;
constexpr auto dim3_tag = specfem::element::dimension_tag::dim3;
constexpr auto forward_field = specfem::simulation::field_type::forward;

/// Pointwise tolerance, host medium: both assemblies run the same self kernel
/// on the same faces with the same interpolants, so only the order of the
/// atomic accumulation differs; this is the tolerance of the strong checker
/// of acoustic_elastic.cpp.
constexpr type_real pointwise_reltol = 1e-5;
constexpr type_real pointwise_abstol = 1e-7;

/// Weak tolerance, guest medium: the tolerances of the conjugate checker of
/// acoustic_elastic_conjugate.cpp.
constexpr type_real weak_reltol = 1e-4;
constexpr type_real weak_abstol = 1e-6;

/// Largest power of the source and test polynomials (NGLL = 5): the sum of
/// both, 6, stays below 2 * NGLL - 3 = 7, the degree up to which the GLL
/// quadrature is exact.
constexpr int max_pow = 3;
constexpr int ngll_fixtures = 5;

/// Counts mismatches, keeps the first few, and tracks the largest compared
/// magnitude so that a comparison of zeros cannot pass unnoticed.
struct Comparison {
  static constexpr int max_listed = 5;
  int count = 0;
  long num_compared = 0;
  double max_magnitude = 0;
  std::vector<std::string> first;

  void add(const std::string &message) {
    if (count < max_listed) {
      first.push_back(message);
    }
    ++count;
  }

  std::string summary(const std::string &title) const {
    std::ostringstream oss;
    oss << title << ": " << count << " mismatch(es) out of " << num_compared
        << " comparisons\n";
    for (const auto &message : first) {
      oss << "    " << message << "\n";
    }
    return oss.str();
  }
};

template <specfem::element::medium_tag medium, int xpow, int ypow>
using initializer_type = std::conditional_t<
    medium == specfem::element::medium_tag::acoustic,
    specfem::nonconforming_test::kernel::acoustic_field_initializer_pow<xpow,
                                                                        ypow>,
    specfem::nonconforming_test::kernel::elastic_field_initializer_pow<xpow,
                                                                       ypow>>;

template <typename MeshType>
std::unique_ptr<assembly_type> build_assembly(const MeshType &mesh,
                                              const meshing_type &type) {
  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::element::dimension_tag::dim3>>>
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
      receivers;

  specfem::element_coupling::flux_scheme_configuration flux_scheme_config;
  flux_scheme_config.set_interfacial_meshing_type(type);

  return std::make_unique<assembly_type>(
      mesh, quadrature, sources, receivers,
      std::vector<specfem::enums::wavefield>{}, 1.0, 0.0, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr, flux_scheme_config);
}

/// Sets the source field, zeroes the target medium, runs the coupling of the
/// target medium and returns its acceleration (host copy).
template <specfem::element::medium_tag target_medium, int NGLL,
          typename SourceInitializer>
auto run_coupling(const assembly_type &assembly,
                  const SourceInitializer &source_initializer) {
  specfem::test_fieldmanip::set_field_values<forward_field>(assembly,
                                                            source_initializer);
  specfem::test_fieldmanip::set_field_values<forward_field>(
      assembly, specfem::test_fieldmanip::PointSetter<dim3_tag, target_medium>(
                    true, true, true) /* zeroes everything out */);
  specfem::compute::impl::compute_coupling<
      NGLL, specfem::tags::Tags<dim3_tag, forward_field, target_medium>>(
      assembly);
  Kokkos::fence();
  const auto values = specfem::test_fieldmanip::get_field_values<
      forward_field, dim3_tag, target_medium,
      specfem::data_access::DataClassType::acceleration>(assembly);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
}

/// Host medium: both assemblies run the same self kernel, so the results must
/// agree degree of freedom by degree of freedom.
template <specfem::element::medium_tag target_medium,
          specfem::element::medium_tag source_medium, int NGLL, int xpow,
          int ypow>
void compare_pointwise(const assembly_type &self_host_assembly,
                       const assembly_type &highres_host_assembly,
                       Comparison &comparison) {
  constexpr int ncomp =
      specfem::element::attributes<dim3_tag, target_medium>::components;
  const type_real xscale = std::max(std::abs(self_host_assembly.mesh.xmax),
                                    std::abs(self_host_assembly.mesh.xmin));
  const type_real yscale = std::max(std::abs(self_host_assembly.mesh.ymax),
                                    std::abs(self_host_assembly.mesh.ymin));
  const initializer_type<source_medium, xpow, ypow> source(xscale, yscale);

  const auto expected =
      run_coupling<target_medium, NGLL>(self_host_assembly, source);
  const auto got =
      run_coupling<target_medium, NGLL>(highres_host_assembly, source);

  if (expected.extent(0) != got.extent(0)) {
    std::ostringstream oss;
    oss << "x^{" << xpow << "} y^{" << ypow << "}: number of dofs "
        << got.extent(0) << " (highres_host) != " << expected.extent(0)
        << " (self_host)";
    comparison.add(oss.str());
    return;
  }

  for (std::size_t iglob = 0; iglob < expected.extent(0); ++iglob) {
    for (int icomp = 0; icomp < ncomp; ++icomp) {
      const double e = expected(iglob, icomp);
      const double g = got(iglob, icomp);
      const double tol = pointwise_reltol * std::abs(e) + pointwise_abstol;
      ++comparison.num_compared;
      comparison.max_magnitude =
          std::max(comparison.max_magnitude, std::abs(e));
      if (!(std::abs(g - e) <= tol)) {
        std::ostringstream oss;
        oss << "x^{" << xpow << "} y^{" << ypow << "} iglob = " << iglob
            << ", component " << icomp << ": highres_host = " << g
            << ", self_host = " << e << ", |diff| = " << std::abs(g - e)
            << " > tol = " << tol;
        comparison.add(oss.str());
      }
    }
  }
}

/// Guest medium: `highres_host` integrates on the host faces, `self_host` on
/// the guest's own faces. The results are only equal as integrals against
/// smooth test functions, so compare sum_iglob accel(iglob) * x^a y^b for all
/// a, b <= max_pow. Since sum_iglob shape(iglob) * poly(x_iglob) = poly(x) for
/// polynomials of degree below NGLL, the sums are the face integrals of
/// poly * (source field), a polynomial of degree <= 2 * max_pow, which the
/// quadrature of either side integrates exactly.
template <specfem::element::medium_tag target_medium,
          specfem::element::medium_tag source_medium, int NGLL, int xpow,
          int ypow>
void compare_weak(const assembly_type &self_host_assembly,
                  const assembly_type &highres_host_assembly,
                  Comparison &comparison) {
  constexpr int ncomp =
      specfem::element::attributes<dim3_tag, target_medium>::components;
  const type_real xscale = std::max(std::abs(self_host_assembly.mesh.xmax),
                                    std::abs(self_host_assembly.mesh.xmin));
  const type_real yscale = std::max(std::abs(self_host_assembly.mesh.ymax),
                                    std::abs(self_host_assembly.mesh.ymin));
  const initializer_type<source_medium, xpow, ypow> source(xscale, yscale);

  const auto expected =
      run_coupling<target_medium, NGLL>(self_host_assembly, source);
  const auto got =
      run_coupling<target_medium, NGLL>(highres_host_assembly, source);
  const auto coords = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),
      specfem::test_fieldmanip::get_coords_per_dof<forward_field,
                                                   target_medium>(
          self_host_assembly));

  if (expected.extent(0) != got.extent(0) ||
      expected.extent(0) != coords.extent(0)) {
    std::ostringstream oss;
    oss << "x^{" << xpow << "} y^{" << ypow << "}: number of dofs "
        << got.extent(0) << " (highres_host) != " << expected.extent(0)
        << " (self_host)";
    comparison.add(oss.str());
    return;
  }

  for (int test_xpow = 0; test_xpow <= max_pow; ++test_xpow) {
    for (int test_ypow = 0; test_ypow <= max_pow; ++test_ypow) {
      double sum_expected[ncomp] = {};
      double sum_got[ncomp] = {};
      double magnitude = 0; // sum of pointwise norms: scale of the error
      for (std::size_t iglob = 0; iglob < expected.extent(0); ++iglob) {
        const double location_factor =
            std::pow(coords(iglob).x / xscale, test_xpow) *
            std::pow(coords(iglob).y / yscale, test_ypow);
        double norm2 = 0;
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          const double addend = expected(iglob, icomp) * location_factor;
          sum_expected[icomp] += addend;
          sum_got[icomp] += got(iglob, icomp) * location_factor;
          norm2 += addend * addend;
        }
        magnitude += std::sqrt(norm2);
      }
      comparison.max_magnitude = std::max(comparison.max_magnitude, magnitude);
      const double tol = weak_reltol * magnitude + weak_abstol;
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        ++comparison.num_compared;
        if (!(std::abs(sum_got[icomp] - sum_expected[icomp]) <= tol)) {
          std::ostringstream oss;
          oss << "source x^{" << xpow << "} y^{" << ypow << "}, test x^{"
              << test_xpow << "} y^{" << test_ypow << "}, component " << icomp
              << ": highres_host = " << sum_got[icomp]
              << ", self_host = " << sum_expected[icomp]
              << ", |diff| = " << std::abs(sum_got[icomp] - sum_expected[icomp])
              << " > tol = " << tol;
          comparison.add(oss.str());
        }
      }
    }
  }
}

/// Expands the source powers 0 <= xpow, ypow <= max_pow.
template <int NGLL, int... Is>
void compare_all_powers(const assembly_type &self_host_assembly,
                        const assembly_type &highres_host_assembly,
                        Comparison &host_medium, Comparison &guest_medium,
                        std::integer_sequence<int, Is...>) {
  // The acoustic side is the finer one in the fixtures: it is the host medium.
  (compare_pointwise<specfem::element::medium_tag::acoustic,
                     specfem::element::medium_tag::elastic, NGLL,
                     Is % (max_pow + 1), Is / (max_pow + 1)>(
       self_host_assembly, highres_host_assembly, host_medium),
   ...);
  (compare_weak<specfem::element::medium_tag::elastic,
                specfem::element::medium_tag::acoustic, NGLL,
                Is % (max_pow + 1), Is / (max_pow + 1)>(
       self_host_assembly, highres_host_assembly, guest_medium),
   ...);
}

template <specfem::element::medium_tag medium>
void expect_same_dofs(const assembly_type &a, const assembly_type &b,
                      const std::string &database_file) {
  const auto ca = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),
      specfem::test_fieldmanip::get_coords_per_dof<forward_field, medium>(a));
  const auto cb = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),
      specfem::test_fieldmanip::get_coords_per_dof<forward_field, medium>(b));
  ASSERT_EQ(ca.extent(0), cb.extent(0)) << database_file;
  int num_different = 0;
  for (std::size_t iglob = 0; iglob < ca.extent(0); ++iglob) {
    // identical meshes, identical numbering: the coordinates are equal
    if (!(std::abs(ca(iglob).x - cb(iglob).x) <= 1e-5 &&
          std::abs(ca(iglob).y - cb(iglob).y) <= 1e-5 &&
          std::abs(ca(iglob).z - cb(iglob).z) <= 1e-5)) {
      ++num_different;
    }
  }
  EXPECT_EQ(num_different, 0)
      << database_file << ": dof numbering differs between the assemblies ("
      << specfem::element::to_string(medium) << ")";
}

inline void test_host_dispatch(const std::string &database_file) {
  const auto mesh =
      specfem::io::read_3d_mesh(database_file, specfem::attenuation::Setup{});
  const auto self_host_assembly = build_assembly(mesh, meshing_type::self_host);
  const auto highres_host_assembly =
      build_assembly(mesh, meshing_type::highres_host);

  constexpr auto connection_tag =
      specfem::element_connections::type::nonconforming;
  constexpr auto boundary_tag = specfem::element::boundary_tag::none;
  constexpr auto flux_scheme_tag =
      specfem::element_coupling::flux_scheme_tag::natural;
  using specfem::element_coupling::interface_tag;

  // data/dim3/interfaces_only/README.md: N x N acoustic elements lie on
  // M x M elastic elements and the acoustic elements are the smaller ones in
  // both fixtures (11 vs 7 and 13 vs 10 elements per side; the kernel
  // failures of the elastic faces being 1000 / 7 wide confirm it), so
  // highres_host makes acoustic_elastic (self medium: acoustic) the host and
  // elastic_acoustic the guest. self_host makes both hosts. Without this the
  // comparison below would not exercise the dispatch.
  const auto &ea_self =
      self_host_assembly->nonconforming_interfaces
          .template get_interface_container<interface_tag::elastic_acoustic,
                                            boundary_tag, connection_tag,
                                            flux_scheme_tag>();
  const auto &ae_self =
      self_host_assembly->nonconforming_interfaces
          .template get_interface_container<interface_tag::acoustic_elastic,
                                            boundary_tag, connection_tag,
                                            flux_scheme_tag>();
  const auto &ea_high =
      highres_host_assembly->nonconforming_interfaces
          .template get_interface_container<interface_tag::elastic_acoustic,
                                            boundary_tag, connection_tag,
                                            flux_scheme_tag>();
  const auto &ae_high =
      highres_host_assembly->nonconforming_interfaces
          .template get_interface_container<interface_tag::acoustic_elastic,
                                            boundary_tag, connection_tag,
                                            flux_scheme_tag>();
  EXPECT_TRUE(ea_self.self_is_host) << database_file;
  EXPECT_TRUE(ae_self.self_is_host) << database_file;
  ASSERT_TRUE(ae_high.self_is_host)
      << database_file << ": acoustic_elastic must host under highres_host";
  ASSERT_FALSE(ea_high.self_is_host)
      << database_file << ": elastic_acoustic must be the guest side";

  const int ngll = highres_host_assembly->mesh
                       .specfem::assembly::mesh_impl::points<dim3_tag>::ngllz;
  ASSERT_EQ(ngll, ngll_fixtures) << database_file;

  expect_same_dofs<specfem::element::medium_tag::elastic>(
      *self_host_assembly, *highres_host_assembly, database_file);
  expect_same_dofs<specfem::element::medium_tag::acoustic>(
      *self_host_assembly, *highres_host_assembly, database_file);

  Comparison host_medium;
  Comparison guest_medium;
  compare_all_powers<ngll_fixtures>(
      *self_host_assembly, *highres_host_assembly, host_medium, guest_medium,
      std::make_integer_sequence<int, (max_pow + 1) * (max_pow + 1)>());

  EXPECT_EQ(host_medium.count, 0)
      << database_file << "\n"
      << host_medium.summary("acoustic (host medium), pointwise");
  EXPECT_EQ(guest_medium.count, 0)
      << database_file << "\n"
      << guest_medium.summary("elastic (guest medium), weak");
  // the coupling must actually have produced something to compare
  EXPECT_GT(host_medium.max_magnitude, 1e-3) << database_file;
  EXPECT_GT(guest_medium.max_magnitude, 1e-3) << database_file;
}

} // namespace specfem::nonconforming_test::host_dispatch_impl

// kerneltest.cpp's checkers cannot run on a highres_host assembly:
//  - test_nonconforming_acoustic_elastic compares every dof with the exact
//    integral of the target field over the target's own faces. On the guest
//    side highres_host integrates on the host's faces instead, whose elements
//    straddle the guest's element boundaries, so the integrand is only
//    piecewise smooth and the quadrature is not exact: the pointwise values
//    differ (observed: up to ~50 % relative on the elastic dofs).
//  - test_nonconforming_acoustic_elastic_conjugate calls the conjugate kernel
//    directly for every tag; it now returns early where the conjugate
//    container is not the host, so for the host medium (acoustic) it yields
//    zeros against a nonzero reference.
// Hence the comparison against a self_host assembly below.
TEST(NonconformingKernel3D, host_dispatch_11_7) {
  specfem::nonconforming_test::host_dispatch_impl::test_host_dispatch(
      "data/dim3/interfaces_only/acoustic_elastic_11-7/database.bin");
}

TEST(NonconformingKernel3D, host_dispatch_13_10) {
  specfem::nonconforming_test::host_dispatch_impl::test_host_dispatch(
      "data/dim3/interfaces_only/acoustic_elastic_13-10/database.bin");
}
