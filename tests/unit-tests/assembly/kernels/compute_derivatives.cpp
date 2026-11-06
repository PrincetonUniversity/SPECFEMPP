#include "specfem/compute/compute_derivatives.tpp"
#include "../test_fixture/test_fixture.hpp"
#include "specfem/element.hpp"
#include "specfem/execution.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Every medium present in the mesh must receive nonzero Frechet derivatives
// from nonzero adjoint and backward wavefields. Guards against the dispatch in
// compute_derivatives silently skipping a medium.
TEST_F(Assembly2D, compute_derivatives_nonzero_for_every_medium) {
  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  for (auto parameters : *this) {
    const auto test = std::get<0>(parameters);
    const auto mesh = std::get<1>(parameters);
    auto sources = std::get<2>(parameters);
    const auto receivers = std::get<3>(parameters);

    std::vector<specfem::enums::wavefield> seismogram_types = {
      specfem::enums::wavefield::displacement
    };
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> assembly(
        mesh, quadrature, sources, receivers, seismogram_types, 1.0, 0.0, 1, 1,
        1, specfem::simulation::type::combined, false, nullptr);

    // SH anisotropic Frechet derivatives are not implemented and abort
    // (core/specfem/medium/dim2/elastic/anisotropic/frechet_derivative.hpp).
    bool has_sh_anisotropic = false;
    for (const auto attenuation :
         { specfem::element::attenuation_tag::none,
           specfem::element::attenuation_tag::constant_isotropic }) {
      has_sh_anisotropic =
          has_sh_anisotropic ||
          assembly.element_types
                  .get_elements_on_host(
                      specfem::element::medium_tag::elastic_sh,
                      specfem::element::property_tag::anisotropic, attenuation)
                  .extent(0) > 0;
    }
    if (has_sh_anisotropic) {
      std::cout << "Skipping " << test.name
                << ": SH anisotropic kernels are not implemented\n";
      continue;
    }

    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim2) *
            MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic),
        [&]<typename MediumTags>() {
          constexpr auto medium = MediumTags::medium_tag;
          for (const auto &field :
               { assembly.fields.adjoint.template get_field<medium>(),
                 assembly.fields.backward.template get_field<medium>() }) {
            for (const auto &view :
                 { field.get_host_field(), field.get_host_field_dot(),
                   field.get_host_field_dot_dot() }) {
              for (std::size_t iglob = 0; iglob < view.extent(0); ++iglob) {
                for (std::size_t icomp = 0; icomp < view.extent(1); ++icomp) {
                  view(iglob, icomp) = static_cast<type_real>(
                      1.0 + 1e-3 * static_cast<double>(iglob) +
                      static_cast<double>(icomp));
                }
              }
            }
          }
        });
    assembly.fields.copy_to_device();

    specfem::compute::compute_derivatives<
        5, specfem::tags::Tags<specfem::element::dimension_tag::dim2>>(assembly,
                                                                       1.0);
    Kokkos::fence();
    assembly.kernels.copy_to_host();

    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim2) *
            MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic) *
            PROPERTY_SET(isotropic, anisotropic) *
            ATTENUATION_SET(none, constant_isotropic),
        [&]<typename ElementTags>() {
          constexpr auto medium = ElementTags::medium_tag;
          constexpr auto property = ElementTags::property_tag;
          constexpr auto attenuation = ElementTags::attenuation_tag;
          const auto elements = assembly.element_types.get_elements_on_host(
              medium, property, attenuation);
          if (elements.extent(0) == 0) {
            return;
          }

          using PointKernelsType = specfem::point::kernels<specfem::tags::Tags<
              specfem::element::dimension_tag::dim2, medium, property, false>>;
          using ParallelConfig =
              specfem::parallel_configuration::default_chunk_config<
                  specfem::element::dimension_tag::dim2,
                  specfem::datatype::simd<type_real, false>,
                  Kokkos::DefaultHostExecutionSpace>;

          bool any_nonzero = false;
          bool any_nan = false;
          const auto &kernels = assembly.kernels;
          specfem::execution::ChunkedDomainIterator policy(
              ParallelConfig(), elements, assembly.mesh.element_grid);
          specfem::execution::for_all(
              "check_kernels", policy,
              [&](const typename decltype(policy)::base_index_type
                      &iterator_index) {
                const auto index = iterator_index.get_index();
                PointKernelsType point;
                specfem::assembly::load_on_host(index, kernels, point);
                for (int iprop = 0; iprop < PointKernelsType::nprops; ++iprop) {
                  any_nan = any_nan || std::isnan(point[iprop]);
                  any_nonzero = any_nonzero || point[iprop] != 0;
                }
              });

          const std::string label = specfem::element::to_string(medium) + " " +
                                    specfem::element::to_string(property) +
                                    " " +
                                    specfem::element::to_string(attenuation) +
                                    " in test " + test.name;
          EXPECT_FALSE(any_nan) << "NaN kernels for " << label;
          EXPECT_TRUE(any_nonzero) << "all kernels zero for " << label;
        });
  }
}
