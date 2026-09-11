#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/element.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace globe_context_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using ElementTypes = specfem::assembly::element_types<dimension>;

const std::string database_path =
    "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
    "proc000000_specfempp_database.bin";

struct GlobeFixture {
  specfem::mesh::globe3d_mesh mesh;
  specfem::assembly::mesh<dimension> assembly_mesh;

  GlobeFixture()
      : mesh(specfem::io::read_globe_mesh(database_path,
                                          specfem::attenuation::Setup{})) {
    specfem::quadrature::gll::gll gll{};
    const specfem::quadrature::quadratures quadrature(gll);
    assembly_mesh = { mesh.nspec,
                      mesh.control_nodes.ngnod,
                      mesh.element_grid.ngllz,
                      mesh.element_grid.nglly,
                      mesh.element_grid.ngllx,
                      mesh.tags,
                      mesh.adjacency_graph,
                      mesh.control_nodes,
                      quadrature };
  }

  ElementTypes element_types() const {
    return { mesh.nspec, assembly_mesh.element_grid, assembly_mesh, mesh.tags,
             mesh.globe.element_context };
  }
};

} // namespace globe_context_test_impl

// One test: building assembly::mesh for the globe fixture dominates the
// run time, so all checks share a single construction.
TEST(GlobeElementContext, CarriesMesherContext) {
  using specfem::element::medium_tag;
  using specfem::element::region_tag;
  const globe_context_test_impl::GlobeFixture fixture;
  const auto element_types = fixture.element_types();
  const int nspec = fixture.mesh.nspec;

  {
    SCOPED_TRACE("views are filled in compute order");
    ASSERT_TRUE(element_types.has_element_context());
    EXPECT_EQ(element_types.regions.extent(0), nspec);
    EXPECT_EQ(element_types.idoubling.extent(0), nspec);
    EXPECT_EQ(element_types.rmin.extent(0), nspec);
    EXPECT_EQ(element_types.rmax.extent(0), nspec);
    EXPECT_EQ(element_types.elem_in_crust.extent(0), nspec);
    EXPECT_EQ(element_types.elem_in_mantle.extent(0), nspec);

    const auto &context = fixture.mesh.globe.element_context;
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const auto &expected =
          context[fixture.assembly_mesh.h_compute_to_mesh(ispec)];
      EXPECT_EQ(element_types.get_region_tag(ispec), expected.region);
      EXPECT_EQ(element_types.idoubling(ispec), expected.idoubling);
      EXPECT_EQ(element_types.rmin(ispec), expected.rmin);
      EXPECT_EQ(element_types.rmax(ispec), expected.rmax);
      EXPECT_EQ(element_types.elem_in_crust(ispec), expected.element_in_crust);
      EXPECT_EQ(element_types.elem_in_mantle(ispec),
                expected.element_in_mantle);
    }
  }

  {
    SCOPED_TRACE("region counts cover the mesh");
    int total = 0;
    for (const auto region : { region_tag::crust_mantle, region_tag::outer_core,
                               region_tag::inner_core }) {
      const int count = element_types.get_number_of_elements(region);
      EXPECT_GT(count, 0) << specfem::element::to_string(region);
      int manual = 0;
      for (int ispec = 0; ispec < nspec; ++ispec) {
        if (element_types.get_region_tag(ispec) == region) {
          ++manual;
        }
      }
      EXPECT_EQ(count, manual) << specfem::element::to_string(region);
      total += count;
    }
    EXPECT_EQ(total, nspec);
  }

  {
    SCOPED_TRACE("medium matches region");
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const auto region = element_types.get_region_tag(ispec);
      const auto expected_medium = (region == region_tag::outer_core)
                                       ? medium_tag::acoustic
                                       : medium_tag::elastic;
      ASSERT_EQ(element_types.get_medium_tag(ispec), expected_medium)
          << "element " << ispec << " in region "
          << specfem::element::to_string(region);
    }
  }

  {
    SCOPED_TRACE("radial shells are ordered and inside the planet");
    const double r_planet = fixture.mesh.globe.planet_radius;
    ASSERT_GT(r_planet, 0.0);
    for (int ispec = 0; ispec < nspec; ++ispec) {
      const double rmin = element_types.rmin(ispec);
      const double rmax = element_types.rmax(ispec);
      ASSERT_GT(rmin, 0.0) << "element " << ispec;
      ASSERT_LT(rmin, rmax) << "element " << ispec;
      ASSERT_LE(rmax, r_planet) << "element " << ispec;
      ASSERT_GT(element_types.idoubling(ispec), 0) << "element " << ispec;
    }
  }

  {
    SCOPED_TRACE("context of the wrong length is rejected");
    std::vector<specfem::mesh::globe_element_context> short_context(
        fixture.mesh.globe.element_context.begin(),
        fixture.mesh.globe.element_context.end() - 1);
    EXPECT_THROW(globe_context_test_impl::ElementTypes(
                     nspec, fixture.assembly_mesh.element_grid,
                     fixture.assembly_mesh, fixture.mesh.tags, short_context),
                 std::runtime_error);
  }
}
