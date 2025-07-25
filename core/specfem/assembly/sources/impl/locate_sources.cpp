
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "algorithms/locate_point.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

void specfem::assembly::sources_impl::locate_sources(
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > > &sources) {

  // Loop over all sources
  for (auto &source : sources) {

    // Get the source coordinates
    const type_real x = source->get_x();
    const type_real z = source->get_z();

    // Get element that that source is located in, as well as the local
    // coordinates
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        coord(x, z);
    const auto lcoord = specfem::algorithms::locate_point(coord, mesh);

    // Set the local coordinates and global element index in the source
    if (lcoord.ispec < 0) {
      throw std::runtime_error("Source is outside of the domain");
    }

    // Giving the local coordinates and global element index to the source
    source->set_element_index(lcoord.ispec);
    source->set_xi(lcoord.xi);
    source->set_gamma(lcoord.gamma);

    // Given the spectral element index provide the medium tag
    source->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));
  }
}
