
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

template<specfem::element::dimension_tag DimensionTag>
void specfem::assembly::sources_impl::locate_sources(
    const specfem::assembly::element_types<DimensionTag>
        &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<
        specfem::sources::source<DimensionTag> > > &sources) {

  // Loop over all sources
  for (auto &source : sources) {

    // Get the source coordinates
    const auto &coord = source->get_global_coordinates();

    // TODO: In MPI runs each partition holds only a subset of the mesh.
    // Sources outside this partition's subdomain will fail to locate here.
    // TODO: Implement proper cross-partition source location so that the
    // owning rank computes and broadcasts the local coordinates.
    specfem::point::local_coordinates<DimensionTag> lcoord;
    try {
      lcoord = specfem::algorithms::locate_point(coord, mesh);
    } catch (const std::exception &) {
      if (specfem::MPI::get_size() > 1) {
        continue;
      }
      throw;
    }

    // Set the local coordinates and global element index in the source
    if (lcoord.ispec < 0) {
      if (specfem::MPI::get_size() > 1) {
        continue;
      }
      throw std::runtime_error("Source is outside of the domain");
    }

    // Giving the local coordinates and global element index to the source
    source->set_local_coordinates(lcoord);

    // Given the spectral element index provide the medium tag
    source->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));
  }
}
