
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

#include <stdexcept>
#include <vector>

template <specfem::element::dimension_tag DimensionTag>
void specfem::assembly::sources_impl::locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources) {

  const int nsources = static_cast<int>(sources.size());
  const int myrank = specfem::MPI::get_rank();

  // Collect global coordinates for all sources.
  std::vector<specfem::point::global_coordinates<DimensionTag>> coords;
  coords.reserve(nsources);
  for (int isrc = 0; isrc < nsources; ++isrc)
    coords.push_back(sources[isrc]->get_global_coordinates());

  // Locate all sources across MPI partitions with inside-preference.
  // The function prefers elements where xi/eta/gamma ∈ [-1,1] and falls back
  // to minimum Cartesian distance when no rank owns an inside element.
  auto [lcoords, islice_selected] =
      specfem::algorithms::locate_point(coords, mesh);

  // Assign local coordinates and medium tags to each source.
  // Only the owning rank sets valid coordinates; all others receive ispec = -1
  // so that downstream filtering skips them.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    sources[isrc]->set_islice(islice_selected[isrc]);
    if (islice_selected[isrc] == myrank) {
      const auto &lcoord = lcoords[isrc];
      sources[isrc]->set_local_coordinates(lcoord);
      sources[isrc]->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));
    } else {
      specfem::point::local_coordinates<DimensionTag> invalid;
      invalid.ispec = -1;
      sources[isrc]->set_local_coordinates(invalid);
    }
  }
}
