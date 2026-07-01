
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

#include <stdexcept>
#include <string>
#include <vector>


template <specfem::element::dimension_tag DimensionTag>
void specfem::assembly::sources_impl::locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources,
    const specfem::mesh::acoustic_free_surface<DimensionTag> &surface,
    const std::optional<specfem::coordinate_systems::utm_projection_config>
        &utm_config) {

  const int nsources = static_cast<int>(sources.size());
  const int myrank = specfem::MPI::get_rank();

  // Resolve any generic coordinates to global coordinates using mesh context,
  // storing the resolution result on the source. Sources constructed with
  // (x,y,z) directly have no read_coordinates_ set and keep their global
  // coordinates unchanged.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (auto *coords = sources[isrc]->get_read_coordinates()) {
      auto resolution = specfem::assembly::resolve_coordinates(
          *coords, mesh, surface, utm_config);
      sources[isrc]->set_resolution_result(resolution);
      sources[isrc]->set_global_coordinates(resolution.global);
    }
  }

  // Collect global coordinates for all sources.
  std::vector<specfem::point::global_coordinates<DimensionTag>> coords;
  coords.reserve(nsources);
  for (int isrc = 0; isrc < nsources; ++isrc)
    coords.push_back(sources[isrc]->get_global_coordinates());

  // Locate all sources across MPI partitions with inside-preference.
  // The function prefers elements where xi/eta/gamma ∈ [-1,1] and falls back
  // to minimum Cartesian distance when no rank owns an inside element.
  const auto located = specfem::algorithms::locate_point(coords, mesh);

  // Assign each source's location. The partition index and location error are
  // replicated on every rank; local coordinates and medium are set only on the
  // owning rank (ispec = -1 elsewhere so downstream filtering skips them).
  for (int isrc = 0; isrc < nsources; ++isrc) {
    sources[isrc]->set_partition_index(located.partition_index[isrc]);
    sources[isrc]->set_location_error(located.error[isrc]);

    if (located.partition_index[isrc] == myrank) {
      const auto &lcoord = located.local[isrc];
      sources[isrc]->set_local_coordinates(lcoord);
      sources[isrc]->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));

      // Warn when the recovered local coordinates land outside the reference
      // element beyond a small tolerance; coordinate resolution need not be
      // exact, but a large excursion signals a mislocated source.
      if (lcoord.outside(type_real(1.001))) {
        specfem::Logger::warning(
            "Source " + std::to_string(isrc) + " (" +
                sources[isrc]->source_name() +
                ") located outside its element: " + lcoord.print(),
            /*root_only=*/false);
      }
    } else {
      specfem::point::local_coordinates<DimensionTag> invalid;
      invalid.ispec = -1;
      sources[isrc]->set_local_coordinates(invalid);
    }
  }
}
