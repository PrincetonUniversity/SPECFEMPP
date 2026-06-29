
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/location_result.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

#include <map>
#include <stdexcept>
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

  // Diagnostic location records, keyed by source index (a map keeps the index
  // aligned with isrc). Discarded once this function returns.
  std::map<int, specfem::assembly::LocationResult<DimensionTag> >
      source_location_results;

  // Resolve any generic coordinates to global coordinates using mesh context.
  // Sources constructed with (x,y,z) directly have no read_coordinates_ set, so
  // they take the direct path and produce a reduced (global-only) record.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (auto *coords = sources[isrc]->get_read_coordinates()) {
      auto resolution = specfem::assembly::resolve_coordinates(
          *coords, mesh, surface, utm_config);
      source_location_results.try_emplace(isrc, coords, resolution);
      sources[isrc]->set_global_coordinates(resolution.global);
    } else {
      source_location_results.try_emplace(
          isrc, sources[isrc]->get_global_coordinates());
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
  auto [lcoords, partition_index_selected] =
      specfem::algorithms::locate_point(coords, mesh);

  // Assign local coordinates and medium tags to each source.
  // Only the owning rank sets valid coordinates; all others receive ispec = -1
  // so that downstream filtering skips them.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    sources[isrc]->set_partition_index(partition_index_selected[isrc]);

    if (partition_index_selected[isrc] == myrank) {
      const auto &lcoord = lcoords[isrc];
      sources[isrc]->set_local_coordinates(lcoord);
      sources[isrc]->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));
      if (auto it = source_location_results.find(isrc);
          it != source_location_results.end()) {
        it->second.set_result(sources[isrc]->get_global_coordinates(),
                              sources[isrc]->get_local_coordinates(),
                              sources[isrc]->get_partition_index(),
                              sources[isrc]->get_medium_tag());
      }
    } else {
      specfem::point::local_coordinates<DimensionTag> invalid;
      invalid.ispec = -1;
      sources[isrc]->set_local_coordinates(invalid);
    }
  }

  specfem::assembly::log_location_results(source_location_results, "Source");
}
