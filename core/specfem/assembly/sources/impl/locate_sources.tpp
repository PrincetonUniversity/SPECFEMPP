
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

#include <limits>
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

  // Per-source local best-fit distances and element coordinates.
  // Initialized as invalid: distance = max, ispec = -1.
  std::vector<type_real> local_dists(nsources,
                                     std::numeric_limits<type_real>::max());
  std::vector<specfem::point::local_coordinates<DimensionTag> > local_coords(
      nsources);
  for (int isrc = 0; isrc < nsources; ++isrc) {
    local_coords[isrc].ispec = -1;
  }

  // Step 1: Each rank tries to locate every source in its local mesh
  // partition. Sources outside the partition cause locate_point to throw; we
  // catch those and leave the corresponding entry invalid.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    const auto &coord = sources[isrc]->get_global_coordinates();

    try {
      const auto lcoord = specfem::algorithms::locate_point(coord, mesh);

      // Back-project to global coordinates to measure the location misfit.
      const auto found_global =
          specfem::algorithms::locate_point(lcoord, mesh);
      local_dists[isrc] = specfem::point::distance(coord, found_global);
      local_coords[isrc] = lcoord;
    } catch (const std::exception &) {
      // Source lies outside this rank's mesh partition; leave entry invalid.
    }
  }

  // Step 2: allreduce(min) so every rank learns the global minimum distance
  // for each source.
  std::vector<type_real> global_dists = local_dists;
  specfem::MPI::allreduce(global_dists.data(), nsources, specfem::min);

  // Step 3: Determine the owning rank for each source.
  // A rank claims ownership when its local distance matches the global
  // minimum. Ties are resolved by allreduce(max), which selects the
  // highest-numbered tied rank (matches reference SPECFEM behavior).
  std::vector<int> islice_selected(nsources, -1);
  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (local_dists[isrc] <= global_dists[isrc]) {
      islice_selected[isrc] = myrank;
    }
  }
  specfem::MPI::allreduce(islice_selected.data(), nsources, specfem::max);

  // Sanity check: every source must have been claimed by at least one rank.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (islice_selected[isrc] < 0) {
      throw std::runtime_error(
          "Source " + std::to_string(isrc) +
          " could not be located in any MPI partition");
    }
  }

  // Step 4: Assign local coordinates and medium tags.
  // Only the owning rank sets valid coordinates; all other ranks receive an
  // invalid entry (ispec = -1) so that sort_sources_per_medium ignores them.
  // islice_selected is identical on all ranks after allreduce, so every rank
  // can record which rank owns each source for informational purposes.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    sources[isrc]->set_islice(islice_selected[isrc]);
    if (islice_selected[isrc] == myrank) {
      const auto &lcoord = local_coords[isrc];
      sources[isrc]->set_local_coordinates(lcoord);
      sources[isrc]->set_medium_tag(element_types.get_medium_tag(lcoord.ispec));
    } else {
      // Mark as non-local so downstream filtering skips this source.
      specfem::point::local_coordinates<DimensionTag> invalid;
      invalid.ispec = -1;
      sources[isrc]->set_local_coordinates(invalid);
    }
  }
}
