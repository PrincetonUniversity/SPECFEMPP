
#include "specfem/assembly/sources/impl/locate_sources.hpp"
#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/source.hpp"

#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


template <specfem::element::dimension_tag DimensionTag>
struct specfem::assembly::sources_impl::SourceLocationResult {

  constexpr static auto dimension_tag = DimensionTag;

  // Non-owning pointer to the source's polymorphic coordinates so we can call
  // its virtual print(). The source retains ownership and outlives this struct.
  SourceLocationResult(
      const specfem::coordinate_systems::coordinates<dimension_tag> *input,
      const specfem::assembly::CoordinateResolutionResult<dimension_tag>
          &resolution)
      : input(input), target(resolution.global),
        found_topography(resolution.topography) {}

  void set_result(
      const specfem::point::global_coordinates<dimension_tag>
          &result,
      const specfem::point::local_coordinates<dimension_tag>
          &local,
      int partition_index_selected,
      specfem::element::medium_tag medium_tag) {
    this->result = result;
    this->local = local;
    this->partition_index_selected = partition_index_selected;
    this->medium_tag = medium_tag;
  }

  std::string print_distance() const {
    auto distance = specfem::point::distance(target, result);
    // The distance is in meters, let's format it to km, m, and mm for better readability
    int km = static_cast<int>(distance / 1000);
    int m = static_cast<int>(distance) % 1000;
    int mm = static_cast<int>(distance * 1000) % 1000;
    return std::to_string(km) + " km, " + std::to_string(m) + " m, " + std::to_string(mm) + " mm";
  }

  std::string print() const {
    constexpr int label_width = 22;
    const auto field = [](const std::string &label) {
      std::ostringstream oss;
      oss << "  " << std::left << std::setw(label_width) << (label + ":");
      return oss.str();
    };

    std::ostringstream oss;
    oss << "Source Location Result\n";
    oss << field("Input coordinates") << input->print() << "\n";
    if (found_topography.has_value()) {
      oss << field("Found topography") << *found_topography << " m\n";
    }
    oss << field("Resolved global") << target.print() << "\n";
    oss << field("Found global") << result.print() << "\n";
    oss << field("Local coordinates") << local.print() << "\n";
    oss << field("Partition index") << partition_index_selected << "\n";
    oss << field("Medium tag") << specfem::element::to_string(medium_tag)
        << "\n";
    oss << field("Target-found distance") << print_distance() << "\n";
    return oss.str();
  }

private:
  const specfem::coordinate_systems::coordinates<dimension_tag> *input = nullptr;
  specfem::point::global_coordinates<dimension_tag> target;
  std::optional<type_real> found_topography;
  specfem::point::global_coordinates<dimension_tag> result;
  specfem::point::local_coordinates<dimension_tag> local;
  int partition_index_selected = -1;
  // No sentinel exists in medium_tag; value-initialized and only read after
  // set_result() has populated it on the owning rank.
  specfem::element::medium_tag medium_tag = {};
};

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

  // Keyed by source index. Only sources whose coordinates needed resolving get
  // an entry, so a map (not a vector) keeps the index aligned with isrc.
  std::map<int, specfem::assembly::sources_impl::SourceLocationResult<DimensionTag> >
      source_location_results;

  // Resolve any generic coordinates to global coordinates using mesh context.
  // Sources constructed with (x,y,z) directly have no read_coordinates_ set,
  // so this is a no-op for them.
  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (auto *coords = sources[isrc]->get_read_coordinates()) {
      auto resolution = specfem::assembly::resolve_coordinates(
          *coords, mesh, surface, utm_config);
      source_location_results.try_emplace(isrc, coords, resolution);
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

  for (int isrc = 0; isrc < nsources; ++isrc) {
    if (partition_index_selected[isrc] == myrank) {
      if (auto it = source_location_results.find(isrc);
          it != source_location_results.end()) {
        specfem::Logger::debug("Source " + std::to_string(isrc) +
                               " location result:\n" + it->second.print());
      }
    }
    specfem::MPI::sync();
  }
}
