#pragma once

#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/point/global_coordinates.hpp"
#include "specfem/point/local_coordinates.hpp"

#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace specfem {
namespace assembly {

/**
 * @brief Transient diagnostic record for a single located source or receiver.
 *
 * Captures where a point was requested (input/resolved coordinates), where it
 * was found in the mesh, and the owning partition/medium, so the information
 * can be printed for debugging and then discarded. Built at the end of source
 * and receiver location; not stored on the assembly objects.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> struct LocationResult {

  constexpr static auto dimension_tag = DimensionTag;

  /**
   * @brief Construct from a resolved coordinate (dim3 sources/receivers).
   *
   * @param input Non-owning pointer to the point's polymorphic input
   * coordinates (the owner outlives this record); may be nullptr.
   * @param resolution Result of @ref resolve_coordinates (global + topography).
   */
  LocationResult(
      const specfem::coordinate_systems::coordinates<dimension_tag> *input,
      const specfem::assembly::CoordinateResolutionResult<dimension_tag>
          &resolution)
      : input_(input), target_(resolution.global),
        found_topography_(resolution.topography) {}

  /**
   * @brief Construct from a direct global coordinate (no resolution step).
   *
   * Used for points specified directly as (x, y, z) and for dim2, which has no
   * generic-coordinate resolution.
   *
   * @param target The point's global coordinates in mesh space.
   */
  explicit LocationResult(
      const specfem::point::global_coordinates<dimension_tag> &target)
      : target_(target) {}

  /**
   * @brief Record where the point was located in the mesh.
   *
   * @param result Located global coordinates.
   * @param local Located reference-element coordinates (ispec, xi, ...).
   * @param partition_index MPI rank owning the located element.
   * @param medium_tag Medium of the located element.
   */
  void
  set_result(const specfem::point::global_coordinates<dimension_tag> &result,
             const specfem::point::local_coordinates<dimension_tag> &local,
             int partition_index, specfem::element::medium_tag medium_tag) {
    result_ = result;
    local_ = local;
    partition_index_ = partition_index;
    medium_tag_ = medium_tag;
    located_ = true;
  }

  /// @brief Whether set_result() has populated the located fields.
  bool located() const { return located_; }

  /**
   * @brief Owner-specific fields of a located point.
   *
   * The input/resolved/topography fields are replicated on every rank, so only
   * these need to be communicated to gather a complete record on the root rank.
   * Trivially copyable for transmission as raw bytes.
   */
  struct LocatedData {
    specfem::point::local_coordinates<dimension_tag> local;
    specfem::element::medium_tag medium = {};
    int partition_index = -1;
  };

  /// @brief Extract the owner-specific located fields (valid if located()).
  LocatedData located_data() const {
    return { local_, medium_tag_, partition_index_ };
  }

  /**
   * @brief Re-attach owner-specific fields received from the owning rank.
   *
   * Completes a record that already holds the replicated input/resolved
   * coordinates, so the root rank can print points it does not own.
   */
  void set_located_data(const LocatedData &data) {
    local_ = data.local;
    medium_tag_ = data.medium;
    partition_index_ = data.partition_index;
    result_ = target_; // located global coincides with the resolved global
    located_ = true;
  }

  std::string print_distance() const {
    auto distance = specfem::point::distance(target_, result_);
    // Format the (meters) distance as km, m, and mm for readability.
    int km = static_cast<int>(distance / 1000);
    int m = static_cast<int>(distance) % 1000;
    int mm = static_cast<int>(distance * 1000) % 1000;
    return std::to_string(km) + " km, " + std::to_string(m) + " m, " +
           std::to_string(mm) + " mm";
  }

  std::string print() const {
    constexpr int label_width = 22;
    const auto field = [](const std::string &label) {
      std::ostringstream oss;
      oss << "    " << std::left << std::setw(label_width) << (label + ":");
      return oss.str();
    };

    std::ostringstream oss;
    if (input_ != nullptr)
      oss << field("Input coordinates") << input_->print() << "\n";
    if (found_topography_.has_value())
      oss << field("Found topography") << *found_topography_ << " m\n";
    oss << field("Resolved global") << target_.print() << "\n";
    oss << field("Found global") << result_.print() << "\n";
    oss << field("Local coordinates") << local_.print() << "\n";
    oss << field("Partition index") << partition_index_ << "\n";
    oss << field("Medium tag") << specfem::element::to_string(medium_tag_)
        << "\n";
    oss << field("Target-found distance") << print_distance() << "\n";
    return oss.str();
  }

private:
  const specfem::coordinate_systems::coordinates<dimension_tag> *input_ =
      nullptr;
  specfem::point::global_coordinates<dimension_tag> target_;
  std::optional<type_real> found_topography_;
  specfem::point::global_coordinates<dimension_tag> result_;
  specfem::point::local_coordinates<dimension_tag> local_;
  int partition_index_ = -1;
  // No sentinel exists in medium_tag; value-initialized and only meaningful
  // once set_result() has populated it on the owning rank.
  specfem::element::medium_tag medium_tag_ = {};
  bool located_ = false;
};

/**
 * @brief Print all located results in global index order from the root rank.
 *
 * Each rank owns a disjoint subset of the points, so per-rank printing
 * interleaves nondeterministically at the terminal. To get deterministic
 * index-ordered output, every rank's located records are gathered to root,
 * which prints them sorted by index. The `std::map` is already index-sorted,
 * so the serial path just prints in iteration order.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param results Per-index location results (keyed by source/receiver index).
 * @param kind Human-readable label, e.g. "Source" or "Receiver".
 */
template <specfem::element::dimension_tag DimensionTag>
void log_location_results(
    const std::map<int, LocationResult<DimensionTag>> &results,
    const std::string &kind) {
  const int nchar = kind.size();
  const auto emit = [&kind](int index, const std::string &body) {
    specfem::Logger::debug("- " + kind + " " + std::to_string(index) + ":\n" +
                           body);
  };

  specfem::Logger::debug(kind + " Location Results:\n" +
                         std::string(nchar + 18, '-') + "\n");
#ifdef SPECFEM_ENABLE_MPI
  // The input/resolved/topography fields are replicated on every rank; only the
  // owner-specific fields differ. Each rank therefore sends just its owned
  // records as compact fixed-size PODs to root, which already holds the
  // replicated fields and prints the complete records in index order.
  struct Packed {
    int index;
    typename LocationResult<DimensionTag>::LocatedData data;
  };

  std::vector<Packed> local;
  for (const auto &[index, result] : results)
    if (result.located())
      local.push_back({ index, result.located_data() });

  const MPI_Comm comm = specfem::MPI::communicator();
  const int size = specfem::MPI::get_size();
  const int rank = specfem::MPI::get_rank();
  const int local_bytes = static_cast<int>(local.size() * sizeof(Packed));

  std::vector<int> byte_counts(rank == 0 ? size : 0);
  SPECFEM_MPI_SAFECALL(MPI_Gather(
      &local_bytes, 1, MPI_INT, rank == 0 ? byte_counts.data() : nullptr, 1,
      MPI_INT, 0, comm));
  std::vector<int> byte_displs(rank == 0 ? size : 0);
  std::vector<Packed> gathered;
  if (rank == 0) {
    int total = 0;
    for (int r = 0; r < size; ++r) {
      byte_displs[r] = total;
      total += byte_counts[r];
    }
    gathered.resize(total / sizeof(Packed));
  }
  SPECFEM_MPI_SAFECALL(MPI_Gatherv(
      local.data(), local_bytes, MPI_BYTE,
      rank == 0 ? gathered.data() : nullptr,
      rank == 0 ? byte_counts.data() : nullptr,
      rank == 0 ? byte_displs.data() : nullptr, MPI_BYTE, 0, comm));

  if (rank == 0) {
    // Complete each record from the replicated map and print in index order.
    std::map<int, std::string> ordered;
    for (const auto &packed : gathered) {
      auto it = results.find(packed.index);
      if (it == results.end())
        continue;
      LocationResult<DimensionTag> record = it->second;
      record.set_located_data(packed.data);
      ordered.emplace(packed.index, record.print());
    }
    for (const auto &[index, body] : ordered)
      emit(index, body);
  }
#else
  for (const auto &[index, result] : results)
    if (result.located())
      emit(index, result.print());
#endif
  specfem::Logger::debug("\n");
}

} // namespace assembly
} // namespace specfem
