#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/element_coupling/tags.hpp"
#include <vector>

namespace specfem::assembly::nonconforming_interfaces_impl {

/** @brief Host selection of one interface container */
struct HostSelection {
  /// Interface tag of the container
  specfem::element_coupling::interface_tag interface_tag = {};
  /// Conjugate of `interface_tag`
  specfem::element_coupling::interface_tag conjugate_interface_tag = {};
  /// Boundary tag of the container
  specfem::element::boundary_tag boundary_tag = {};
  /// Number of pairs stored
  int num_faces = 0;
  /// Whether the self side is the host
  bool self_is_host = false;
};

/**
 * @brief Throws unless every pair is integrated on exactly one side.
 *
 * The host is selected per container. That selection must agree within one
 * interface tag, and exactly one of an interface tag and its conjugate must
 * be the host (under `self_host` both are). Empty containers are ignored.
 *
 * Throws `std::runtime_error` when containers with pairs of one interface tag
 * disagree on `self_is_host`, or, unless `interfacial_meshing_type` is
 * `self_host`, when an interface tag and its conjugate are both host or both
 * non-host. The message names the meshing type and, for the two offending
 * containers, the interface tag, boundary tag, pair count and host flag.
 *
 * @param selections Host selection of every container
 * @param interfacial_meshing_type Meshing type used to select the hosts
 */
void check_host_selection(
    const std::vector<HostSelection> &selections,
    const specfem::element_coupling::interfacial_meshing_type
        &interfacial_meshing_type);

} // namespace specfem::assembly::nonconforming_interfaces_impl
