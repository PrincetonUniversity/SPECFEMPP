#pragma once

#include "specfem/globe_model/model_config.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace specfem::mesh {

/** @brief Values retained only to detect mesher/evaluator catalog skew. */
struct globe_model_verification {
  std::vector<int> codes;
  std::vector<bool> flags;
  double attenuation_source_frequency = 0.0;
};

/** @brief Context needed to evaluate one globe element in the model oracle. */
struct globe_element_context {
  int region = 0;
  int idoubling = 0;
  double rmin = 0.0;
  double rmax = 0.0;
  bool element_in_crust = false;
  bool element_in_mantle = false;
};

/** @brief One named surface from the thin globe database. */
struct globe_boundary_surface {
  std::vector<int> elements;
  std::vector<specfem::mesh_entity::dim3::type> faces;
};

/** @brief Raw anchor-node interface to one neighboring MPI rank. */
struct globe_mpi_interface {
  int neighbor_rank = -1;
  std::vector<int> node_ids;
};

/** @brief Globe-only mesh payload retained until assembly property setup. */
struct globe_mesh_data {
  using CoordinatesViewType =
      Kokkos::View<type_real *[3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

  int format_version = 0;
  double planet_radius = 0.0;
  double average_density = 0.0;
  int nregions = 0;
  bool full_gravity = false;
  bool has_reference_geometry = false;
  int material_mode = 0;

  specfem::globe_model::ModelConfig model_config;
  globe_model_verification model_verification;
  CoordinatesViewType reference_coordinates;
  std::vector<globe_element_context> element_context;
  globe_boundary_surface free_surface;
  globe_boundary_surface cmb;
  globe_boundary_surface icb;
  globe_boundary_surface ocean_load;
  std::vector<globe_mpi_interface> mpi_interfaces;
};

} // namespace specfem::mesh
