#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/globe/model_config.hpp"
#include "specfem/globe/planet_constants.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace specfem::mesh {

/**
 * @brief Per-element context required by the SPECFEM3D_GLOBE model evaluator.
 *
 * The thin globe mesh stores enough metadata to re-evaluate material properties
 * at assembly time. These fields are passed to the Fortran-backed evaluator for
 * each spectral element together with the element's reference coordinates.
 */
struct globe_element_context {
  /** @brief Radial region of the element; always set by the reader. */
  specfem::element::region_tag region =
      specfem::element::region_tag::crust_mantle;

  /** @brief SPECFEM3D_GLOBE radial doubling flag for the element. */
  int idoubling = 0;

  /** @brief Lower radius of the element's radial shell, in metres. */
  double rmin = 0.0;

  /** @brief Upper radius of the element's radial shell, in metres. */
  double rmax = 0.0;

  /** @brief True when the element intersects the crustal model region. */
  bool element_in_crust = false;

  /** @brief True when the element intersects the mantle model region. */
  bool element_in_mantle = false;
};

/**
 * @brief Boundary surface represented as element/face pairs.
 *
 * The globe database records several named surfaces independently, such as the
 * free surface, CMB, ICB, and ocean-load surface. Element indices are stored
 * zero-based after reading.
 */
struct globe_boundary_surface {
  /** @brief Zero-based mesh element index for each boundary face. */
  std::vector<int> elements;

  /** @brief Local face identifier for each boundary face. */
  std::vector<specfem::mesh_entity::dim3::type> faces;
};

/**
 * @brief Globe-specific raw mesh payload retained for assembly setup.
 *
 * @c specfem::mesh::globe3d_mesh shares the standard 3-D raw mesh fields
 * through @c mesh_dim3_base, but the thin globe database also carries metadata
 * needed to evaluate pointwise material properties through the
 * SPECFEM3D_GLOBE model evaluator. This structure stores that globe-only
 * payload.
 */
struct globe_mesh_data {
  /** @brief Host view of xyz coordinates indexed by global anchor node. */
  using CoordinatesViewType =
      Kokkos::View<type_real *[3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

  /** @brief Thin globe database format version. */
  int format_version = 0;

  /** @brief Number of radial/material regions in the globe model. */
  int nregions = 0;

  /** @brief True when full-gravity terms were enabled in the globe mesher. */
  bool full_gravity = false;

  /**
   * @brief True when the database stores separate reference coordinates.
   *
   * If false, @ref reference_coordinates is initialized from the physical
   * control-node coordinates.
   */
  bool has_reference_geometry = false;

  /** @brief Encoded globe material mode; currently only oracle mode is read. */
  int material_mode = 0;

  /** @brief Resolved planet metadata written by the globe mesher. */
  specfem::globe::PlanetConstants planet_constants;

  /** @brief Configuration used to initialize the globe model evaluator. */
  specfem::globe::ModelConfig model_config;

  /**
   * @brief Reference xyz coordinates indexed by global anchor node.
   *
   * These coordinates are interpolated to GLL points before calling the globe
   * material evaluator.
   */
  CoordinatesViewType reference_coordinates;

  /** @brief Per-element metadata required by the globe material evaluator. */
  std::vector<globe_element_context> element_context;

  /** @brief Surface entries belonging to the exterior acoustic free surface. */
  globe_boundary_surface free_surface;

  /** @brief Surface entries on the core-mantle boundary. */
  globe_boundary_surface cmb;

  /** @brief Surface entries on the inner-core boundary. */
  globe_boundary_surface icb;

  /** @brief Surface entries used for ocean-load metadata. */
  globe_boundary_surface ocean_load;
};

} // namespace specfem::mesh
