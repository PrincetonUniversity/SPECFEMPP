#pragma once

#include "specfem/globe_model/model_config.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

namespace specfem::mesh {

/**
 * @brief Model-catalog values written by the globe mesher for consistency
 * checks.
 *
 * SPECFEM3D_GLOBE material properties are evaluated later through the globe
 * model oracle rather than stored directly in the thin mesh database. These
 * values identify the model catalog state used by the mesher so the C++ reader
 * can detect obvious mismatches with the linked evaluator implementation.
 */
struct globe_model_verification {
  /**
   * @brief Verification-only Fortran model codes derived from @c MODEL.
   *
   * Entries are written by SPECFEM3D_GLOBE in this order:
   * - @c REFERENCE_1D_MODEL
   * - @c THREE_D_MODEL
   * - @c THREE_D_MODEL_IC
   * - @c REFERENCE_CRUSTAL_MODEL
   * - @c MODEL_GLL_TYPE
   *
   * These are raw SPECFEM3D_GLOBE setup/constants values, not stable
   * SPECFEM++ encodings. They are retained to detect mesher/evaluator catalog
   * skew, not replayed into the evaluator.
   */
  std::vector<int> codes;

  /**
   * @brief Verification-only Fortran model flags derived from @c MODEL.
   *
   * Entries are written by SPECFEM3D_GLOBE in this order:
   * - @c TRANSVERSE_ISOTROPY
   * - @c CRUSTAL
   * - @c ONE_CRUST
   * - @c CASE_3D
   * - @c ANISOTROPIC_3D_MANTLE
   * - @c ANISOTROPIC_INNER_CORE
   * - @c MODEL_3D_MANTLE_PERTUBATIONS
   * - @c HETEROGEN_3D_MANTLE
   * - @c ATTENUATION_3D
   * - @c ATTENUATION_3D_BERKELEY
   * - @c ATTENUATION_GLL
   * - @c HONOR_1D_SPHERICAL_MOHO
   * - @c MODEL_GLL
   * - @c USE_FULL_TISO_MANTLE
   * - @c REGIONAL_MOHO_MESH
   * - @c EMC_MODEL
   *
   * The linked evaluator derives its own flags from @c model_config.model_name;
   * these stored values are only for consistency checking.
   */
  std::vector<bool> flags;

  /**
   * @brief Attenuation source frequency used by the mesher.
   *
   * This is checked against the logarithmic center of the stored attenuation
   * period band when attenuation is enabled.
   */
  double attenuation_source_frequency = 0.0;
};

/**
 * @brief Per-element context required by the SPECFEM3D_GLOBE model oracle.
 *
 * The thin globe mesh stores enough metadata to re-evaluate material properties
 * at assembly time. These fields are passed to the Fortran-backed evaluator for
 * each spectral element together with the element's reference coordinates.
 */
struct globe_element_context {
  /** @brief Globe radial region index for the element. */
  int region = 0;

  /** @brief SPECFEM3D_GLOBE radial doubling flag for the element. */
  int idoubling = 0;

  /** @brief Minimum nondimensional radius represented by the element. */
  double rmin = 0.0;

  /** @brief Maximum nondimensional radius represented by the element. */
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
 * @brief Raw anchor-node interface to one neighboring MPI rank.
 *
 * SPECFEM3D_GLOBE stores MPI interfaces as the set of shared anchor nodes with
 * a neighbor rank. The reader reconstructs element-level MPI adjacency from
 * these node sets after the control-node connectivity has been read.
 */
struct globe_mpi_interface {
  /** @brief Neighbor MPI rank sharing this interface. */
  int neighbor_rank = -1;

  /** @brief Zero-based global anchor-node ids shared with @ref neighbor_rank.
   */
  std::vector<int> node_ids;
};

/**
 * @brief Globe-specific raw mesh payload retained for assembly setup.
 *
 * @c specfem::mesh::globe3d_mesh shares the standard 3-D raw mesh fields
 * through @c mesh_dim3_base, but the thin globe database also carries metadata
 * needed to evaluate pointwise material properties through the
 * SPECFEM3D_GLOBE model oracle. This structure stores that globe-only payload.
 */
struct globe_mesh_data {
  /** @brief Host view of xyz coordinates indexed by global anchor node. */
  using CoordinatesViewType =
      Kokkos::View<type_real *[3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

  /** @brief Thin globe database format version. */
  int format_version = 0;

  /** @brief Planet radius used by the mesher and evaluator length scale. */
  double planet_radius = 0.0;

  /** @brief Average density used by the mesher and evaluator density scale. */
  double average_density = 0.0;

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

  /** @brief Configuration used to initialize the globe model evaluator. */
  specfem::globe_model::ModelConfig model_config;

  /** @brief Mesher-side model identifiers used for consistency checks. */
  globe_model_verification model_verification;

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

  /** @brief Raw MPI node-interface descriptions from the globe database. */
  std::vector<globe_mpi_interface> mpi_interfaces;
};

} // namespace specfem::mesh
