#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/fields/impl/field_impl.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief 2D simulation field container for spectral element wave computations
 *
 * This template specialization provides storage and management for simulation
 * fields in 2D spectral element meshes. It handles different wavefield types
 * (forward, adjoint, backward, buffer) and supports multiple physical media
 * (elastic P-SV, elastic SH, acoustic, poroelastic).
 *
 * The class manages field components appropriate for each medium:
 * - Elastic P-SV: displacement (ux, uz), velocity, acceleration
 * - Elastic SH: displacement (uy), velocity, acceleration
 * - Acoustic: potential, velocity potential, acceleration potential
 * - Poroelastic: solid and fluid phase displacements and pressures
 *
 * @tparam SimulationWavefieldType Type of simulation field (forward, adjoint,
 * backward, buffer)
 */
template <specfem::wavefield::simulation_field SimulationWavefieldType>
struct simulation_field<specfem::dimension::type::dim2,
                        SimulationWavefieldType> {

private:
  /**
   * @brief 3D index mapping view type for element-to-global indexing.
   *
   * Maps (element, z-quad, x-quad) triplets to linear indices for 2D field
   * access.
   */
  using IndexViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Assembly index mapping view type for global degree of freedom
   * indexing.
   *
   * Maps local field indices to global assembled system indices.
   */
  using AssemblyIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag
  constexpr static auto simulation_wavefield =
      SimulationWavefieldType; ///< Simulation wavefield type

  /**
   * @brief Default constructor.
   *
   * Initializes an empty 2D simulation field with no allocated storage.
   */
  simulation_field() = default;

  /**
   * @brief Construct 2D simulation field from mesh and element information.
   *
   * Initializes the simulation field by allocating storage for all medium
   * types present in the mesh. Creates global indexing mappings and allocates
   * appropriate field components based on the 2D mesh structure and element
   * classifications.
   *
   * @param mesh 2D assembly mesh containing global numbering and connectivity
   * @param element_types Element classification determining field allocation
   */
  simulation_field(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Copy 2D simulation field data from device to host memory.
   *
   * Synchronizes all field components and index mappings from device-accessible
   * memory to host memory for post-processing, I/O, or debugging operations.
   */
  void copy_to_host() { sync_fields<specfem::sync::kind::DeviceToHost>(); }

  /**
   * @brief Copy 2D simulation field data from host to device memory.
   *
   * Synchronizes all field components and index mappings from host memory
   * to device-accessible memory for GPU-accelerated computations.
   */
  void copy_to_device() { sync_fields<specfem::sync::kind::HostToDevice>(); }

  /**
   * @brief Assignment operator for copying from different wavefield types.
   *
   * Enables copying field data between different simulation field types
   * (e.g., from forward to buffer, adjoint to backward) while preserving
   * the field structure and indexing information.
   *
   * @tparam DestinationWavefieldType Source wavefield type to copy from
   * @param rhs Source simulation field to copy data from
   */
  template <specfem::wavefield::simulation_field DestinationWavefieldType>
  void operator=(
      const simulation_field<dimension_tag, DestinationWavefieldType> &rhs) {
    this->nglob = rhs.nglob;
    this->nspec = rhs.nspec;
    this->ngllz = rhs.ngllz;
    this->ngllx = rhs.ngllx;
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(field, (rhs_field, rhs.field), assembly_index_mapping,
                (rhs_assembly_index_mapping, rhs.assembly_index_mapping),
                h_assembly_index_mapping,
                (rhs_h_assembly_index_mapping, rhs.h_assembly_index_mapping)) {
          _field_ = _rhs_field_;
          _assembly_index_mapping_ = _rhs_assembly_index_mapping_;
          _h_assembly_index_mapping_ = _rhs_h_assembly_index_mapping_;
        })
  }

  /**
   * @brief Get number of global points for a specific medium type.
   *
   * Returns the total number of global degrees of freedom for the specified
   * medium type in the 2D simulation field.
   *
   * @tparam MediumTag Medium type to query (elastic_psv, elastic_sh, acoustic,
   * etc.)
   * @return Number of global points for the specified medium
   *
   */
  template <specfem::element::medium_tag MediumTag>
  KOKKOS_FORCEINLINE_FUNCTION int get_nglob() const {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(field) {
          if constexpr (MediumTag == _medium_tag_) {
            return _field_.nglob;
          }
        })

    Kokkos::abort("Medium type not supported");
    return 0;
  }

  /**
   * @brief Get field implementation for a specific medium type.
   *
   * Provides access to the underlying field storage for the specified medium
   * type, containing the appropriate field components (displacement, velocity,
   * acceleration for elastic; potential derivatives for acoustic, etc.).
   *
   * @tparam MediumTag Medium type to access (elastic_psv, elastic_sh, acoustic,
   * etc.)
   * @return Const reference to the field implementation for the specified
   * medium
   *
   * @code
   * auto elastic_field = field.get_field<specfem::element::medium_tag::elastic_psv>();
   * auto displacement = elastic_field.displacement;
   * // Now you can use displacement for further computations
   * @endcode
   */
  template <specfem::element::medium_tag MediumTag>
  KOKKOS_INLINE_FUNCTION
      constexpr specfem::assembly::fields_impl::field_impl<dimension_tag,
                                                           MediumTag> const &
      get_field() const {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        CAPTURE(field) {
          if constexpr (MediumTag == _medium_tag_) {
            return _field_;
          }
        })

    Kokkos::abort("Medium type not supported");
    /// Code path should never be reached

    auto return_value =
        new specfem::assembly::fields_impl::field_impl<dimension_tag,
                                                       MediumTag>();

    return *return_value;
  }

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION constexpr int
  get_iglob(const int &ispec, const int &iz, const int &ix,
            const specfem::element::medium_tag MediumTag) const {
    if constexpr (on_device) {
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                           POROELASTIC, ELASTIC_PSV_T)),
          CAPTURE(assembly_index_mapping) {
            if (MediumTag == _medium_tag_) {
              return _assembly_index_mapping_(index_mapping(ispec, iz, ix));
            }
          })

    } else {
      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                           POROELASTIC, ELASTIC_PSV_T)),
          CAPTURE(h_assembly_index_mapping) {
            if (MediumTag == _medium_tag_) {
              return _h_assembly_index_mapping_(h_index_mapping(ispec, iz, ix));
            }
          })
    }

    // If we reach here, it means the medium type is not defined in the macro
    Kokkos::abort("Medium type not defined in the macro");

    return -1;
  }

  template <bool on_device, typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    IndexType::using_simd == false &&
                    IndexType::dimension_tag == specfem::dimension::type::dim2,
                int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr int
  get_iglob(const IndexType &index,
            const specfem::element::medium_tag MediumTag) const {
    return get_iglob<on_device>(index.ispec, index.iz, index.ix, MediumTag);
  }

  template <bool on_device, typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    IndexType::using_simd == true &&
                    IndexType::dimension_tag == specfem::dimension::type::dim2,
                int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr int
  get_iglob(const IndexType &index, const int &lane,
            const specfem::element::medium_tag MediumTag) const {
    return get_iglob<on_device>(index.ispec + lane, index.iz, index.ix,
                                MediumTag);
  }

  int nglob = 0; ///< Total number of global points across all media
  int nspec;     ///< Number of spectral elements in the 2D mesh
  int ngllz;     ///< Number of quadrature points in z-direction
  int ngllx;     ///< Number of quadrature points in x-direction
  IndexViewType index_mapping; ///< Device index mapping from (ispec,iz,ix) to
                               ///< linear index
  IndexViewType::HostMirror h_index_mapping; ///< Host mirror of index mapping
                                             ///< for CPU operations

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      DECLARE(((specfem::assembly::fields_impl::field_impl,
                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
               field),
              (AssemblyIndexViewType, assembly_index_mapping),
              (AssemblyIndexViewType::HostMirror, h_assembly_index_mapping)))

  /**
   * @brief Get total degrees of freedom across all medium types.
   *
   * Computes the total number of degrees of freedom in the 2D simulation field,
   * summing over all active medium types and their respective field components.
   *
   * @return Total number of degrees of freedom in the assembled system
   */
  int get_total_degrees_of_freedom();

private:
  /**
   * @brief Synchronize field data between host and device memory.
   *
   * @tparam sync Synchronization direction (HostToDevice or DeviceToHost)
   */
  template <specfem::sync::kind sync> void sync_fields();

  int total_degrees_of_freedom = 0; ///< Cached total degrees of freedom count
};

/**
 * @brief Deep copy between 2D simulation fields of different types.
 *
 * Performs a complete deep copy of field data, index mappings, and metadata
 * between two 2D simulation fields. This enables copying between different
 * wavefield types (e.g., forward to buffer, adjoint to backward) while
 * preserving all field structure and indexing information.
 *
 * @tparam SimulationWavefieldType1 Destination field type
 * @tparam SimulationWavefieldType2 Source field type
 * @param dst Destination simulation field
 * @param src Source simulation field to copy from
 *
 * @code
 * // Copy forward field to buffer for checkpointing
 * deep_copy(buffer_field, forward_field);
 * @endcode
 */
template <typename SimulationWavefieldType1, typename SimulationWavefieldType2,
          typename std::enable_if_t<((SimulationWavefieldType1::dimension_tag ==
                                      specfem::dimension::type::dim2) &&
                                     (SimulationWavefieldType2::dimension_tag ==
                                      specfem::dimension::type::dim2)),
                                    int> = 0>
inline void deep_copy(SimulationWavefieldType1 &dst,
                      const SimulationWavefieldType2 &src) {
  dst.nglob = src.nglob;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE((src_assembly_index_mapping, src.assembly_index_mapping),
              (dst_assembly_index_mapping, dst.assembly_index_mapping),
              (src_h_assembly_index_mapping, src.h_assembly_index_mapping),
              (dst_h_assembly_index_mapping, dst.h_assembly_index_mapping),
              (src_field, src.field), (dst_field, dst.field)) {
        Kokkos::deep_copy(_dst_assembly_index_mapping_,
                          _src_assembly_index_mapping_);
        Kokkos::deep_copy(_dst_h_assembly_index_mapping_,
                          _src_h_assembly_index_mapping_);
        specfem::assembly::fields_impl::deep_copy(_dst_field_, _src_field_);
      })
}

} // namespace specfem::assembly
