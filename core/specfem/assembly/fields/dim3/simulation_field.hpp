#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/fields/impl/field_impl.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief 3D simulation field container for spectral element wave computations
 *
 * This template specialization provides storage and management for simulation
 * fields in 3D spectral element meshes. It handles different wavefield types
 * (forward, adjoint, backward, buffer) with support for elastic media.
 *
 * The 3D implementation currently focuses on elastic media with displacement,
 * velocity, and acceleration field components (ux, uy, uz and derivatives).
 *
 * @tparam SimulationWavefieldType Type of simulation field (forward, adjoint,
 * backward, buffer)
 */
template <specfem::wavefield::simulation_field SimulationWavefieldType>
struct simulation_field<specfem::dimension::type::dim3,
                        SimulationWavefieldType> {

private:
  /**
   * @brief 4D index mapping view type for 3D element-to-global indexing.
   *
   * Maps (element, z-quad, y-quad, x-quad) quadruplets to linear indices
   * for 3D field access in hexahedral spectral elements.
   */
  using IndexViewType =
      Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Assembly index mapping view type for 3D global degree of freedom
   * indexing.
   *
   * Maps local 3D field indices to global assembled system indices for
   * 3D finite element assembly operations.
   */
  using AssemblyIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  constexpr static auto simulation_wavefield =
      SimulationWavefieldType; ///< Simulation wavefield type

  /**
   * @brief Default constructor.
   *
   * Initializes an empty 3D simulation field with no allocated storage.
   */
  simulation_field() = default;

  /**
   * @brief Construct 3D simulation field from mesh and element information.
   *
   * Initializes the 3D simulation field by allocating storage for elastic
   * medium present in the 3D mesh. Creates global indexing mappings for
   * the full 3D tensor grid and allocates appropriate field components
   * based on the 3D mesh structure and element classifications.
   *
   * @param mesh 3D assembly mesh containing global numbering and connectivity
   * @param element_types 3D element classification determining field allocation
   *
   * @code
   * specfem::assembly::simulation_field<specfem::dimension::type::dim3,
   *     specfem::wavefield::simulation_field::forward> field(mesh,
   * element_types);
   * @endcode
   */
  simulation_field(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  /**
   * @brief Copy 3D simulation field data from device to host memory.
   *
   * Synchronizes all 3D field components and index mappings from
   * device-accessible memory to host memory for post-processing, I/O, or
   * debugging operations. Handles the full 3D tensor structure efficiently.
   */
  void copy_to_host() { sync_fields<specfem::sync::kind::DeviceToHost>(); }

  /**
   * @brief Copy 3D simulation field data from host to device memory.
   *
   * Synchronizes all 3D field components and index mappings from host memory
   * to device-accessible memory for GPU-accelerated 3D computations.
   */
  void copy_to_device() { sync_fields<specfem::sync::kind::HostToDevice>(); }

  /**
   * @brief Assignment operator for copying between 3D wavefield types.
   *
   * Enables copying 3D field data between different simulation field types
   * (e.g., from forward to buffer, adjoint to backward) while preserving
   * the 3D field structure and indexing information.
   *
   * @tparam DestinationWavefieldType Source wavefield type to copy from
   * @param rhs Source 3D simulation field to copy data from
   */
  template <specfem::wavefield::simulation_field DestinationWavefieldType>
  void operator=(
      const simulation_field<dimension_tag, DestinationWavefieldType> &rhs) {
    this->nglob = rhs.nglob;
    this->nspec = rhs.nspec;
    this->ngllz = rhs.ngllz;
    this->nglly = rhs.nglly;
    this->ngllx = rhs.ngllx;
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
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
   * @brief Get number of global points for 3D elastic medium.
   *
   * Returns the total number of global degrees of freedom for the elastic
   * medium type in the 3D simulation field.
   *
   * @tparam MediumTag Medium type to query (currently elastic for 3D)
   * @return Number of global points for the specified 3D medium
   *
   * @code
   * int nglob_elastic =
   * field.get_nglob<specfem::element::medium_tag::elastic>();
   * @endcode
   */
  template <specfem::element::medium_tag MediumTag>
  KOKKOS_FORCEINLINE_FUNCTION int get_nglob() const {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                        CAPTURE(field) {
                          if constexpr (MediumTag == _medium_tag_) {
                            return _field_.nglob;
                          }
                        })

    Kokkos::abort("Medium type not supported");
    return 0;
  }

  /**
   * @brief Get 3D field implementation for elastic medium.
   *
   * Provides access to the underlying 3D field storage for the elastic medium,
   * containing displacement, velocity, and acceleration components (ux, uy, uz
   * and their time derivatives) appropriate for 3D wave propagation.
   *
   * @tparam MediumTag Medium type to access (elastic for 3D applications)
   * @return Const reference to the 3D field implementation for elastic medium
   *
   * @code
   * const auto& elastic_field =
   * field.get_field<specfem::element::medium_tag::elastic>(); 
   * const auto& displacement = elastic_field.displacement;
   * @endcode
   */
  template <specfem::element::medium_tag MediumTag>
  KOKKOS_INLINE_FUNCTION
      constexpr specfem::assembly::fields_impl::field_impl<dimension_tag,
                                                           MediumTag> const &
      get_field() const {
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
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
  get_iglob(const int &ispec, const int &iz, const int &iy, const int &ix,
            const specfem::element::medium_tag MediumTag) const {
    if constexpr (on_device) {
      FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                          CAPTURE(assembly_index_mapping) {
                            if (MediumTag == _medium_tag_) {
                              return _assembly_index_mapping_(
                                  index_mapping(ispec, iz, iy, ix));
                            }
                          })

    } else {
      FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                          CAPTURE(h_assembly_index_mapping) {
                            if (MediumTag == _medium_tag_) {
                              return _h_assembly_index_mapping_(
                                  h_index_mapping(ispec, iz, iy, ix));
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
                    IndexType::dimension_tag == specfem::dimension::type::dim3,
                int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr int
  get_iglob(const IndexType &index,
            const specfem::element::medium_tag MediumTag) const {
    return get_iglob<on_device>(index.ispec, index.iz, index.iy, index.ix,
                                MediumTag);
  }

  template <bool on_device, typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    IndexType::using_simd == true &&
                    IndexType::dimension_tag == specfem::dimension::type::dim3,
                int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr int
  get_iglob(const IndexType &index, const int &lane,
            const specfem::element::medium_tag MediumTag) const {
    return get_iglob<on_device>(index.ispec + lane, index.iz, index.iy,
                                index.ix, MediumTag);
  }

  int nglob = 0;               ///< Number of global degrees of freedom
  int nspec;                   ///< Number of spectral elements
  int ngllz;                   ///< Number of quadrature points in z direction
  int nglly;                   ///< Number of quadrature points in y direction
  int ngllx;                   ///< Number of quadrature points in x direction
  IndexViewType index_mapping; ///< Device 3D index mapping from
                               ///< (ispec,iz,iy,ix) to linear index
  IndexViewType::HostMirror h_index_mapping; ///< Host mirror of 3D index
                                             ///< mapping for CPU operations

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                      DECLARE(((specfem::assembly::fields_impl::field_impl,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               field),
                              (AssemblyIndexViewType, assembly_index_mapping),
                              (AssemblyIndexViewType::HostMirror,
                               h_assembly_index_mapping)))

  /**
   * @brief Get total degrees of freedom in the 3D elastic system.
   *
   * Computes the total number of degrees of freedom in the 3D simulation field,
   * accounting for the elastic medium and its field components (3 displacement
   * components × number of global points).
   *
   * @return Total number of degrees of freedom in the assembled 3D system
   */
  int get_total_degrees_of_freedom();

private:
  /**
   * @brief Synchronize 3D field data between host and device memory.
   *
   * @tparam sync Synchronization direction (HostToDevice or DeviceToHost)
   */
  template <specfem::sync::kind sync> void sync_fields();
  int total_degrees_of_freedom = 0; ///< Total number of degrees of freedom
};

/**
 * @brief Deep copy between 3D simulation fields of different types.
 *
 * Performs a complete deep copy of 3D field data, index mappings, and metadata
 * between two 3D simulation fields. This enables copying between different
 * wavefield types (e.g., forward to buffer, adjoint to backward) while
 * preserving all 3D field structure and indexing information.
 *
 * @tparam SimulationWavefieldType1 Destination 3D field type
 * @tparam SimulationWavefieldType2 Source 3D field type
 * @param dst Destination 3D simulation field
 * @param src Source 3D simulation field to copy from
 *
 * @code
 * // Copy 3D forward field to buffer for checkpointing
 * deep_copy(buffer_field, forward_field);
 * @endcode
 */
template <typename SimulationWavefieldType1, typename SimulationWavefieldType2,
          typename std::enable_if_t<((SimulationWavefieldType1::dimension_tag ==
                                      specfem::dimension::type::dim3) &&
                                     (SimulationWavefieldType2::dimension_tag ==
                                      specfem::dimension::type::dim3)),
                                    int> = 0>
inline void deep_copy(SimulationWavefieldType1 &dst,
                      const SimulationWavefieldType2 &src) {
  dst.nglob = src.nglob;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
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
