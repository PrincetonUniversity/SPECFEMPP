#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/fields/impl/field_impl.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
/**
 * @brief Store fields for a given simulation type
 *
 * @tparam WavefieldType Wavefield type.
 */
template <specfem::wavefield::simulation_field SimulationWavefieldType>
struct simulation_field<specfem::dimension::type::dim3,
                        SimulationWavefieldType> {

private:
  using IndexViewType =
      Kokkos::View<int ****, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store field values
  using AssemblyIndexViewType =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< assembled indices

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  constexpr static auto simulation_wavefield =
      SimulationWavefieldType; ///< Simulation wavefield type
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  simulation_field() = default;

  /**
   * @brief Construct a new simulation field object from assebled mesh
   *
   * @param mesh Assembled mesh
   * @param properties Material properties
   */
  simulation_field(
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types);
  ///@}

  /**
   * @brief Copy fields to the device
   *
   */
  void copy_to_host() { sync_fields<specfem::sync::kind::DeviceToHost>(); }

  /**
   * @brief Copy fields to the host
   *
   */
  void copy_to_device() { sync_fields<specfem::sync::kind::HostToDevice>(); }

  /**
   * @brief Copy fields from another simulation field
   *
   * @tparam DestinationWavefieldType Destination wavefield type
   * @param rhs Simulation field to copy from
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
   * @brief Get the number of global degrees of freedom within a medium
   *
   * @tparam MediumTag Medium type
   * @return int Number of global degrees of freedom
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
   * @brief Returns the field for a given medium
   *
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

  /**
   * @brief Returns the assembled index given element index.
   *
   */
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
  }

  int nglob = 0; ///< Number of global degrees of freedom
  int nspec;     ///< Number of spectral elements
  int ngllz;     ///< Number of quadrature points in z direction
  int nglly;     ///< Number of quadrature points in y direction
  int ngllx;     ///< Number of quadrature points in x direction
  IndexViewType index_mapping;
  IndexViewType::HostMirror h_index_mapping;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC)),
                      DECLARE(((specfem::assembly::fields_impl::field_impl,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               field),
                              (AssemblyIndexViewType, assembly_index_mapping),
                              (AssemblyIndexViewType::HostMirror,
                               h_assembly_index_mapping)))

  int get_total_degrees_of_freedom();

private:
  template <specfem::sync::kind sync> void sync_fields();
  int total_degrees_of_freedom = 0; ///< Total number of degrees of freedom
};

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
        specfem::assembly::deep_copy(_dst_field_, _src_field_);
      })
}

} // namespace specfem::assembly
