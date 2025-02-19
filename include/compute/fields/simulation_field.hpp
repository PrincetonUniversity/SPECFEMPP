#pragma once

#include "compute/fields/impl/field_impl.hpp"
#include "data_access.tpp"
#include "element/field.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "point/assembly_index.hpp"
#include "point/field.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Store fields for a given simulation type
 *
 * @tparam WavefieldType Wavefield type.
 */
template <specfem::wavefield::simulation_field WavefieldType>
struct simulation_field {
private:
  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store field values

public:
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
  simulation_field(const specfem::compute::mesh &mesh,
                   const specfem::compute::element_types &element_types);
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
  void operator=(const simulation_field<DestinationWavefieldType> &rhs) {
    this->nglob = rhs.nglob;
    this->assembly_index_mapping = rhs.assembly_index_mapping;
    this->h_assembly_index_mapping = rhs.h_assembly_index_mapping;
    this->elastic = rhs.elastic;
    this->acoustic = rhs.acoustic;
  }

  /**
   * @brief Get the number of global degrees of freedom within a medium
   *
   * @tparam MediumType Medium type
   * @return int Number of global degrees of freedom
   */
  template <specfem::element::medium_tag MediumType>
  KOKKOS_FORCEINLINE_FUNCTION int get_nglob() const {
    if constexpr (MediumType == specfem::element::medium_tag::elastic_sv) {
      return elastic.nglob;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return acoustic.nglob;
    } else {
      static_assert("medium type not supported");
    }
  }

  int nglob = 0; ///< Number of global degrees of freedom
  int nspec;     ///< Number of spectral elements
  int ngllz;     ///< Number of quadrature points in z direction
  int ngllx;     ///< Number of quadrature points in x direction
  ViewType index_mapping;
  ViewType::HostMirror h_index_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      assembly_index_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      h_assembly_index_mapping;
  specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sv>
      elastic; ///< Elastic field
  specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::acoustic>
      acoustic; ///< Acoustic field

private:
  template <specfem::sync::kind sync> void sync_fields() {
    elastic.sync_fields<sync>();
    acoustic.sync_fields<sync>();
  }
};

template <specfem::wavefield::simulation_field WavefieldType1,
          specfem::wavefield::simulation_field WavefieldType2>
void deep_copy(simulation_field<WavefieldType1> &dst,
               const simulation_field<WavefieldType2> &src) {
  dst.nglob = src.nglob;
  Kokkos::deep_copy(dst.assembly_index_mapping, src.assembly_index_mapping);
  Kokkos::deep_copy(dst.h_assembly_index_mapping, src.h_assembly_index_mapping);
  specfem::compute::deep_copy(dst.elastic, src.elastic);
  specfem::compute::deep_copy(dst.acoustic, src.acoustic);
}

/**
 * @defgroup FieldDataAccess
 */

/**
 * @brief Load fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param field Wavefield container
 * @param point_field Point field to store the field values (output)
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const WavefieldContainer &field,
                                                ViewType &point_field) {
  impl_load_on_device(index, field, point_field);
}

/**
 * @brief Load fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param field Wavefield container
 * @param point_field Point field to store the field values (output)
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
inline void load_on_host(const IndexType &index,
                         const WavefieldContainer &field,
                         ViewType &point_field) {
  impl_load_on_host(index, field, point_field);
}

/**
 * @brief Store fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to store the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
store_on_device(const IndexType &index, const ViewType &point_field,
                const WavefieldContainer &field) {
  impl_store_on_device(index, point_field, field);
}

/**
 * @brief Store fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to store the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
inline void store_on_host(const IndexType &index, const ViewType &point_field,
                          const WavefieldContainer &field) {
  impl_store_on_host(index, point_field, field);
}

/**
 * @brief Add fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
add_on_device(const IndexType &index, const ViewType &point_field,
              const WavefieldContainer &field) {
  impl_add_on_device(index, point_field, field);
}

/**
 * @brief Add fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
inline void add_on_host(const IndexType &index, const ViewType &point_field,
                        const WavefieldContainer &field) {
  impl_add_on_host(index, point_field, field);
}

/**
 * @brief Atomic add fields at a given quadrature point index on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
atomic_add_on_device(const IndexType &index, const ViewType &point_field,
                     const WavefieldContainer &field) {
  impl_atomic_add_on_device(index, point_field, field);
}

/**
 * @brief Atomic add fields at a given quadrature point index on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index, @ref
 * specfem::point::simd_index, @ref specfem::point::assembly_index, or @ref
 * specfem::point::simd_assembly_index
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::point::field
 * @param index Index of the quadrature point
 * @param point_field Point field to add to the field values from
 * @param field Wavefield container
 */
template <typename IndexType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isPointFieldType, int> = 0>
inline void atomic_add_on_host(const IndexType &index,
                               const ViewType &point_field,
                               const WavefieldContainer &field) {
  impl_atomic_add_on_host(index, point_field, field);
}

/**
 * @brief Load fields at a given element on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param index Spectral element index
 * @param field Wavefield container
 * @param element_field Element field to store the field values (output)
 */
template <typename MemberType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isElementFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const MemberType &member, const int &index,
               const WavefieldContainer &field, ViewType &element_field) {
  impl_load_on_device(member, index, field, element_field);
}

/**
 * @brief Load fields at a given element on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param index Spectral element index
 * @param field Wavefield container
 * @param element_field Element field to store the field values (output)
 */
template <typename MemberType, typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isElementFieldType, int> = 0>
inline void load_on_host(const MemberType &member, const int &index,
                         const WavefieldContainer &field,
                         ViewType &element_field) {
  impl_load_on_host(member, index, field, element_field);
}

/**
 * @brief Store fields for a given chunk of elements on the device
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam ChunkIteratorType Chunk iterator type. Needs to be of @ref
 * specfem::iterator::chunk_iterator
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param iterator Chunk iterator specifying the elements
 * @param field Wavefield container
 * @param chunk_field Chunk field to store the field values (output)
 */
template <typename MemberType, typename ChunkIteratorType,
          typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const MemberType &member, const ChunkIteratorType &iterator,
               const WavefieldContainer &field, ViewType &chunk_field) {
  impl_load_on_device(member, iterator, field, chunk_field);
}

/**
 * @brief Store fields for a given chunk of elements on the host
 *
 * @ingroup FieldDataAccess
 *
 * @tparam MemberType Member type. Needs to be of @ref Kokkos::TeamPolicy
 * @tparam ChunkIteratorType Chunk iterator type. Needs to be of @ref
 * specfem::iterator::chunk_iterator
 * @tparam WavefieldContainer Wavefield container type. Needs to be of @ref
 * specfem::compute::fields::simulation_field
 * @tparam ViewType View type. Needs to be of @ref specfem::element::field
 * @param member Team member
 * @param iterator Chunk iterator specifying the elements
 * @param field Wavefield container
 * @param chunk_field Chunk field to store the field values (output)
 */
template <typename MemberType, typename ChunkIteratorType,
          typename WavefieldContainer, typename ViewType,
          typename std::enable_if_t<ViewType::isChunkFieldType, int> = 0>
inline void
load_on_host(const MemberType &member, const ChunkIteratorType &iterator,
             const WavefieldContainer &field, ViewType &chunk_field) {
  impl_load_on_host(member, iterator, field, chunk_field);
}

} // namespace compute
} // namespace specfem
