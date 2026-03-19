#pragma once

#include "accessor.hpp"
#include "data_class.hpp"
#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

/**
 * @brief Storage layout types for simulation data containers.
 */
enum class ContainerType {
  boundary, ///< Boundary element storage
  edge,     ///< Edge-based storage
  face,     ///< Face-based storage
  domain    ///< Domain element storage
};

namespace impl {

/**
 * @brief Maps container and dimension types to appropriate Kokkos view layouts.
 */
template <specfem::data_access::ContainerType ContainerType,
          specfem::element::dimension_tag DimensionTag>
struct ContainerValueType;

template <>
struct ContainerValueType<specfem::data_access::ContainerType::domain,
                          specfem::element::dimension_tag::dim2> {
  template <typename T, typename MemorySpace>
  using scalar_type = specfem::datatype::DomainView2d<T, 3, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = specfem::datatype::DomainView2d<T, 4, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = specfem::datatype::DomainView2d<T, 5, MemorySpace>;
};

template <>
struct ContainerValueType<specfem::data_access::ContainerType::domain,
                          specfem::element::dimension_tag::dim3> {
  template <typename T, typename MemorySpace>
  using scalar_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = Kokkos::View<T *****, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = Kokkos::View<T ******, Kokkos::LayoutLeft, MemorySpace>;
};

template <>
struct ContainerValueType<specfem::data_access::ContainerType::edge,
                          specfem::element::dimension_tag::dim2> {
  template <typename T, typename MemorySpace>
  using scalar_type = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = Kokkos::View<T ***, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace>;
};

template <>
struct ContainerValueType<specfem::data_access::ContainerType::face,
                          specfem::element::dimension_tag::dim3> {
  template <typename T, typename MemorySpace>
  using scalar_type = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using vector_type = Kokkos::View<T ***, Kokkos::LayoutLeft, MemorySpace>;
  template <typename T, typename MemorySpace>
  using tensor_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace>;
};
} // namespace impl

/**
 * @brief Type-safe data container for spectral element simulation data.
 *
 * Provides scalar, vector, and tensor storage types based on container layout,
 * data classification, and spatial dimension. Used throughout SPECFEM for
 * mesh data, field variables, and material properties.
 *
 * **Usage:**
 * For example, to store scalar data (e.g., density) for the whole simulation
 * domain in 2D:
 * @code{cpp}
 * using density_container = specfem::data_access::Container<
 *     specfem::data_access::ContainerType::domain,
 *     // density data-class not explicitly defined
 *     specfem::element::dimension_tag::dim2>;
 * using scalar_view = density_container::scalar_type<float,
 *                                                     Kokkos::HostSpace>;
 * scalar_view density("density", nspec, ngllz, ngllx);
 * @endcode
 *
 * @tparam ContainerType Storage layout (domain/edge/face/boundary)
 * @tparam DataClass Type of data being stored (properties/fields/indices)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 *
 */
template <specfem::data_access::ContainerType ContainerType,
          specfem::data_access::DataClassType DataClass,
          specfem::element::dimension_tag DimensionTag>
struct Container {
  constexpr static auto container_type =
      ContainerType;                            ///< Container layout type
  constexpr static auto data_class = DataClass; ///< Data classification type
  constexpr static auto dimension_tag = DimensionTag; ///< Spatial dimension

  /**
   * @brief Container to be used when storing scalar data.
   *
   * @tparam T Base data type to be stored within the container
   * @tparam MemorySpace Kokkos memory space for storage
   */
  template <typename T, typename MemorySpace>
  using scalar_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template scalar_type<T, MemorySpace>;

  /**
   * @brief Container to be used when storing vector data.
   *
   * @tparam T Base data type to be stored within the container
   * @tparam MemorySpace Kokkos memory space for storage
   */
  template <typename T, typename MemorySpace>
  using vector_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template vector_type<T, MemorySpace>;

  /**
   * @brief Container to be used when storing tensor data.
   *
   * @tparam T Base data type to be stored within the container
   * @tparam MemorySpace Kokkos memory space for storage
   */
  template <typename T, typename MemorySpace>
  using tensor_type = typename impl::ContainerValueType<
      ContainerType, dimension_tag>::template tensor_type<T, MemorySpace>;
};

/**
 * @brief Type trait to detect container types.
 *
 * Checks if a type has the container_type static member, indicating
 * it follows the Container interface.
 */
template <typename T, typename = void> struct is_container : std::false_type {};

template <typename T>
struct is_container<T, std::void_t<decltype(T::container_type)> >
    : std::true_type {};

} // namespace specfem::data_access
