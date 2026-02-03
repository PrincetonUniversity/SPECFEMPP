
#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {

/**
 * @brief Container for 2D nonconforming interface data storage and access
 *
 * Manages interface data between different physical media (elastic-acoustic)
 * with specific boundary conditions. Stores edge factors and normal vectors
 * for interface computations in 2D spectral element simulations.
 *
 * TODO: consider same physical media
 *
 * @tparam InterfaceTag Type of interface (ELASTIC_ACOUSTIC or ACOUSTIC_ELASTIC)
 * @tparam BoundaryTag Boundary condition type (NONE, STACEY, etc.)
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container<specfem::dimension::type::dim2, InterfaceTag,
                           BoundaryTag,
                           specfem::connections::type::nonconforming>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::dimension::type::dim2> {
public:
  /** @brief Dimension tag for 2D specialization */
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  constexpr static auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  constexpr static auto boundary_tag = BoundaryTag;
  /** @brief Medium type on the self side of the interface */
  constexpr static auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  /** @brief Medium type on the coupled side of the interface */
  constexpr static auto coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();

public:
  /** @brief Base container type alias */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::edge,
      specfem::data_access::DataClassType::nonconforming_interface,
      specfem::dimension::type::dim2>;
  /** @brief View type for edge scaling factors */
  using EdgeFactorView = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for edge normal vectors */
  using EdgeNormalView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for transfer function */
  using TransferFunctionView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for edge scaling factors */
  EdgeFactorView intersection_factor;
  /** @brief Device view for edge normal vectors */
  EdgeNormalView intersection_normal;
  /** @brief Device view for transfer function on self */
  TransferFunctionView transfer_function;
  /** @brief Device view for transfer function on coupled side */
  TransferFunctionView transfer_function_other;

  /** @brief Host mirror for edge scaling factors */
  EdgeFactorView::HostMirror h_intersection_factor;
  /** @brief Host mirror for edge normal vectors */
  EdgeNormalView::HostMirror h_intersection_normal;
  /** @brief Device view for transfer function on self */
  TransferFunctionView::HostMirror h_transfer_function;
  /** @brief Device view for transfer function on coupled side */
  TransferFunctionView::HostMirror h_transfer_function_other;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @param edge_types Edge type information from mesh
   * @param mesh Mesh connectivity and geometry
   */
  interface_container(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<specfem::dimension::type::dim2>
          &edge_types,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  /** @brief Default constructor */
  interface_container() = default;

private:
  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION auto
  get_value(std::integral_constant<
                specfem::data_access::DataClassType,
                specfem::data_access::DataClassType::transfer_function_self>,
            const int iedge, const int ipoint, const int iquad) const {
    if constexpr (on_device) {
      return transfer_function(iedge, iquad, ipoint);
    } else {
      return h_transfer_function(iedge, iquad, ipoint);
    }
  }

  template <bool on_device>
  KOKKOS_FORCEINLINE_FUNCTION auto
  get_value(std::integral_constant<
                specfem::data_access::DataClassType,
                specfem::data_access::DataClassType::transfer_function_coupled>,
            const int iedge, const int ipoint, const int iquad) const {
    if constexpr (on_device) {
      return transfer_function_other(iedge, iquad, ipoint);
    } else {
      return h_transfer_function_other(iedge, iquad, ipoint);
    }
  }

  template <bool on_device, typename IndexType, typename... PointTypes>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load_after_expansion(
      const std::integral_constant<
          specfem::data_access::AccessorType,
          specfem::data_access::AccessorType::point> /* AccessorType */,
      const IndexType &index, PointTypes &...points) const {

    static_assert((specfem::data_access::is_point<PointTypes>::value && ...),
                  "impl_load only supports point accessors");

    static_assert(specfem::data_access::is_point<IndexType>::value,
                  "impl_load requires point type for IndexType");

    static_assert(specfem::data_access::is_edge_index<IndexType>::value,
                  "impl_load requires edge_index type for IndexType");

    constexpr int nquad_intersection =
        std::tuple_element_t<0,
                             std::tuple<PointTypes...> >::n_quad_intersection;

    const auto assign = [&](auto &point, auto &iquad, auto &index) {
      using PointType = std::decay_t<decltype(point)>;

      point(iquad) = this->get_value<on_device>(
          std::integral_constant<specfem::data_access::DataClassType,
                                 PointType::data_class>(),
          index.iedge, index.ipoint, iquad);
    };

    for (int iquad = 0; iquad < nquad_intersection; iquad++) {
      (assign(points, iquad, index), ...);
    }
  }

  template <bool on_device, typename IndexType, typename... EdgeTypes>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load_after_expansion(
      const std::integral_constant<
          specfem::data_access::AccessorType,
          specfem::data_access::AccessorType::chunk_edge> /* AccessorType */,
      const IndexType &index, EdgeTypes &...edges) const {

    static_assert(
        (specfem::data_access::is_chunk_edge<EdgeTypes>::value && ...),
        "impl_load only supports chunk_edge accessors");

    static_assert(specfem::data_access::is_edge_index<IndexType>::value,
                  "impl_load requires edge_index type for IndexType");

    static_assert(specfem::data_access::is_chunk_edge<IndexType>::value,
                  "impl_load requires chunk_edge type for IndexType");

    constexpr int nquad_intersection =
        std::tuple_element_t<0, std::tuple<EdgeTypes...> >::n_quad_intersection;

    const auto factor_subview = [&]() {
      if constexpr (on_device) {
        return Kokkos::subview(intersection_factor, index.get_range(),
                               Kokkos::ALL);
      } else {
        return Kokkos::subview(h_intersection_factor, index.get_range(),
                               Kokkos::ALL);
      }
    }();

    const auto normal_subview = [&]() {
      if constexpr (on_device) {
        return Kokkos::subview(intersection_normal, index.get_range(),
                               Kokkos::ALL, Kokkos::ALL);
      } else {
        return Kokkos::subview(h_intersection_normal, index.get_range(),
                               Kokkos::ALL, Kokkos::ALL);
      }
    }();

    const auto factor_call = [&](auto &edge, auto &index) {
      using edge_t = std::decay_t<decltype(edge)>;

      if constexpr (specfem::data_access::is_intersection_factor<
                        edge_t>::value) {
        edge(index(0), index(1)) = factor_subview(index(0), index(1));
      }
    };

    const auto normal_call = [&](auto &edge, auto &index) {
      using edge_t = std::decay_t<decltype(edge)>;

      if constexpr (specfem::data_access::is_intersection_normal<
                        edge_t>::value) {
        for (int icomponent = 0; icomponent < 2; icomponent++) {
          edge(index(0), index(1), icomponent) =
              normal_subview(index(0), index(1), icomponent);
        }
      }
    };

    const auto transfer_call = [&](auto &edge, auto &local_index, auto &index) {
      using edge_t = std::decay_t<decltype(edge)>;

      if constexpr (specfem::data_access::is_transfer_function_self<
                        edge_t>::value ||
                    specfem::data_access::is_transfer_function_coupled<
                        edge_t>::value) {
        for (int iquad = 0; iquad < nquad_intersection; iquad++) {
          edge(local_index.iedge, local_index.ipoint, iquad) =
              this->get_value<on_device>(
                  std::integral_constant<specfem::data_access::DataClassType,
                                         edge_t::data_class>(),
                  index.iedge, index.ipoint, iquad);
        }
      }
    };

    specfem::execution::for_each_level(
        specfem::execution::TeamThreadMDRangeIterator(
            index.get_policy_index(), index.nedges(), nquad_intersection),
        [&](const auto index) {
          (factor_call(edges, index), ...);
          (normal_call(edges, index), ...);
        });

    specfem::execution::for_each_level(
        index.get_iterator(),
        [&](const typename IndexType::iterator_type::index_type
                &iterator_index) {
          const auto &local_index = iterator_index.get_local_index();
          const auto &index = iterator_index.get_index();

          (transfer_call(edges, local_index, index), ...);
        });
  }

  template <bool on_device, typename DispatchType, typename IndexType,
            typename AccessorType, std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION void
  impl_load_expand(const DispatchType dispatch, const IndexType &index,
                   AccessorType &accessor,
                   std::index_sequence<Is...> /* index sequence */) const {
    impl_load_after_expansion<on_device>(
        dispatch, index,
        static_cast<typename std::tuple_element_t<
            Is, typename AccessorType::packed_accessors> &>(accessor)...);
  }

public:
  template <bool on_device, typename DispatchType, typename IndexType,
            typename AccessorType,
            std::enable_if_t<
                (specfem::data_access::is_packed_accessor<AccessorType>::value),
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load(const DispatchType dispatch,
                                             const IndexType &index,
                                             AccessorType &accessor) const {

    impl_load_expand<on_device>(
        dispatch, index, accessor,
        std::make_index_sequence<AccessorType::n_accessors>{});
  }

  template <bool on_device, typename DispatchType, typename IndexType,
            typename AccessorType,
            std::enable_if_t<(!specfem::data_access::is_packed_accessor<
                                 AccessorType>::value),
                             int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load(const DispatchType dispatch,
                                             const IndexType &index,
                                             AccessorType &accessor) const {
    impl_load_after_expansion<on_device>(dispatch, index, accessor);
  }
};
} // namespace specfem::assembly::nonconforming_interfaces_impl
