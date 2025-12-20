#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/coupled_interface.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_coupling.hpp"
#include "specfem/point/field_derivatives.hpp"

#include "utilities/include/fixture/nonconforming_interface.hpp"
#include "utilities/interface.hpp"

#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>

/**
 * @brief Test index type for chunk edge operations.
 */
class ChunkEdgeIndex {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  /**
   * @brief Get Kokkos team member index.
   * @return Reference to Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Construct chunk edge index.
   * @param nedges Number of edges in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;                  ///< Number of edges in the chunk
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
};

constexpr static auto dimension_tag = specfem::dimension::type::dim2;
constexpr static auto boundary_tag = specfem::element::boundary_tag::none;

template <specfem::interface::interface_tag interface_tag,
          typename IntersectionData2D, typename EdgeFunction2D>
std::array<
    std::array<
        std::array<type_real,
                   specfem::element::attributes<
                       dimension_tag, specfem::interface::attributes<
                                          dimension_tag, interface_tag>::
                                          self_medium()>::components>,
        std::tuple_element_t<0, typename IntersectionData2D::packed_accessors>::
            nquad_intersection>,
    1 /*num_edges*/>
expected_solution(
    std::integral_constant<specfem::interface::interface_tag, interface_tag>,
    const IntersectionData2D &intersection_data,
    const EdgeFunction2D &coupled_field);

/**
 * @brief Execute transfer function test with validation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param function Input function data
 */
template <specfem::interface::interface_tag interface_tag,
          typename IntersectionData2D, typename EdgeFunction2D>
void execute(std::integral_constant<decltype(interface_tag), interface_tag>,
             const IntersectionData2D &intersection_data,
             const EdgeFunction2D &coupled_field) {

  static constexpr auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  static constexpr auto coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();
  static constexpr int ndim = specfem::dimension::dimension<dimension_tag>::dim;
  static constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<dimension_tag, coupled_medium>::components;
  static constexpr int num_edges = 1;
  static constexpr int chunk_size = 1;

  using TransferFunction2D =
      std::tuple_element_t<0, typename IntersectionData2D::packed_accessors>;

  static constexpr int nquad_edge = TransferFunction2D::nquad_edge;
  static constexpr int nquad_intersection =
      TransferFunction2D::nquad_intersection;
  using memory_space = typename TransferFunction2D::memory_space;
  using SelfFieldType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, num_edges, nquad_edge, ncomp_self, false,
      memory_space, Kokkos::MemoryTraits<> >;
  using CoupledFieldType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, num_edges, nquad_edge, ncomp_coupled, false,
      memory_space, Kokkos::MemoryTraits<> >;

  auto expected = expected_solution(
      std::integral_constant<specfem::interface::interface_tag,
                             interface_tag>(),
      intersection_data, coupled_field);

  const auto coupled_function_view = coupled_field.get_view();

  const auto results_view_name = "result_view";

  Kokkos::View<type_real *[nquad_intersection][ncomp_self], memory_space>
      result_view(results_view_name, num_edges);

  Kokkos::parallel_for(
      "transfer_function_test", Kokkos::TeamPolicy<>(num_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge = team_member.league_rank();
        // const TransferFunctionType TF(Kokkos::subview(
        //     transfer_function_view, Kokkos::make_pair(iedge, iedge + 1),
        //     Kokkos::ALL(), Kokkos::ALL()));
        // const CoupledFieldType F(Kokkos::subview(
        //     coupled_function_view, Kokkos::make_pair(iedge, iedge + 1),
        //     Kokkos::ALL(), Kokkos::ALL()));

        specfem::medium::compute_coupling(
            ChunkEdgeIndex(chunk_size, team_member), intersection_data,
            coupled_field, result_view);
      });

  Kokkos::fence();

  auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);

  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < nquad_intersection; ++j) {
      if (!specfem::utilities::is_close(result_host(i, j, 0),
                                        expected[i][j][0])) {
        std::ostringstream oss;
        oss << "-- Intersection Data --\n"
            << IntersectionData2D::description() << std::endl
            << "-- Edge Function --\n"
            << EdgeFunction2D::description() << std::endl
            << "\n-- Failure --\n"
            << "compute_coupling<"
            << specfem::interface::to_string(interface_tag)
            << "> test failed at edge " << i << ", intersection point " << j
            << " ("
            << TransferFunction2D::TransferFunctionInitializer::
                   intersection_quadrature_points[j]
            << "):\n expected " << expected[i][j][0] << "\n got "
            << result_host(i, j, 0) << std::endl;

        ADD_FAILURE() << oss.str();
      }
    }
  }
}
