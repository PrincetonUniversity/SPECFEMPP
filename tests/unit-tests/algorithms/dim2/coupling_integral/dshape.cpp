#include "algorithms/integrate/coupling_integral1d_dnshape.hpp"

#include "SPECFEM_Environment.hpp"
#include <gtest/gtest.h>

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

// temporary test for purposes of uncombined coupling_integral
TEST(CouplingIntegral, SimpleDShapeTest) {
  constexpr int nquad_element = 5;
  constexpr int nquad_intersection = 5;
  constexpr int ngllx = nquad_element;
  constexpr int ngllz = nquad_element;
  constexpr int num_edges = 1;

  constexpr auto interface_tag =
      specfem::interface::interface_tag::acoustic_elastic;
  constexpr auto boundary_tag = specfem::element::boundary_tag::none;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  constexpr auto medium_self =
      specfem::interface::attributes<specfem::dimension::type::dim2,
                                     interface_tag>::self_medium();
  constexpr auto ncomp_self =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   medium_self>::components;

  specfem::assembly::nonconforming_interfaces<specfem::dimension::type::dim2>
      nonconforming_interfaces;

  nonconforming_interfaces.template get_interface_container<
      interface_tag, boundary_tag,
      specfem::connections::type::nonconforming>() =
      specfem::assembly::nonconforming_interfaces_impl::interface_container<
          specfem::dimension::type::dim2, interface_tag, boundary_tag,
          specfem::connections::type::nonconforming>(ngllx, ngllz, num_edges);

  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, nquad_intersection, nquad_edge,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, memory_space, Kokkos::MemoryTraits<> >;
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, nquad_intersection, ncomp_self, false,
      memory_space, Kokkos::MemoryTraits<> >;

  Kokkos::parallel_for(
      "SimpleDShapeTest", Kokkos::TeamPolicy<>(num_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge = team_member.league_rank();
        const TransferFunctionType TF(Kokkos::subview(
            transfer_function_view, Kokkos::make_pair(iedge, iedge + 1),
            Kokkos::ALL(), Kokkos::ALL()));
        const FunctionType F(
            Kokkos::subview(function_view, Kokkos::make_pair(iedge, iedge + 1),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::coupling_integral_dnshape(
            nonconforming_interfaces, ChunkEdgeIndex(num_edges, team_member),
            TF, F, [&](const auto &index, const auto &point) {
              for (int icomp = 0; icomp < ncomp_self; ++icomp) {
                Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                  result_view(index(0), index(1), icomp) = point(icomp);
                });
              }
            });
      });
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
