#include "containers_test.hpp"
#include "enumerations/dimension.hpp"
#include "parallel_configuration/edge_config.hpp"
#include "policies/edge.hpp"
#include "specfem/point.hpp"

using ParallelConfig = specfem::parallel_config::default_edge_config<
    specfem::dimension::type::dim2, Kokkos::DefaultExecutionSpace>;
using EdgePolicyType = specfem::policy::element_edge<ParallelConfig>;

template <typename ContainerType>
void test_edge_policy(
    ContainerType &container,
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  using Medium1PointType = specfem::point::field<specfem::dimension::type::dim2,
                                                 ContainerType::Medium1, true,
                                                 false, false, false, false>;

  // single_edge containers bind medium1,medium2 to the same value.
  // EdgePolicyType edge_policy = [&](){

  // if constexpr(ContainerType::is_single_edge){
  //   return EdgePolicyType(container.index_mapping,
  //                            container.index_mapping,
  //                            container.edge_type,
  //                            container.edge_type, assembly.mesh.ngllz);
  // }else{
  //   return EdgePolicyType(container.medium1_index_mapping,
  //                            container.medium2_index_mapping,
  //                            container.medium1_edge_type,
  //                            container.medium2_edge_type,
  //                            assembly.mesh.ngllz);
  // }
  // }();
  EdgePolicyType edge_policy(container.template get_edge_index_view<1>(),
                             container.template get_edge_index_view<2>(),
                             container.template get_edge_type_view<1>(),
                             container.template get_edge_type_view<2>(),
                             assembly.mesh.ngllz);

  Kokkos::parallel_for(
      "edge_policy_test.tpp:test_edge_policy",
      static_cast<typename EdgePolicyType::policy_type &>(edge_policy),
      KOKKOS_LAMBDA(const typename EdgePolicyType::member_type &team_member) {
        const auto iterator =
            edge_policy.league_iterator(team_member.league_rank());

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, iterator.edge_size()),
            [=](const int ipoint) {
              const auto index = iterator(ipoint);
              const auto self_index = index.self_index;
              const auto coupled_index = index.coupled_index;
              const int iedge = index.iedge;

              Medium1PointType disp;
              disp(0) = 1;

              specfem::assembly::store_on_device(self_index, disp,
                                                 assembly.fields.forward);

              // TODO verify normals/edge geometry here, too
            });
      });
}
