#include "specfem/assembly/nonconforming_interfaces/dim3/impl/host_selection.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/setup.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

namespace specfem::testing::host_selection_impl {

using HostSelection =
    specfem::assembly::nonconforming_interfaces_impl::HostSelection;
using interface_tag = specfem::element_coupling::interface_tag;
using meshing_type = specfem::element_coupling::interfacial_meshing_type;

/**
 * @brief Host flags of the two sides of a nonconforming interface
 * @param hs Host selection to query
 * @return (acoustic side hosts, elastic side hosts)
 */
std::pair<bool, bool> self_host_pair(const HostSelection &hs);

} // namespace specfem::testing::host_selection_impl

std::pair<bool, bool>
specfem::testing::host_selection_impl::self_host_pair(const HostSelection &hs) {
  return { hs.self_is_host(interface_tag::acoustic_elastic,
                           interface_tag::elastic_acoustic),
           hs.self_is_host(interface_tag::elastic_acoustic,
                           interface_tag::acoustic_elastic) };
}

namespace hs_impl = specfem::testing::host_selection_impl;

TEST(NonconformingHostSelection3D, FinerSideHosts) {
  // Acoustic mean 1 (10 / 10), elastic mean 3 (12 / 4): acoustic hosts.
  hs_impl::HostSelection acoustic_finer(hs_impl::meshing_type::highres_host);
  acoustic_finer.add_faces(hs_impl::interface_tag::acoustic_elastic, 10.0, 10);
  acoustic_finer.add_faces(hs_impl::interface_tag::elastic_acoustic, 12.0, 4);
  const auto [a_host, e_host] = hs_impl::self_host_pair(acoustic_finer);
  EXPECT_TRUE(a_host) << "Finer acoustic side should host";
  EXPECT_FALSE(e_host) << "Coarser elastic side should not host";

  // Mirrored: acoustic mean 3, elastic mean 1.
  hs_impl::HostSelection elastic_finer(hs_impl::meshing_type::highres_host);
  elastic_finer.add_faces(hs_impl::interface_tag::acoustic_elastic, 12.0, 4);
  elastic_finer.add_faces(hs_impl::interface_tag::elastic_acoustic, 10.0, 10);
  const auto [a_host2, e_host2] = hs_impl::self_host_pair(elastic_finer);
  EXPECT_FALSE(a_host2) << "Coarser acoustic side should not host";
  EXPECT_TRUE(e_host2) << "Finer elastic side should host";
}

TEST(NonconformingHostSelection3D, TotalsAccumulateAcrossCalls) {
  // Acoustic totals: 11 / 8 = 1.375. Elastic totals: 10 / 4 = 2.5.
  // The first acoustic call alone has mean 4 and the first elastic call has
  // mean 2, so a single-call decision would pick the elastic side.
  hs_impl::HostSelection hs(hs_impl::meshing_type::highres_host);
  hs.add_faces(hs_impl::interface_tag::acoustic_elastic, 8.0, 2);
  hs.add_faces(hs_impl::interface_tag::acoustic_elastic, 0.0, 0); // empty
  hs.add_faces(hs_impl::interface_tag::acoustic_elastic, 3.0, 6);
  hs.add_faces(hs_impl::interface_tag::elastic_acoustic, 4.0, 2);
  hs.add_faces(hs_impl::interface_tag::elastic_acoustic, 6.0, 2);
  const auto [a_host, e_host] = hs_impl::self_host_pair(hs);
  EXPECT_TRUE(a_host) << "Host must follow accumulated acoustic mean";
  EXPECT_FALSE(e_host) << "Host must follow accumulated elastic mean";
}

TEST(NonconformingHostSelection3D, ExactlyOneHostOnTie) {
  // Both means are exactly 2. elastic_acoustic compares smaller in the enum.
  hs_impl::HostSelection hs(hs_impl::meshing_type::highres_host);
  hs.add_faces(hs_impl::interface_tag::acoustic_elastic, 6.0, 3);
  hs.add_faces(hs_impl::interface_tag::elastic_acoustic, 8.0, 4);
  const auto [a_host, e_host] = hs_impl::self_host_pair(hs);
  EXPECT_NE(a_host, e_host) << "Exactly one side must host";
  EXPECT_FALSE(a_host) << "Tie should go to the smaller tag";
  EXPECT_TRUE(e_host) << "Tie should go to the smaller tag";
}

TEST(NonconformingHostSelection3D, ExactlyOneHostWhenBothEmpty) {
  hs_impl::HostSelection hs(hs_impl::meshing_type::highres_host);
  const auto [a_host, e_host] = hs_impl::self_host_pair(hs);
  EXPECT_NE(a_host, e_host) << "Exactly one side must host";
  EXPECT_TRUE(e_host) << "Empty tie should go to the smaller tag";
}

TEST(NonconformingHostSelection3D, PopulatedSideHostsWhenOtherIsEmpty) {
  hs_impl::HostSelection acoustic_only(hs_impl::meshing_type::highres_host);
  acoustic_only.add_faces(hs_impl::interface_tag::acoustic_elastic, 4.0, 2);
  const auto [a_host, e_host] = hs_impl::self_host_pair(acoustic_only);
  EXPECT_TRUE(a_host) << "Populated acoustic side should host";
  EXPECT_FALSE(e_host) << "Empty elastic side should not host";

  hs_impl::HostSelection elastic_only(hs_impl::meshing_type::highres_host);
  elastic_only.add_faces(hs_impl::interface_tag::elastic_acoustic, 4.0, 2);
  const auto [a_host2, e_host2] = hs_impl::self_host_pair(elastic_only);
  EXPECT_FALSE(a_host2) << "Empty acoustic side should not host";
  EXPECT_TRUE(e_host2) << "Populated elastic side should host";
}

TEST(NonconformingHostSelection3D, SelfHostBothHost) {
  hs_impl::HostSelection hs(hs_impl::meshing_type::self_host);
  hs.add_faces(hs_impl::interface_tag::acoustic_elastic, 100.0, 1);
  hs.add_faces(hs_impl::interface_tag::elastic_acoustic, 1.0, 50);
  const auto [a_host, e_host] = hs_impl::self_host_pair(hs);
  EXPECT_TRUE(a_host) << "self_host: acoustic side always hosts";
  EXPECT_TRUE(e_host) << "self_host: elastic side always hosts";

  hs_impl::HostSelection defaulted;
  const auto [a_host2, e_host2] = hs_impl::self_host_pair(defaulted);
  EXPECT_TRUE(a_host2) << "Default object behaves as self_host";
  EXPECT_TRUE(e_host2) << "Default object behaves as self_host";
}

TEST(NonconformingHostSelection3D, UnsupportedTypeThrows) {
  hs_impl::HostSelection hs(hs_impl::meshing_type::intersections);
  EXPECT_THROW(hs.self_is_host(hs_impl::interface_tag::acoustic_elastic,
                               hs_impl::interface_tag::elastic_acoustic),
               std::runtime_error)
      << "intersections must throw for the acoustic tag";
  EXPECT_THROW(hs.self_is_host(hs_impl::interface_tag::elastic_acoustic,
                               hs_impl::interface_tag::acoustic_elastic),
               std::runtime_error)
      << "intersections must throw for the elastic tag";
}
