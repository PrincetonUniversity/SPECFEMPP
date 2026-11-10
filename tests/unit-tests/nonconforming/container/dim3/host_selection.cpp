#include "specfem/assembly/nonconforming_interfaces/dim3/impl/host_selection.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::nonconforming_test::host_selection_impl {

using specfem::assembly::nonconforming_interfaces_impl::check_host_selection;
using specfem::assembly::nonconforming_interfaces_impl::HostSelection;
using specfem::element::boundary_tag;
using specfem::element_coupling::interface_tag;
using specfem::element_coupling::interfacial_meshing_type;

constexpr auto acoustic_elastic = interface_tag::acoustic_elastic;
constexpr auto elastic_acoustic = interface_tag::elastic_acoustic;

/// Selection of one container; the conjugate tag is derived from `tag`.
HostSelection select(const interface_tag tag, const boundary_tag boundary,
                     const int num_faces, const bool self_is_host) {
  const auto conjugate =
      tag == acoustic_elastic ? elastic_acoustic : acoustic_elastic;
  return { tag, conjugate, boundary, num_faces, self_is_host };
}

/// Returns the `what()` of the `std::runtime_error` thrown by the check, or an
/// empty string if it does not throw.
std::string thrown_message(const std::vector<HostSelection> &selections,
                           const interfacial_meshing_type meshing_type) {
  try {
    check_host_selection(selections, meshing_type);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return {};
}

/// Whether `message` contains `text`.
bool contains(const std::string &message, const std::string &text) {
  return message.find(text) != std::string::npos;
}

} // namespace specfem::nonconforming_test::host_selection_impl

TEST(NonconformingHostSelection3D, AcceptsHighresHostWithOneHostTag) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 5, true),
    t::select(t::acoustic_elastic, bt::stacey, 3, true),
    t::select(t::elastic_acoustic, bt::none, 5, false),
    t::select(t::elastic_acoustic, bt::stacey, 3, false),
  };
  EXPECT_NO_THROW(t::check_host_selection(
      selections,
      specfem::element_coupling::interfacial_meshing_type::highres_host));
}

TEST(NonconformingHostSelection3D, AcceptsEmptyContainersWithAnyFlag) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  // The situation that once rejected a valid mesh: empty containers of the
  // host tag carry self_is_host == false.
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 0, false),
    t::select(t::acoustic_elastic, bt::stacey, 0, false),
    t::select(t::acoustic_elastic, bt::acoustic_free_surface, 4, true),
    t::select(t::elastic_acoustic, bt::none, 4, false),
    t::select(t::elastic_acoustic, bt::stacey, 0, true),
  };
  EXPECT_NO_THROW(t::check_host_selection(
      selections,
      specfem::element_coupling::interfacial_meshing_type::highres_host));
}

TEST(NonconformingHostSelection3D, AcceptsSelfHostWithEveryContainerHost) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 5, true),
    t::select(t::acoustic_elastic, bt::stacey, 3, true),
    t::select(t::elastic_acoustic, bt::none, 5, true),
    t::select(t::elastic_acoustic, bt::stacey, 3, true),
  };
  EXPECT_NO_THROW(t::check_host_selection(
      selections,
      specfem::element_coupling::interfacial_meshing_type::self_host));
}

TEST(NonconformingHostSelection3D, AcceptsNoContainers) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  const std::vector<t::HostSelection> selections;
  EXPECT_NO_THROW(t::check_host_selection(
      selections,
      specfem::element_coupling::interfacial_meshing_type::highres_host));
  EXPECT_NO_THROW(t::check_host_selection(
      selections,
      specfem::element_coupling::interfacial_meshing_type::self_host));
}

TEST(NonconformingHostSelection3D, AcceptsOnlyOneInterfaceTagPopulated) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  // Without any container of the conjugate tag, rules (b) and (c) have
  // nothing to compare, so neither a host nor a non-host lone tag is rejected.
  const auto highres =
      specfem::element_coupling::interfacial_meshing_type::highres_host;
  const std::vector<t::HostSelection> host_only = {
    t::select(t::acoustic_elastic, bt::none, 5, true),
    t::select(t::elastic_acoustic, bt::none, 0, false),
  };
  const std::vector<t::HostSelection> non_host_only = {
    t::select(t::acoustic_elastic, bt::none, 5, false),
    t::select(t::elastic_acoustic, bt::none, 0, false),
  };
  EXPECT_NO_THROW(t::check_host_selection(host_only, highres));
  EXPECT_NO_THROW(t::check_host_selection(non_host_only, highres));
}

TEST(NonconformingHostSelection3D, RejectsSameTagDisagreement) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  const auto highres =
      specfem::element_coupling::interfacial_meshing_type::highres_host;
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 5, true),
    t::select(t::acoustic_elastic, bt::stacey, 3, false),
    t::select(t::elastic_acoustic, bt::none, 5, false),
  };
  EXPECT_THROW(t::check_host_selection(selections, highres),
               std::runtime_error);
  const std::string message = t::thrown_message(selections, highres);
  EXPECT_TRUE(
      t::contains(message, "containers of the same interface tag disagree"))
      << message;
  EXPECT_TRUE(t::contains(message, "highres-host")) << message;
  EXPECT_TRUE(t::contains(message, "interface = acoustic_elastic")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 5")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 3")) << message;
  EXPECT_TRUE(t::contains(message, "self_is_host = true")) << message;
  EXPECT_TRUE(t::contains(message, "self_is_host = false")) << message;
}

TEST(NonconformingHostSelection3D, RejectsBothTagsHost) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  const auto highres =
      specfem::element_coupling::interfacial_meshing_type::highres_host;
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 5, true),
    t::select(t::elastic_acoustic, bt::stacey, 7, true),
  };
  EXPECT_THROW(t::check_host_selection(selections, highres),
               std::runtime_error);
  const std::string message = t::thrown_message(selections, highres);
  EXPECT_TRUE(
      t::contains(message, "an interface tag and its conjugate are both host"))
      << message;
  EXPECT_TRUE(t::contains(message, "highres-host")) << message;
  EXPECT_TRUE(t::contains(message, "interface = acoustic_elastic")) << message;
  EXPECT_TRUE(t::contains(message, "interface = elastic_acoustic")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 5")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 7")) << message;
}

TEST(NonconformingHostSelection3D, RejectsNeitherTagHost) {
  namespace t = specfem::nonconforming_test::host_selection_impl;
  using bt = specfem::element::boundary_tag;
  const auto highres =
      specfem::element_coupling::interfacial_meshing_type::highres_host;
  const std::vector<t::HostSelection> selections = {
    t::select(t::acoustic_elastic, bt::none, 5, false),
    t::select(t::elastic_acoustic, bt::stacey, 7, false),
  };
  EXPECT_THROW(t::check_host_selection(selections, highres),
               std::runtime_error);
  const std::string message = t::thrown_message(selections, highres);
  EXPECT_TRUE(t::contains(message,
                          "neither an interface tag nor its conjugate is host"))
      << message;
  EXPECT_TRUE(t::contains(message, "highres-host")) << message;
  EXPECT_TRUE(t::contains(message, "interface = acoustic_elastic")) << message;
  EXPECT_TRUE(t::contains(message, "interface = elastic_acoustic")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 5")) << message;
  EXPECT_TRUE(t::contains(message, "pairs = 7")) << message;
}
