#pragma once

#include "specfem/constants.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/element_coupling/tags.hpp"

#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace specfem::assembly::nonconforming_interfaces_impl {

/**
 * @brief Decides, once per interface direction, which side hosts the
 * quadrature of a nonconforming interface.
 *
 * Every touching face pair is filed under both interface tags
 * (acoustic_elastic and elastic_acoustic). Under
 * interfacial_meshing_type::highres_host the side with the smaller mean face
 * length hosts the quadrature. The decision must be the same for every
 * container of an interface tag, so the face lengths of all containers of the
 * tag are accumulated here first and the decision is made from the totals.
 */
class HostSelection {
public:
  /** @brief Default: self_host (every side hosts its own faces). */
  HostSelection() = default;

  /**
   * @brief Constructs a selection for the given meshing type.
   * @param type interfacial meshing type
   */
  explicit HostSelection(
      specfem::element_coupling::interfacial_meshing_type type)
      : type_(type) {}

  /**
   * @brief Adds unique self faces of one container to the total of its
   * interface tag.
   * @param tag interface tag of the container
   * @param length_sum sum of the characteristic lengths of the faces
   * @param count number of faces
   */
  void add_faces(specfem::element_coupling::interface_tag tag,
                 type_real length_sum, int count) {
    auto &total = totals_[tag];
    total.length_sum += length_sum;
    total.count += count;
  }

  /**
   * @brief Whether the self side of `tag` hosts the quadrature.
   *
   * With self_host this is always true. With highres_host the side with the
   * smaller mean face length hosts; the mean of a tag with no faces is the
   * largest representable value, so an empty direction never hosts unless both
   * are empty, and equal means are broken by the order of the tags. Both
   * directions compare the same two numbers, so exactly one of `tag` and
   * `conjugate_tag` is the host.
   *
   * @param tag interface tag of the self side
   * @param conjugate_tag the conjugate interface tag
   * @return true if the self side of `tag` hosts the quadrature
   * @throws std::runtime_error for any other meshing type
   */
  bool
  self_is_host(specfem::element_coupling::interface_tag tag,
               specfem::element_coupling::interface_tag conjugate_tag) const {
    switch (type_) {
    case specfem::element_coupling::interfacial_meshing_type::self_host:
      return true;
    case specfem::element_coupling::interfacial_meshing_type::highres_host: {
      const type_real self = mean(tag);
      const type_real conjugate = mean(conjugate_tag);
      return self < conjugate || (self == conjugate && tag < conjugate_tag);
    }
    default:
      std::ostringstream oss;
      oss << "Unsupported interface meshing type: "
          << specfem::element_coupling::to_string(type_);
      throw std::runtime_error(oss.str());
    }
  }

private:
  /** @brief Running total of face lengths and face count of one tag. */
  struct Total {
    type_real length_sum = 0;
    int count = 0;
  };

  /** @brief Mean face length of `tag`, or the largest value if it has none. */
  type_real mean(specfem::element_coupling::interface_tag tag) const {
    const auto it = totals_.find(tag);
    if (it == totals_.end() || it->second.count == 0) {
      return std::numeric_limits<type_real>::max();
    }
    return it->second.length_sum / it->second.count;
  }

  specfem::element_coupling::interfacial_meshing_type type_ =
      specfem::element_coupling::interfacial_meshing_type::self_host;
  std::map<specfem::element_coupling::interface_tag, Total> totals_{};
};

} // namespace specfem::assembly::nonconforming_interfaces_impl
