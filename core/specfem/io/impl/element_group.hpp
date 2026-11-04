#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/io/impl/coordinates.hpp"
#include "specfem/io/impl/sub_block.hpp"
#include <string>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief Transform that leaves the staged dataset untouched.
 */
struct NoTransform {
  /**
   * @brief No-op.
   *
   * @tparam ViewType Scratch view type
   */
  template <typename ViewType>
  void operator()(ViewType &, const std::string &) const {}
};

/**
 * @brief Write the coordinates and every host view of a container for one
 *        element group.
 *
 * The container may span more elements than the group. The group's sub-block
 * of each view is staged into a plain row-major scratch view (see
 * sub_block.hpp) before being written.
 *
 * @tparam GroupType Output library group type
 * @tparam DimensionTag Dimension of the mesh
 * @tparam RangeType Element index range type
 * @tparam ContainerType Container providing `for_each_host_view` and
 *         `element_range`
 * @tparam Transform Callable `(scratch, view_name)` applied before writing
 * @param group File group receiving the datasets
 * @param mesh Assembly mesh providing coordinates
 * @param element_range Elements of the group
 * @param container Value container holding the data
 * @param transform Applied to each scratch view before it is written
 */
template <typename GroupType, specfem::element::dimension_tag DimensionTag,
          typename RangeType, typename ContainerType,
          typename Transform = NoTransform>
void write_element_group(GroupType &group,
                         const specfem::assembly::mesh<DimensionTag> &mesh,
                         const RangeType &element_range,
                         const ContainerType &container,
                         Transform &&transform = Transform{}) {
  specfem::io::impl::write_coordinates(group, mesh, element_range);

  const int offset =
      element_range.begin_index() - container.element_range.begin_index();
  const int count = element_range.size();

  container.for_each_host_view([&](const auto view,
                                   const std::string view_name) {
    auto scratch =
        specfem::io::impl::extract_sub_block(view, view_name, offset, count);
    transform(scratch, view_name);
    group.createDataset(view_name, scratch).write();
  });
}

/**
 * @brief Read every host view of a container for one element group.
 *
 * Inverse of write_element_group: each dataset is read into a plain scratch
 * view and unpacked into the group's sub-block of the container view.
 *
 * @tparam GroupType Input library group type
 * @tparam RangeType Element index range type
 * @tparam ContainerType Container providing `for_each_host_view` and
 *         `element_range`
 * @param group File group holding the datasets
 * @param element_range Elements of the group
 * @param container Value container receiving the data
 */
template <typename GroupType, typename RangeType, typename ContainerType>
void read_element_group(GroupType &group, const RangeType &element_range,
                        const ContainerType &container) {
  const int offset =
      element_range.begin_index() - container.element_range.begin_index();
  const int count = element_range.size();

  container.for_each_host_view([&](const auto view,
                                   const std::string view_name) {
    auto scratch = specfem::io::impl::make_sub_block(view, view_name, count);
    group.openDataset(view_name, scratch).read();
    specfem::io::impl::insert_sub_block(view, scratch, offset);
  });
}

} // namespace impl
} // namespace io
} // namespace specfem
