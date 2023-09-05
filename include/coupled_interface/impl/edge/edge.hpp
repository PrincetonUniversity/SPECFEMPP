#ifndef _COUPLED_INTERFACE_IMPL_EDGE_HPP
#define _COUPLED_INTERFACE_IMPL_EDGE_HPP

namespace specfem {
namespace coupled_interface {
namespace impl {

using self_iterator_type = decltype(
    specfem::compute::coupled_interfaces::iterator::get_points_along_the_edges<
        specfem::compute::coupled_interfaces::iterator::enums::edge::self>);

using coupled_iterator_type = decltype(
    specfem::compute::coupled_interfaces::iterator::get_points_along_the_edges<
        specfem::compute::coupled_interfaces::iterator::enums::edge::coupled>);

self_iterator_type &self_iterator =
    specfem::compute::coupled_interfaces::iterator::get_points_along_the_edges<
        specfem::compute::coupled_interfaces::iterator::enums::edge::self>;

coupled_iterator_type &coupled_iterator =
    specfem::compute::coupled_interfaces::iterator::get_points_along_the_edges<
        specfem::compute::coupled_interfaces::iterator::enums::edge::coupled>;

template <class self_domain, class coupled_domain> class edge {};
} // namespace impl
} // namespace coupled_interface
} // namespace specfem

#endif // _COUPLED_INTERFACE_IMPL_EDGE_HPP
