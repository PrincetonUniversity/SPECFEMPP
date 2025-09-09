#pragma once

#include "quadrature/interface.hpp"
#include "quadrature/quadrature.hpp"
#include "specfem/assembly.hpp"

#include <vector>

namespace specfem::assembly {
namespace nonconforming_interfaces {

std::vector<std::pair<type_real, type_real> > compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const boost::graph_traits<specfem::mesh::adjacency_graph<
        specfem::dimension::type::dim2>::GraphType>::edge_descriptor &edge,
    const specfem::quadrature::quadrature &mortar_quadrature);
} // namespace nonconforming_interfaces
} // namespace specfem::assembly
