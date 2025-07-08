#include "enumerations/interface.hpp"
#include "specfem/point/jacobian_matrix.hpp"

namespace specfem::assembly::impl::boundaries {

bool is_on_boundary(specfem::enums::boundaries::type type, int iz, int ix,
                    int ngllz, int ngllx);

std::tuple<std::array<type_real, 2>, type_real> get_boundary_edge_and_weight(
    specfem::enums::boundaries::type type,
    const std::array<type_real, 2> &weights,
    const specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                          false> &point_jacobian_matrix);

} // namespace specfem::assembly::impl::boundaries
