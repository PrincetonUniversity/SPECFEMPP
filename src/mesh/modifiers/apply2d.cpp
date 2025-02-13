#include "mesh/modifiers/modifiers.hpp"
#include <stdexcept>
#include <utility>

template <int ngnod>
static std::pair<type_real, type_real>
interp_control_nodes(const std::vector<type_real> &cnodes, const type_real x,
                     const type_real z);

template <>
std::pair<type_real, type_real>
interp_control_nodes<4>(const std::vector<type_real> &cnodes, const type_real x,
                        const type_real z) {
  throw std::runtime_error("specfem::mesh::modifiers::apply() for 4 contol "
                           "nodes not yet supported.");
  return std::make_pair(0, 0);
}

template <>
std::pair<type_real, type_real>
interp_control_nodes<9>(const std::vector<type_real> &cnodes, const type_real x,
                        const type_real z) {
  return std::make_pair(0, 0);
}

static void subdivide(specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
                      const specfem::mesh::modifiers &modifiers) {
  const int nspec_original = mesh.nspec;
  // find nspec for each material
  std::vector<int> mat_nspec(mesh.materials.n_materials, 0);
  for (int ispec = 0; ispec < nspec_original; ispec++) {
    const auto mat = mesh.materials.material_index_mapping(ispec);
    mat_nspec[mat.database_index]++;
  }
  // recompute new nspec
  int nspec_new = 0;
  for (int imat = 0; imat < mesh.materials.n_materials; imat++) {
    const auto subdivs = modifiers.get_subdivision(imat);
    nspec_new += mat_nspec[imat] * subdivs.first * subdivs.second;
  }
}

template <>
void specfem::mesh::modifiers::apply(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) const {
  subdivide(mesh, *this);

  // forcing boundary skipped
}
