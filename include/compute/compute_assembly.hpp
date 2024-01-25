#ifndef _COMPUTE_ASSEMBLY_HPP
#define _COMPUTE_ASSEMBLY_HPP

namespace specfem {
namespace compute {

struct assembly {
  specfem::compute::mesh mesh;
  specfem::quadrature::quadrature quadrature;
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::compute::properties properties;
  specfem::compute::sources sources;
  specfem::compute::receivers receivers;
  specfem::compute::boundaries boundaries;
  specfem::compute::coupled_interfaces coupled_interfaces;

  assembly(
      const specfem::mesh::mesh &mesh,
      const std::vector<std::shared_ptr<specfem::source::source> > &sources,
      const int nspec);
}

} // namespace compute
} // namespace specfem

#endif
