#include "specfem/assembly/info.hpp"
#include "specfem/assembly/info.tpp"

template <>
std::string
specfem::assembly::Info<specfem::element::dimension_tag::dim2>::string() const {
  std::ostringstream oss;
  oss << "Mesh Information (2D):\n";
  oss << " Domain X: ............... [" << domain_bounds.x().min << ", "
      << domain_bounds.x().max << "]\n";
  oss << " Domain Z: ............... [" << domain_bounds.z().min << ", "
      << domain_bounds.z().max << "]\n";
  oss << " VP: ..................... [" << vp.min << ", " << vp.max << "]\n";
  oss << " VS: ..................... [" << vs.min << ", " << vs.max << "]\n";
  oss << " V: .......................[" << v.min << ", " << v.max << "]\n";
  oss << " Rho: .................... [" << rho.min << ", " << rho.max << "]\n";
  oss << " Element Size: ........... [" << element_size.min << ", "
      << element_size.max << "]\n";
  oss << " GLL Distance: ........... [" << gll_distance.min << ", "
      << gll_distance.max << "]\n";
  oss << " Largest Minimum Period: . " << largest_minimum_period << "\n";
  oss << " Suggested Time Step: .... " << suggested_time_step << "\n";
  return oss.str();
}

template <>
std::string
specfem::assembly::Info<specfem::element::dimension_tag::dim3>::string() const {
  std::ostringstream oss;
  oss << "Mesh Information (3D):\n";
  oss << " Domain X: ............... [" << domain_bounds.x().min << ", "
      << domain_bounds.x().max << "]\n";
  oss << " Domain Y: ............... [" << domain_bounds.y().min << ", "
      << domain_bounds.y().max << "]\n";
  oss << " Domain Z: ............... [" << domain_bounds.z().min << ", "
      << domain_bounds.z().max << "]\n";
  oss << " VP: ..................... [" << vp.min << ", " << vp.max << "]\n";
  oss << " VS: ..................... [" << vs.min << ", " << vs.max << "]\n";
  oss << " V: .......................[" << v.min << ", " << v.max << "]\n";
  oss << " Rho: .................... [" << rho.min << ", " << rho.max << "]\n";
  oss << " Element Size: ........... [" << element_size.min << ", "
      << element_size.max << "]\n";
  oss << " GLL Distance: ........... [" << gll_distance.min << ", "
      << gll_distance.max << "]\n";
  oss << " Largest Minimum Period: . " << largest_minimum_period << "\n";
  oss << " Suggested Time Step: .... " << suggested_time_step << "\n";
  return oss.str();
}

template specfem::assembly::Info<specfem::element::dimension_tag::dim2>::Info(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim2>
        &,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &);
template specfem::assembly::Info<specfem::element::dimension_tag::dim3>::Info(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &);
