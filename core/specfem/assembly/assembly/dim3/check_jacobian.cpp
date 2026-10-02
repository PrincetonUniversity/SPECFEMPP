#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/globe/radial_flags.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

void specfem::assembly::assembly<
    specfem::element::dimension_tag::dim3>::check_jacobian_matrix() const {
  if (this->element_types.has_element_context()) {
    const int fictitious_flag =
        static_cast<int>(specfem::globe::radial_flag::fictitious_cube);
    for (int ispec = 0; ispec < this->mesh.nspec; ++ispec) {
      if (this->element_types.idoubling(ispec) == fictitious_flag) {
        std::ostringstream message;
        message << "Globe mesh contains fictitious central-cube element "
                << ispec << " (mesh element "
                << this->mesh.h_compute_to_mesh(ispec)
                << ", idoubling=" << fictitious_flag
                << "). The globe database writer must exclude these elements.";
        throw std::runtime_error(message.str());
      }
    }
  }

  const auto result = this->jacobian_matrix.check_small_jacobian();
  if (!result.found) {
    return;
  }

  std::vector<specfem::assembly::small_jacobian_diagnostic> diagnostics =
      result.diagnostics;
  std::sort(diagnostics.begin(), diagnostics.end(),
            [](const auto &left, const auto &right) {
              const type_real left_relative = left.relative_jacobian();
              const type_real right_relative = right.relative_jacobian();
              if (!std::isfinite(left_relative)) {
                return std::isfinite(right_relative);
              }
              if (!std::isfinite(right_relative)) {
                return false;
              }
              return left_relative < right_relative;
            });

  int central_cube_failures = 0;
  if (this->element_types.has_element_context()) {
    for (const auto &diagnostic : diagnostics) {
      if (specfem::globe::is_central_cube(
              this->element_types.idoubling(diagnostic.element_index))) {
        ++central_cube_failures;
      }
    }
  }

  constexpr std::size_t maximum_reported_elements = 10;
  std::ostringstream message;
  message << "Invalid Jacobian mapping in " << diagnostics.size()
          << " element(s): determinants must be finite, strictly positive, and "
             "at least 1e-6 of the element-local mean absolute determinant.\n";
  if (central_cube_failures > 0) {
    message << "Central-cube mapping check failed for " << central_cube_failures
            << " element(s) containing or adjacent to r=0.\n";
  }
  message << "Worst elements:\n";

  const std::size_t report_count =
      std::min(maximum_reported_elements, diagnostics.size());
  for (std::size_t i = 0; i < report_count; ++i) {
    const auto &diagnostic = diagnostics[i];
    const int ispec = diagnostic.element_index;
    message << "  element=" << ispec
            << " mesh_element=" << this->mesh.h_compute_to_mesh(ispec)
            << " gll=(" << diagnostic.ix << ',' << diagnostic.iy << ','
            << diagnostic.iz << ") jacobian=" << diagnostic.jacobian
            << " scale=" << diagnostic.scale
            << " relative=" << diagnostic.relative_jacobian();
    if (this->element_types.has_element_context()) {
      const type_real rmin = this->element_types.rmin(ispec);
      const type_real rmax = this->element_types.rmax(ispec);
      message << " region="
              << specfem::element::to_string(
                     this->element_types.get_region_tag(ispec))
              << " radius=" << type_real(0.5) * (rmin + rmax)
              << " radial_range=[" << rmin << ',' << rmax << ']'
              << " idoubling=" << this->element_types.idoubling(ispec);
    }
    message << '\n';
  }
  if (diagnostics.size() > report_count) {
    message << "  ... " << diagnostics.size() - report_count
            << " additional invalid element(s) omitted\n";
  }

  throw std::runtime_error(message.str());
}
