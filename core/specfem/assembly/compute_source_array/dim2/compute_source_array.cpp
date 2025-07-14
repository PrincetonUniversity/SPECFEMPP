#include "specfem/assembly/compute_source_array.hpp"
#include "../impl/compute_source_array.hpp"
#include "compute_source_array.hpp"
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem_setup.hpp"

template <>
void specfem::assembly::compute_source_array<specfem::dimension::type::dim2>(
    const std::shared_ptr<specfem::sources::source> &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  // Marker to indicate that the source array has been computed
  bool computed_source = false;

  // Get the compute source array functions for the specific dimension.
  compute_source_array_impl::SourceArrayFunctionVector
      compute_source_array_functions =
          compute_source_array_impl::get_compute_source_array_functions();

  // Loop through the function that compute the source.
  for (const auto &compute_source_array : compute_source_array_functions) {
    computed_source = compute_source_array(source, mesh, jacobian_matrix,
                                           element_types, source_array);
    if (computed_source) {
      return;
    }
  }

  KOKKOS_ABORT_WITH_LOCATION(
      "For this source type, the compute_source_array "
      "function is not implemented. Please implement "
      "it in the source class or provide a specialization "
      "for this function.");
}
