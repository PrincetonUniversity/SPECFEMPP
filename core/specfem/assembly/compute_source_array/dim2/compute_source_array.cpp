#include "specfem/assembly/compute_source_array.hpp"
#include "enumerations/dimension.hpp"
#include "impl/compute_source_array_from_tensor.hpp"
#include "impl/compute_source_array_from_vector.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"

void specfem::assembly::compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    specfem::kokkos::HostView3d<type_real> source_array) {

  switch (source->get_source_type()) {
  case specfem::sources::source_type::vector_source: {

    // Cast to derived class to access specific methods
    auto vector_source = static_cast<const specfem::sources::vector_source<
        specfem::dimension::type::dim2> *>(source.get());

    if (!vector_source) {
      KOKKOS_ABORT_WITH_LOCATION(
          "Source is not of vector type. Cannot compute vector source "
          "array.");
    }

    specfem::assembly::compute_source_array_impl::from_vector(*vector_source,
                                                              source_array);
    break;
  }
  case specfem::sources::source_type::tensor_source: {

    // Cast to derived class to access specific methods
    auto tensor_source = static_cast<const specfem::sources::tensor_source<
        specfem::dimension::type::dim2> *>(source.get());

    if (!tensor_source) {
      KOKKOS_ABORT_WITH_LOCATION(
          "Source is not of tensor type. Cannot compute tensor source "
          "array.");
    }

    specfem::assembly::compute_source_array_impl::from_tensor(
        *tensor_source, mesh, jacobian_matrix, source_array);
    break;
  }
  default:
    // Handle unsupported source types
    KOKKOS_ABORT_WITH_LOCATION(
        "Unsupported source type for compute_source_array.");
  }
  return;
}
