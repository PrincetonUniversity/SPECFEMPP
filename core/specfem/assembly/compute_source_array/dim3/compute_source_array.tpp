#pragma once

#include "specfem/assembly/compute_source_array.hpp"
#include "enumerations/dimension.hpp"
#include "impl/compute_source_array_from_tensor.hpp"
#include "impl/compute_source_array_from_vector.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template<typename SourceArrayViewType>
void specfem::assembly::compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim3> > &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    SourceArrayViewType &source_array) {

  // Ensure source_array is a 4D view
  static_assert(SourceArrayViewType::rank() == 4, "Source array must be in rank 4.");


  switch (source->get_source_type()) {
  case specfem::sources::source_type::vector_source: {

    // Cast to derived class to access specific methods
    auto vector_source = static_cast<const specfem::sources::vector_source<
        specfem::dimension::type::dim3> *>(source.get());

    if (!vector_source) {
      KOKKOS_ABORT_WITH_LOCATION(
          "Source is not of vector type. Cannot compute vector source "
          "array.");
    }

    specfem::assembly::compute_source_array_impl::from_vector(*vector_source,
                                                              source_array);
    break;
  }
  // case specfem::sources::source_type::tensor_source: {

  //   // Cast to derived class to access specific methods
  //   auto tensor_source = static_cast<const specfem::sources::tensor_source<
  //       specfem::dimension::type::dim3> *>(source.get());

  //   if (!tensor_source) {
  //     KOKKOS_ABORT_WITH_LOCATION(
  //         "Source is not of tensor type. Cannot compute tensor source "
  //         "array.");
  //   }

  //   specfem::assembly::compute_source_array_impl::from_tensor(
  //       *tensor_source, mesh, jacobian_matrix, source_array);
  //   break;
  // }
  default:
    // Handle unsupported source types
    KOKKOS_ABORT_WITH_LOCATION(
        "Unsupported source type for compute_source_array.");
  }
  return;
}
