#pragma once

#include "enumerations/interface.hpp"
#include "medium/kernels_container.hpp"
#include "specfem/assembly/assembly/value_containers.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <>
struct kernels<specfem::dimension::type::dim2>
    : public impl::value_containers<specfem::dimension::type::dim2,
                                    specfem::medium::kernels_container> {
public:
  /**
   * @name Constructors
   *
   */

  ///@{
  /**
   * @brief Default constructor
   *
   */
  kernels() = default;

  /**
   * @brief Construct a new kernels object
   *
   * @param nspec Total number of spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   * @param mapping mesh to compute mapping
   * @param tags Tags for every element in spectral element mesh
   */
  kernels(const int nspec, const int ngllz, const int ngllx,
          const specfem::assembly::element_types<dimension_tag> &element_types);
  ///@}

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    impl::value_containers<dimension_tag,
                           specfem::medium::kernels_container>::copy_to_host();
  }

  void copy_to_device() {
    impl::value_containers<
        dimension_tag, specfem::medium::kernels_container>::copy_to_device();
  }
};

} // namespace specfem::assembly
