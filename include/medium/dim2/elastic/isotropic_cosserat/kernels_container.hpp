#pragma once

#include "medium/impl/data_container.hpp"

namespace specfem::medium::kernels {

template <specfem::element::medium_tag MediumTag>
struct data_container<
    MediumTag, specfem::element::property_tag::isotropic_cosserat,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic_cosserat;

  DATA_CONTAINER(rho) ///< density @f$ \rho @f$
};

} // namespace specfem::medium::kernels
