#pragma once

#include "enumerations/medium.hpp"

namespace specfem {
namespace element {
namespace impl {

template <specfem::element::medium_tag MediumTag,
          typename std::enable_if_t<
              MediumTag == specfem::element::medium_tag::elastic_sh ||
                  MediumTag == specfem::element::medium_tag::elastic_sv,
              int> = 0>
class is_elastic_2d {};

} // namespace impl
} // namespace element
} // namespace specfem
