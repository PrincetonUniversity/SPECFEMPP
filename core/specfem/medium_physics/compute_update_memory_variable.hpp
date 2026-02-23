#include "specfem/medium/dim3/elastic/isotropic/update_memory_variable.hpp"

namespace specfem::medium_physics {

template <typename Tags, bool UseSIMD,
          std::enable_if_t<
              (Tags::dimension_tag == specfem::element::dimension_tag::dim3) &&
                  (Tags::medium_tag == specfem::element::medium_tag::elastic) &&
                  (Tags::attenuation_tag ==
                   specfem::element::attenuation_tag::constant_isotropic),
              int> = 0>
void compute_update_memory_variable(
    specfem::point::memory<Tags::dimension_tag, Tags::medium_tag,
                           Tags::attenuation_tag, UseSIMD>
        &point_memory_variable,
    specfem::point::stress<Tags::dimension_tag, Tags::medium_tag, UseSIMD>
        &point_stress) {

  impl::compute_update_memory_variable<Tags, UseSIMD>(point_memory_variable,
                                                      point_stress);
}

} // namespace specfem::medium_physics
