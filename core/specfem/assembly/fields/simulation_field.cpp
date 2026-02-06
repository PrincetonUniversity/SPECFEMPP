#include "dim2/simulation_field.tpp"
#include "dim3/simulation_field.tpp"
#include "specfem/assembly/fields.hpp"
#include "specfem/enums.hpp"
#include <boost/preprocessor.hpp>

#define DIMENSION_TAGS                                                         \
  (specfem::element::dimension_tag::dim2)(specfem::element::dimension_tag::dim3)

#define WAVEFIELD_TYPES                                                        \
  (specfem::simulation::field_type::forward)(                                  \
      specfem::simulation::field_type::adjoint)(                               \
      specfem::simulation::field_type::backward)(                              \
      specfem::simulation::field_type::buffer)

#define INSTANTIATE_SIMULATION_FIELD(r, elem)                                  \
  template class specfem::assembly::simulation_field<                          \
      BOOST_PP_SEQ_ELEM(0, elem), BOOST_PP_SEQ_ELEM(1, elem)>;

#define INSTANTIATE_SIMULATION_COPY_MEMBERS(r, elem)                           \
  template void specfem::assembly::simulation_field<                           \
      BOOST_PP_SEQ_ELEM(0, elem), BOOST_PP_SEQ_ELEM(1, elem)>::copy_to_host(); \
                                                                               \
  template void specfem::assembly::simulation_field<                           \
      BOOST_PP_SEQ_ELEM(0, elem),                                              \
      BOOST_PP_SEQ_ELEM(1, elem)>::copy_to_device();

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_SIMULATION_FIELD,
                              (DIMENSION_TAGS)(WAVEFIELD_TYPES))
