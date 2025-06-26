
#include "jacobian_matrix.hpp"
#include "jacobian_matrix.tpp"
// Explicit template instantiation

template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                                false, false>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                                true, false>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                                false, true>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                                true, true>;

template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
                                                false, false>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
                                                true, false>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
                                                false, true>;
template struct specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
                                                true, true>;
