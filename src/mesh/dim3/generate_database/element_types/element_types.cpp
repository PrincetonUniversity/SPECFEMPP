#include "mesh/dim3/generate_database/element_types/element_types.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <iostream>

void specfem::mesh::element_types<specfem::dimension::type::dim3>::set_elements(
    Kokkos::View<int *, Kokkos::HostSpace> &ispec_type_in) {

  // Create the vectors
  // std::vector<int> ispec_elastic;
  // std::vector<int> ispec_acoustic;
  // std::vector<int> ispec_poroelastic;

  // Initialize the number of elements
  this->nelastic = 0;
  this->nacoustic = 0;
  this->nporoelastic = 0;

  // Initialize the vectors
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec_type_in[ispec] == 0) {
      this->ispec_type(ispec) = specfem::element::medium_tag::acoustic;
      this->ispec_acoustic(nacoustic) = ispec;
      this->nacoustic++;
    } else if (ispec_type_in[ispec] == 1) {
      this->ispec_type(ispec) = specfem::element::medium_tag::elastic;
      this->ispec_elastic(nelastic) = ispec;
      this->nelastic++;
    } else if (ispec_type_in[ispec] == 2) {
      this->ispec_type(ispec) = specfem::element::medium_tag::poroelastic;
      this->ispec_poroelastic(nporoelastic) = ispec;
      this->nporoelastic++;
    } else {
      std::ostringstream message;
      message << "Error: Spectral element " << ispec
              << " is not assigned to any material";
      throw std::runtime_error(message.str());
    }
  };
}

std::string
specfem::mesh::element_types<specfem::dimension::type::dim3>::print() const {

  std::ostringstream message;
  message << "Number of acoustic elements: " << nacoustic << ".\n";
  message << "Number of elastic elements: " << nelastic << ".\n";
  message << "Number of poroelastic elements: " << nporoelastic << ".\n";

  return message.str();
}

std::string specfem::mesh::element_types<specfem::dimension::type::dim3>::print(
    const int ispec) const {

  std::ostringstream message;

  if (ispec >= nspec) {
    std::ostringstream message;
    message << "Error: Spectral element " << ispec
            << " does not exist.\nNumber of spectral elements is :" << nspec
            << ". "
            << "(" << __FILE__ << ":" << __LINE__ << ")\n";
    throw std::runtime_error(message.str());
  }

  message << "Element " << ispec << " is ";
  message << specfem::element::to_string(ispec_type(ispec)) << ".\n";

  return message.str();
}

template <specfem::element::medium_tag MediumTag>
std::string specfem::mesh::element_types<specfem::dimension::type::dim3>::print(
    const int i) const {

  std::ostringstream message;

  if (MediumTag == specfem::element::medium_tag::elastic) {
    if (i >= nelastic) {
      std::ostringstream message;
      message << "Error: Elastic element " << i
              << " does not exist.\nNumber "
                 "of elastic elements is :"
              << nelastic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }
    message << "Elastic element " << i << " is global element "
            << ispec_elastic(i) << ".\n";
  } else if (MediumTag == specfem::element::medium_tag::acoustic) {

    if (i >= nacoustic) {
      std::ostringstream message;
      message << "Error: Acoustic element " << i
              << " does not exist.\nNumber "
                 " of acoustic elements: "
              << nacoustic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }

    message << "Acoustic element " << i << " is global element "
            << ispec_acoustic(i) << ".\n";

  } else if (MediumTag == specfem::element::medium_tag::poroelastic) {

    if (i >= nporoelastic) {
      std::ostringstream message;
      message << "Error: Poroelastic element " << i
              << " does not exist.\nNumber "
                 " of poroelastic elements: "
              << nporoelastic << ". "
              << "(" << __FILE__ << ":" << __LINE__ << ")\n";
      throw std::runtime_error(message.str());
    }

    message << "Poroelastic element " << i << " is global element "
            << ispec_poroelastic(i) << ".\n";
  }

  return message.str();
}

// Explicit instantiations
template std::string
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::elastic>(const int) const;
template std::string
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::acoustic>(const int) const;
template std::string
specfem::mesh::element_types<specfem::dimension::type::dim3>::print<
    specfem::element::medium_tag::poroelastic>(const int) const;
